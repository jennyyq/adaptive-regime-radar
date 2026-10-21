"""Stage 7: causal portfolio risk estimation.

Return units
  The split files hold daily LOG returns r = log(P_t / P_t-1) of adjusted closes. Portfolio arithmetic needs SIMPLE
  returns R = exp(r) - 1, because a portfolio's simple return is the weighted sum of asset simple returns,
  R_p,t = sum_i w_i,t-1 R_i,t, whereas a weighted sum of log returns is not the portfolio log return.
  Covariances are estimated on daily simple returns and annualized by 252.

Estimators (all use only the rows passed in; callers pass returns through the decision date t)
  sample       unbiased sample covariance
  ledoit_wolf  Ledoit & Wolf (2004) shrinkage toward a scaled identity, analytic optimal intensity
               (the same estimator as sklearn.covariance.LedoitWolf; implemented here so the formula is explicit)
  ewma         exponentially weighted covariance with a fixed half-life, weights normalized to sum to one,
               demeaned with the weighted mean; effective sample size (sum w)^2 / sum w^2 is reported
PSD stabilization: symmetrize, then floor eigenvalues at psd_floor_relative * mean variance; the number of
floored eigenvalues is reported (an estimator that needs flooring is flagged).
"""

from __future__ import annotations

STAGE7_CODE_VERSION = "stage7-2026-10-21"  # must match scripts/09_optimize_portfolio.py

from dataclasses import dataclass

import numpy as np
import pandas as pd

ANNUALIZATION = 252


def to_simple(log_returns: pd.DataFrame) -> pd.DataFrame:
    """Exact conversion of log returns to simple returns."""
    return np.expm1(log_returns)


@dataclass
class CovEstimate:
    cov_daily: np.ndarray
    method: str
    n_obs: int
    shrinkage: float = np.nan
    effective_n: float = np.nan
    floored_eigenvalues: int = 0

    @property
    def cov_annual(self) -> np.ndarray:
        return self.cov_daily * ANNUALIZATION


def sample_cov(x: np.ndarray) -> np.ndarray:
    return np.cov(x, rowvar=False, ddof=1)


def ledoit_wolf(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf (2004) shrinkage of the (1/n, demeaned) sample covariance toward mu*I, mu = trace(S)/p.
    Returns (covariance, shrinkage intensity in [0, 1])."""
    x = np.asarray(x, float)
    n, p = x.shape
    xc = x - x.mean(axis=0)
    s = xc.T @ xc / n
    mu = np.trace(s) / p
    target = mu * np.eye(p)
    d2 = np.sum((s - target) ** 2)                       # ||S - mu I||_F^2
    x2 = xc ** 2
    b2_bar = (np.sum(x2.T @ x2) / n - np.sum(s ** 2)) / n  # average ||x x' - S||_F^2 / n
    b2 = min(b2_bar, d2)
    delta = b2 / d2 if d2 > 0 else 1.0
    return delta * target + (1 - delta) * s, float(delta)


def ewma_cov(x: np.ndarray, halflife: float) -> tuple[np.ndarray, float]:
    x = np.asarray(x, float)
    n = len(x)
    lam = 0.5 ** (1.0 / halflife)
    w = lam ** np.arange(n - 1, -1, -1)                  # most recent observation has weight 1
    w = w / w.sum()
    m = w @ x
    xc = x - m
    cov = (xc * w[:, None]).T @ xc
    ess = 1.0 / np.sum(w ** 2)
    return cov, float(ess)


def psd_stabilize(cov: np.ndarray, floor_relative: float = 1e-8) -> tuple[np.ndarray, int]:
    c = 0.5 * (cov + cov.T)
    vals, vecs = np.linalg.eigh(c)
    floor = floor_relative * float(np.mean(np.diag(c)))
    n_floor = int((vals < floor).sum())
    if n_floor:
        vals = np.maximum(vals, floor)
        c = (vecs * vals) @ vecs.T
        c = 0.5 * (c + c.T)
    return c, n_floor


def estimate_cov(returns: pd.DataFrame, method: str, lookback, floor_relative: float = 1e-8, halflife: float = 63,
                 min_obs: int = 2) -> CovEstimate:
    """Covariance from the LAST `lookback` rows of `returns` (or all rows if 'expanding'). Callers must pass only
    returns observed through the decision date."""
    x = returns.to_numpy(float)
    if lookback != "expanding":
        x = x[-int(lookback):]
    x = x[np.isfinite(x).all(axis=1)]
    if len(x) < min_obs:
        raise ValueError(f"not enough observations ({len(x)}) for {method}")
    shrink, ess = np.nan, float(len(x))
    if method == "sample":
        cov = sample_cov(x)
    elif method == "ledoit_wolf":
        cov, shrink = ledoit_wolf(x)
    elif method == "ewma":
        cov, ess = ewma_cov(x, halflife)
    else:
        raise ValueError(f"unknown covariance method {method}")
    cov, n_floor = psd_stabilize(cov, floor_relative)
    return CovEstimate(cov, method, len(x), shrink, ess, n_floor)


def cov_diagnostics(cov_annual: np.ndarray, assets: list[str]) -> dict:
    vals = np.linalg.eigvalsh(cov_annual)
    vol = np.sqrt(np.diag(cov_annual))
    corr = cov_annual / np.outer(vol, vol)
    iu = np.triu_indices(len(assets), 1)
    out = {"min_eigenvalue": float(vals.min()), "max_eigenvalue": float(vals.max()),
           "condition_number": float(vals.max() / vals.min()) if vals.min() > 0 else np.inf,
           "mean_pairwise_correlation": float(corr[iu].mean()), "max_pairwise_correlation": float(corr[iu].max()),
           "min_pairwise_correlation": float(corr[iu].min())}
    for a, v in zip(assets, vol):
        out[f"vol_{a}"] = float(v)
    return out


def portfolio_vol(w: np.ndarray, cov: np.ndarray) -> float:
    return float(np.sqrt(max(w @ cov @ w, 0.0)))


def risk_contributions(w: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Marginal contribution MCR_i = (Sigma w)_i / sigma_p and component contribution CCR_i = w_i * MCR_i
    (sum CCR = sigma_p)."""
    s = portfolio_vol(w, cov)
    mcr = cov @ w / s if s > 0 else np.zeros_like(w)
    return mcr, w * mcr


def historical_var_cvar(scenarios: np.ndarray, w: np.ndarray, alpha: float = 0.95) -> tuple[float, float]:
    """Empirical VaR / CVaR of portfolio LOSS L = -R_p over equally weighted historical scenarios of simple returns.
    VaR = alpha-quantile of L (lower empirical quantile convention: the ceil(alpha*n)-th smallest loss);
    CVaR = Rockafellar-Uryasev value min_z z + E[(L - z)+] / (1 - alpha), evaluated at z = VaR (exact for the
    empirical distribution)."""
    loss = -(np.asarray(scenarios, float) @ np.asarray(w, float))
    n = len(loss)
    srt = np.sort(loss)
    k = int(np.ceil(alpha * n)) - 1
    var = float(srt[min(max(k, 0), n - 1)])
    cvar = var + float(np.mean(np.maximum(loss - var, 0.0))) / (1 - alpha)
    return var, cvar


def herfindahl(w: np.ndarray) -> float:
    w = np.asarray(w, float)
    return float(np.sum(w ** 2))


def effective_n(w: np.ndarray) -> float:
    h = herfindahl(w)
    return 1.0 / h if h > 0 else np.nan


def tracking_error(w: np.ndarray, wb: np.ndarray, cov: np.ndarray) -> float:
    d = np.asarray(w) - np.asarray(wb)
    return float(np.sqrt(max(d @ cov @ d, 0.0)))
