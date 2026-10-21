"""Stage 8: uncertainty for serially dependent daily returns (settings frozen in configs/stage8_protocol.json).

* Newey-West (Bartlett) HAC standard error of a mean, lag L = floor(4 (T/100)^(2/9)).
* OLS with Newey-West covariance (alpha / beta of excess returns on the reference's excess returns).
* Stationary bootstrap (Politis & Romano 1994), paired: one index path resamples every portfolio and BIL jointly,
  so cross-sectional dependence is preserved. Percentile intervals.
* Holm step-down adjustment for a family of p-values.
"""

from __future__ import annotations

STAGE8_CODE_VERSION = "stage8-2026-10-21"  # must match scripts/10_backtest_portfolios.py

import math

import numpy as np
import pandas as pd

ANN = 252


def nw_lag(n: int) -> int:
    return int(math.floor(4 * (n / 100.0) ** (2.0 / 9.0)))


def nw_variance_of_mean(x: np.ndarray, lag: int | None = None) -> float:
    x = np.asarray(x, float)
    n = len(x)
    lag = nw_lag(n) if lag is None else lag
    u = x - x.mean()
    s = float(u @ u) / n
    for k in range(1, lag + 1):
        s += 2 * (1 - k / (lag + 1)) * float(u[k:] @ u[:-k]) / n
    return s / n


def mean_test(x: np.ndarray, lag: int | None = None) -> dict:
    """Mean of daily x, annualized (x 252), with a Newey-West standard error, 95% interval and two-sided p."""
    from scipy.stats import norm
    x = np.asarray(x, float)
    lag = nw_lag(len(x)) if lag is None else lag
    se = math.sqrt(max(nw_variance_of_mean(x, lag), 0.0))
    m = float(x.mean())
    z = m / se if se > 0 else np.nan
    return {"n": len(x), "nw_lag": lag, "mean_daily": m, "mean_annualized": m * ANN, "se_annualized": se * ANN,
            "ci_low": (m - 1.959963984540054 * se) * ANN, "ci_high": (m + 1.959963984540054 * se) * ANN,
            "t_stat": z, "p_value": float(2 * norm.sf(abs(z))) if np.isfinite(z) else np.nan}


def ols_hac(y: np.ndarray, x: np.ndarray, lag: int | None = None) -> dict:
    """y = a + b x + e; Newey-West covariance of (a, b). Daily alpha is reported annualized (x 252)."""
    from scipy.stats import norm
    y, x = np.asarray(y, float), np.asarray(x, float)
    n = len(y)
    lag = nw_lag(n) if lag is None else lag
    X = np.column_stack([np.ones(n), x])
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    e = y - X @ beta
    g = X * e[:, None]
    s = g.T @ g
    for k in range(1, lag + 1):
        w = 1 - k / (lag + 1)
        gk = g[k:].T @ g[:-k]
        s += w * (gk + gk.T)
    cov = xtx_inv @ s @ xtx_inv
    se = np.sqrt(np.diag(cov))
    za, zb = beta[0] / se[0], (beta[1] - 1) / se[1]
    return {"alpha_annualized": beta[0] * ANN, "alpha_se_annualized": se[0] * ANN,
            "alpha_ci_low": (beta[0] - 1.96 * se[0]) * ANN, "alpha_ci_high": (beta[0] + 1.96 * se[0]) * ANN,
            "alpha_p_value": float(2 * norm.sf(abs(za))), "beta": beta[1], "beta_se": se[1],
            "beta_ci_low": beta[1] - 1.96 * se[1], "beta_ci_high": beta[1] + 1.96 * se[1],
            "beta_p_value_vs_1": float(2 * norm.sf(abs(zb))), "r_squared": 1 - float(e @ e) / float(((y - y.mean()) ** 2).sum()),
            "nw_lag": lag, "n": n}


def stationary_indices(n: int, mean_block: float, reps: int, seed: int) -> np.ndarray:
    """(reps, n) index paths of the stationary bootstrap: each step continues the current block with probability
    1 - 1/mean_block (wrapping around the sample) or starts a new block at a uniform random position."""
    rng = np.random.default_rng(seed)
    p = 1.0 / mean_block
    idx = np.empty((reps, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, reps)
    new = rng.random((reps, n)) < p
    starts = rng.integers(0, n, (reps, n))
    for t in range(1, n):
        idx[:, t] = np.where(new[:, t], starts[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def _path_stats(r: np.ndarray, rf: np.ndarray) -> dict:
    """Vectorized statistics of (reps, n) return paths."""
    ex = r - rf
    n = r.shape[1]
    mean_ex = ex.mean(axis=1)
    sd_ex = ex.std(axis=1, ddof=1)
    dd = np.sqrt(np.mean(np.minimum(ex, 0) ** 2, axis=1))
    w = np.cumprod(1 + r, axis=1)
    w = np.concatenate([np.ones((r.shape[0], 1)), w], axis=1)
    mdd = (w / np.maximum.accumulate(w, axis=1) - 1).min(axis=1)
    cagr = w[:, -1] ** (ANN / n) - 1
    return {"sharpe": mean_ex * ANN / (sd_ex * np.sqrt(ANN)), "sortino": mean_ex * ANN / (dd * np.sqrt(ANN)),
            "cagr": cagr, "volatility": r.std(axis=1, ddof=1) * np.sqrt(ANN), "max_drawdown": mdd,
            "mean_annualized": r.mean(axis=1) * ANN}


def point_stats(r: np.ndarray, rf: np.ndarray) -> dict:
    return {k: float(v[0]) for k, v in _path_stats(np.asarray(r, float)[None, :], np.asarray(rf, float)[None, :]).items()}


def bootstrap_differences(returns: pd.DataFrame, rf: pd.Series, pairs: list[tuple[str, str]], mean_block: float,
                          reps: int, seed: int, chunk: int = 500) -> pd.DataFrame:
    """Paired stationary bootstrap of statistic differences A - B for each pair. Returns one row per pair and
    statistic with the point estimate, percentile 95% interval, bootstrap standard error and the two-sided
    bootstrap p-value 2 * min(P(diff* <= 0), P(diff* >= 0))."""
    cols = sorted({p for pair in pairs for p in pair})
    R = returns[cols].to_numpy(float)
    f = rf.reindex(returns.index).to_numpy(float)
    n = len(R)
    draws = {(a, b, s): [] for a, b in pairs for s in ("sharpe", "sortino", "cagr", "volatility", "max_drawdown",
                                                         "mean_annualized")}
    done = 0
    k = 0
    while done < reps:
        m = min(chunk, reps - done)
        idx = stationary_indices(n, mean_block, m, seed + k)
        rf_b = f[idx]
        stats = {c: _path_stats(R[:, j][idx], rf_b) for j, c in enumerate(cols)}
        for a, b in pairs:
            for s in ("sharpe", "sortino", "cagr", "volatility", "max_drawdown", "mean_annualized"):
                draws[(a, b, s)].append(stats[a][s] - stats[b][s])
        done += m
        k += 1
    rows = []
    for (a, b, s), lst in draws.items():
        d = np.concatenate(lst)
        pa, pb = point_stats(returns[a].to_numpy(), f), point_stats(returns[b].to_numpy(), f)
        est = pa[s] - pb[s]
        p = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
        rows.append({"portfolio_a": a, "portfolio_b": b, "statistic": f"{s}_difference", "estimate": est,
                     "ci_low": float(np.quantile(d, 0.025)), "ci_high": float(np.quantile(d, 0.975)),
                     "bootstrap_se": float(d.std(ddof=1)), "bootstrap_p_value": p, "mean_block_length": mean_block,
                     "replications": reps, "seed": seed, "n": n})
    return pd.DataFrame(rows)


def holm(pvalues: list[float]) -> list[float]:
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj.tolist()
