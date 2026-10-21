"""Stage 7: robust expected returns (annualized SIMPLE returns).

Primary - benchmark equilibrium (reverse optimization, low-alpha):
    pi_t = delta_t * Sigma_t w_b,   delta_t = SR_b / sigma_b,t,   sigma_b,t = sqrt(w_b' Sigma_t w_b)
    mu_t = rf_t + pi_t
  SR_b is a fixed, PROPOSED benchmark Sharpe ratio. The implied benchmark excess return is SR_b * sigma_b,t and an
  unconstrained mean-variance investor with risk aversion delta_t holds exactly w_b. No historical mean enters.
  rf_t = trailing mean BIL simple return (63 sessions through t), annualized.

Sensitivity only - shrunk historical mean:
    mu_t = (1 - s) * mean_hist,t + s * mu_eq,t     (mean over the trailing window through t)
  and +/- 1 standard error perturbations of the historical mean (se_i = sd_i / sqrt(n), annualized).
Everything is computed from returns observed through the decision date; no return after t is used.
"""

from __future__ import annotations

STAGE7_CODE_VERSION = "stage7-2026-10-21"  # must match scripts/09_optimize_portfolio.py

import numpy as np
import pandas as pd

from .portfolio_risk import ANNUALIZATION


def risk_free(simple_returns: pd.DataFrame, cash: str = "BIL", window: int = 63) -> float:
    r = simple_returns[cash].to_numpy(float)[-window:]
    return float(np.nanmean(r) * ANNUALIZATION)


def equilibrium(cov_annual: np.ndarray, wb: np.ndarray, sharpe: float, rf: float) -> tuple[np.ndarray, float, float]:
    """Returns (mu, delta, sigma_b)."""
    sigma_b = float(np.sqrt(wb @ cov_annual @ wb))
    delta = sharpe / sigma_b
    return rf + delta * (cov_annual @ wb), delta, sigma_b


def historical_mean(simple_returns: pd.DataFrame, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    x = simple_returns.to_numpy(float)[-lookback:]
    m = np.nanmean(x, axis=0) * ANNUALIZATION
    se = np.nanstd(x, axis=0, ddof=1) / np.sqrt(len(x)) * ANNUALIZATION
    return m, se


def shrunk_historical(hist_mean: np.ndarray, prior: np.ndarray, shrinkage: float) -> np.ndarray:
    return (1 - shrinkage) * hist_mean + shrinkage * prior
