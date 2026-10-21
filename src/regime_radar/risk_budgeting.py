"""Stage 7: translate the frozen Stage 6 signals into a portfolio risk budget.

One scalar defensiveness d in [0, 1] drives every budget quantity through the SAME fixed mapping for P1 and P2
(only the signal -> d step differs):
    vol_multiplier(d) = 1 - vol_slope * d          target ex-ante vol = vol_multiplier * benchmark ex-ante vol
    equity_max(d)     = equity_max0 - equity_slope * d
    credit_max(d)     = credit_max0 - credit_slope * d          (credit exposure can only fall as risk rises)
    liquidity_min(d)  = liquidity_min0 + liquidity_slope * d    (minimum BIL sleeve)
    turnover_cap(d)   = turnover_cap0 + turnover_slope * d      (more room to de-risk when the signal is severe)
d = 0 is the normal-market budget and is identical for P1 and P2.

Signal -> d
  P1 (role baseline_regime)  d1 = min(1, a * baseline_p_non_normal); a is calibrated on Train so that the mean
                             Train d1 equals the mean Train d2 (equal average defensiveness, so P1 and P2 differ in
                             timing rather than in average risk). Lipschitz in p: no jump near p = 1.
  P2 (role challenger_ews)   d2 = level map of challenger_alert_level: GREEN 0, AMBER 1/3, ORANGE 2/3, RED 1,
                             RECOVERY 1/3 (keeps the AMBER posture until the frozen state machine returns to GREEN).
"""

from __future__ import annotations

STAGE7_CODE_VERSION = "stage7-2026-10-21"  # must match scripts/09_optimize_portfolio.py

from dataclasses import dataclass

import numpy as np
import pandas as pd

LEVEL_ORDER = ("GREEN", "RECOVERY", "AMBER", "ORANGE", "RED")


@dataclass(frozen=True)
class RiskBudget:
    d: float
    vol_multiplier: float
    equity_max: float
    credit_max: float
    liquidity_min: float
    turnover_cap: float

    def to_dict(self) -> dict:
        return dict(vars(self))


def budget_from_d(d: float, coef: dict) -> RiskBudget:
    if not (0.0 <= d <= 1.0) or not np.isfinite(d):
        raise ValueError(f"defensiveness must be in [0, 1], got {d}")
    return RiskBudget(float(d), 1 - coef["vol_slope"] * d, coef["equity_max0"] - coef["equity_slope"] * d,
                      coef["credit_max0"] - coef["credit_slope"] * d, coef["liquidity_min0"] + coef["liquidity_slope"] * d,
                      coef["turnover_cap0"] + coef["turnover_slope"] * d)


def p2_defensiveness(level: str, level_map: dict) -> float:
    if level not in level_map:
        raise ValueError(f"unknown alert level {level!r}")
    return float(level_map[level])


def p2_defensiveness_continuous(score: float, amber_out: float, red_in: float) -> float:
    if not np.isfinite(score):
        return 0.0
    return float(np.clip((score - amber_out) / (red_in - amber_out), 0.0, 1.0))


def p1_defensiveness(p_non_normal: float, scale: float) -> float:
    if not np.isfinite(p_non_normal):
        raise ValueError("P(non-normal) must be finite")
    return float(min(1.0, max(0.0, scale * p_non_normal)))


def parity_scale(p_train: pd.Series, levels_train: pd.Series, level_map: dict) -> float:
    """a such that mean_Train(min(1, a p)) = mean_Train(d2). Solved by bisection (monotone in a)."""
    target = float(levels_train.map(level_map).mean())
    p = p_train.to_numpy(float)
    if target <= 0:
        return 0.0
    lo, hi = 0.0, 1.0
    while np.minimum(1, hi * p).mean() < target and hi < 1e6:
        hi *= 2
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if np.minimum(1, mid * p).mean() < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
