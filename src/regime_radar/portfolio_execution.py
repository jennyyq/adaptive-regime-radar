"""Stage 7: weekly decision calendar, execution timing, pre-trade weights, turnover and transaction-cost estimates.

Calendar (frozen convention conservative_close_t_plus_1)
  * Signals are observed every session. A FORMAL decision is taken on the last trading session of each calendar
    week (Monday-Sunday) of the known Train + Validation calendar.
  * Execution is at the close of the next session t+1 (always in the following week, so at most one execution per
    calendar week); the first close-to-close holding return is that of session t+2.
  * The last known week can be confirmed as complete only if its last known session is a Friday; otherwise the
    Test calendar would be needed, so that week has no decision (no Test date is ever read).
  * The decision taken on the last Train session executes on the first Validation session; it is labelled by
    its signal date (Train) and flagged as split-crossing.
Pre-trade weights
  The exact pre-trade weights at the execution close t+1 need return_(t+1), unknown at decision time. The ex-ante
  proxy drifts the previous target from its first holding session through t (all returns known at t):
      w_pre,i = w_prev,i * prod(1 + R_i) / sum_j w_prev,j * prod(1 + R_j).
  Realized turnover at t+1 is left to Stage 8.
Turnover and costs
  one-way turnover TO = 0.5 * sum |w_new - w_pre|; traded notional = 2 * TO = sum |w_new - w_pre|;
  cost = rate_bps / 1e4 * traded notional (* stress multiplier when the causal stress indicator is on).
"""

from __future__ import annotations

STAGE7_CODE_VERSION = "stage7-2026-10-21"  # must match scripts/09_optimize_portfolio.py

import numpy as np
import pandas as pd


def weekly_schedule(calendar: pd.DatetimeIndex, split: pd.Series) -> pd.DataFrame:
    """One row per formal decision: signal_date, execution_date, first_holding_return_date (NaT if unknown),
    split of the signal date, week start, split_crossing flag."""
    cal = pd.DatetimeIndex(calendar)
    if not cal.is_monotonic_increasing or cal.has_duplicates:
        raise ValueError("calendar must be strictly increasing")
    week = cal - pd.to_timedelta(cal.weekday, unit="D")
    s = pd.Series(cal, index=cal)
    last_in_week = s.groupby(week).max()
    rows = []
    pos = {d: i for i, d in enumerate(cal)}
    for wk, d in last_in_week.items():
        i = pos[d]
        if i == len(cal) - 1 and d.weekday() != 4:
            continue                          # last known week incomplete without the Test calendar
        if i + 1 >= len(cal):
            continue                          # no known execution session
        exe = cal[i + 1]
        hold = cal[i + 2] if i + 2 < len(cal) else pd.NaT
        rows.append({"signal_date": d, "execution_date": exe, "first_holding_return_date": hold, "week_start": wk,
                     "split": split.loc[d], "split_crossing": split.loc[d] != split.loc[exe]})
    out = pd.DataFrame(rows)
    ex_week = out["execution_date"] - pd.to_timedelta(out["execution_date"].dt.weekday, unit="D")
    if ex_week.duplicated().any():
        raise AssertionError("more than one execution in a calendar week")
    return out


def drift(w: np.ndarray, simple_returns: np.ndarray) -> np.ndarray:
    """Buy-and-hold drift of weights through the given rows of simple returns (rows = sessions)."""
    w = np.asarray(w, float)
    if len(simple_returns) == 0:
        return w.copy()
    growth = np.prod(1.0 + np.asarray(simple_returns, float), axis=0)
    v = w * growth
    return v / v.sum()


def one_way_turnover(w_new: np.ndarray, w_pre: np.ndarray) -> float:
    return float(0.5 * np.abs(np.asarray(w_new) - np.asarray(w_pre)).sum())


def transaction_cost(turnover: float, bps: float, stress: bool = False, stress_multiplier: float = 1.0) -> float:
    """Cost as a fraction of portfolio value: bps of traded notional (= 2 x one-way turnover)."""
    return float(bps / 1e4 * 2.0 * turnover * (stress_multiplier if stress else 1.0))


def portfolio_simple_return(w: np.ndarray, simple_returns_row: np.ndarray) -> float:
    """R_p = sum_i w_i R_i with SIMPLE returns (never a weighted sum of log returns)."""
    return float(np.asarray(w) @ np.asarray(simple_returns_row))
