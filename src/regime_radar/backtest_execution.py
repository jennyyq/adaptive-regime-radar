"""Stage 8: orders, the execution-day rule and transaction costs (frozen before Test access).

Decision at the close of t  -> Order (no-trade hold, or a decision-time target with the decision's turnover cap)
Execution at the close of t+1:
  pre    = actual holdings drifted through the return of t+1 (known at the close of t+1)
  hold   -> no trade
  trade  -> TO = 0.5 * sum |target - pre|
            TO <= cap : executed = target                                    (achievable target = decision target)
            TO >  cap : executed = pre + (cap / TO) * (target - pre)         (cap-preserving straight-line projection,
                                                                              the Stage 7 transition rule)
  No signal dated t+1 is used; only execution-day prices (pre-trade weights) enter the feasible trade size.
Costs: cost fraction = bps / 1e4 * (stress multiplier if the decision-date stress flag is on) * gross notional,
gross notional = sum_i |executed_i - pre_i| = 2 * one-way turnover. Costs are deducted pro rata at the execution
close, so post-trade weights equal the executed weights.
"""

from __future__ import annotations

STAGE8_CODE_VERSION = "stage8-2026-10-21"  # must match scripts/10_backtest_portfolios.py

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADE_EPS = 1e-14


@dataclass
class Order:
    portfolio: str
    signal_date: pd.Timestamp
    execution_date: pd.Timestamp
    first_holding_return_date: pd.Timestamp | None
    kind: str                       # "hold" | "trade"
    target: np.ndarray               # decision-time target (for a hold: the decision-time holdings)
    turnover_cap: float
    stress_on: bool
    decision_status: str
    budget_d: float = np.nan


@dataclass
class Execution:
    portfolio: str
    signal_date: pd.Timestamp
    execution_date: pd.Timestamp
    kind: str
    pre: np.ndarray
    decision_target: np.ndarray
    achievable: np.ndarray
    executed: np.ndarray
    turnover: float
    gross_notional: float
    decision_time_turnover_to_target: float
    turnover_cap: float
    projected: bool
    theta: float
    stress_on: bool
    decision_status: str
    exceptions: list = field(default_factory=list)


def one_way(a: np.ndarray, b: np.ndarray) -> float:
    return float(0.5 * np.abs(np.asarray(a, float) - np.asarray(b, float)).sum())


def execute(order: Order, pre: np.ndarray) -> Execution:
    """Apply the frozen execution-day rule to an order given the actual pre-trade weights at the execution close."""
    pre = np.asarray(pre, float)
    target = np.asarray(order.target, float)
    to_target = one_way(target, pre)
    exceptions = []
    if order.kind == "hold":
        executed, achievable, projected, theta = pre.copy(), pre.copy(), False, 0.0
    elif order.kind == "trade":
        if to_target <= order.turnover_cap + 1e-12:
            executed, projected, theta = target.copy(), False, 1.0
        else:
            theta = order.turnover_cap / to_target
            executed = pre + theta * (target - pre)
            projected = True
            exceptions.append(f"execution-day turnover {to_target:.4f} > cap {order.turnover_cap:.4f}: "
                              f"cap-preserving projection, {theta:.3f} of the move executed")
        executed = np.where(np.abs(executed) < 1e-15, 0.0, executed)
        executed = executed / executed.sum()
        achievable = executed.copy()
    else:
        raise ValueError(f"unknown order kind {order.kind!r}")
    to = one_way(executed, pre)
    if to < TRADE_EPS:                       # numerically no trade: keep the holdings exactly
        executed, to = pre.copy(), 0.0
    return Execution(order.portfolio, order.signal_date, order.execution_date, order.kind, pre, target, achievable,
                     executed, to, 2.0 * to, to_target, order.turnover_cap, projected, theta, order.stress_on,
                     order.decision_status, exceptions)


def cost_fraction(gross_notional: float, bps: float, stress_on: bool, stress_multiplier: float) -> float:
    """bps of GROSS traded notional (10 bps = 0.001), doubled (x multiplier) when the stress flag is on."""
    return float(bps / 1e4 * (stress_multiplier if stress_on else 1.0) * gross_notional)


def stress_indicator(simple: pd.DataFrame, threshold: float, window: int = 21, annualization: int = 252) -> pd.Series:
    """Causal cost-stress flag: trailing `window`-session SPY realized volatility through t > Train threshold.
    Identical for every portfolio; uses no HMM / EWS role signal."""
    vol = simple["SPY"].rolling(window).std() * np.sqrt(annualization)
    return (vol > threshold).fillna(False)
