"""Stage 8: sequential decision + execution engine (reuses the frozen Stage 7 optimizer; no copy of it).

At every scheduled weekly decision date t (Stage 7 calendar rule, extended over the Test calendar):
  * risk inputs from simple returns dated <= t (portfolio_engine.risk_inputs: Ledoit-Wolf 504, equilibrium mu);
  * the portfolio's own defensiveness d_t (one role only; constant for B2; none for P0);
  * pre-trade weights = ACTUAL ledger holdings at the close of t (known at t);
  * portfolio_optimizer.solve_policy (Stage 7 target_then_trade rule) -> an Order executed at the close of t+1.
The ledger (backtest_accounting) earns returns with the holdings actually held, executes orders at t+1 with the
execution-day rule (backtest_execution) and deducts costs. Nothing dated after t enters a date-t decision.

`decide_stage7_row` reproduces one portfolio_engine.run_portfolio row; with the Stage 7 ex-ante proxy as w_pre
it is bit-identical to Stage 7 (tested), with the actual holdings it is the Stage 8 decision.
"""

from __future__ import annotations

STAGE8_CODE_VERSION = "stage8-2026-10-21"  # must match scripts/10_backtest_portfolios.py

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .backtest_accounting import Ledger
from .backtest_execution import Order, one_way
from .portfolio_constraints import check_weights, hard_violations
from .portfolio_diagnostics import decision_row
from .portfolio_optimizer import solve_policy
from .risk_budgeting import budget_from_d


@dataclass
class PortfolioSpec:
    pid: str
    kind: str                                  # "optimizer" | "fixed" | "buy_and_hold"
    role: str | None = None                    # signal role (P1 / P2 only)
    defensiveness: object = None               # pd.Series (one role), float (constant), or None (no budget)
    fixed_target: np.ndarray | None = None     # kind "fixed"
    fixed_cap: float = 1.0                     # kind "fixed": turnover cap (1.0 = none)
    start_weights: np.ndarray | None = None

    def d_at(self, t) -> float | None:
        if self.defensiveness is None:
            return None
        if isinstance(self.defensiveness, (int, float, np.floating)):
            return float(self.defensiveness)
        return float(self.defensiveness.loc[t])


@dataclass
class BacktestRun:
    spec: PortfolioSpec
    ledger: Ledger
    decisions: pd.DataFrame
    executions: pd.DataFrame
    pending: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def decide_stage7_row(policy, cfg, cost_cfg, ri, w_pre, prev_target, d, stress_on) -> tuple[dict, object, np.ndarray, object]:
    """One Stage 7 decision (the body of portfolio_engine.run_portfolio for target_then_trade)."""
    budget = None if d is None else budget_from_d(float(d), cfg.budget_coef)
    gamma = ri.delta * cfg.gamma_multiplier
    res, posture = solve_policy(policy, ri.mu, ri.cov, gamma, cfg.lambda_tracking, w_pre, budget, ri.sigma_b,
                                cfg.no_trade_band, prev_target, cfg.solver, cfg.lambda_turnover)
    w = res.weights
    hard = [r for r in res.checks if r["kind"] == "hard"]
    info = {"solver_status": res.status, "solver_stage": res.stage, "converged": res.converged,
            "start_disagreement": res.start_disagreement, "unmet_budget": ";".join(res.unmet_budget),
            "active_constraints": ";".join(r["constraint"] for r in res.checks if r["active"]),
            "min_hard_slack": min(r["slack"] for r in hard) if hard else np.nan,
            "hard_violations": ";".join(r["constraint"] for r in hard if r["violated"]),
            "solver_attempts": len(res.attempts)}
    info["distance_to_posture"] = float(0.5 * np.abs(w - posture).sum())
    info["posture_vol_ex_ante"] = float(np.sqrt(max(posture @ ri.cov @ posture, 0)))
    row = {"n_obs": ri.n_obs, "lw_shrinkage": ri.shrinkage, "risk_free": ri.rf, "gamma": gamma}
    row.update(decision_row(policy, w, ri.mu, ri.cov, ri.scenarios, w_pre, budget, ri.sigma_b, cost_cfg, stress_on,
                            cfg.cvar_alpha))
    row.update({f"pretrade_{a}": float(x) for a, x in zip(policy.assets, w_pre)})
    row.update({f"posture_{a}": float(x) for a, x in zip(policy.assets, posture)})
    row.update(info)
    return row, res, posture, budget


def _fixed_row(policy, target, w_pre, band, ri) -> tuple[dict, str]:
    to = one_way(target, w_pre)
    status = "hold_within_no_trade_band" if to <= band else "trade_to_target"
    w = w_pre.copy() if status.startswith("hold") else target.copy()
    row = {f"w_{a}": float(x) for a, x in zip(policy.assets, w)}
    row.update({f"pretrade_{a}": float(x) for a, x in zip(policy.assets, w_pre)})
    row.update({f"posture_{a}": float(x) for a, x in zip(policy.assets, target)})
    row.update({"sum_weights": float(w.sum()), "min_weight": float(w.min()), "turnover_ex_ante_proxy": to,
                "vol_ex_ante": float(np.sqrt(max(w @ ri.cov @ w, 0))) if ri is not None else np.nan,
                "solver_status": status, "solver_stage": 1, "converged": True, "hard_violations": "",
                "unmet_budget": "", "distance_to_posture": float(0.5 * np.abs(w - target).sum())})
    for g, e in policy.exposures(w).items():
        row[f"exposure_{g}"] = e
    return row, status


def run_sequential(spec: PortfolioSpec, simple: pd.DataFrame, schedule: pd.DataFrame, inputs: dict, policy, cfg,
                   cost_cfg: dict, stress: pd.Series, start_date: pd.Timestamp, bps=(5, 10, 25), primary_bps=10,
                   end_date: pd.Timestamp | None = None, initial_prev_target: np.ndarray | None = None) -> BacktestRun:
    """Run one portfolio from `start_date` (holdings = spec.start_weights at its close) to `end_date`."""
    cal = simple.index if end_date is None else simple.index[simple.index <= end_date]
    sched = schedule.set_index("signal_date")
    i0 = cal.get_loc(start_date)
    w0 = np.asarray(spec.start_weights if spec.start_weights is not None else policy.benchmark, float)
    ledger = Ledger(spec.pid, policy.assets, start_date, w0, bps, primary_bps, cost_cfg["stress_friction_multiplier"])
    R = simple.loc[cal].to_numpy(float)
    pending: Order | None = None
    prev_target = None if initial_prev_target is None else np.asarray(initial_prev_target, float)
    rows, exe_rows, warn_rows = [], [], []
    for i in range(i0, len(cal)):
        s = cal[i]
        if i > i0:
            order = pending if (pending is not None and pending.execution_date == s) else None
            exe = ledger.session(s, R[i], order)
            if order is not None:
                pending = None
                exe_rows.append(_execution_row(exe, policy, s))
        if spec.kind == "buy_and_hold" or s not in sched.index:
            continue
        if spec.kind in ("optimizer", "fixed") and s not in inputs:
            continue                                    # warm-up: not enough past returns
        srow = sched.loc[s]
        exe_date = srow["execution_date"]
        if pd.isna(exe_date) or exe_date > cal[-1]:
            continue
        w_pre = ledger.w.copy()
        stress_on = bool(stress.loc[s]) if s in stress.index and pd.notna(stress.loc[s]) else False
        ri = inputs.get(s)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if spec.kind == "optimizer":
                d = spec.d_at(s)
                row, res, posture, budget = decide_stage7_row(policy, cfg, cost_cfg, ri, w_pre, prev_target, d, stress_on)
                w = res.weights
                status = res.status
                hold = status == "hold_within_no_trade_band" or np.array_equal(w, w_pre)
                cap = budget.turnover_cap if budget is not None else policy.turnover_cap
                d_val = np.nan if d is None else d
            else:
                row, status = _fixed_row(policy, spec.fixed_target, w_pre, cfg.no_trade_band, ri)
                w = spec.fixed_target.copy() if status == "trade_to_target" else w_pre.copy()
                hold, cap, d_val = status.startswith("hold"), spec.fixed_cap, np.nan
        msgs = sorted({f"{x.category.__name__}: {x.message}" for x in caught})
        for m in msgs:
            warn_rows.append({"portfolio": spec.pid, "signal_date": s, "warning": m})
        row.update({"portfolio": spec.pid, "signal_role": spec.role or "none", "signal_date": s,
                    "execution_date": exe_date, "first_holding_return_date": srow["first_holding_return_date"],
                    "split": srow["split"], "split_crossing": bool(srow["split_crossing"]), "budget_d_input": d_val,
                    "order_kind": "hold" if hold else "trade", "order_turnover_cap": cap,
                    "stress_friction_on": stress_on, "solver_warnings": " | ".join(msgs)})
        rows.append(row)
        pending = Order(spec.pid, s, exe_date, srow["first_holding_return_date"], "hold" if hold else "trade", w.copy(),
                        float(cap), stress_on, status, d_val)
        prev_target = w.copy()
    left = [pending] if pending is not None else []
    return BacktestRun(spec, ledger, pd.DataFrame(rows), pd.DataFrame(exe_rows), left, warn_rows)


def _execution_row(exe, policy, date) -> dict:
    hard = check_weights(policy, exe.executed)
    row = {"portfolio": exe.portfolio, "signal_date": exe.signal_date, "execution_date": exe.execution_date,
           "ledger_date": date, "order_kind": exe.kind, "decision_status": exe.decision_status,
           "execution_day_turnover_to_target": exe.decision_time_turnover_to_target, "turnover_cap": exe.turnover_cap,
           "projected_to_cap": exe.projected, "fraction_of_move_executed": exe.theta, "turnover": exe.turnover,
           "gross_notional": exe.gross_notional, "stress_on": exe.stress_on,
           "executed_hard_violations": ";".join(hard_violations(hard)), "exceptions": " | ".join(exe.exceptions)}
    for a, p, t, x in zip(policy.assets, exe.pre, exe.decision_target, exe.executed):
        row[f"pre_{a}"], row[f"target_{a}"], row[f"executed_{a}"] = float(p), float(t), float(x)
    for g in policy.groups:
        gv = policy.group_vector(g)
        row[f"pre_exposure_{g}"], row[f"executed_exposure_{g}"] = float(gv @ exe.pre), float(gv @ exe.executed)
    return row
