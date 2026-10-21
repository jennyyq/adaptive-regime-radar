"""Stage 7: chronological, causal portfolio decision engine shared by P0, P1 and P2 (and reusable in Stage 8).

For every scheduled decision date t (see portfolio_execution.weekly_schedule):
  1. returns_through_t = simple returns with index <= t   (return_t is known at the close of t)
  2. covariance, risk-free rate, equilibrium expected returns and historical CVaR scenarios from returns_through_t
  3. pre-trade proxy = previous target drifted from its first holding session through t
  4. posture target: P0 = strategic weights; P1 / P2 = robust MVO under the risk budget from the role's own
     defensiveness d_t (path independent); then the identical trade rule for all three portfolios: hold inside the
     no-trade band, else trade to the target, else move toward it as far as the turnover cap allows
  5. ex-ante diagnostics; the target is executed at the close of t+1 and first earns return_(t+2)
No step reads a return dated after t, a forward outcome or any Test-period file. The engine receives the
defensiveness series of exactly ONE role (or none for P0).
"""

from __future__ import annotations

STAGE7_CODE_VERSION = "stage7-2026-10-21"  # must match scripts/09_optimize_portfolio.py

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from . import expected_returns as er
from .portfolio_diagnostics import contribution_rows, decision_row
from .portfolio_execution import drift
from .portfolio_optimizer import solve_mvo, solve_policy
from .portfolio_risk import ANNUALIZATION, estimate_cov
from .risk_budgeting import budget_from_d


@dataclass(frozen=True)
class EngineConfig:
    cov_method: str = "ledoit_wolf"
    lookback: object = 504
    halflife: float = 63
    min_obs: int = 252
    psd_floor: float = 1e-8
    er_method: str = "equilibrium"          # equilibrium | shrunk_historical | plus_se | minus_se
    sharpe: float = 0.30
    rf_window: int = 63
    er_shrinkage: float = 0.75
    gamma_multiplier: float = 1.0
    lambda_turnover: float = 0.008
    lambda_tracking: float = 2.0
    cvar_alpha: float = 0.95
    execution_mode: str = "target_then_trade"   # primary; "single_step" = turnover-penalized MVO (sensitivity only)
    no_trade_band: float = 0.01
    cvar_lookback: int = 504
    budget_coef: dict = field(default_factory=dict)
    solver: dict = field(default_factory=dict)

    def variant(self, **kw) -> "EngineConfig":
        return replace(self, **kw)


@dataclass
class RiskInputs:
    cov: np.ndarray
    mu: np.ndarray
    sigma_b: float
    delta: float
    rf: float
    scenarios: np.ndarray
    n_obs: int
    shrinkage: float
    effective_n: float
    floored: int
    hist_mean: np.ndarray
    hist_se: np.ndarray
    mu_eq: np.ndarray


def risk_inputs(simple_through_t: pd.DataFrame, wb: np.ndarray, cfg: EngineConfig) -> RiskInputs:
    est = estimate_cov(simple_through_t, cfg.cov_method, cfg.lookback, cfg.psd_floor, cfg.halflife, cfg.min_obs)
    cov = est.cov_annual
    rf = er.risk_free(simple_through_t, window=cfg.rf_window)
    mu_eq, delta, sigma_b = er.equilibrium(cov, wb, cfg.sharpe, rf)
    look = len(simple_through_t) if cfg.lookback == "expanding" else int(cfg.lookback)
    hm, se = er.historical_mean(simple_through_t, min(look, len(simple_through_t)))
    if cfg.er_method == "equilibrium":
        mu = mu_eq
    elif cfg.er_method == "shrunk_historical":
        mu = er.shrunk_historical(hm, mu_eq, cfg.er_shrinkage)
    elif cfg.er_method == "plus_se":
        mu = mu_eq + se
    elif cfg.er_method == "minus_se":
        mu = mu_eq - se
    else:
        raise ValueError(cfg.er_method)
    scen = simple_through_t.to_numpy(float)[-cfg.cvar_lookback:]
    return RiskInputs(cov, mu, sigma_b, delta, rf, scen, est.n_obs, est.shrinkage, est.effective_n, est.floored_eigenvalues,
                      hm, se, mu_eq)


def prepare_inputs(simple: pd.DataFrame, schedule: pd.DataFrame, wb: np.ndarray, cfg: EngineConfig) -> dict:
    """Risk inputs for every eligible decision date (dates with fewer than min_obs past returns are skipped)."""
    out = {}
    for t in schedule["signal_date"]:
        hist = simple.loc[:t]
        if len(hist) < cfg.min_obs:
            continue
        out[t] = risk_inputs(hist, wb, cfg)
    return out


def run_portfolio(pid: str, role: str | None, simple: pd.DataFrame, schedule: pd.DataFrame, inputs: dict, policy,
                  cfg: EngineConfig, cost_cfg: dict, stress: pd.Series, defensiveness: pd.Series | None = None,
                  start_weights: np.ndarray | None = None, contributions: bool = True
                  ) -> tuple[pd.DataFrame, pd.DataFrame, list[dict], list[dict]]:
    """Sequential decisions. `defensiveness` (indexed by signal date) must come from ONE role; None for P0."""
    if (defensiveness is None) != (role is None):
        raise ValueError("P0 takes no signal; P1 / P2 need exactly one role's defensiveness series")
    wb = policy.benchmark
    rows, contrib, attempts, checks = [], [], [], []
    prev_target, prev_hold = None, None
    for _, s in schedule.iterrows():
        t = s["signal_date"]
        if t not in inputs:
            continue
        ri = inputs[t]
        if prev_target is None:
            w_pre = (wb if start_weights is None else np.asarray(start_weights, float)).copy()
        else:
            held = simple.loc[prev_hold:t] if pd.notna(prev_hold) and prev_hold <= t else simple.iloc[0:0]
            w_pre = drift(prev_target, held.to_numpy(float))
        budget = None if defensiveness is None else budget_from_d(float(defensiveness.loc[t]), cfg.budget_coef)
        gamma = ri.delta * cfg.gamma_multiplier
        if cfg.execution_mode == "target_then_trade":
            res, posture = solve_policy(policy, ri.mu, ri.cov, gamma, cfg.lambda_tracking, w_pre, budget, ri.sigma_b,
                                        cfg.no_trade_band, prev_target, cfg.solver, cfg.lambda_turnover)
        elif cfg.execution_mode == "single_step":
            if budget is None:
                res, posture = None, wb.copy()
            else:
                res = solve_mvo(policy, ri.mu, ri.cov, gamma, cfg.lambda_turnover, cfg.lambda_tracking, w_pre, budget,
                                ri.sigma_b, prev_target, cfg.solver)
                posture = res.weights
        else:
            raise ValueError(cfg.execution_mode)
        if res is None:                                   # single-step P0: rebalance to the benchmark
            from .portfolio_constraints import check_weights
            rows_ = check_weights(policy, wb, None, w_pre, policy.turnover_cap)
            w = wb.copy()
            info = {"solver_status": "benchmark", "solver_stage": 1, "converged": True, "start_disagreement": 0.0,
                    "unmet_budget": "", "active_constraints": ";".join(r["constraint"] for r in rows_ if r["active"]),
                    "min_hard_slack": min(r["slack"] for r in rows_),
                    "hard_violations": ";".join(r["constraint"] for r in rows_ if r["violated"]), "solver_attempts": 0}
            checks += [{"portfolio": pid, "signal_date": t, "split": s["split"], **r} for r in rows_]
        else:
            w = res.weights
            hard = [r for r in res.checks if r["kind"] == "hard"]
            info = {"solver_status": res.status, "solver_stage": res.stage, "converged": res.converged,
                    "start_disagreement": res.start_disagreement, "unmet_budget": ";".join(res.unmet_budget),
                    "active_constraints": ";".join(r["constraint"] for r in res.checks if r["active"]),
                    "min_hard_slack": min(r["slack"] for r in hard) if hard else np.nan,
                    "hard_violations": ";".join(r["constraint"] for r in hard if r["violated"]),
                    "solver_attempts": len(res.attempts)}
            attempts += [{"portfolio": pid, "signal_date": t, "attempt": a} for a in res.attempts]
            checks += [{"portfolio": pid, "signal_date": t, "split": s["split"], **r} for r in res.checks]
        info["distance_to_posture"] = float(0.5 * np.abs(w - posture).sum())
        info["posture_vol_ex_ante"] = float(np.sqrt(max(posture @ ri.cov @ posture, 0)))
        st = bool(stress.loc[t]) if t in stress.index and pd.notna(stress.loc[t]) else False
        row = {"portfolio": pid, "signal_role": role or "none", "signal_date": t, "execution_date": s["execution_date"],
               "first_holding_return_date": s["first_holding_return_date"], "split": s["split"],
               "split_crossing": s["split_crossing"], "n_obs": ri.n_obs, "lw_shrinkage": ri.shrinkage,
               "risk_free": ri.rf, "gamma": ri.delta * cfg.gamma_multiplier}
        row.update(decision_row(policy, w, ri.mu, ri.cov, ri.scenarios, w_pre, budget, ri.sigma_b, cost_cfg, st,
                                cfg.cvar_alpha))
        row.update({f"pretrade_{a}": float(x) for a, x in zip(policy.assets, w_pre)})
        row.update({f"posture_{a}": float(x) for a, x in zip(policy.assets, posture)})
        row.update(info)
        rows.append(row)
        if contributions:
            contrib.append({"portfolio": pid, "signal_date": t, "split": s["split"], **contribution_rows(policy, w, ri.cov)})
        prev_target, prev_hold = w, s["first_holding_return_date"]
    return pd.DataFrame(rows), pd.DataFrame(contrib), attempts, checks


def annualized(x: float) -> float:
    return x * ANNUALIZATION
