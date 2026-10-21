"""Stage 7: robust mean-variance optimizer with turnover and tracking penalties, a deterministic fallback, a
Rockafellar-Uryasev CVaR linear program (secondary specification) and an efficient-frontier helper.

Primary problem (annualized simple-return units throughout), variables x = [w, u, v], w - u + v = w_pre:
    maximize   w'mu - (gamma/2) w'Sigma w - lambda_to * 0.5 * sum(u + v) - (lambda_te/2) (w - w_b)'Sigma (w - w_b)
    subject to sum w = 1; lo <= w <= hi (long only, single-ETF caps, |w - w_b| <= max_active);
               static group limits tightened by the risk budget (equity_max, credit_max, liquidity_min);
               0.5 * sum(u + v) <= turnover_cap(d)                                    (hard)
               w'Sigma w <= (vol_multiplier(d) * sigma_b)^2                           (risk budget)
The split w - w_pre = u - v (u, v >= 0) makes the L1 turnover term smooth; at the optimum u_i v_i = 0.
Solver: SLSQP from two starting points (pre-trade weights and benchmark). The problem is convex, so both starts
should agree; the max weight difference is reported as an objective-stability diagnostic.

Deterministic fallback (decided in advance):
  stage 1  full problem;
  stage 2  if stage 1 is infeasible (LP pre-check of the linear constraints, or a solver failure, or a residual
           violation): keep all hard constraints including the turnover cap and give each risk-budget constraint a
           non-negative slack with a linear penalty (10 per unit) -> the turnover-limited move toward the budget;
           the unmet part of the budget is reported;
  stage 3  if stage 2 also fails: hold the pre-trade weights, else the previous target, else the benchmark - the
           first candidate that satisfies every hard constraint. Never equal weights. A fallback that violates a
           hard constraint is reported as such (and fails the audit checks).
"""

from __future__ import annotations

STAGE7_CODE_VERSION = "stage7-2026-10-21"  # must match scripts/09_optimize_portfolio.py

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linprog, minimize

from .portfolio_constraints import (Policy, budget_violations, bounds, check_weights, group_limits, hard_violations,
                                    linear_feasible)



@dataclass
class SolveResult:
    weights: np.ndarray
    status: str                 # optimal / transition_fallback / hold_fallback
    stage: int
    converged: bool
    message: str
    objective: float
    start_disagreement: float
    checks: list[dict] = field(default_factory=list)
    attempts: list[str] = field(default_factory=list)
    unmet_budget: list[str] = field(default_factory=list)


def _objective(x, n, mu, cov, gamma, lam_to, lam_te, wb):
    w, u, v = x[:n], x[n:2 * n], x[2 * n:]
    d = w - wb
    f = w @ mu - 0.5 * gamma * w @ cov @ w - lam_to * 0.5 * (u.sum() + v.sum()) - 0.5 * lam_te * d @ cov @ d
    g = np.r_[mu - gamma * cov @ w - lam_te * cov @ d, -0.5 * lam_to * np.ones(2 * n)]
    return -f, -g


OBJECTIVE_SCALE = 100.0   # posture objective x100 inside SLSQP (numerical scaling only; 1e-6 start agreement)
SLACK_PENALTY = 10.0       # per unit of budget shortfall (annualized return units); dominates any objective gain


def _run_stage2(policy, mu, cov, gamma, lam_to, lam_te, w_pre, budget, vol_target, starts, solver_cfg):
    """Turnover-limited transition: every hard constraint kept; the risk-budget constraints get non-negative slack
    variables s = (equity, credit, liquidity, volatility) with a linear penalty, so the solver moves as far toward
    the budget as the turnover cap allows and the slacks measure the unmet part."""
    n = len(policy.assets)
    lo, hi = bounds(policy)
    wb = policy.benchmark
    gmin, gmax = group_limits(policy, None)
    eq_v, cr_v, lq_v = (policy.group_vector(g) for g in ("equity", "credit", "liquidity"))
    nx = 3 * n + 4

    def fun(x):
        f, g = _objective(x[:3 * n], n, mu, cov, gamma, lam_to, lam_te, wb)
        return f + SLACK_PENALTY * x[3 * n:].sum(), np.r_[g, SLACK_PENALTY * np.ones(4)]

    rows, rhs = [], []
    for g in policy.groups:
        gv = np.r_[policy.group_vector(g), np.zeros(2 * n + 4)]
        rows += [-gv, gv]
        rhs += [gmax[g], -gmin[g]]
    rows.append(np.r_[np.zeros(n), -0.5 * np.ones(2 * n), np.zeros(4)])
    rhs.append(budget.turnover_cap)
    s0, s1, s2 = (np.eye(4)[k] for k in range(3))
    rows += [np.r_[-eq_v, np.zeros(2 * n), s0], np.r_[-cr_v, np.zeros(2 * n), s1], np.r_[lq_v, np.zeros(2 * n), s2]]
    rhs += [budget.equity_max, budget.credit_max, -budget.liquidity_min]
    a, b = np.array(rows), np.array(rhs)
    cons = [{"type": "eq", "fun": lambda x: np.array([x[:n].sum() - 1.0]),
             "jac": lambda x: np.r_[np.ones(n), np.zeros(2 * n + 4)][None, :]},
            {"type": "eq", "fun": lambda x: x[:n] - x[n:2 * n] + x[2 * n:3 * n] - w_pre,
             "jac": lambda x: np.hstack([np.eye(n), -np.eye(n), np.eye(n), np.zeros((n, 4))])},
            {"type": "ineq", "fun": lambda x: b + a @ x, "jac": lambda x: a},
            {"type": "ineq", "fun": lambda x: np.array([(vol_target + x[-1]) ** 2 - x[:n] @ cov @ x[:n]]),
             "jac": lambda x: np.r_[-2 * cov @ x[:n], np.zeros(2 * n + 3), 2 * (vol_target + x[-1])][None, :]}]
    bnds = [(lo[i], hi[i]) for i in range(n)] + [(0.0, 1.0)] * (2 * n) + [(0.0, 1.0)] * 4
    sols = []
    for w0 in starts:
        w0 = np.clip(w0, lo, hi)
        d = w0 - w_pre
        e = np.r_[max(eq_v @ w0 - budget.equity_max, 0), max(cr_v @ w0 - budget.credit_max, 0),
                  max(budget.liquidity_min - lq_v @ w0, 0), max(np.sqrt(w0 @ cov @ w0) - vol_target, 0)]
        x0 = np.r_[w0, np.maximum(d, 0), np.maximum(-d, 0), e]
        r = minimize(fun, x0, jac=True, method="SLSQP", bounds=bnds, constraints=cons,
                     options={"ftol": solver_cfg.get("ftol", 1e-12), "maxiter": solver_cfg.get("maxiter", 500)})
        sols.append(r)
    assert nx == len(bnds)
    return sols


def _constraints(n, policy, gmin, gmax, w_pre, turnover_cap, cov=None, vol_target=None):
    cons = [{"type": "eq", "fun": lambda x: np.array([x[:n].sum() - 1.0]),
             "jac": lambda x: np.r_[np.ones(n), np.zeros(2 * n)][None, :]},
            {"type": "eq", "fun": lambda x: x[:n] - x[n:2 * n] + x[2 * n:] - w_pre,
             "jac": lambda x: np.hstack([np.eye(n), -np.eye(n), np.eye(n)])}]
    rows, rhs = [], []
    for g in policy.groups:
        gv = np.r_[policy.group_vector(g), np.zeros(2 * n)]
        rows += [-gv, gv]
        rhs += [gmax[g], -gmin[g]]            # rhs + row @ x >= 0
    rows.append(np.r_[np.zeros(n), -0.5 * np.ones(2 * n)])
    rhs.append(turnover_cap)
    a, b = np.array(rows), np.array(rhs)
    cons.append({"type": "ineq", "fun": lambda x: b + a @ x, "jac": lambda x: a})
    if cov is not None and vol_target is not None:
        cons.append({"type": "ineq", "fun": lambda x: np.array([vol_target ** 2 - x[:n] @ cov @ x[:n]]),
                     "jac": lambda x: np.r_[-2 * cov @ x[:n], np.zeros(2 * n)][None, :]})
    return cons


def _start(w0, w_pre):
    d = w0 - w_pre
    return np.r_[w0, np.maximum(d, 0), np.maximum(-d, 0)]


def _run(stage, policy, mu, cov, gamma, lam_to, lam_te, w_pre, budget, vol_target, starts, solver_cfg):
    n = len(policy.assets)
    lo, hi = bounds(policy)
    wb = policy.benchmark
    if stage == 2:
        return _run_stage2(policy, mu, cov, gamma, lam_to, lam_te, w_pre, budget, vol_target, starts, solver_cfg)
    gmin, gmax = group_limits(policy, budget)
    cons = _constraints(n, policy, gmin, gmax, w_pre, budget.turnover_cap, cov, vol_target)

    def fun(x):
        return _objective(x, n, mu, cov, gamma, lam_to, lam_te, wb)
    bnds = [(lo[i], hi[i]) for i in range(n)] + [(0.0, 1.0)] * (2 * n)
    sols = []
    for w0 in starts:
        x0 = _start(np.clip(w0, lo, hi), w_pre)
        r = minimize(fun, x0, jac=True, method="SLSQP", bounds=bnds, constraints=cons,
                     options={"ftol": solver_cfg.get("ftol", 1e-12), "maxiter": solver_cfg.get("maxiter", 500)})
        sols.append(r)
    return sols


def solve_mvo(policy: Policy, mu: np.ndarray, cov: np.ndarray, gamma: float, lam_to: float, lam_te: float,
              w_pre: np.ndarray, budget, sigma_b: float, w_prev_target: np.ndarray | None = None,
              solver_cfg: dict | None = None) -> SolveResult:
    solver_cfg = solver_cfg or {}
    tol = solver_cfg.get("constraint_tolerance", 1e-7)
    n = len(policy.assets)
    vol_target = budget.vol_multiplier * sigma_b
    starts = [w_pre, policy.benchmark]
    attempts = []

    def evaluate(sols, stage):
        good = []
        for r in sols:
            w = r.x[:n]
            rows = check_weights(policy, w, budget, w_pre, budget.turnover_cap, cov, vol_target, tol)
            hard = hard_violations(rows)
            bud = budget_violations(rows)
            ok = r.success and not hard and (stage == 2 or not bud)
            attempts.append(f"stage{stage}: success={r.success} hard_violations={hard} budget_violations={bud} "
                            f"msg={r.message}")
            if ok:
                good.append((r.fun, w, rows, bud, r.message))
        if not good:
            return None
        good.sort(key=lambda t: t[0])
        disagreement = max(float(np.abs(g[1] - good[0][1]).max()) for g in good) if len(good) > 1 else np.nan
        return good[0], disagreement

    lin_ok, _ = linear_feasible(policy, budget, w_pre, budget.turnover_cap)
    if lin_ok:
        out = evaluate(_run(1, policy, mu, cov, gamma, lam_to, lam_te, w_pre, budget, vol_target, starts, solver_cfg), 1)
        if out:
            (f, w, rows, bud, msg), dis = out
            return SolveResult(w, "optimal", 1, True, str(msg), -f, dis, rows, attempts, [])
    else:
        attempts.append("stage1: skipped - linear risk-budget constraints infeasible with the turnover cap")
    out = evaluate(_run(2, policy, mu, cov, gamma, lam_to, lam_te, w_pre, budget, vol_target, starts, solver_cfg), 2)
    if out:
        (f, w, rows, bud, msg), dis = out
        return SolveResult(w, "transition_fallback", 2, True, str(msg), np.nan, dis, rows, attempts, bud)
    for name, cand in (("pretrade", w_pre), ("previous_target", w_prev_target), ("benchmark", policy.benchmark)):
        if cand is None:
            continue
        rows = check_weights(policy, cand, budget, w_pre, budget.turnover_cap, cov, vol_target, tol)
        if not hard_violations(rows):
            attempts.append(f"stage3: hold {name}")
            return SolveResult(np.asarray(cand, float), f"hold_fallback_{name}", 3, False, "fallback", np.nan, np.nan,
                               rows, attempts, budget_violations(rows))
    rows = check_weights(policy, policy.benchmark, budget, w_pre, budget.turnover_cap, cov, vol_target, tol)
    attempts.append("stage3: no candidate satisfies the hard constraints; benchmark returned and flagged")
    return SolveResult(policy.benchmark.copy(), "fallback_violates_hard_constraints", 3, False, "fallback", np.nan,
                       np.nan, rows, attempts, budget_violations(rows))


def solve_cvar_lp(policy: Policy, mu: np.ndarray, scenarios: np.ndarray, w_pre: np.ndarray, budget, cvar_cap: float,
                  lam_to: float, lam_l1: float, alpha: float = 0.95) -> tuple[np.ndarray | None, str]:
    """Rockafellar-Uryasev LP: max w'mu - lam_to TO - lam_l1 |w - w_b|_1  s.t.  CVaR_alpha(w) <= cvar_cap and the
    same linear hard / budget constraints (daily scenario losses; CVaR in daily units)."""
    n, t = len(policy.assets), len(scenarios)
    lo, hi = bounds(policy)
    gmin, gmax = group_limits(policy, budget)
    nv = 4 * n + t + 1                                   # w, u, v, a, z, zeta
    iw, iu, iv, ia, iz, izeta = 0, n, 2 * n, 3 * n, 4 * n, 4 * n + t
    c = np.zeros(nv)
    c[iw:iw + n] = -mu
    c[iu:iu + n] = c[iv:iv + n] = 0.5 * lam_to
    c[ia:ia + n] = lam_l1
    a_ub, b_ub = [], []
    for g in policy.groups:
        row = np.zeros(nv)
        row[iw:iw + n] = policy.group_vector(g)
        a_ub += [row, -row]
        b_ub += [gmax[g], -gmin[g]]
    row = np.zeros(nv)
    row[iu:iu + n] = row[iv:iv + n] = 0.5
    a_ub.append(row)
    b_ub.append(budget.turnover_cap)
    for i in range(n):                                    # a_i >= |w_i - w_b,i|
        r1 = np.zeros(nv)
        r1[iw + i], r1[ia + i] = 1, -1
        r2 = np.zeros(nv)
        r2[iw + i], r2[ia + i] = -1, -1
        a_ub += [r1, r2]
        b_ub += [policy.benchmark[i], -policy.benchmark[i]]
    sc = np.zeros((t, nv))                                # loss_k - zeta - z_k <= 0, loss_k = -R_k w
    sc[:, iw:iw + n] = -scenarios
    sc[np.arange(t), iz + np.arange(t)] = -1
    sc[:, izeta] = -1
    a_ub_m = np.vstack([np.array(a_ub), sc])
    b_ub_v = np.r_[np.array(b_ub), np.zeros(t)]
    cv = np.zeros(nv)                                     # zeta + sum z / ((1-alpha) T) <= cap
    cv[izeta] = 1
    cv[iz:iz + t] = 1.0 / ((1 - alpha) * t)
    a_ub_m = np.vstack([a_ub_m, cv])
    b_ub_v = np.r_[b_ub_v, cvar_cap]
    a_eq = np.zeros((n + 1, nv))
    a_eq[0, iw:iw + n] = 1
    for i in range(n):
        a_eq[1 + i, iw + i], a_eq[1 + i, iu + i], a_eq[1 + i, iv + i] = 1, -1, 1
    b_eq = np.r_[1.0, w_pre]
    bnds = ([(lo[i], hi[i]) for i in range(n)] + [(0, None)] * (3 * n) + [(0, None)] * t + [(None, None)])
    res = linprog(c, A_ub=a_ub_m, b_ub=b_ub_v, A_eq=a_eq, b_eq=b_eq, bounds=bnds, method="highs")
    if res.status != 0:
        return None, f"infeasible_or_failed: {res.message}"
    return res.x[:n], "optimal"


def efficient_frontier(policy: Policy, mu: np.ndarray, cov: np.ndarray, budget=None, points: int = 15
                       ) -> list[dict]:
    """Max expected return for a grid of volatility caps under the linear constraints (no turnover limit)."""
    n = len(policy.assets)
    lo, hi = bounds(policy)
    gmin, gmax = group_limits(policy, budget)
    rows, rhs = [], []
    for g in policy.groups:
        gv = policy.group_vector(g)
        rows += [-gv, gv]
        rhs += [gmax[g], -gmin[g]]
    a, b = np.array(rows), np.array(rhs)
    base = [{"type": "eq", "fun": lambda w: np.array([w.sum() - 1]), "jac": lambda w: np.ones((1, n))},
            {"type": "ineq", "fun": lambda w: b + a @ w, "jac": lambda w: a}]
    bnds = list(zip(lo, hi))
    ok, w_feas = linear_feasible(policy, budget)
    if not ok:
        return []
    mv = minimize(lambda w: (w @ cov @ w, 2 * cov @ w), w_feas, jac=True, method="SLSQP", bounds=bnds,
                  constraints=base, options={"ftol": 1e-14, "maxiter": 500})
    mr = minimize(lambda w: (-(w @ mu), -mu), w_feas, jac=True, method="SLSQP", bounds=bnds, constraints=base,
                  options={"ftol": 1e-14, "maxiter": 500})
    s_min, s_max = np.sqrt(mv.x @ cov @ mv.x), np.sqrt(mr.x @ cov @ mr.x)
    out = []
    for s in np.linspace(s_min, s_max, points):
        cons = base + [{"type": "ineq", "fun": lambda w, s=s: np.array([s ** 2 - w @ cov @ w]),
                        "jac": lambda w: (-2 * cov @ w)[None, :]}]
        r = minimize(lambda w: (-(w @ mu), -mu), mv.x, jac=True, method="SLSQP", bounds=bnds, constraints=cons,
                     options={"ftol": 1e-14, "maxiter": 500})
        if r.success:
            out.append({"vol": float(np.sqrt(r.x @ cov @ r.x)), "expected_return": float(r.x @ mu),
                        **{f"w_{x}": float(v) for x, v in zip(policy.assets, r.x)}})
    return out


# --------------------------------------------------------------------------------------
# Primary execution rule: path-independent posture target, then a turnover-limited trade
# --------------------------------------------------------------------------------------
def solve_target(policy: Policy, mu: np.ndarray, cov: np.ndarray, gamma: float, lam_te: float, budget,
                 sigma_b: float, solver_cfg: dict | None = None) -> tuple[np.ndarray, str, float, list[str]]:
    """Posture target: maximize w'mu - (gamma/2) w'Sigma w - (lam_te/2)(w-w_b)'Sigma(w-w_b) subject to the static
    hard constraints and the risk-budget constraints (no turnover term, no dependence on current holdings).
    Returns (weights, status, start disagreement, attempts). If the budget is unattainable even without a turnover
    limit, the closest point (slack formulation from the benchmark, turnover cap 1) is returned and flagged."""
    solver_cfg = solver_cfg or {}
    tol = solver_cfg.get("constraint_tolerance", 1e-7)
    n = len(policy.assets)
    lo, hi = bounds(policy)
    wb = policy.benchmark
    gmin, gmax = group_limits(policy, budget)
    vol_target = budget.vol_multiplier * sigma_b
    rows, rhs = [], []
    for g in policy.groups:
        gv = policy.group_vector(g)
        rows += [-gv, gv]
        rhs += [gmax[g], -gmin[g]]
    a, b = np.array(rows), np.array(rhs)
    cons = [{"type": "eq", "fun": lambda w: np.array([w.sum() - 1.0]), "jac": lambda w: np.ones((1, n))},
            {"type": "ineq", "fun": lambda w: b + a @ w, "jac": lambda w: a},
            {"type": "ineq", "fun": lambda w: np.array([vol_target ** 2 - w @ cov @ w]),
             "jac": lambda w: (-2 * cov @ w)[None, :]}]

    scale = OBJECTIVE_SCALE        # rescaled objective: SLSQP's ftol is absolute, the objective is O(1e-2)

    def fun(w):
        d = w - wb
        f = w @ mu - 0.5 * gamma * w @ cov @ w - 0.5 * lam_te * d @ cov @ d
        return -scale * f, -scale * (mu - gamma * cov @ w - lam_te * cov @ d)

    attempts, good = [], []
    ok_lin, w_lin = linear_feasible(policy, budget)
    starts = [wb] + ([w_lin] if ok_lin else [])
    for w0 in starts:
        r = minimize(fun, np.clip(w0, lo, hi), jac=True, method="SLSQP", bounds=list(zip(lo, hi)), constraints=cons,
                     options={"ftol": solver_cfg.get("ftol", 1e-12), "maxiter": solver_cfg.get("maxiter", 500)})
        rws = check_weights(policy, r.x, budget, cov=cov, vol_target=vol_target, tol=tol)
        ok = r.success and not hard_violations(rws) and not budget_violations(rws)
        attempts.append(f"target: success={r.success} feasible={ok} msg={r.message}")
        if ok:
            good.append((r.fun, r.x))
    if good:
        good.sort(key=lambda t: t[0])
        dis = max(float(np.abs(g[1] - good[0][1]).max()) for g in good) if len(good) > 1 else np.nan
        return good[0][1], "target_optimal", dis, attempts
    from dataclasses import replace as _replace
    loose = _replace(budget, turnover_cap=1.0)
    sols = _run_stage2(policy, mu, cov, gamma, 0.0, lam_te, wb, loose, vol_target, [wb], solver_cfg)
    r = min(sols, key=lambda s_: s_.fun)
    attempts.append(f"target: budget unattainable without turnover limit; closest point success={r.success}")
    return r.x[:n], "target_budget_unattainable", np.nan, attempts


def solve_policy(policy: Policy, mu: np.ndarray, cov: np.ndarray, gamma: float, lam_te: float, w_pre: np.ndarray,
                 budget, sigma_b: float, no_trade_band: float, w_prev_target: np.ndarray | None = None,
                 solver_cfg: dict | None = None, fallback_lam_to: float = 0.008) -> tuple[SolveResult, np.ndarray]:
    """Primary rule (identical for P0, P1, P2; P0 passes budget=None and its posture is the benchmark):
      1. posture target w* (path independent);
      2. if 0.5 |w* - w_pre| <= no_trade_band and w_pre satisfies every hard constraint: hold w_pre (any risk-budget
         deviation is then bounded by the band and reported as "within no-trade band");
         elif 0.5 |w* - w_pre| <= turnover cap: trade to w*;
         else: w = w_pre + k (w* - w_pre), k = cap / TO (turnover-limited transition along the straight line; a convex
               combination of two feasible points satisfies every linear hard constraint), unmet budget reported;
      3. if the result still violates a hard constraint (possible only if drift pushed w_pre outside the policy):
         slack optimizer from w_pre (stage 2), then hold fallbacks (stage 3).
    Returns (result, posture target)."""
    solver_cfg = solver_cfg or {}
    tol = solver_cfg.get("constraint_tolerance", 1e-7)
    attempts: list[str] = []
    if budget is None:
        target, tstatus, dis = policy.benchmark.copy(), "benchmark", 0.0
        cap, vol_target = policy.turnover_cap, None
    else:
        target, tstatus, dis, attempts = solve_target(policy, mu, cov, gamma, lam_te, budget, sigma_b, solver_cfg)
        cap, vol_target = budget.turnover_cap, budget.vol_multiplier * sigma_b

    def rows_for(w):
        return check_weights(policy, w, budget, w_pre, cap, cov if budget is not None else None, vol_target, tol)

    to = 0.5 * float(np.abs(target - w_pre).sum())
    pre_rows = rows_for(w_pre)
    if to <= no_trade_band and not hard_violations(pre_rows):
        w, status, stage = w_pre.copy(), "hold_within_no_trade_band", 1
    elif to <= cap + 1e-12:
        w, status, stage = target.copy(), "trade_to_target", 1
    else:
        k = cap / to
        w = w_pre + k * (target - w_pre)
        status, stage = "turnover_limited_transition", 2
    if tstatus == "target_budget_unattainable":
        status, stage = status + "_budget_unattainable", 2
    rows = rows_for(w)
    if hard_violations(rows):
        attempts.append(f"rule result violates {hard_violations(rows)}; solving the slack problem from w_pre")
        if budget is None:
            from .risk_budgeting import RiskBudget
            b_eff = RiskBudget(0.0, 1e6, policy.group_max["equity"], policy.group_max["credit"],
                               policy.group_min["liquidity"], cap)
            res = solve_mvo(policy, mu, cov, gamma, fallback_lam_to, lam_te, w_pre, b_eff, sigma_b, w_prev_target,
                            solver_cfg)
        else:
            res = solve_mvo(policy, mu, cov, gamma, fallback_lam_to, lam_te, w_pre, budget, sigma_b, w_prev_target,
                            solver_cfg)
        res.attempts = attempts + res.attempts
        res.stage = max(res.stage, 2)
        return res, target
    unmet = budget_violations(rows)
    if status == "hold_within_no_trade_band":
        unmet = [f"{u} (within no-trade band)" for u in unmet]
    return SolveResult(w, status, stage, True, tstatus, np.nan, dis, rows, attempts, unmet), target
