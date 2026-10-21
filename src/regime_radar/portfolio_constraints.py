"""Stage 7: long-only investment policy constraints, feasibility checks and constraint diagnostics.

Hard constraints (always enforced):
  sum w = 1; w_i >= 0 (no shorts, no leverage); w_i <= max_weight_i; |w_i - w_b,i| <= max_active_weight;
  static group limits (equity, duration, credit, real assets, liquidity); one-way turnover <= turnover cap.
Risk-budget constraints (state dependent, from risk_budgeting.RiskBudget):
  equity <= equity_max(d); credit <= credit_max(d); liquidity (BIL) >= liquidity_min(d);
  ex-ante volatility <= vol_multiplier(d) * benchmark ex-ante volatility.
The effective group limit is the tighter of the static limit and the risk-budget limit.
"""

from __future__ import annotations

STAGE7_CODE_VERSION = "stage7-2026-10-21"  # must match scripts/09_optimize_portfolio.py

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

TOL = 1e-7


@dataclass
class Policy:
    assets: list[str]
    groups: dict[str, list[str]]
    benchmark: np.ndarray
    max_weight: np.ndarray
    min_weight: np.ndarray
    max_active: float
    group_min: dict[str, float]
    group_max: dict[str, float]
    turnover_cap: float
    raw: dict = field(default_factory=dict)

    def group_vector(self, g: str) -> np.ndarray:
        return np.array([1.0 if a in self.groups[g] else 0.0 for a in self.assets])

    def exposures(self, w: np.ndarray) -> dict[str, float]:
        return {g: float(self.group_vector(g) @ w) for g in self.groups}


def load_policy(policy_path: Path, project_path: Path) -> Policy:
    """Policy from configs/portfolio_policy.json; the benchmark comes from configs/project.json (frozen) and is
    verified, never overwritten."""
    cfg = json.loads(Path(policy_path).read_text(encoding="utf-8"))
    proj = json.loads(Path(project_path).read_text(encoding="utf-8"))
    assets = cfg["universe"]["etfs"]
    if assets != proj["investment_etfs"]:
        raise ValueError(f"policy universe {assets} differs from configs/project.json {proj['investment_etfs']}")
    wb = np.array([proj["strategic_weights"][a] for a in assets], float)
    if abs(wb.sum() - 1) > 1e-12 or (wb < 0).any():
        raise ValueError("strategic weights must be non-negative and sum to one")
    c = cfg["constraints"]
    return Policy(assets, cfg["universe"]["groups"], wb,
                  np.array([c["max_weight"][a] for a in assets], float),
                  np.array([c["min_weight"][a] for a in assets], float), float(c["max_active_weight"]),
                  {g: float(v["min"]) for g, v in c["group_limits"].items()},
                  {g: float(v["max"]) for g, v in c["group_limits"].items()},
                  float(c["max_turnover_per_rebalance"]), cfg)


def bounds(policy: Policy) -> tuple[np.ndarray, np.ndarray]:
    lo = np.maximum(policy.min_weight, policy.benchmark - policy.max_active)
    hi = np.minimum(policy.max_weight, policy.benchmark + policy.max_active)
    return np.maximum(lo, 0.0), hi


def group_limits(policy: Policy, budget=None) -> tuple[dict, dict]:
    gmin, gmax = dict(policy.group_min), dict(policy.group_max)
    if budget is not None:
        gmax["equity"] = min(gmax["equity"], budget.equity_max)
        gmax["credit"] = min(gmax["credit"], budget.credit_max)
        gmin["liquidity"] = max(gmin["liquidity"], budget.liquidity_min)
    return gmin, gmax


def linear_feasible(policy: Policy, budget=None, pretrade: np.ndarray | None = None, turnover_cap: float | None = None
                    ) -> tuple[bool, np.ndarray | None]:
    """LP feasibility of all LINEAR constraints (bounds, groups, budget groups, turnover cap). The volatility
    constraint is checked by the optimizer. Variables [w, u, v] with w - u + v = pretrade."""
    n = len(policy.assets)
    lo, hi = bounds(policy)
    gmin, gmax = group_limits(policy, budget)
    a_ub, b_ub = [], []
    for g in policy.groups:
        gv = np.r_[policy.group_vector(g), np.zeros(2 * n)]
        a_ub += [gv, -gv]
        b_ub += [gmax[g], -gmin[g]]
    a_eq = [np.r_[np.ones(n), np.zeros(2 * n)]]
    b_eq = [1.0]
    bnds = [(lo[i], hi[i]) for i in range(n)] + [(0, None)] * (2 * n)
    if pretrade is not None and turnover_cap is not None:
        for i in range(n):
            row = np.zeros(3 * n)
            row[i], row[n + i], row[2 * n + i] = 1, -1, 1
            a_eq.append(row)
            b_eq.append(pretrade[i])
        a_ub.append(np.r_[np.zeros(n), 0.5 * np.ones(2 * n)])
        b_ub.append(turnover_cap)
    else:
        bnds = bnds[:n] + [(0, 0)] * (2 * n)
    res = linprog(np.zeros(3 * n), A_ub=np.array(a_ub), b_ub=np.array(b_ub), A_eq=np.array(a_eq), b_eq=np.array(b_eq),
                  bounds=bnds, method="highs")
    return bool(res.status == 0), (res.x[:n] if res.status == 0 else None)


def check_weights(policy: Policy, w: np.ndarray, budget=None, pretrade: np.ndarray | None = None,
                  turnover_cap: float | None = None, cov: np.ndarray | None = None, vol_target: float | None = None,
                  tol: float = TOL) -> list[dict]:
    """Every constraint as a row: name, kind (hard / budget), value, limit, slack (>= 0 means satisfied), active,
    violated."""
    rows = []

    def add(name, kind, value, limit, sense):
        slack = (limit - value) if sense == "<=" else (value - limit)
        rows.append({"constraint": name, "kind": kind, "value": float(value), "limit": float(limit), "sense": sense,
                     "slack": float(slack), "active": bool(abs(slack) <= 1e-6), "violated": bool(slack < -tol)})

    lo, hi = bounds(policy)
    add("sum_weights", "hard", abs(w.sum() - 1), 0.0, "<=")
    for i, a in enumerate(policy.assets):
        add(f"min_{a}", "hard", w[i], lo[i], ">=")
        add(f"max_{a}", "hard", w[i], hi[i], "<=")
    gmin, gmax = group_limits(policy, budget)
    for g in policy.groups:
        e = float(policy.group_vector(g) @ w)
        add(f"{g}_max", "hard", e, policy.group_max[g], "<=")          # static policy limits are always hard
        add(f"{g}_min", "hard", e, policy.group_min[g], ">=")
        if gmax[g] < policy.group_max[g]:
            add(f"{g}_max_budget", "budget", e, gmax[g], "<=")          # risk-budget tightening
        if gmin[g] > policy.group_min[g]:
            add(f"{g}_min_budget", "budget", e, gmin[g], ">=")
    if pretrade is not None and turnover_cap is not None:
        add("turnover_cap", "hard", 0.5 * np.abs(w - pretrade).sum(), turnover_cap, "<=")
    if cov is not None and vol_target is not None:
        add("volatility_target", "budget", float(np.sqrt(max(w @ cov @ w, 0))), vol_target, "<=")
    return rows


def hard_violations(rows: list[dict]) -> list[str]:
    return [r["constraint"] for r in rows if r["violated"] and r["kind"] == "hard"]


def budget_violations(rows: list[dict]) -> list[str]:
    return [r["constraint"] for r in rows if r["violated"] and r["kind"] == "budget"]
