"""Stage 7: ex-ante portfolio diagnostics, historical stress scenarios and summary tables.

Every quantity here is EX ANTE (predicted from information through the decision date) unless its name says
otherwise. Realized portfolio performance (returns, Sharpe, drawdowns, wealth) is NOT computed in Stage 7.
"""

from __future__ import annotations

STAGE7_CODE_VERSION = "stage7-2026-10-21"  # must match scripts/09_optimize_portfolio.py

import numpy as np
import pandas as pd

from .portfolio_execution import one_way_turnover, transaction_cost
from .portfolio_risk import (effective_n, herfindahl, historical_var_cvar, portfolio_vol, risk_contributions,
                             tracking_error)

SCENARIO_WINDOW = 21


def decision_row(policy, w, mu, cov, scenarios, w_pre, budget, sigma_b, cost_cfg, stress_on, alpha=0.95) -> dict:
    wb = policy.benchmark
    vol = portfolio_vol(w, cov)
    var, cvar = historical_var_cvar(scenarios, w, alpha)
    var_b, cvar_b = historical_var_cvar(scenarios, wb, alpha)
    to = one_way_turnover(w, w_pre)
    row = {f"w_{a}": float(x) for a, x in zip(policy.assets, w)}
    row.update({"sum_weights": float(w.sum()), "min_weight": float(w.min()),
                "expected_return_ex_ante": float(w @ mu), "vol_ex_ante": vol,
                "var95_daily_ex_ante": var, "cvar95_daily_ex_ante": cvar,
                "benchmark_vol_ex_ante": float(sigma_b), "benchmark_cvar95_daily_ex_ante": cvar_b,
                "tracking_error_ex_ante": tracking_error(w, wb, cov),
                "active_share": float(0.5 * np.abs(w - wb).sum()),
                "herfindahl": herfindahl(w), "effective_n": effective_n(w),
                "turnover_ex_ante_proxy": to, "stress_friction_on": bool(stress_on)})
    for g, e in policy.exposures(w).items():
        row[f"exposure_{g}"] = e
    for bps in cost_cfg["scenarios_bps_of_traded_notional"]:
        row[f"cost_{int(bps)}bps"] = transaction_cost(to, bps, stress_on, cost_cfg["stress_friction_multiplier"])
    if budget is not None:
        row.update({"budget_d": budget.d, "budget_vol_target": budget.vol_multiplier * sigma_b,
                    "budget_vol_multiplier": budget.vol_multiplier, "budget_equity_max": budget.equity_max,
                    "budget_credit_max": budget.credit_max, "budget_liquidity_min": budget.liquidity_min,
                    "budget_turnover_cap": budget.turnover_cap,
                    "budget_vol_achieved_ratio": vol / (budget.vol_multiplier * sigma_b)})
    return row


def contribution_rows(policy, w, cov) -> dict:
    mcr, ccr = risk_contributions(w, cov)
    s = portfolio_vol(w, cov)
    out = {}
    for a, m, c in zip(policy.assets, mcr, ccr):
        out[f"mcr_{a}"] = float(m)
        out[f"ccr_{a}"] = float(c)
        out[f"pct_risk_{a}"] = float(c / s) if s > 0 else np.nan
    return out


def historical_scenarios(simple_train: pd.DataFrame, wb: np.ndarray, window: int = SCENARIO_WINDOW) -> pd.DataFrame:
    """Actual Train windows (cumulative SIMPLE returns over `window` sessions) selected by fixed, interpretable
    rules: worst SPY window (equity crash), worst HYG (credit), worst IEF (rates shock), worst DBC (commodity),
    worst TIP (inflation-linked / real-rate shock), worst benchmark window (broad multi-asset drawdown) and the
    window with the highest SPY realized volatility (volatility spike). Diagnostics only; never signal inputs."""
    growth = (1 + simple_train).rolling(window).apply(np.prod, raw=True) - 1
    bench = pd.Series(simple_train.to_numpy() @ wb, index=simple_train.index)   # daily-rebalanced benchmark
    bench_cum = (1 + bench).rolling(window).apply(np.prod, raw=True) - 1
    vol = simple_train["SPY"].rolling(window).std()
    picks = {"equity_crash (worst SPY)": growth["SPY"].idxmin(), "credit_deterioration (worst HYG)": growth["HYG"].idxmin(),
             "rates_shock (worst IEF)": growth["IEF"].idxmin(), "commodity_shock (worst DBC)": growth["DBC"].idxmin(),
             "real_rate_shock (worst TIP)": growth["TIP"].idxmin(),
             "broad_multi_asset_drawdown (worst benchmark)": bench_cum.idxmin(),
             "volatility_spike (max SPY realized vol)": vol.idxmax()}
    rows = []
    for name, end in picks.items():
        start = simple_train.index[simple_train.index.get_loc(end) - window + 1]
        rows.append({"scenario": name, "type": "historical (Train)", "start": start.date(), "end": end.date(),
                     **{a: float(growth.at[end, a]) for a in simple_train.columns}})
    return pd.DataFrame(rows)


HYPOTHETICAL = {
    "hypothetical_stagflation (assumption)": {"SPY": -0.15, "EFA": -0.15, "EEM": -0.18, "IEF": -0.05, "TIP": -0.02,
                                              "HYG": -0.08, "GLD": 0.05, "DBC": 0.10, "BIL": 0.0},
    "hypothetical_liquidity_crisis (assumption)": {"SPY": -0.20, "EFA": -0.22, "EEM": -0.25, "IEF": 0.03, "TIP": -0.01,
                                                   "HYG": -0.12, "GLD": -0.03, "DBC": -0.15, "BIL": 0.0},
}


def scenario_losses(weights: dict[str, np.ndarray], scenarios: pd.DataFrame, assets: list[str]) -> pd.DataFrame:
    rows = []
    for _, s in scenarios.iterrows():
        r = s[assets].to_numpy(float)
        for pid, w in weights.items():
            rows.append({"scenario": s["scenario"], "type": s["type"], "portfolio": pid,
                         "scenario_return": float(w @ r), "scenario_loss": float(-(w @ r))})
    return pd.DataFrame(rows)
