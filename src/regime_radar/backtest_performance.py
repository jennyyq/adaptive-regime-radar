"""Stage 8: realized performance and risk metrics from daily wealth / return series (definitions frozen in
configs/stage8_protocol.json before Test access).

Conventions: daily SIMPLE returns; 252 sessions per year for every strategy; excess returns over the realized
daily BIL return (evaluation benchmark only; it is never added to portfolio returns); maximum drawdown on the
daily wealth path including the initial 1.0; VaR / CVaR 95% with the Stage 7 historical convention.
"""

from __future__ import annotations

STAGE8_CODE_VERSION = "stage8-2026-10-21"  # must match scripts/10_backtest_portfolios.py

import numpy as np
import pandas as pd

from .portfolio_risk import historical_var_cvar

ANN = 252


def wealth(r: pd.Series) -> pd.Series:
    return (1 + r).cumprod()


def drawdown(r: pd.Series) -> pd.Series:
    w = np.r_[1.0, (1 + r.to_numpy(float)).cumprod()]
    dd = w / np.maximum.accumulate(w) - 1
    return pd.Series(dd[1:], index=r.index)


def max_drawdown(r) -> float:
    r = np.asarray(r, float)
    w = np.r_[1.0, np.cumprod(1 + r)]
    return float((w / np.maximum.accumulate(w) - 1).min())


def drawdown_episodes(r: pd.Series, min_depth: float = 0.0) -> pd.DataFrame:
    """Peak -> trough -> recovery episodes of the wealth path (depth as a negative fraction)."""
    w = pd.Series(np.r_[1.0, (1 + r.to_numpy(float)).cumprod()],
                  index=pd.DatetimeIndex([r.index[0] - pd.Timedelta(days=1)]).append(r.index))
    peak = w.cummax()
    under = w < peak
    rows, start = [], None
    idx = w.index
    for i in range(len(w)):
        if under.iloc[i] and start is None:
            start = i - 1
        if start is not None and (not under.iloc[i] or i == len(w) - 1):
            seg = w.iloc[start:i + 1]
            trough_i = int(np.argmin(seg.to_numpy())) + start
            depth = float(w.iloc[trough_i] / w.iloc[start] - 1)
            recovered = not under.iloc[i]
            rows.append({"peak_date": idx[start], "trough_date": idx[trough_i],
                         "recovery_date": idx[i] if recovered else pd.NaT, "depth": depth,
                         "sessions_peak_to_trough": trough_i - start,
                         "sessions_trough_to_recovery": (i - trough_i) if recovered else np.nan,
                         "sessions_underwater": i - start if recovered else len(w) - 1 - start,
                         "recovered": recovered})
            start = None
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out[out["depth"] <= -abs(min_depth)].reset_index(drop=True)
    return out


def period_returns(r: pd.Series, freq: str) -> pd.Series:
    """Compounded calendar-period returns: 'W' (Monday-Sunday weeks), 'M' (months), 'Y' (years)."""
    key = {"W": r.index.to_period("W-SUN"), "M": r.index.to_period("M"), "Y": r.index.to_period("Y")}[freq]
    return (1 + r).groupby(key).prod() - 1


def metrics(r: pd.Series, rf: pd.Series, ref: dict[str, pd.Series] | None = None) -> dict:
    """Return / risk / risk-adjusted metrics of one daily simple-return series."""
    r = r.astype(float)
    rf = rf.reindex(r.index).astype(float)
    n = len(r)
    w_end = float((1 + r).prod())
    years = n / ANN
    ex = r - rf
    vol = float(r.std(ddof=1) * np.sqrt(ANN))
    ex_sd = float(ex.std(ddof=1) * np.sqrt(ANN))
    dd_dev = float(np.sqrt(np.mean(np.minimum(ex.to_numpy(), 0) ** 2)) * np.sqrt(ANN))
    mdd = max_drawdown(r)
    cagr = w_end ** (1 / years) - 1 if years > 0 else np.nan
    var, cvar = historical_var_cvar(r.to_numpy()[:, None], np.array([1.0]), 0.95)
    eps = drawdown_episodes(r)
    longest = int(eps["sessions_underwater"].max()) if not eps.empty else 0
    if not eps.empty:
        worst = eps.loc[eps["depth"].idxmin()]
        mdd_peak, mdd_trough, mdd_rec = worst["peak_date"], worst["trough_date"], worst["recovery_date"]
        mdd_days = worst["sessions_underwater"]
    else:
        mdd_peak = mdd_trough = mdd_rec = pd.NaT
        mdd_days = 0
    out = {"sessions": n, "years": years, "cumulative_return": w_end - 1, "cagr": cagr,
           "arithmetic_annual_return": float(r.mean() * ANN), "annual_volatility": vol,
           "excess_return_annual": float(ex.mean() * ANN),
           "sharpe": float(ex.mean() * ANN / ex_sd) if ex_sd > 0 else np.nan,
           "downside_deviation": dd_dev, "sortino": float(ex.mean() * ANN / dd_dev) if dd_dev > 0 else np.nan,
           "max_drawdown": mdd, "calmar": float(cagr / abs(mdd)) if mdd < 0 else np.nan,
           "max_drawdown_peak": mdd_peak, "max_drawdown_trough": mdd_trough, "max_drawdown_recovery": mdd_rec,
           "max_drawdown_sessions_underwater": mdd_days, "longest_underwater_sessions": longest,
           "worst_day": float(r.min()), "worst_week": float(period_returns(r, "W").min()),
           "worst_month": float(period_returns(r, "M").min()), "var95_daily": var, "cvar95_daily": cvar,
           "best_day": float(r.max()), "positive_day_share": float((r > 0).mean())}
    for name, rr in (ref or {}).items():
        a = r - rr.reindex(r.index)
        te = float(a.std(ddof=1) * np.sqrt(ANN))
        out[f"active_return_vs_{name}"] = float(a.mean() * ANN)
        out[f"tracking_error_vs_{name}"] = te
        out[f"information_ratio_vs_{name}"] = float(a.mean() * ANN / te) if te > 0 else np.nan
    return out


def rolling_sharpe(r: pd.Series, rf: pd.Series, window: int) -> pd.Series:
    ex = r - rf.reindex(r.index)
    return ex.rolling(window).mean() * ANN / (ex.rolling(window).std() * np.sqrt(ANN))


def rolling_vol(r: pd.Series, window: int) -> pd.Series:
    return r.rolling(window).std() * np.sqrt(ANN)


def implementation_stats(ledger: pd.DataFrame, decisions: pd.DataFrame, weights: pd.DataFrame, groups: dict,
                         years: float, primary_bps: float) -> dict:
    """Turnover, notional, costs, trading frequency and exposures over the evaluation window."""
    traded = ledger["gross_notional"] > 0
    to = float(ledger["turnover"].sum())
    out = {"executions_with_trades": int(traded.sum()), "rebalances_per_year": float(traded.sum() / years),
           "one_way_turnover_total": to, "annualized_one_way_turnover": to / years,
           "gross_traded_notional_total": float(ledger["gross_notional"].sum()),
           "average_holding_period_years": (years / to) if to > 0 else np.nan}
    for c in [c for c in ledger.columns if c.startswith("cost_fraction_")]:
        b = c.replace("cost_fraction_", "")
        out[f"total_cost_fraction_{b}"] = float(ledger[c].sum())
    st = ledger["stress_on"] & traded
    c10 = f"cost_fraction_{primary_bps:g}bps"
    out["cost_primary_in_stress"] = float(ledger.loc[st, c10].sum())
    out["cost_primary_in_normal"] = float(ledger.loc[traded & ~ledger["stress_on"], c10].sum())
    out["executions_in_stress"] = int(st.sum())
    for g, members in groups.items():
        out[f"mean_weight_{g}"] = float(weights[members].sum(axis=1).mean())
    out["mean_weight_BIL"] = float(weights["BIL"].mean()) if "BIL" in weights else np.nan
    if decisions is not None and len(decisions) and "budget_d_input" in decisions:
        d = decisions["budget_d_input"].astype(float)
        out["decisions"] = int(len(decisions))
        out["mean_defensiveness_d"] = float(d.mean()) if d.notna().any() else np.nan
        out["share_decisions_defensive_d_gt_0"] = float((d > 1e-12).mean()) if d.notna().any() else np.nan
        out["hold_decisions_share"] = float((decisions["order_kind"] == "hold").mean())
    return out
