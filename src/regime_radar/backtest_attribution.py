"""Stage 8: reconciled daily return attribution, crisis / stress-window analysis and risk matching.

Attribution of A vs B (both from the same ledger conventions):
    active_gross_t = gross_A,t - gross_B,t = sum_i (w_A,i,t-1 - w_B,i,t-1) R_i,t            (exact)
    per group g:    sum_{i in g} dw_i,t-1 R_i,t = average-exposure part + timing part
                    average-exposure part = mean_Test(dw_i) R_i,t   (descriptive: uses the whole Test window)
                    timing part           = (dw_i,t-1 - mean_Test(dw_i)) R_i,t
    cost effect_t   = (net_A,t - net_B,t) - active_gross_t
    sum over t of the daily net active returns differs from the terminal wealth difference by compounding; the
    residual is reported explicitly.
"""

from __future__ import annotations

STAGE8_CODE_VERSION = "stage8-2026-10-21"  # must match scripts/10_backtest_portfolios.py

import numpy as np
import pandas as pd

from .backtest_performance import ANN, max_drawdown


def holdings_for_returns(weights_post: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Weights that earn the return of each date = post-trade weights of the previous session."""
    w = weights_post.shift(1)
    return w.loc[dates]


def attribution(a: str, b: str, held: dict, simple: pd.DataFrame, gross: pd.DataFrame, net: pd.DataFrame,
                groups: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(summary rows, daily frame) for A - B. held[p] = weights earning each Test date's return."""
    dates = gross.index
    R = simple.loc[dates]
    dw = held[a] - held[b]
    contrib = dw * R
    avg = dw.mean()
    static = R * avg
    timing = contrib - static
    daily = pd.DataFrame(index=dates)
    for g, members in groups.items():
        daily[f"{g}_total"] = contrib[members].sum(axis=1)
        daily[f"{g}_average_exposure"] = static[members].sum(axis=1)
        daily[f"{g}_timing"] = timing[members].sum(axis=1)
    daily["gross_active"] = gross[a] - gross[b]
    daily["sum_of_groups"] = contrib.sum(axis=1)
    daily["net_active"] = net[a] - net[b]
    daily["cost_effect"] = daily["net_active"] - daily["gross_active"]
    n = len(dates)
    rows = []

    def add(component, group, series, exposure=np.nan):
        rows.append({"portfolio_a": a, "portfolio_b": b, "component": component, "group": group,
                     "sum_daily": float(series.sum()), "annualized_mean": float(series.mean() * ANN),
                     "mean_weight_difference": exposure, "sessions": n})

    for g, members in groups.items():
        e = float(dw[members].sum(axis=1).mean())
        add("asset allocation (total)", g, daily[f"{g}_total"], e)
        add("average-exposure part", g, daily[f"{g}_average_exposure"], e)
        add("timing part", g, daily[f"{g}_timing"], e)
    add("gross active return (sum of groups)", "all", daily["sum_of_groups"])
    add("gross active return (ledger)", "all", daily["gross_active"])
    add("average-exposure part (all groups)", "all", static.sum(axis=1))
    add("timing part (all groups)", "all", timing.sum(axis=1))
    add("transaction-cost effect", "all", daily["cost_effect"])
    add("net active return", "all", daily["net_active"])
    wa, wb = float((1 + net[a]).prod()), float((1 + net[b]).prod())
    rows.append({"portfolio_a": a, "portfolio_b": b, "component": "terminal wealth difference (compounded)",
                 "group": "all", "sum_daily": wa - wb, "annualized_mean": (wa ** (ANN / n) - 1) - (wb ** (ANN / n) - 1),
                 "mean_weight_difference": np.nan, "sessions": n})
    rows.append({"portfolio_a": a, "portfolio_b": b, "component": "compounding residual (terminal difference - sum of "
                 "daily net active)", "group": "all", "sum_daily": (wa - wb) - float(daily["net_active"].sum()),
                 "annualized_mean": np.nan, "mean_weight_difference": np.nan, "sessions": n})
    out = pd.DataFrame(rows)
    out["reconciliation_error"] = float(np.max(np.abs(daily["sum_of_groups"] - daily["gross_active"])))
    return out, daily


# --------------------------------------------------------------------------------------
# Stress windows and objective events
# --------------------------------------------------------------------------------------
def _window_slice(index: pd.DatetimeIndex, start, end) -> pd.DatetimeIndex:
    s = pd.Timestamp(start)
    e = index.max() if end is None else pd.Timestamp(end)
    return index[(index >= s) & (index <= e)]


def window_analysis(windows: dict, net: pd.DataFrame, gross: pd.DataFrame, held: dict, weights_post: dict,
                    ledgers: dict, groups: dict, signals: dict, primary: str) -> pd.DataFrame:
    """Per fixed calendar window and portfolio: returns, drawdown, exposures, signals, costs."""
    rows = []
    for name, (start, end) in windows.items():
        d = _window_slice(net.index, start, end)
        if len(d) == 0:
            continue
        for p in net.columns:
            r = net.loc[d, p]
            g = gross.loc[d, p]
            h = held[p].loc[d]
            eq = h[groups["equity"]].sum(axis=1)
            lg = ledgers[p].set_index("date").loc[d]
            row = {"window": name, "start": d[0], "end": d[-1], "sessions": len(d), "portfolio": p,
                   "gross_cumulative_return": float((1 + g).prod() - 1), "net_cumulative_return": float((1 + r).prod() - 1),
                   "max_drawdown_in_window": max_drawdown(r.to_numpy()),
                   "annualized_volatility": float(r.std(ddof=1) * np.sqrt(ANN)) if len(d) > 2 else np.nan,
                   "equity_weight_at_start": float(eq.iloc[0]), "equity_weight_min": float(eq.min()),
                   "equity_weight_mean": float(eq.mean()), "bil_weight_at_start": float(h["BIL"].iloc[0]),
                   "bil_weight_max": float(h["BIL"].max()), "bil_weight_mean": float(h["BIL"].mean()),
                   "trades": int((lg["gross_notional"] > 0).sum()), "one_way_turnover": float(lg["turnover"].sum()),
                   "cost_primary": float(lg[f"cost_fraction_{primary}"].sum())}
            if p == "P1_regime_aware" and "p" in signals:
                s = signals["p"].loc[d]
                row.update({"hmm_p_non_normal_mean": float(s.mean()), "hmm_p_non_normal_max": float(s.max())})
            if p == "P2_ews_enhanced" and "level" in signals:
                lv = signals["level"].loc[d]
                row.update({"ews_share_ge_amber": float(lv.isin(["AMBER", "ORANGE", "RED"]).mean()),
                            "ews_share_red": float((lv == "RED").mean()),
                            "ews_level_at_start": lv.iloc[0], "ews_max_level": _max_level(lv)})
            rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    for ref in ("P0_strategic", "P1_regime_aware"):
        if ref in net.columns:
            m = out.set_index(["window", "portfolio"])
            base = m.xs(ref, level="portfolio")
            out[f"net_return_minus_{ref}"] = [r["net_cumulative_return"] - base.loc[r["window"], "net_cumulative_return"]
                                              for _, r in out.iterrows()]
            out[f"drawdown_protection_vs_{ref}"] = [r["max_drawdown_in_window"] - base.loc[r["window"], "max_drawdown_in_window"]
                                                    for _, r in out.iterrows()]
    return out


SEV = {"GREEN": 0, "RECOVERY": 1, "AMBER": 2, "ORANGE": 3, "RED": 4}


def _max_level(levels: pd.Series) -> str:
    return max(levels, key=lambda s: SEV.get(s, -1)) if len(levels) else ""


def event_analysis(events: pd.DataFrame, net: pd.DataFrame, holdings: dict, executions: dict, groups: dict,
                   signals: dict, calendar: pd.DatetimeIndex, lookback: int = 21, recovery: int = 63,
                   min_cut: float = 0.01, strategic_equity: float = 0.50) -> pd.DataFrame:
    """Objective SPY drawdown events: protection before onset vs reaction, losses and recovery participation.
    `holdings[p]` = end-of-session (post-trade) weights; events need peak_date, onset_date, trough_date, end_date."""
    rows = []
    pos = {d: i for i, d in enumerate(calendar)}
    for k, ev in events.reset_index(drop=True).iterrows():
        peak, onset, end = ev["peak_date"], ev["onset_date"], ev["end_date"]
        seg = net.index[(net.index >= peak) & (net.index <= end)]
        if len(seg) == 0:
            continue
        spy_trough = ev.get("trough_date")
        search_from = calendar[max(pos.get(peak, 0) - lookback, 0)]
        for p in net.columns:
            r = net[p]
            h = holdings[p]
            eq = h[groups["equity"]].sum(axis=1)
            w_seg = (1 + r.loc[seg]).cumprod()
            trough = spy_trough if spy_trough is not None and spy_trough in r.index else w_seg.idxmin()
            loss_peak_to_trough = float((1 + r.loc[(r.index > peak) & (r.index <= trough)]).prod() - 1)
            after = r.index[r.index > trough][:recovery]
            rec = float((1 + r.loc[after]).prod() - 1) if len(after) else np.nan
            ex = executions.get(p)
            first, kind, sig_date = pd.NaT, "none", pd.NaT
            if ex is not None and len(ex):
                cand = ex[(ex["execution_date"] >= search_from) & (ex["execution_date"] <= trough)]
                lower = cand["executed_exposure_equity"] < cand["pre_exposure_equity"]
                defensive = cand["executed_exposure_equity"] <= strategic_equity - min_cut
                cand = cand[lower & defensive]
                if len(cand):
                    first = cand["execution_date"].iloc[0]
                    sig_date = cand["signal_date"].iloc[0]
                    kind = "before onset (executable protection)" if first < onset else "after onset (reaction)"
            before_onset = calendar[max(pos[onset] - 1, 0)] if onset in pos else onset
            row = {"event": k + 1, "spy_peak_date": peak, "spy_onset_date (-5%)": onset, "spy_trough_date": trough,
                   "spy_end_date": end, "spy_trough_loss_pct": ev.get("trough_loss_pct"),
                   "censored": bool(ev.get("censored", False)), "portfolio": p,
                   "loss_peak_to_trough": loss_peak_to_trough, "max_drawdown_peak_to_end": max_drawdown(r.loc[seg[1:]].to_numpy())
                   if len(seg) > 1 else np.nan,
                   "recovery_return_63_sessions": rec,
                   "equity_at_peak": float(eq.loc[peak]) if peak in eq.index else np.nan,
                   "equity_before_onset": float(eq.loc[before_onset]) if before_onset in eq.index else np.nan,
                   "equity_at_trough": float(eq.loc[trough]) if trough in eq.index else np.nan,
                   "bil_at_peak": float(h["BIL"].loc[peak]) if peak in h.index else np.nan,
                   "bil_at_trough": float(h["BIL"].loc[trough]) if trough in h.index else np.nan,
                   "first_de_risking_signal_date": sig_date, "first_de_risking_execution_date": first,
                   "de_risking_timing": kind}
            if p == "P1_regime_aware" and "p" in signals:
                row["hmm_p_non_normal_at_peak"] = float(signals["p"].get(peak, np.nan))
                row["hmm_p_non_normal_at_onset"] = float(signals["p"].get(onset, np.nan))
            if p == "P2_ews_enhanced" and "level" in signals:
                lv = signals["level"]
                window = lv.loc[(lv.index >= calendar[max(pos.get(peak, 0) - 63, 0)]) & (lv.index <= trough)]
                ge = window[window.isin(["AMBER", "ORANGE", "RED"])]
                row["ews_level_at_peak"] = lv.get(peak, "")
                row["ews_level_at_onset"] = lv.get(onset, "")
                row["ews_first_ge_amber_signal (63 sessions before peak .. trough)"] = ge.index[0] if len(ge) else pd.NaT
            rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    base = out[out["portfolio"] == "P0_strategic"].set_index("event")
    out["protection_vs_P0"] = [r["loss_peak_to_trough"] - base.loc[r["event"], "loss_peak_to_trough"] for _, r in out.iterrows()]
    out["recovery_participation_vs_P0"] = [r["recovery_return_63_sessions"] - base.loc[r["event"], "recovery_return_63_sessions"]
                                           for _, r in out.iterrows()]
    if "P1_regime_aware" in set(out["portfolio"]):
        b1 = out[out["portfolio"] == "P1_regime_aware"].set_index("event")
        out["protection_vs_P1"] = [r["loss_peak_to_trough"] - b1.loc[r["event"], "loss_peak_to_trough"]
                                   for _, r in out.iterrows()]
    return out


def capture_ratios(r: pd.Series, ref: pd.Series, worst_share: float = 0.05) -> dict:
    up, down = ref > 0, ref < 0
    q = ref.quantile(worst_share)
    return {"up_capture": float(r[up].mean() / ref[up].mean()) if up.any() else np.nan,
            "down_capture": float(r[down].mean() / ref[down].mean()) if down.any() else np.nan,
            "mean_return_on_reference_worst_5pct_days": float(r[ref <= q].mean()),
            "reference_mean_on_its_worst_5pct_days": float(ref[ref <= q].mean())}


def vol_matched(r: pd.Series, rf: pd.Series, target_vol: float) -> pd.Series:
    """DESCRIPTIVE ex-post normalization (not tradable): excess returns scaled to a target realized volatility."""
    ex = r - rf.reindex(r.index)
    k = target_vol / (r.std(ddof=1) * np.sqrt(ANN))
    return rf.reindex(r.index) + k * ex
