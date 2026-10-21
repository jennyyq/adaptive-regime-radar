"""Stage 8: prespecified conclusion rules, figures, final research report and console summary.

Every number is read from the computed Stage 8 tables. The conclusion labels follow the decision rules frozen in
configs/stage8_protocol.json (they were written before the Test split was opened).
"""

from __future__ import annotations

STAGE8_CODE_VERSION = "stage8-2026-10-21"  # must match scripts/10_backtest_portfolios.py

from pathlib import Path

import numpy as np
import pandas as pd

from .quality import _md_table
from .regime_report import GRID, INK, INK_2, MUTED, SURFACE, _save, _style, _title

PIDS = ("P0_strategic", "P1_regime_aware", "P2_ews_enhanced")
SUPP = ("B1_60_40", "B2_constant_defensiveness", "B3_strategic_buy_and_hold")
PCOL = {"P0_strategic": "#898781", "P1_regime_aware": "#2a78d6", "P2_ews_enhanced": "#eb6834",
        "B1_60_40": "#1baf7a", "B2_constant_defensiveness": "#6250d6", "B3_strategic_buy_and_hold": "#eda100"}
SHORT = {"P0_strategic": "P0 strategic", "P1_regime_aware": "P1 HMM regime", "P2_ews_enhanced": "P2 Full EWS",
         "B1_60_40": "B1 60/40", "B2_constant_defensiveness": "B2 constant d", "B3_strategic_buy_and_hold": "B3 buy & hold"}
GROUP_COL = {"equity": "#2a78d6", "duration": "#1baf7a", "credit": "#eda100", "real_assets": "#e34948",
             "liquidity": "#6250d6"}
LEVEL_COL = {"GREEN": "#1baf7a", "AMBER": "#eda100", "ORANGE": "#eb6834", "RED": "#c93131", "RECOVERY": "#6aa4e6"}


def _pct(x, d=1):
    return "n/a" if x is None or not np.isfinite(x) else f"{100 * x:.{d}f}%"


def _num(x, d=2):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


# --------------------------------------------------------------------------------------
# Prespecified conclusion rules
# --------------------------------------------------------------------------------------
def _boot(stats, a, b, stat, setting="primary"):
    s = stats[(stats["portfolio_a"] == a) & (stats["portfolio_b"] == b) & (stats["statistic"] == stat) &
              stats["method"].str.contains(f"({setting})", regex=False)]
    if s.empty:
        return np.nan, np.nan, np.nan
    r = s.iloc[0]
    return float(r["estimate"]), float(r["ci_low"]), float(r["ci_high"])


def conclusions(protocol, perf, stats, attribution, cost_sens, reg_rows, events, windows, prim_bps) -> dict:
    dr = protocol["decision_rules"]
    msh = dr["economically_meaningful_sharpe_difference"]
    mdd_pp = dr["economically_meaningful_drawdown_reduction_pct_points"] / 100
    net = perf[perf["basis"] == f"net_{prim_bps:g}bps"].set_index("portfolio")
    out = {"rules": "configs/stage8_protocol.json decision_rules (frozen before Test)"}

    def sharpe(a, b):
        e, lo, hi = _boot(stats, a, b, "sharpe_difference")
        return {"estimate": e, "ci_low": lo, "ci_high": hi, "excludes_zero": bool(lo > 0 or hi < 0)}

    def robust(a, b):
        sub = cost_sens[cost_sens["portfolio"] == f"{a} - {b}"]
        signs = {r["basis"]: np.sign(r["sharpe"]) for _, r in sub.iterrows() if r["basis"].startswith("net_")}
        return len(set(signs.values())) == 1, {k: float(v) for k, v in signs.items()}

    s10, s20, s21 = sharpe("P1_regime_aware", "P0_strategic"), sharpe("P2_ews_enhanced", "P0_strategic"), \
        sharpe("P2_ews_enhanced", "P1_regime_aware")
    rob = {k: robust(*k.split("|")) for k in ("P1_regime_aware|P0_strategic", "P2_ews_enhanced|P0_strategic",
                                              "P2_ews_enhanced|P1_regime_aware")}

    def verdict(s):
        if s["estimate"] > 0 and s["ci_low"] > 0:
            return "YES - higher net Sharpe ratio and the 95% interval excludes 0"
        if s["estimate"] > 0:
            return "NOT DEMONSTRATED - higher point estimate, but the 95% interval contains 0"
        return "NO - net Sharpe ratio not higher"

    out["A"] = {"question": "Does P1 outperform P0 on risk-adjusted performance after costs?", **s10, "verdict": verdict(s10)}
    out["B"] = {"question": "Does P2 outperform P1 on risk-adjusted performance after costs?", **s21, "verdict": verdict(s21)}
    m2, m1, m0 = net.loc["P2_ews_enhanced", "max_drawdown"], net.loc["P1_regime_aware", "max_drawdown"], \
        net.loc["P0_strategic", "max_drawdown"]
    e20, lo20, hi20 = _boot(stats, "P2_ews_enhanced", "P0_strategic", "max_drawdown_difference")
    e21, lo21, hi21 = _boot(stats, "P2_ews_enhanced", "P1_regime_aware", "max_drawdown_difference")
    meaningful = (m2 - m0) >= mdd_pp
    out["C"] = {"question": "Does P2 reduce maximum drawdown by an economically meaningful amount?",
                "p2_max_drawdown": m2, "p1_max_drawdown": m1, "p0_max_drawdown": m0,
                "reduction_vs_p0_pct_points": 100 * (m2 - m0), "reduction_vs_p1_pct_points": 100 * (m2 - m1),
                "bootstrap_ci_vs_p0": [lo20, hi20], "bootstrap_ci_vs_p1": [lo21, hi21],
                "verdict": ("YES vs P0" if meaningful else "NO vs P0") + f" (threshold {dr['economically_meaningful_drawdown_reduction_pct_points']} "
                           f"pp); vs P1: {'YES' if (m2 - m1) >= mdd_pp else 'NO'}"}
    wrows = []
    if windows is not None and len(windows):
        for w, g in windows.groupby("window", sort=False):
            gi = g.set_index("portfolio")
            if {"P2_ews_enhanced", "P1_regime_aware", "P0_strategic"} <= set(gi.index):
                wrows.append({"window": w, "p2_minus_p0_net": gi.loc["P2_ews_enhanced", "net_cumulative_return"] -
                              gi.loc["P0_strategic", "net_cumulative_return"],
                              "p2_minus_p1_net": gi.loc["P2_ews_enhanced", "net_cumulative_return"] -
                              gi.loc["P1_regime_aware", "net_cumulative_return"],
                              "p2_drawdown_vs_p1": gi.loc["P2_ews_enhanced", "max_drawdown_in_window"] -
                              gi.loc["P1_regime_aware", "max_drawdown_in_window"]})
    ev_p2 = events[events["portfolio"] == "P2_ews_enhanced"] if events is not None and len(events) else pd.DataFrame()
    before = int((ev_p2["de_risking_timing"] == "before onset (executable protection)").sum()) if len(ev_p2) else 0
    after = int((ev_p2["de_risking_timing"] == "after onset (reaction)").sum()) if len(ev_p2) else 0
    better_p1 = sum(1 for r in wrows if r["p2_minus_p1_net"] > 0)
    better_p0 = sum(1 for r in wrows if r["p2_minus_p0_net"] > 0)
    shallower = sum(1 for r in wrows if r["p2_drawdown_vs_p1"] > 0)
    out["D"] = {"question": "Does P2 provide additional value in the predeclared stress periods?", "windows": wrows,
                "events": int(len(ev_p2)), "p2_de_risked_before_onset": before, "p2_de_risked_after_onset": after,
                "verdict": f"net return: P2 above P1 in {better_p1} and above P0 in {better_p0} of {len(wrows)} fixed "
                           f"windows; shallower in-window drawdown than P1 in {shallower}; in the {len(ev_p2)} objective "
                           f"SPY drawdown events P2 was already defensive (executed) before the onset in {before}, "
                           f"de-risked only after the onset in {after}, and did not de-risk in {len(ev_p2) - before - after}"}
    def sign_text(v):
        vals = set(v[1].values())
        if vals == {1.0}:
            return "higher net Sharpe at 5, 10 and 25 bps"
        if vals <= {-1.0, 0.0}:
            return "not higher at any cost level"
        return "sign of the net Sharpe difference changes with the cost level"
    out["E"] = {"question": "Are the improvements robust across transaction-cost assumptions?",
                "sign_of_net_sharpe_difference": {k: v[1] for k, v in rob.items()},
                "verdict": "; ".join(f"{SHORT[k.split('|')[0]].split()[0]} vs {SHORT[k.split('|')[1]].split()[0]}: "
                                     f"{sign_text(v)}" for k, v in rob.items())}
    att = attribution.set_index(["portfolio_a", "portfolio_b", "component", "group"])["annualized_mean"]

    def split(a, b):
        avg = att.get((a, b, "average-exposure part (all groups)", "all"), np.nan)
        tim = att.get((a, b, "timing part (all groups)", "all"), np.nan)
        eq_avg = att.get((a, b, "average-exposure part", "equity"), np.nan)
        eq_tim = att.get((a, b, "timing part", "equity"), np.nan)
        return avg, tim, eq_avg, eq_tim
    regs = {f"{r['portfolio']}|{r['reference']}": r for r in reg_rows}
    f_rows = {}
    for a, b in (("P2_ews_enhanced", "P0_strategic"), ("P2_ews_enhanced", "P1_regime_aware"), ("P1_regime_aware", "P0_strategic")):
        avg, tim, ea, et = split(a, b)
        r = regs.get(f"{a}|{b}", {})
        lo, hi = r.get("alpha_ci_low", np.nan), r.get("alpha_ci_high", np.nan)
        if np.isfinite(avg) and np.isfinite(tim) and abs(avg) > abs(tim):
            label = "exposure-driven (average weight difference dominates)"
        elif np.isfinite(lo) and (lo > 0 or hi < 0):
            label = "timing-driven, alpha interval excludes 0"
        else:
            label = "timing component larger but not statistically distinguishable (alpha interval contains 0)"
        f_rows[f"{a} vs {b}"] = {"average_exposure_part": avg, "timing_part": tim, "equity_average_exposure": ea,
                                 "equity_timing": et, "alpha_annualized": r.get("alpha_annualized", np.nan),
                                 "alpha_ci": [lo, hi], "beta": r.get("beta", np.nan), "classification": label,
                                 "explained_by_exposure": label.startswith("exposure")}
    out["F"] = {"question": "Is any advantage mainly explained by reduced equity exposure rather than superior timing?",
                "pairs": f_rows, "verdict": "; ".join(
                    f"{SHORT[k.split(' vs ')[0]]} vs {SHORT[k.split(' vs ')[1]]}: {v['classification']} (gross active "
                    f"{_pct(v['average_exposure_part'] + v['timing_part'], 2)}/yr = exposure {_pct(v['average_exposure_part'], 2)} "
                    f"+ timing {_pct(v['timing_part'], 2)}; beta {_num(v['beta'])})" for k, v in f_rows.items())}
    cd = net.loc["P2_ews_enhanced", "total_cost_fraction_%gbps" % prim_bps] - net.loc["P1_regime_aware", "total_cost_fraction_%gbps" % prim_bps]
    justified = s21["estimate"] >= msh and s21["ci_low"] > 0 and rob["P2_ews_enhanced|P1_regime_aware"][0]
    out["G"] = {"question": "Does additional EWS complexity justify its operational cost?",
                "extra_turnover_per_year": net.loc["P2_ews_enhanced", "annualized_one_way_turnover"] -
                net.loc["P1_regime_aware", "annualized_one_way_turnover"], "extra_cost_total": cd,
                "verdict": "YES" if justified else "NO - the prespecified bar (net Sharpe +0.10 over P1 with an interval "
                                                   "excluding 0, robust to costs) is not met"}
    supported = [k for k, s in (("P1-P0", s10), ("P2-P0", s20), ("P2-P1", s21)) if s["excludes_zero"]]
    out["H"] = {"question": "Do statistical confidence intervals support the economic conclusions?",
                "comparisons_with_sharpe_interval_excluding_zero": supported,
                "verdict": ("intervals exclude 0 for " + ", ".join(supported)) if supported else
                "no primary net Sharpe difference has a 95% interval excluding 0"}
    const = {}
    for a in ("P1_regime_aware", "P2_ews_enhanced"):
        e, lo, hi = _boot(stats, a, "B2_constant_defensiveness", "sharpe_difference")
        const[a] = {"estimate": e, "ci_low": lo, "ci_high": hi, "similar": bool(abs(e) < msh and lo <= 0 <= hi)}
    out["constant_defensiveness"] = const
    # outcome classification
    o1 = (s21["estimate"] >= msh and s21["ci_low"] > 0 and s20["ci_low"] > 0 and rob["P2_ews_enhanced|P1_regime_aware"][0]
          and rob["P2_ews_enhanced|P0_strategic"][0])
    o3 = s10["estimate"] <= 0 and s20["estimate"] <= 0
    o2 = (not o1) and s10["estimate"] > 0 and s10["ci_low"] > 0 and (s21["estimate"] < msh or s21["ci_low"] <= 0 <= s21["ci_high"])
    outcome = ("Outcome 1 - Full EWS adds credible incremental value" if o1 else
               "Outcome 3 - Strategic allocation remains preferable" if o3 else
               "Outcome 2 - Simple HMM allocation is sufficient" if o2 else "Outcome 4 - Evidence is inconclusive")
    out["outcome"] = outcome
    out["sharpe"] = {"P1-P0": s10, "P2-P0": s20, "P2-P1": s21}
    return out


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
def _legend(ax, **kw):
    leg = ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, **kw)
    return leg


def _test_axis(ax):
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def fig_wealth(c, path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(13, 6), facecolor=SURFACE)
    _style(ax)
    b = c["prim_bps"]
    for p in SUPP:
        w = (1 + c["net"][b][p]).cumprod()
        ax.plot(w.index, w, color=PCOL[p], lw=1.0, alpha=0.75, ls=(0, (4, 2)), label=f"{SHORT[p]} (net, context)")
    for p in PIDS:
        w = (1 + c["net"][b][p]).cumprod()
        g = (1 + c["gross"][p]).cumprod()
        ax.plot(g.index, g, color=PCOL[p], lw=1.0, ls=":", alpha=0.9)
        ax.plot(w.index, w, color=PCOL[p], lw=2.0,
                label=f"{SHORT[p]} net {b:g} bps (end {w.iloc[-1]:.3f}; gross {g.iloc[-1]:.3f})")
    ax.axhline(1.0, color=MUTED, lw=0.7)
    ax.set_ylabel("wealth (1.0 at the last Validation close)", fontsize=8, color=INK_2)
    _test_axis(ax)
    _legend(ax, loc="upper right", ncol=2)
    _title(fig, "Cumulative wealth in the reserved Test period - net of costs (solid) and gross (dotted)",
           f"Daily self-financing ledger, execution at the close of t+1, {b:g} bps of gross traded notional (x2 in cost "
           "stress). Gross and net share one trading path; the dotted-solid gap is the cost drag.")
    _save(fig, path)


def fig_drawdowns(c, path):
    import matplotlib.pyplot as plt
    from .backtest_performance import drawdown
    fig, ax = plt.subplots(figsize=(13, 5), facecolor=SURFACE)
    _style(ax)
    b = c["prim_bps"]
    for p in PIDS + ("B2_constant_defensiveness",):
        dd = drawdown(c["net"][b][p])
        ax.plot(dd.index, 100 * dd, color=PCOL[p], lw=1.8 if p in PIDS else 1.0,
                ls="-" if p in PIDS else (0, (4, 2)), label=f"{SHORT[p]} (max {100 * dd.min():.1f}%)")
    ax.set_ylabel("drawdown from running peak, %", fontsize=8, color=INK_2)
    _test_axis(ax)
    _legend(ax, loc="lower left", ncol=2)
    _title(fig, "Drawdowns of net wealth through the Test period", f"Net of {b:g} bps costs. B2 = same optimizer with a "
           "constant pre-Test defensiveness (no timing).")
    _save(fig, path)


def fig_rolling(c, path, kind):
    import matplotlib.pyplot as plt
    from .backtest_performance import rolling_sharpe, rolling_vol
    win = c["protocol"]["metrics"]["rolling"]["sharpe_window_sessions" if kind == "sharpe" else "volatility_window_sessions"]
    fig, ax = plt.subplots(figsize=(13, 4.8), facecolor=SURFACE)
    _style(ax)
    b = c["prim_bps"]
    for p in PIDS + ("B2_constant_defensiveness",):
        r = c["net"][b][p]
        s = rolling_sharpe(r, c["rf"], win) if kind == "sharpe" else 100 * rolling_vol(r, win)
        ax.plot(s.index, s, color=PCOL[p], lw=1.6 if p in PIDS else 1.0, ls="-" if p in PIDS else (0, (4, 2)), label=SHORT[p])
    ax.axhline(0, color=MUTED, lw=0.7)
    ax.set_ylabel("rolling Sharpe ratio (excess over BIL)" if kind == "sharpe" else "annualized volatility, %",
                  fontsize=8, color=INK_2)
    _test_axis(ax)
    _legend(ax, loc="upper left", ncol=4)
    if kind == "sharpe":
        _title(fig, f"Rolling {win}-session Sharpe ratio (net)", "Descriptive only: a one-year window holds few independent "
               "observations; short-window swings are noise-dominated and are not tested.")
    else:
        _title(fig, f"Rolling {win}-session realized volatility (net returns)", "Realized volatility of the actual "
               "holdings; compare with the ex-ante risk budgets of Stage 7.")
    _save(fig, path)


def fig_allocation(c, path):
    import matplotlib.pyplot as plt
    pol = c["policy"]
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), facecolor=SURFACE, sharex=True)
    for ax, p in zip(axes, PIDS):
        _style(ax, grid_y=False)
        w = c["post"][p].loc[c["netp"].index]
        bottom = np.zeros(len(w))
        for g, members in pol.groups.items():
            v = w[members].sum(axis=1).to_numpy()
            ax.fill_between(w.index, bottom, bottom + v, color=GROUP_COL[g], lw=0, label=g, step="post")
            bottom += v
        ax.set_ylim(0, 1)
        ax.set_ylabel(SHORT[p], fontsize=9, color=INK_2)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    _legend(axes[0], loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.25))
    _test_axis(axes[-1])
    _title(fig, "Realized asset-class allocation - actual end-of-day holdings (executed and drifted)",
           "Not Stage 7 targets: weights change at executions (t+1 close) and drift every session in between.")
    _save(fig, path)


def fig_response(c, path):
    import matplotlib.pyplot as plt
    idx = c["netp"].index
    fig, axes = plt.subplots(4, 1, figsize=(13, 10), facecolor=SURFACE, sharex=True,
                             gridspec_kw={"height_ratios": [1, 0.6, 1.1, 1.1]})
    ax = axes[0]
    _style(ax)
    p = c["roles"]["baseline_regime"]["baseline_p_non_normal"].loc[idx]
    ax.plot(idx, p, color=PCOL["P1_regime_aware"], lw=1.0, label="P1 signal: HMM P(non-normal) (filtered)")
    ax.set_ylim(-0.02, 1.05)
    _legend(ax, loc="lower left", bbox_to_anchor=(0, 1.0))
    ax = axes[1]
    _style(ax, grid_y=False)
    lv = c["roles"]["challenger_ews"]["challenger_alert_level"].loc[idx]
    for name, col in LEVEL_COL.items():
        m = (lv == name).to_numpy()
        ax.fill_between(idx, 0, 1, where=m, color=col, step="post", lw=0, label=name)
    ax.set_yticks([])
    ax.set_ylabel("P2 signal:\nEWS level", fontsize=8, color=INK_2)
    _legend(ax, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.35))
    for ax, col, lab in ((axes[2], "equity", "equity weight"), (axes[3], "BIL", "BIL weight")):
        _style(ax)
        for pid in PIDS:
            w = c["post"][pid].loc[idx]
            v = w[c["policy"].groups["equity"]].sum(axis=1) if col == "equity" else w["BIL"]
            ax.plot(idx, 100 * v, color=PCOL[pid], lw=1.4, label=SHORT[pid], drawstyle="steps-post")
        ax.set_ylabel(f"actual {lab}, %", fontsize=8, color=INK_2)
        _legend(ax, loc="lower right", ncol=3, bbox_to_anchor=(1.0, 1.0))
    _test_axis(axes[-1])
    _title(fig, "Frozen signals and the portfolios' actual exposures in Test",
           "Signals are observed after the close of t; any change in holdings appears at the close of t+1 at the earliest "
           "and only at the weekly decision. Exposures are actual end-of-day holdings.")
    _save(fig, path)


def fig_crisis(c, path):
    import matplotlib.pyplot as plt
    win = c["windows"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), facecolor=SURFACE)
    if win is None or win.empty:
        _save(fig, path)
        return
    names = list(dict.fromkeys(win["window"]))
    ports = list(PIDS) + ["B2_constant_defensiveness"]
    x = np.arange(len(names))
    wdt = 0.8 / len(ports)
    for ax, col, lab in ((axes[0], "net_cumulative_return", "net cumulative return in window, %"),
                         (axes[1], "max_drawdown_in_window", "maximum drawdown in window, %")):
        _style(ax)
        for k, p in enumerate(ports):
            vals = [100 * float(win[(win["window"] == n) & (win["portfolio"] == p)][col].iloc[0])
                    if len(win[(win["window"] == n) & (win["portfolio"] == p)]) else np.nan for n in names]
            ax.bar(x + (k - (len(ports) - 1) / 2) * wdt, vals, width=wdt * 0.92, color=PCOL[p], label=SHORT[p])
        ax.axhline(0, color=MUTED, lw=0.7)
        ax.set_xticks(x, [n.replace("_", "\n", 1) for n in names], fontsize=8, color=INK_2)
        ax.set_ylabel(lab, fontsize=8, color=INK_2)
    _legend(axes[0], loc="best")
    _title(fig, "Stress and recovery windows fixed before Test: realized net outcome",
           "Windows from the pre-Test protocol (calendar months of the public macro chronology). Recovery windows show "
           "the cost of staying defensive.")
    _save(fig, path)


def fig_attribution(c, path):
    import matplotlib.pyplot as plt
    att = c["attribution"]
    a, b = "P2_ews_enhanced", "P1_regime_aware"
    s = att[(att["portfolio_a"] == a) & (att["portfolio_b"] == b)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    _style(ax, grid_y=False)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    groups = list(c["protocol"]["attribution"]["groups"])
    labels, avg, tim = [], [], []
    for g in groups:
        r = s[s["group"] == g].set_index("component")["annualized_mean"]
        labels.append(g)
        avg.append(100 * r.get("average-exposure part", 0))
        tim.append(100 * r.get("timing part", 0))
    tot = s.set_index("component")["annualized_mean"]
    labels += ["transaction costs", "NET active"]
    avg += [0, 0]
    tim += [0, 0]
    extra = [100 * tot.get("transaction-cost effect", 0), 100 * tot.get("net active return", 0)]
    y = np.arange(len(labels))[::-1]
    ng = len(groups)
    ax.barh(y[:ng] + 0.18, avg[:ng], height=0.34, color="#86b6ef", label="average-exposure part")
    ax.barh(y[:ng] - 0.18, tim[:ng], height=0.34, color=PCOL[a], label="timing part")
    ax.barh(y[ng:], extra, height=0.5, color=[INK_2, INK])
    ax.axvline(0, color=MUTED, lw=0.7)
    ax.set_yticks(y, labels, fontsize=9, color=INK_2)
    ax.set_xlabel("annualized contribution to P2 - P1, % per year", fontsize=8, color=INK_2)
    _legend(ax, loc="lower right")
    ax = axes[1]
    _style(ax)
    d = c["att_daily"][(a, b)]
    ax.plot(d.index, 100 * d["net_active"].cumsum(), color=INK, lw=1.8, label="net active", zorder=2)
    ax.plot(d.index, 100 * d["gross_active"].cumsum(), color=PCOL[a], lw=1.2, ls=(0, (4, 2)),
            label="gross active (before costs)", zorder=3)
    ax.plot(d.index, 100 * d["cost_effect"].cumsum(), color=INK_2, lw=1.2, label="cost effect", zorder=2)
    ax.axhline(0, color=MUTED, lw=0.7)
    ax.set_ylabel("cumulative sum of daily active return, %", fontsize=8, color=INK_2)
    _test_axis(ax)
    _legend(ax, loc="best")
    _title(fig, "Where does P2 differ from P1? Holdings-based daily attribution",
           "Exact daily identity: gross active = sum over assets of (w_P2 - w_P1) x return; the average-exposure / timing "
           "split uses the Test-average weight difference (descriptive). Arithmetic sums, not compounded.")
    _save(fig, path)


def fig_turnover(c, path):
    import matplotlib.pyplot as plt
    perf = c["perf"]
    ports = list(PIDS) + ["B2_constant_defensiveness", "B1_60_40"]
    net = perf[perf["basis"] == f"net_{c['prim_bps']:g}bps"].set_index("portfolio")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), facecolor=SURFACE)
    ax = axes[0]
    _style(ax)
    v = [net.loc[p, "annualized_one_way_turnover"] for p in ports]
    ax.bar(range(len(ports)), v, color=[PCOL[p] for p in ports])
    for i, x in enumerate(v):
        ax.text(i, x, f"{x:.2f}", ha="center", va="bottom", fontsize=8, color=INK_2)
    ax.set_xticks(range(len(ports)), [SHORT[p] for p in ports], fontsize=8, color=INK_2, rotation=20)
    ax.set_title("annualized one-way turnover (realized)", fontsize=9, color=INK, loc="left")
    ax = axes[1]
    _style(ax)
    for k, bb in enumerate(c["bps_all"]):
        vals = [100 * net.loc[p, f"total_cost_fraction_{bb:g}bps"] / c["years"] for p in ports]
        ax.bar(np.arange(len(ports)) + (k - 1) * 0.27, vals, width=0.25, color=["#cde2fb", "#5598e7", "#184f95"][k % 3],
               label=f"{bb:g} bps")
    ax.set_xticks(range(len(ports)), [SHORT[p] for p in ports], fontsize=8, color=INK_2, rotation=20)
    ax.set_title("cost drag, % of wealth per year", fontsize=9, color=INK, loc="left")
    _legend(ax, loc="upper left")
    ax = axes[2]
    _style(ax)
    for p in ports[:4]:
        lg = c["ledgers"][p].set_index("date")
        ax.plot(lg.index, 100 * lg[f"cost_fraction_{c['prim_bps']:g}bps"].cumsum(), color=PCOL[p], lw=1.5, label=SHORT[p])
    ax.set_title(f"cumulative cost at {c['prim_bps']:g} bps, % of wealth", fontsize=9, color=INK, loc="left")
    _test_axis(ax)
    _legend(ax, loc="upper left")
    _title(fig, "Turnover and trading costs: is EWS responsiveness expensive?",
           "Realized execution-day turnover (pre-trade weights drifted through t+1); costs on gross traded notional, x2 "
           "when the pre-specified SPY volatility stress flag is on.")
    _save(fig, path)


def fig_risk_adjusted(c, path):
    import matplotlib.pyplot as plt
    perf = c["perf"]
    ports = list(PIDS) + list(SUPP)
    net = perf[perf["basis"] == f"net_{c['prim_bps']:g}bps"].set_index("portfolio")
    mets = [("cagr", "CAGR, %", 100), ("annual_volatility", "volatility, %", 100), ("sharpe", "Sharpe", 1),
            ("sortino", "Sortino", 1), ("max_drawdown", "max drawdown, %", 100)]
    fig, axes = plt.subplots(1, 5, figsize=(16, 4.8), facecolor=SURFACE)
    for ax, (m, lab, k) in zip(axes, mets):
        _style(ax, grid_y=False)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        vals = [k * net.loc[p, m] for p in ports]
        y = np.arange(len(ports))[::-1]
        ax.barh(y, vals, color=[PCOL[p] for p in ports], height=0.6)
        for yy, v in zip(y, vals):
            ax.text(v, yy, f" {v:.2f}", va="center", fontsize=7.5, color=INK_2, ha="left" if v >= 0 else "right")
        ax.set_yticks(y, [SHORT[p] for p in ports] if ax is axes[0] else [""] * len(ports), fontsize=8, color=INK_2)
        ax.set_title(lab, fontsize=9, color=INK, loc="left")
        ax.axvline(0, color=MUTED, lw=0.7)
        lo_, hi_ = min(min(vals), 0), max(max(vals), 0)
        span = (hi_ - lo_) or 1.0
        ax.set_xlim(lo_ - 0.3 * span * (lo_ < 0), hi_ + 0.3 * span * (hi_ > 0))
    _title(fig, f"Risk-adjusted performance in Test (net of {c['prim_bps']:g} bps)",
           "Sharpe / Sortino use excess returns over realized BIL. B1-B3 are context benchmarks defined before Test.")
    _save(fig, path)


def fig_annual(c, path):
    import matplotlib.pyplot as plt
    from .regime_report import _diverging
    an = c["annual"]
    an = an[an["basis"] == f"net_{c['prim_bps']:g}bps"]
    ports = list(PIDS) + list(SUPP)
    tab = an.pivot(index="portfolio", columns="year", values="return").loc[ports]
    fig, ax = plt.subplots(figsize=(13, 4.6), facecolor=SURFACE)
    vmax = float(np.nanmax(np.abs(tab.to_numpy()))) or 0.1
    ax.imshow(100 * tab.to_numpy(), cmap=_diverging().reversed(), vmin=-100 * vmax, vmax=100 * vmax, aspect="auto")
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            v = tab.iat[i, j]
            ax.text(j, i, f"{100 * v:.1f}%", ha="center", va="center", fontsize=8,
                    color=INK if abs(v) < 0.6 * vmax else "#fcfcfb")
    sess = an.groupby("year")["sessions"].max()
    ax.set_xticks(range(tab.shape[1]), [f"{y}\n({sess.get(y, 0)} sessions)" if sess.get(y, 252) < 240 else y
                                        for y in tab.columns], fontsize=8, color=INK_2)
    ax.set_yticks(range(len(ports)), [SHORT[p] for p in ports], fontsize=9, color=INK_2)
    for s in ax.spines.values():
        s.set_visible(False)
    _title(fig, "Calendar-year net returns", "Partial years are marked with their session count (not annualized).")
    _save(fig, path)


def fig_cost_sensitivity(c, path):
    import matplotlib.pyplot as plt
    cs = c["cost_sens"]
    order = ["gross"] + [f"net_{b:g}bps" for b in c["bps_all"]]
    xl = ["0 (gross)"] + [f"{b:g}" for b in c["bps_all"]]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), facecolor=SURFACE)
    for ax, m, lab in ((axes[0], "sharpe", "Sharpe ratio"), (axes[1], "cagr", "CAGR, %")):
        _style(ax)
        for p in list(PIDS) + ["B2_constant_defensiveness"]:
            s = cs[cs["portfolio"] == p].set_index("basis").loc[order, m]
            k = 100 if m == "cagr" else 1
            ax.plot(range(len(order)), k * s.to_numpy(), marker="o", ms=6, color=PCOL[p], lw=1.8, label=SHORT[p])
        ax.set_xticks(range(len(order)), xl, fontsize=8, color=INK_2)
        ax.set_xlabel("cost assumption, bps of gross traded notional", fontsize=8, color=INK_2)
        ax.set_title(lab, fontsize=9, color=INK, loc="left")
    _legend(axes[0], loc="best")
    _title(fig, "Transaction-cost sensitivity (5 / 10 / 25 bps, frozen before Test)",
           "Identical trading path in every scenario; only the cost deducted at each execution changes.")
    _save(fig, path)


def fig_uncertainty(c, path):
    import matplotlib.pyplot as plt
    st = c["stats"]
    pairs = [("P1_regime_aware", "P0_strategic"), ("P2_ews_enhanced", "P0_strategic"), ("P2_ews_enhanced", "P1_regime_aware"),
             ("P1_regime_aware", "B2_constant_defensiveness"), ("P2_ews_enhanced", "B2_constant_defensiveness")]
    panels = [("mean_active_return", "Newey-West HAC", "mean active return, % per year", 100),
              ("sharpe_difference", "(primary)", "net Sharpe difference", 1),
              ("max_drawdown_difference", "(primary)", "max drawdown difference, pp (+ = shallower)", 100)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=SURFACE)
    for ax, (stat, meth, lab, k) in zip(axes, panels):
        _style(ax, grid_y=False)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        y = np.arange(len(pairs))[::-1]
        for yy, (a, b) in zip(y, pairs):
            r = st[(st["portfolio_a"] == a) & (st["portfolio_b"] == b) & (st["statistic"] == stat) &
                   st["method"].str.contains(meth, regex=False)]
            if r.empty:
                continue
            r = r.iloc[0]
            col = PCOL[a]
            ax.plot([k * r["ci_low"], k * r["ci_high"]], [yy, yy], color=col, lw=2.2, solid_capstyle="round")
            ax.plot(k * r["estimate"], yy, "o", color=col, ms=8, mec=SURFACE, mew=2)
        ax.axvline(0, color=INK_2, lw=0.9)
        ax.set_yticks(y, [f"{SHORT[a].split()[0]} - {SHORT[b].split()[0]}" for a, b in pairs] if ax is axes[0]
                      else [""] * len(pairs), fontsize=9, color=INK_2)
        ax.set_title(lab, fontsize=9, color=INK, loc="left")
    _title(fig, "Statistical uncertainty of the prespecified comparisons (95% intervals, net of costs)",
           "Mean: Newey-West HAC. Sharpe and drawdown: paired stationary bootstrap, mean block 21 sessions, 5000 draws. "
           "An interval containing 0 is not evidence of equivalence.")
    _save(fig, path)


def fig_summary(c, path):
    import matplotlib.pyplot as plt
    from .backtest_performance import drawdown
    b = c["prim_bps"]
    fig = plt.figure(figsize=(15, 8.5), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.3, 1], width_ratios=[1.6, 1, 1])
    ax = fig.add_subplot(gs[0, :2])
    _style(ax)
    for p in PIDS:
        w = (1 + c["net"][b][p]).cumprod()
        ax.plot(w.index, w, color=PCOL[p], lw=2, label=f"{SHORT[p]} (end {w.iloc[-1]:.3f})")
    ax.set_title(f"Net wealth, Test {c['test_start'].date()} to {c['test_end'].date()}", fontsize=10, color=INK, loc="left")
    _test_axis(ax)
    _legend(ax, loc="upper right")
    ax = fig.add_subplot(gs[1, :2])
    _style(ax)
    for p in PIDS:
        dd = drawdown(c["net"][b][p])
        ax.plot(dd.index, 100 * dd, color=PCOL[p], lw=1.5)
    ax.set_title("Drawdown, %", fontsize=10, color=INK, loc="left")
    _test_axis(ax)
    ax = fig.add_subplot(gs[:, 2])
    ax.axis("off")
    net = c["perf_p"]
    lines = [("", "P0", "P1", "P2")]
    for m, lab, f in (("cagr", "CAGR", _pct), ("annual_volatility", "Volatility", _pct), ("sharpe", "Sharpe", _num),
                      ("sortino", "Sortino", _num), ("max_drawdown", "Max drawdown", _pct),
                      ("annualized_one_way_turnover", "Turnover / yr", _num)):
        lines.append((lab,) + tuple(f(net.loc[p, m]) for p in PIDS))
    y = 0.97
    for row in lines:
        for k, cell in enumerate(row):
            ax.text([0.0, 0.42, 0.62, 0.82][k], y, cell, fontsize=9.5, color=INK if k == 0 or y == 0.97 else INK_2,
                    fontweight="bold" if y == 0.97 else "normal", transform=ax.transAxes)
        y -= 0.065
    con = c["concl"]
    y -= 0.03
    ax.text(0.0, y, "Prespecified verdicts", fontsize=10, color=INK, fontweight="bold", transform=ax.transAxes)
    y -= 0.06
    for k in ("A", "B", "C", "E"):
        import textwrap
        for j, line in enumerate(textwrap.wrap(f"{k}: {con[k]['verdict']}", 60)[:3]):
            ax.text(0.0, y, line, fontsize=8, color=INK_2, transform=ax.transAxes)
            y -= 0.035
        y -= 0.015
    y -= 0.02
    ax.text(0.0, y, con["outcome"], fontsize=10, color=INK, fontweight="bold", transform=ax.transAxes, wrap=True)
    _title(fig, "Adaptive Regime Radar - final Test evaluation (net of costs)",
           f"Frozen models, conservative t+1 execution, {b:g} bps (x2 in stress). Outcome label follows the decision "
           "rules frozen before Test access.")
    _save(fig, path)


FIGURES = ("cumulative_wealth_gross_net", "portfolio_drawdowns", "rolling_sharpe", "rolling_volatility",
           "realized_asset_allocation", "regime_warning_portfolio_response", "crisis_performance_comparison",
           "ews_incremental_attribution", "turnover_and_costs", "risk_adjusted_performance_comparison",
           "annual_returns_heatmap", "transaction_cost_sensitivity", "performance_uncertainty",
           "investment_decision_summary")


def save_figures(directory: Path, c: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fns = {"cumulative_wealth_gross_net": fig_wealth, "portfolio_drawdowns": fig_drawdowns,
           "rolling_sharpe": lambda c_, p: fig_rolling(c_, p, "sharpe"),
           "rolling_volatility": lambda c_, p: fig_rolling(c_, p, "vol"), "realized_asset_allocation": fig_allocation,
           "regime_warning_portfolio_response": fig_response, "crisis_performance_comparison": fig_crisis,
           "ews_incremental_attribution": fig_attribution, "turnover_and_costs": fig_turnover,
           "risk_adjusted_performance_comparison": fig_risk_adjusted, "annual_returns_heatmap": fig_annual,
           "transaction_cost_sensitivity": fig_cost_sensitivity, "performance_uncertainty": fig_uncertainty,
           "investment_decision_summary": fig_summary}
    out = []
    for name in FIGURES:
        path = directory / f"{name}.png"
        fns[name](c, path)
        out.append(str(path.relative_to(directory.parents[2])) if len(directory.parents) > 2 else str(path))
    return out


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------
def _f(frame: pd.DataFrame, pct_cols=(), digits=3) -> pd.DataFrame:
    out = frame.copy()
    for col in out.columns:
        if col in pct_cols:
            out[col] = out[col].map(lambda v: _pct(v, 2) if isinstance(v, (float, np.floating)) else v)
        elif pd.api.types.is_float_dtype(out[col]):
            vals = out[col].dropna()
            if len(vals) and (vals == vals.round()).all() and vals.abs().max() >= 2:
                out[col] = out[col].map(lambda v: "" if pd.isna(v) else f"{v:.0f}")
            else:
                out[col] = out[col].map(lambda v: "" if pd.isna(v) else f"{v:.{digits}f}")
        elif pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d")
    return out


def render_report(c: dict) -> str:
    P = c["protocol"]
    b = c["prim_bps"]
    perf, perf_p, con = c["perf"], c["perf_p"], c["concl"]
    ports = list(PIDS) + list(SUPP)
    L = []
    L.append("# Stage 8 - Final sequential out-of-sample portfolio backtest\n")
    L.append(f"## STAGE 8: {c['stage8_status']} - FINAL RESEARCH STATUS: {c['final_status']}\n")
    cnt = c["counts"]
    L.append(f"Checks: {cnt.get('PASS', 0)} PASS, {cnt.get('WARN', 0)} WARN, {cnt.get('FAIL', 0)} FAIL. Evaluation mode: "
             f"**{c['mode']}**" + (f" (amendment: {c['amendment']})" if c["amendment"] else "") +
             f". Pre-Test lock `{c['lock']['fingerprint'][:16]}...`. Test {c['test_start'].date()} to "
             f"{c['test_end'].date()} ({len(c['netp'])} sessions, {c['years']:.2f} years).\n")
    L.append(f"**Prespecified outcome classification: {con['outcome']}**\n")

    # 1
    L.append("## 1. Executive summary\n")
    s = con["sharpe"]
    L.append(f"- Net of {b:g} bps (x2 in cost stress), Test CAGR / volatility / Sharpe / max drawdown: " + "; ".join(
        f"{SHORT[p]} {_pct(perf_p.loc[p, 'cagr'])} / {_pct(perf_p.loc[p, 'annual_volatility'])} / "
        f"{_num(perf_p.loc[p, 'sharpe'])} / {_pct(perf_p.loc[p, 'max_drawdown'])}" for p in PIDS) + ".")
    for k, lab in (("P1-P0", "P1 vs P0"), ("P2-P0", "P2 vs P0"), ("P2-P1", "P2 vs P1")):
        L.append(f"- {lab}: net Sharpe difference {_num(s[k]['estimate'])} (95% stationary-bootstrap interval "
                 f"{_num(s[k]['ci_low'])} to {_num(s[k]['ci_high'])}).")
    L.append("- Implementation: annualized one-way turnover / cost drag at " + f"{b:g} bps: " + "; ".join(
        f"{SHORT[p]} {_num(perf_p.loc[p, 'annualized_one_way_turnover'])} / "
        f"{_pct(perf_p.loc[p, f'total_cost_fraction_{b:g}bps'] / c['years'], 2)} per year" for p in PIDS) + ".")
    L.append(f"- Maximum drawdown: {con['C']['verdict']}.")
    L.append(f"- Stress periods: {con['D']['verdict']}.")
    L.append(f"- Exposure vs timing: {con['F']['verdict']}.")
    L.append(f"- Costs: {con['E']['verdict']}.")
    L.append(f"- Independent evidence is limited: {c['n_events']} SPY drawdown events (>= 5%) in Test, {c['n_major']} of "
             "them >= 10%; a single crisis can dominate the comparison.\n")

    # 2
    L.append("## 2. Research question and investment mandate\n")
    L.append(P["research_question"] + "\n")
    L.append("Moderate-risk multi-asset client; nine US-listed ETFs; long only, no leverage; at most one rebalance per "
             "week. Benchmark (frozen): " + ", ".join(f"{a} {w:.0%}" for a, w in zip(c["assets"], c["policy"].benchmark))
             + ". Stage 7 constraints are PROPOSED research assumptions (not an approved mandate).\n")
    # 3
    L.append("## 3. Research pipeline (Stages 0-8)\n")
    L.append(_md_table(pd.DataFrame([
        ("0-2", "data download, point-in-time panels (ALFRED first releases where available)"),
        ("3", "five causal factors; Train 2007-2015 / Validation 2016-2019 / Test 2020- split (protocol v1.2)"),
        ("4", "regime exploration (GMM / K-Means)"), ("5", "frozen Gaussian HMM K=3 (hmm_std_diag_k3), forward filter"),
        ("5.5", "AR(1) and AR-HMM robustness (variance-floor M-step), frozen"),
        ("6 / 6.1", "BOCPD, EWS score, 5-level alert machine; audit and role-separated handoff"),
        ("7", "robust MVO, dynamic risk budgets, weekly target_then_trade policy (Train/Validation only)"),
        ("8", "pre-Test lock, then ONE sequential Test evaluation with a daily ledger")],
        columns=["stage", "content"]).set_index("stage")) + "\n")
    # 4
    L.append("## 4. Data partitions and Test protocol\n")
    L.append(f"- Train {c['t_first'].date()} is the first decision (warm-up 252 sessions); Train/Validation history "
             f"is used only to carry holdings forward to {c['last_val'].date()} (no pre-Test performance is reported).")
    L.append(f"- Test {c['test_start'].date()} - {c['test_end'].date()}: last valid session of the frozen local split "
             "(not chosen). Opened only through the lock-gated evaluation loader; access log: " +
             "; ".join(f"#{i + 1} {e['utc']} ({e['mode']})" for i, e in enumerate(c["access_log"])) + ".")
    bd = c["boundary"]
    L.append(f"- Boundary: the week of {bd['week_start'].date()} was incomplete in Stage 7; the frozen weekly rule "
             f"completes it with Test sessions, so the first Test decision is {bd['signal_date'].date()} (executed "
             f"{bd['execution_date'].date()}).")
    L.append(f"- Wealth of every strategy = 1.0 at the close of {c['last_val'].date()}; the first Test return is "
             f"{c['test_start'].date()}.\n")
    # 5
    L.append("## 5. Model freeze and pre-Test lock\n")
    L.append(f"`data/processed/backtesting/stage8_pretest_lock.json` (fingerprint `{c['lock']['fingerprint']}`) was "
             "written before Test was opened and records the protocol, every code / configuration / frozen-artifact hash, "
             "the frozen policies and constants, the initial state and the preflight results. Before reading Test the "
             "loader re-verified every hash; after the evaluation the hashes were checked again (section 25 checks).")
    pre = c["pre_checks"]
    L.append(f"Preflight: {sum(x.status == 'PASS' for x in pre)} PASS, {sum(x.status == 'WARN' for x in pre)} WARN, "
             f"{sum(x.status == 'FAIL' for x in pre)} FAIL (details in `reports/stage8_pretest_protocol.md`).\n")
    L.append(_md_table(c["rep_df"].set_index("portfolio")) + "\n")
    L.append("Frozen constants: " + ", ".join(f"{k} = {v!r}" for k, v in c["lock"]["frozen"]["frozen_constants"].items()) + "\n")
    # 6
    L.append("## 6. Point-in-time limitations\n")
    L.append("- Factors are point-in-time from Stage 2/3 (ALFRED first releases where available; rule-based lags "
             "otherwise); revisions not captured by first-release data remain a limitation.")
    L.append("- ETF returns are adjusted-close total returns. Executing at the adjusted close of t+1 is an accounting "
             "convention; it does not prove that those prices were obtainable fills (no bid-ask, no market impact "
             "beyond the cost assumption, dividends assumed reinvested).")
    L.append("- Train signals come from models fitted on all of Train (retrospective); Validation was reused in Stages 5-7. "
             "Only the Test period is an untouched evaluation.\n")
    # 7
    L.append("## 7. Frozen portfolio strategies\n")
    rows = [{"id": k, "rule": v} for k, v in P["primary_portfolios"].items()]
    rows += [{"id": k, "rule": v["rule"]} for k, v in P["supplementary_benchmarks"].items()]
    L.append(_md_table(pd.DataFrame(rows).set_index("id")) + "\n")
    L.append(f"P1 scale a = {c['a_scale']!r} (Train-calibrated in Stage 7); B2 constant d = {c['d_const']!r} (mean Train "
             "defensiveness of P1 and P2).\n")
    # 8
    L.append("## 8. Sequential signal-generation framework\n")
    L.append(P["terminology"] + " The Stage 6 generator `early_warning.build_signals` is reused unchanged and run "
             "once over Train + Validation + Test: the HMM forward filter, AR(1) innovations, AR-HMM surprise, BOCPD "
             "run-length posterior and the alert state machine all continue chronologically; nothing is reset at the "
             "Test boundary. This is NOT monthly HMM retraining: no parameter is re-estimated in Test. Verified: prefix "
             "invariance through a Test date, invariance of earlier signals to perturbed later factors, and bit-identity "
             "with the Stage 6.1 handoff on Train/Validation.\n")
    # 9
    L.append("## 9. Portfolio optimization and execution conventions\n")
    for k, v in {**P["decision_rule"], **P["execution"]}.items():
        L.append(f"- **{k}**: {v}")
    L.append("\nImplementation ambiguities resolved before Test:\n")
    for a in c["ambiguities"]:
        L.append(f"- {a}")
    L.append("")
    # 10
    L.append("## 10. Actual daily holdings and rebalancing\n")
    impl_cols = ["decisions", "hold_decisions_share", "executions_with_trades", "rebalances_per_year",
                 "mean_defensiveness_d", "share_decisions_defensive_d_gt_0", "mean_weight_equity", "mean_weight_duration",
                 "mean_weight_credit", "mean_weight_real_assets", "mean_weight_BIL"]
    t = perf_p.loc[ports, [x for x in impl_cols if x in perf_p.columns]]
    L.append(_md_table(_f(t)) + "\n")
    L.append("Holdings at the start of Test (carried from the pre-Test continuation; recorded in the lock):\n")
    L.append(_md_table(c["init_tab"]) + "\n")
    # 11
    L.append("## 11. Transaction-cost methodology\n")
    for k, v in P["transaction_costs"].items():
        L.append(f"- **{k}**: {v}")
    L.append(f"- Train-calibrated stress threshold (annualized 21-session SPY volatility): {c['stress_thr']:.4f}.\n")
    # 12
    L.append("## 12. Test-period performance\n")
    cols = ["cumulative_return", "cagr", "arithmetic_annual_return", "annual_volatility", "sharpe", "sortino", "calmar",
            "max_drawdown", "max_drawdown_sessions_underwater", "worst_day", "worst_week", "worst_month", "var95_daily",
            "cvar95_daily", "downside_deviation"]
    pct = {"cumulative_return", "cagr", "arithmetic_annual_return", "annual_volatility", "max_drawdown", "worst_day",
           "worst_week", "worst_month", "var95_daily", "cvar95_daily", "downside_deviation"}
    L.append(f"Net of {b:g} bps:\n")
    L.append(_md_table(_f(perf_p.loc[ports, cols], pct)) + "\n")
    L.append("Calendar-year net returns:\n")
    an = c["annual"][c["annual"]["basis"] == f"net_{b:g}bps"].pivot(index="year", columns="portfolio", values="return")[ports]
    L.append(_md_table(_f(an, set(ports))) + "\n")
    # 13 / 14
    bench = c["bench"].set_index(["portfolio_a", "portfolio_b"])
    bcols = ["active_return_annualized", "tracking_error", "information_ratio", "cagr_difference", "sharpe_difference",
             "sortino_difference", "max_drawdown_difference", "up_capture_vs_b", "down_capture_vs_b"]
    st = c["stats"]
    for sec, (a, bb) in (("13. P0 versus P1", ("P1_regime_aware", "P0_strategic")),
                         ("14. P1 versus P2", ("P2_ews_enhanced", "P1_regime_aware"))):
        L.append(f"## {sec}\n")
        r = bench.loc[(a, bb)]
        L.append(_md_table(_f(pd.DataFrame([r[bcols].to_numpy()], columns=bcols, index=[f"{SHORT[a]} - {SHORT[bb]}"]),
                              {"active_return_annualized", "tracking_error", "cagr_difference", "max_drawdown_difference"})) + "\n")
        sub = st[(st["portfolio_a"] == a) & (st["portfolio_b"] == bb)]
        L.append(_md_table(_f(sub[["statistic", "method", "estimate", "ci_low", "ci_high", "p_value", "p_value_holm"]]
                              .set_index("statistic"), digits=4)) + "\n")
        q = con["A"] if a == "P1_regime_aware" else con["B"]
        L.append(f"**{q['question']}** {q['verdict']}.\n")
    # 15
    L.append("## 15. Gross versus net performance\n")
    g = perf[perf["basis"] == "gross"].set_index("portfolio")
    tab = pd.DataFrame({"gross CAGR": g.loc[ports, "cagr"], f"net CAGR ({b:g} bps)": perf_p.loc[ports, "cagr"],
                        "cost drag (CAGR points)": g.loc[ports, "cagr"] - perf_p.loc[ports, "cagr"],
                        "gross Sharpe": g.loc[ports, "sharpe"], "net Sharpe": perf_p.loc[ports, "sharpe"]})
    L.append(_md_table(_f(tab, {"gross CAGR", f"net CAGR ({b:g} bps)", "cost drag (CAGR points)"})) + "\n")
    # 16
    L.append("## 16. Drawdown and downside-risk comparison\n")
    dd = c["dd_events"]
    if len(dd):
        top = dd.sort_values("depth").groupby("portfolio").head(3)
        top = top[top["portfolio"].isin(PIDS)].sort_values(["portfolio", "depth"])
        L.append(_md_table(_f(top[["portfolio", "peak_date", "trough_date", "recovery_date", "depth", "sessions_peak_to_trough",
                                   "sessions_underwater"]].set_index("portfolio"), {"depth"})) + "\n")
    L.append(f"{con['C']['question']} {con['C']['verdict']}. Reduction vs P0 {con['C']['reduction_vs_p0_pct_points']:.2f} pp "
             f"(bootstrap 95% interval of the drawdown difference {_pct(con['C']['bootstrap_ci_vs_p0'][0], 2)} to "
             f"{_pct(con['C']['bootstrap_ci_vs_p0'][1], 2)}; approximate, resampled paths do not keep the crash sequence).\n")
    # 17
    L.append("## 17. Risk-adjusted performance\n")
    L.append(_md_table(_f(perf_p.loc[ports, ["sharpe", "sortino", "calmar", "information_ratio_vs_P0",
                                             "information_ratio_vs_P1"]])) + "\n")
    L.append("Risk matching (regression of excess returns on the reference's excess returns, Newey-West):\n")
    rg = c["regressions"][["portfolio", "reference", "alpha_annualized", "alpha_ci_low", "alpha_ci_high", "alpha_p_value",
                           "beta", "beta_ci_low", "beta_ci_high", "r_squared"]]
    L.append(_md_table(_f(rg.set_index("portfolio"), {"alpha_annualized", "alpha_ci_low", "alpha_ci_high"})) + "\n")
    vm = c["bench"][["portfolio_a", "portfolio_b", "descriptive_vol_matched_cagr_a", "descriptive_vol_matched_max_drawdown_a"]]
    L.append("DESCRIPTIVE ex-post volatility normalization (scaled to P0's realized volatility; not a tradable strategy):\n")
    L.append(_md_table(_f(vm[vm["portfolio_b"].isin(["P0_strategic", "P1_regime_aware", "B2_constant_defensiveness"])]
                          .set_index("portfolio_a"), {"descriptive_vol_matched_cagr_a",
                                                      "descriptive_vol_matched_max_drawdown_a"})) + "\n")
    # 18
    L.append("## 18. Crisis-event analysis\n")
    w = c["windows"]
    if w is not None and len(w):
        wc = ["window", "portfolio", "gross_cumulative_return", "net_cumulative_return", "max_drawdown_in_window",
              "equity_weight_at_start", "equity_weight_min", "bil_weight_max", "trades", "cost_primary"]
        L.append(_md_table(_f(w[wc].set_index("window"), {"gross_cumulative_return", "net_cumulative_return",
                                                          "max_drawdown_in_window", "cost_primary"})) + "\n")
        sig = w[w["portfolio"].isin(["P1_regime_aware", "P2_ews_enhanced"])]
        sc = [x for x in ("hmm_p_non_normal_mean", "ews_share_ge_amber", "ews_share_red", "ews_level_at_start",
                          "ews_max_level") if x in sig.columns]
        L.append("Signals inside each window:\n")
        L.append(_md_table(_f(sig[["window", "portfolio"] + sc].set_index("window"))) + "\n")
    e = c["events"]
    if e is not None and len(e):
        ec = ["event", "portfolio", "spy_peak_date", "spy_onset_date (-5%)", "spy_trough_date", "spy_trough_loss_pct",
              "loss_peak_to_trough", "equity_at_peak", "equity_before_onset", "equity_at_trough",
              "first_de_risking_signal_date", "first_de_risking_execution_date", "de_risking_timing",
              "recovery_return_63_sessions", "protection_vs_P0", "recovery_participation_vs_P0"]
        ec = [x for x in ec if x in e.columns]
        L.append(f"Objective SPY drawdown events (frozen Stage 6 rule), {c['n_events']} in Test:\n")
        L.append(_md_table(_f(e[e["portfolio"].isin(PIDS)][ec].set_index("event"),
                              {"loss_peak_to_trough", "recovery_return_63_sessions", "protection_vs_P0",
                               "recovery_participation_vs_P0"})) + "\n")
        L.append("Protection is credited only when the de-risking EXECUTION precedes the onset date; an alert executed "
                 "after the -5% threshold is a reaction. Negative recovery participation shows a missed rebound.\n")
    L.append(f"{con['D']['question']} {con['D']['verdict']}.\n")
    # 19
    L.append("## 19. Portfolio return attribution\n")
    at = c["attribution"]
    for a, bb in P["attribution"]["pairs"]:
        s_ = at[(at["portfolio_a"] == a) & (at["portfolio_b"] == bb)]
        L.append(f"**{SHORT[a]} - {SHORT[bb]}** (annualized; reconciliation error "
                 f"{s_['reconciliation_error'].iloc[0]:.1e}):\n")
        L.append(_md_table(_f(s_[["component", "group", "annualized_mean", "sum_daily", "mean_weight_difference"]]
                              .set_index("component"), {"annualized_mean", "sum_daily", "mean_weight_difference"})) + "\n")
    L.append(f"{con['F']['question']} {con['F']['verdict']}.\n")
    # 20
    L.append("## 20. Turnover and implementation costs\n")
    tc = ["annualized_one_way_turnover", "gross_traded_notional_total", "rebalances_per_year", "average_holding_period_years",
          f"total_cost_fraction_{b:g}bps", "cost_primary_in_stress", "cost_primary_in_normal", "executions_in_stress"]
    L.append(_md_table(_f(perf_p.loc[ports, [x for x in tc if x in perf_p.columns]], digits=4)) + "\n")
    L.append(f"{con['G']['question']} {con['G']['verdict']} (extra one-way turnover {_num(con['G']['extra_turnover_per_year'])} "
             f"per year; extra total cost {_pct(con['G']['extra_cost_total'], 2)} of wealth).\n")
    # 21
    L.append("## 21. Constant-defensiveness comparison\n")
    for a, v in con["constant_defensiveness"].items():
        L.append(f"- {SHORT[a]} - B2: net Sharpe difference {_num(v['estimate'])} (95% interval {_num(v['ci_low'])} to "
                 f"{_num(v['ci_high'])}); {'similar to permanent defensiveness' if v['similar'] else 'not classified as similar'}.")
    L.append("B2 holds the same average defensiveness every week without timing; it isolates the value of dynamic risk "
             "timing from simply holding less risk.\n")
    # 22
    L.append("## 22. Statistical significance and uncertainty\n")
    L.append(f"{len(c['netp'])} daily observations; {c['n_events']} independent SPY drawdown events ({c['n_major']} >= 10%). "
             "Annualized Sharpe, CAGR and maximum drawdown are path statistics; their intervals come from a paired "
             "stationary bootstrap (block 21 primary, 63 secondary, 5000 draws, fixed seed) and are approximate.\n")
    sc = st[st["family"].isin(["primary", "dynamic_vs_constant"]) &
            (st["method"].str.contains("Newey") | st["method"].str.contains("(primary)", regex=False))]
    L.append("Primary settings (Newey-West for the mean; stationary bootstrap with mean block 21 for path statistics):\n")
    L.append(_md_table(_f(sc[["portfolio_a", "portfolio_b", "statistic", "estimate", "ci_low", "ci_high", "p_value",
                              "p_value_holm"]].set_index("portfolio_a"), digits=4)) + "\n")
    sec = st[st["method"].str.contains("(secondary)", regex=False) & (st["statistic"] == "sharpe_difference")]
    L.append("Secondary block length (63 sessions), net Sharpe difference:\n")
    L.append(_md_table(_f(sec[["portfolio_a", "portfolio_b", "estimate", "ci_low", "ci_high", "p_value"]]
                          .set_index("portfolio_a"), digits=4)) + "\n")
    L.append(f"{con['H']['question']} {con['H']['verdict']}. A 95% interval that contains 0 is not evidence that the "
             "strategies are equivalent.\n")
    # 23
    L.append("## 23. Prespecified sensitivity analyses\n")
    cs = c["cost_sens"]
    piv = cs[cs["portfolio"].isin(ports)].pivot(index="portfolio", columns="basis", values="sharpe").loc[ports]
    piv = piv[["gross"] + [f"net_{x:g}bps" for x in c["bps_all"]]]
    L.append("Net Sharpe ratio by cost assumption:\n")
    L.append(_md_table(_f(piv)) + "\n")
    L.append(f"{con['E']['verdict']}.\n")
    ms = c["model_sens"]
    L.append("Model and initialization sensitivity (robustness only; the primary policy is never replaced):\n")
    L.append(_md_table(_f(ms[["variant", "initialization", "portfolio", "cagr", "annual_volatility", "sharpe", "max_drawdown",
                              "annualized_one_way_turnover", "max_abs_weight_difference_vs_restart_reference"]]
                          .set_index("variant"), {"cagr", "annual_volatility", "max_drawdown"})) + "\n")
    # 24
    L.append("## 24. Model failures and unfavorable market conditions\n")
    if e is not None and len(e):
        bad = e[(e["portfolio"] == "P2_ews_enhanced")].sort_values("recovery_participation_vs_P0").head(3)
        for r in bad.itertuples():
            L.append(f"- Event {r.event} (SPY peak {pd.Timestamp(r.spy_peak_date).date()}): P2 recovery over 63 sessions after "
                     f"the trough {_pct(r.recovery_return_63_sessions)} vs P0 {_pct(r.recovery_return_63_sessions - r.recovery_participation_vs_P0)}; "
                     f"de-risking: {r.de_risking_timing}.")
        late = e[(e["portfolio"] == "P2_ews_enhanced") & (e["de_risking_timing"] == "after onset (reaction)")]
        L.append(f"- P2 reacted only after the -5% onset in {len(late)} event(s).")
    if len(c["exceptions"]):
        ex = c["exceptions"]
        kinds = ex.groupby(["portfolio", "reason"]).size().reset_index(name="count")
        L.append(f"- {len(ex)} reported policy exceptions (portfolio_constraint_audit.csv):")
        L.append("")
        L.append(_md_table(kinds.set_index("portfolio")))
        L.append("")
    if len(c["warnings_te"]):
        L.append(f"- Solver warnings in Test decisions: {len(c['warnings_te'])} (documented; final solutions verified).")
    L.append("")
    # 25
    L.append("## 25. Research limitations\n")
    for lim in ["One Test period (about seven years) with few independent stress episodes; results are dominated by "
                "a handful of events.",
                "Validation was reused in Stages 5-7; Train signals are retrospective reconstructions.",
                "Adjusted-close execution and the 5 / 10 / 25 bps cost scenarios are assumptions, not observed fills.",
                "P(non-normal) is saturated (often exactly 0 or 1), so P1 behaves almost like a two-state switch.",
                "Stage 7 constraints and risk-budget slopes are proposed research assumptions; they were not optimized but "
                "they shape every dynamic result.",
                "Gross / net returns ignore taxes, fund fees beyond the adjusted-close series and market impact."]:
        L.append(f"- {lim}")
    L.append("")
    # 26
    L.append("## 26. Economic interpretation\n")
    L.append("Forecasting quality, regime classification, early warning, portfolio implementation and realized investment "
             "performance are different claims. Stage 6 measured ranking quality of the signals; Stage 7 showed that the "
             "policies change ex-ante risk; only this stage measures realized money outcomes after delay and costs. A "
             "better early-warning score does not by itself imply better portfolio performance, and lower volatility is "
             "not alpha.\n")
    L.append(f"- Attribution: {con['F']['verdict']}.")
    L.append(f"- Implementation: {con['G']['verdict']}.\n")
    # 27
    L.append("## 27. Final investment conclusion\n")
    for k in ("A", "B", "C", "D", "E", "F", "G", "H"):
        L.append(f"- **{k}. {con[k]['question']}** {con[k]['verdict']}.")
    L.append(f"\n**{con['outcome']}** (rule-based classification frozen before Test).\n")
    L.append(f"**STAGE 8: {c['stage8_status']}** - **FINAL RESEARCH STATUS: {c['final_status']}**\n")
    # 28
    L.append("## 28. Next steps for presentation and potential deployment\n")
    L.append("- Presentation: use investment_decision_summary, cumulative_wealth_gross_net, portfolio_drawdowns, "
             "ews_incremental_attribution and performance_uncertainty; state the prespecified outcome label and the "
             "number of independent events.")
    L.append("- Deployment would require a live, append-only signal log (true point-in-time), paper trading with real "
             "quotes, ETF-specific cost estimates, a governance rule for model refits, and an approved client mandate "
             "for the proposed constraints. None of these is claimed here.\n")
    L.append("## Checks\n")
    L.append(_md_table(pd.DataFrame([vars(x) for x in c["checks"]]).set_index("name")) + "\n")
    L.append("Figures: " + ", ".join(f"`{f}`" for f in c["figures"]) + "\n")
    return "\n".join(L)


def print_console(c: dict) -> None:
    b = c["prim_bps"]
    perf_p = c["perf_p"]
    ports = list(PIDS) + list(SUPP)
    cols = ["cagr", "annual_volatility", "sharpe", "sortino", "max_drawdown", "annualized_one_way_turnover",
            f"total_cost_fraction_{b:g}bps"]
    print(f"\nTest {c['test_start'].date()}..{c['test_end'].date()} ({len(c['netp'])} sessions), net of {b:g} bps:")
    print(perf_p.loc[ports, cols].round(4).to_string())
    g = c["perf"][c["perf"]["basis"] == "gross"].set_index("portfolio")
    print("\nGross CAGR / Sharpe: " + "; ".join(f"{p.split('_')[0]} {g.loc[p, 'cagr']:.4f} / {g.loc[p, 'sharpe']:.3f}"
                                              for p in ports))
    st = c["stats"]
    sub = st[st["method"].str.contains("primary", regex=False) | (st["statistic"] == "mean_active_return")]
    sub = sub[sub["statistic"].isin(["mean_active_return", "sharpe_difference", "max_drawdown_difference"])]
    print("\nPrespecified comparisons (net):")
    print(sub[["portfolio_a", "portfolio_b", "statistic", "estimate", "ci_low", "ci_high", "p_value", "p_value_holm"]]
          .round(4).to_string(index=False))
    if c["windows"] is not None and len(c["windows"]):
        w = c["windows"]
        print("\nStress windows (net cumulative return / max drawdown):")
        print(w.pivot(index="window", columns="portfolio", values="net_cumulative_return")[list(PIDS)].round(4).to_string())
    con = c["concl"]
    print("\nPrespecified verdicts:")
    for k in ("A", "B", "C", "D", "E", "F", "G", "H"):
        print(f"  {k}. {con[k]['verdict']}")
    print(f"  => {con['outcome']}")
