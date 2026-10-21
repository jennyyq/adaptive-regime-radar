"""Stage 7: tables, figures, research report and Stage 8 handoff manifest for the portfolio-construction stage.

All numbers are generated from the computed tables. Every quantity is EX ANTE (predicted at the decision date) or a
retrospective research diagnostic; no realized strategy performance is computed or plotted.
"""

from __future__ import annotations

STAGE7_CODE_VERSION = "stage7-2026-10-21"  # must match scripts/09_optimize_portfolio.py

from pathlib import Path

import numpy as np
import pandas as pd

from .quality import _md_table
from .regime_report import GRID, INK, INK_2, MUTED, SURFACE, _fmt, _save, _style, _title

PIDS = ("P0_strategic", "P1_regime_aware", "P2_ews_enhanced")
PCOL = {"P0_strategic": "#898781", "P1_regime_aware": "#2a78d6", "P2_ews_enhanced": "#eb6834"}
GROUP_COL = {"equity": "#2a78d6", "duration": "#1baf7a", "credit": "#eda100", "real_assets": "#c93131",
             "liquidity": "#8a5cd6"}
LEVEL_COL = {"GREEN": "#1baf7a", "AMBER": "#eda100", "ORANGE": "#eb6834", "RED": "#c93131", "RECOVERY": "#6aa4e6"}


# --------------------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------------------
def representative_dates(vt: pd.DataFrame) -> dict:
    """Validation decision dates chosen by fixed rules (no performance information): the last decision at each
    P2 alert level, and the last decisions with the lowest / highest P1 defensiveness."""
    p2 = vt[vt["portfolio"] == "P2_ews_enhanced"]
    p1 = vt[vt["portfolio"] == "P1_regime_aware"]
    out = {}
    for lv in ("GREEN", "AMBER", "ORANGE", "RED", "RECOVERY"):
        d = p2.loc[p2["signal_value"] == lv, "signal_date"]
        if len(d):
            out[f"P2 {lv}"] = d.max()
    if len(p1):
        out["P1 lowest d"] = p1.loc[p1["budget_d"] == p1["budget_d"].min(), "signal_date"].max()
        out["P1 highest d"] = p1.loc[p1["budget_d"] == p1["budget_d"].max(), "signal_date"].max()
    return dict(sorted(out.items(), key=lambda kv: kv[1]))


def representative_table(targets: pd.DataFrame, rep_dates: dict, assets: list, policy) -> pd.DataFrame:
    rows = []
    for label, t in rep_dates.items():
        for pid in PIDS:
            r = targets[(targets["portfolio"] == pid) & (targets["signal_date"] == t)]
            if r.empty:
                continue
            r = r.iloc[0]
            rows.append({"condition": label, "signal_date": t.date(), "portfolio": pid,
                         "signal": r.get("signal_value", ""), "budget_d": r.get("budget_d", np.nan),
                         **{a: r[f"w_{a}"] for a in assets},
                         **{g: r[f"exposure_{g}"] for g in policy.groups},
                         "vol_ex_ante": r["vol_ex_ante"], "cvar95_daily": r["cvar95_daily_ex_ante"],
                         "effective_n": r["effective_n"], "solver_status": r["solver_status"]})
    return pd.DataFrame(rows)


def turnover_table(targets: pd.DataFrame, cost_cfg: dict) -> pd.DataFrame:
    rows = []
    for (pid, sp), g in targets.groupby(["portfolio", "split"]):
        years = len(g) / 52.0
        row = {"portfolio": pid, "split": sp, "decisions": len(g),
               "mean_turnover": g["turnover_ex_ante_proxy"].mean(), "median_turnover": g["turnover_ex_ante_proxy"].median(),
               "max_turnover": g["turnover_ex_ante_proxy"].max(),
               "annualized_turnover": g["turnover_ex_ante_proxy"].sum() / years,
               "share_at_turnover_cap": float((g["turnover_ex_ante_proxy"] >= g.get("budget_turnover_cap", np.inf) - 1e-6).mean())
               if "budget_turnover_cap" in g and g["budget_turnover_cap"].notna().any() else np.nan,
               "stress_friction_share": g["stress_friction_on"].mean()}
        for bps in cost_cfg["scenarios_bps_of_traded_notional"]:
            row[f"annual_cost_{int(bps)}bps"] = g[f"cost_{int(bps)}bps"].sum() / years
        rows.append(row)
    return pd.DataFrame(rows)


def stability_table(targets: pd.DataFrame, assets: list) -> pd.DataFrame:
    rows = []
    wc = [f"w_{a}" for a in assets]
    for (pid, sp), g in targets.groupby(["portfolio", "split"]):
        g = g.sort_values("signal_date")
        W = g[wc].to_numpy()
        rows.append({"portfolio": pid, "split": sp, "decisions": len(g),
                     "mean_abs_weekly_weight_change": float(np.abs(np.diff(W, axis=0)).mean()) if len(W) > 1 else np.nan,
                     **{f"mean_{a}": float(W[:, i].mean()) for i, a in enumerate(assets)},
                     **{f"sd_{a}": float(W[:, i].std()) for i, a in enumerate(assets)},
                     "mean_vol": g["vol_ex_ante"].mean(), "mean_equity": g["exposure_equity"].mean(),
                     "mean_bil": g["w_BIL"].mean(), "mean_effective_n": g["effective_n"].mean()})
    p1 = targets[targets["portfolio"] == "P1_regime_aware"].set_index("signal_date")
    p2 = targets[targets["portfolio"] == "P2_ews_enhanced"].set_index("signal_date")
    for sp in ("train", "validation"):
        idx = p1.index[p1["split"] == sp].intersection(p2.index)
        if len(idx):
            diff = np.abs(p1.loc[idx, wc].to_numpy() - p2.loc[idx, wc].to_numpy())
            rows.append({"portfolio": "P1_vs_P2", "split": sp, "decisions": len(idx),
                         "mean_abs_weekly_weight_change": np.nan,
                         **{f"mean_{a}": float(diff[:, i].mean()) for i, a in enumerate(assets)},
                         "mean_vol": float((p1.loc[idx, "vol_ex_ante"] - p2.loc[idx, "vol_ex_ante"]).abs().mean()),
                         "mean_equity": float((p1.loc[idx, "exposure_equity"] - p2.loc[idx, "exposure_equity"]).abs().mean()),
                         "mean_bil": float((p1.loc[idx, "w_BIL"] - p2.loc[idx, "w_BIL"]).abs().mean()),
                         "share_dates_identical": float((diff.max(axis=1) < 1e-6).mean()),
                         "mean_active_share_between": float(0.5 * diff.sum(axis=1).mean())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
def _vline(ax, t):
    ax.axvline(t, color=INK, linewidth=0.9)


def _val_start(targets):
    v = targets.loc[targets["split"] == "validation", "signal_date"]
    return v.min() if len(v) else None


def fig_strategic(policy, path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    _style(ax, grid_y=False)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    grp = {a: g for g, m in policy.groups.items() for a in m}
    y = np.arange(len(policy.assets))[::-1]
    ax.barh(y, policy.benchmark, color=[GROUP_COL[grp[a]] for a in policy.assets], height=0.65)
    for yy, a, w, hi in zip(y, policy.assets, policy.benchmark, policy.max_weight):
        ax.text(w + 0.004, yy, f"{w:.0%}", va="center", fontsize=8, color=INK_2)
        ax.plot([hi, hi], [yy - 0.35, yy + 0.35], color=INK, linewidth=1.0)
    ax.set_yticks(y, policy.assets, fontsize=9, color=INK_2)
    ax.set_xlabel("weight (black tick = proposed single-ETF cap)", fontsize=8, color=INK_2)
    ax.set_title("Strategic benchmark (frozen, configs/project.json)", fontsize=10, color=INK, loc="left")
    ax = axes[1]
    _style(ax, grid_y=False)
    ex = policy.exposures(policy.benchmark)
    gs = list(policy.groups)
    ax.barh(range(len(gs))[::-1], [ex[g] for g in gs], color=[GROUP_COL[g] for g in gs], height=0.6)
    for i, g in enumerate(gs):
        yy = len(gs) - 1 - i
        ax.text(ex[g] + 0.005, yy, f"{ex[g]:.0%}  [limit {policy.group_min[g]:.0%}-{policy.group_max[g]:.0%}]",
                va="center", fontsize=8, color=INK_2)
    ax.set_yticks(range(len(gs))[::-1], gs, fontsize=9, color=INK_2)
    ax.set_xlim(0, 0.8)
    ax.set_title("Asset-class exposure and proposed static limits", fontsize=10, color=INK, loc="left")
    _title(fig, "Strategic asset allocation (P0 benchmark)",
           "Equities 50%, duration 25%, credit 5%, real assets 15%, BIL 5%. Limits are PROPOSED Stage 7 research "
           "assumptions, not an approved mandate.")
    _save(fig, path)


def fig_weight_heatmap(targets, assets, path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(14, 7.5), facecolor=SURFACE, sharex=True)
    vs = _val_start(targets)
    for ax, pid in zip(axes, ("P1_regime_aware", "P2_ews_enhanced")):
        g = targets[targets["portfolio"] == pid].sort_values("signal_date")
        W = g[[f"w_{a}" for a in assets]].to_numpy().T
        t = g["signal_date"].to_numpy()
        edges = np.r_[t, t[-1] + np.timedelta64(7, "D")]                 # each target holds until the next decision
        im = ax.pcolormesh(edges, np.arange(len(assets) + 1), W, shading="flat", cmap="Blues", vmin=0, vmax=0.45)
        ax.set_yticks(np.arange(len(assets)) + 0.5, assets, fontsize=8, color=INK_2)
        if vs is not None:
            _vline(ax, vs)
            ax.text(vs, len(assets) + 0.2, "  Validation", fontsize=8, color=INK, va="bottom")
        ax.set_title(f"{pid}: target weights at each weekly decision (Train = retrospective reconstruction)",
                     fontsize=10, color=INK, loc="left")
        fig.colorbar(im, ax=ax, pad=0.01, fraction=0.02)
    _title(fig, "Dynamic target weights", "Weights are ex-ante targets executed at the close of t+1; not realized holdings.")
    _save(fig, path)


def fig_asset_class(targets, policy, path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(14, 7.5), facecolor=SURFACE, sharex=True)
    vs = _val_start(targets)
    b = policy.exposures(policy.benchmark)
    for ax, pid in zip(axes, ("P1_regime_aware", "P2_ews_enhanced")):
        _style(ax)
        g = targets[targets["portfolio"] == pid].sort_values("signal_date")
        gs = list(policy.groups)
        ax.stackplot(g["signal_date"], *[g[f"exposure_{x}"] for x in gs], colors=[GROUP_COL[x] for x in gs], alpha=0.85,
                     labels=gs, linewidth=0)
        cum = 0
        for x in gs:
            cum += b[x]
            ax.axhline(cum, color="white", linewidth=0.6, linestyle="--")
        if vs is not None:
            _vline(ax, vs)
        ax.set_ylim(0, 1)
        ax.set_title(f"{pid} (dashed white = P0 benchmark group boundaries)", fontsize=10, color=INK, loc="left")
        ax.legend(fontsize=8, frameon=False, loc="lower left", ncol=5)
    _title(fig, "Asset-class allocation over time", "Equity = SPY+EFA+EEM, duration = IEF+TIP, credit = HYG, real assets = "
           "GLD+DBC, liquidity = BIL. Ex-ante targets.")
    _save(fig, path)


def fig_regime_vs_ews(targets, path):
    import matplotlib.pyplot as plt
    p1 = targets[targets["portfolio"] == "P1_regime_aware"].sort_values("signal_date")
    p2 = targets[targets["portfolio"] == "P2_ews_enhanced"].sort_values("signal_date")
    vs = _val_start(targets)
    fig, axes = plt.subplots(4, 1, figsize=(14, 11), facecolor=SURFACE, sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 1, 1]})
    ax = axes[0]
    _style(ax)
    ax.step(p1["signal_date"], p1["budget_d"], where="post", color=PCOL["P1_regime_aware"], label="P1 d (from P(non-normal))")
    ax.step(p2["signal_date"], p2["budget_d"], where="post", color=PCOL["P2_ews_enhanced"], label="P2 d (from EWS level)")
    ax.set_ylabel("defensiveness d", fontsize=8, color=INK_2)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.set_title("Signal -> defensiveness (same mapping d -> budget for both)", fontsize=10, color=INK, loc="left")
    for ax, col, lab in zip(axes[1:], ("exposure_equity", "exposure_duration", "w_BIL"),
                            ("equity (SPY+EFA+EEM)", "duration (IEF+TIP)", "BIL")):
        _style(ax)
        ax.plot(p1["signal_date"], p1[col], color=PCOL["P1_regime_aware"], linewidth=1.0, label="P1")
        ax.plot(p2["signal_date"], p2[col], color=PCOL["P2_ews_enhanced"], linewidth=1.0, label="P2")
        p0 = targets[targets["portfolio"] == "P0_strategic"][col].iloc[0]
        ax.axhline(p0, color=PCOL["P0_strategic"], linestyle="--", linewidth=0.9, label="P0")
        ax.set_ylabel(lab, fontsize=8, color=INK_2)
        ax.legend(fontsize=8, frameon=False, loc="upper right", ncol=3)
    for ax in axes:
        if vs is not None:
            _vline(ax, vs)
    _title(fig, "Regime-aware (P1) versus EWS-enhanced (P2) allocation",
           "Same optimizer, covariance, expected returns and constraints; only the signal -> d step differs. P1's scale is "
           "Train-calibrated so both have the same average Train defensiveness.")
    _save(fig, path)


def fig_risk_budget(targets, path):
    import matplotlib.pyplot as plt
    vs = _val_start(targets)
    fig, axes = plt.subplots(2, 1, figsize=(14, 7.5), facecolor=SURFACE, sharex=True)
    for ax, pid in zip(axes, ("P1_regime_aware", "P2_ews_enhanced")):
        _style(ax)
        g = targets[targets["portfolio"] == pid].sort_values("signal_date")
        ax.plot(g["signal_date"], g["benchmark_vol_ex_ante"], color=PCOL["P0_strategic"], linewidth=1.0,
                label="P0 benchmark ex-ante vol")
        ax.step(g["signal_date"], g["budget_vol_target"], where="post", color=INK, linewidth=0.8, linestyle="--",
                label="risk-budget target vol")
        ax.plot(g["signal_date"], g["vol_ex_ante"], color=PCOL[pid], linewidth=1.1, label=f"{pid} ex-ante vol")
        tr = g[g["solver_stage"] == 2]
        ax.scatter(tr["signal_date"], tr["vol_ex_ante"], s=12, color="#c93131", zorder=3,
                   label="turnover-limited transition (budget not yet met)")
        if vs is not None:
            _vline(ax, vs)
        ax.set_ylabel("annualized volatility", fontsize=8, color=INK_2)
        ax.legend(fontsize=8, frameon=False, loc="upper right", ncol=2)
        ax.set_title(pid, fontsize=10, color=INK, loc="left")
    _title(fig, "Risk-budget timeline (ex ante)", "Ledoit-Wolf 504-session covariance through t; target = vol multiplier(d) x "
           "benchmark ex-ante vol. Not realized volatility.")
    _save(fig, path)


def fig_risk_contribution(contribs, assets, path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=SURFACE)
    for ax, sp in zip(axes, ("train", "validation")):
        _style(ax)
        g = contribs[contribs["split"] == sp]
        x = np.arange(len(assets))
        for k, pid in enumerate(PIDS):
            m = g[g["portfolio"] == pid][[f"pct_risk_{a}" for a in assets]].mean()
            ax.bar(x + (k - 1) * 0.27, m.to_numpy(), width=0.27, color=PCOL[pid], label=pid)
        ax.axhline(0, color=MUTED, linewidth=0.7)
        ax.set_xticks(x, assets, fontsize=8, color=INK_2)
        ax.set_ylabel("mean share of ex-ante portfolio volatility", fontsize=8, color=INK_2)
        ax.set_title(f"{sp.capitalize()}{' (retrospective)' if sp == 'train' else ''}", fontsize=10, color=INK, loc="left")
        ax.legend(fontsize=8, frameon=False)
    _title(fig, "Component risk contribution by ETF", "CCR_i = w_i (Sigma w)_i / sigma_p; shares sum to 100% per decision. "
           "Equity dominates portfolio risk even at 50% capital weight.")
    _save(fig, path)


def fig_frontier(frontier, targets, last, path):
    import matplotlib.pyplot as plt
    t, ri = last
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=SURFACE)
    _style(ax)
    for (name, pts), col in zip(frontier.items(), ("#2a78d6", "#c93131")):
        if pts:
            f = pd.DataFrame(pts)
            ax.plot(f["vol"], f["expected_return"], color=col, marker="o", markersize=3, label=f"feasible frontier, {name}")
    for k, pid in enumerate(PIDS):
        r = targets[(targets["portfolio"] == pid) & (targets["signal_date"] == t)]
        if len(r):
            ax.scatter(r["vol_ex_ante"], r["expected_return_ex_ante"], s=70, color=PCOL[pid], zorder=4,
                       label=f"{pid} (d = {r['budget_d'].iloc[0]:.2f})" if pid != "P0_strategic" else pid, edgecolor="white")
    ax.set_xlabel("ex-ante annualized volatility", fontsize=9, color=INK_2)
    ax.set_ylabel("equilibrium expected return (annualized, assumption)", fontsize=9, color=INK_2)
    ax.legend(fontsize=8, frameon=False)
    _title(fig, f"Efficient frontier under the policy constraints at {t.date()}",
           "Expected returns are benchmark-implied equilibrium ASSUMPTIONS (SR_b = 0.30), not forecasts; the frontier shows "
           "feasible trade-offs, not achievable performance.")
    _save(fig, path)


def fig_turnover(turn, targets, path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=SURFACE, gridspec_kw={"width_ratios": [1, 1.4]})
    ax = axes[0]
    _style(ax)
    x = np.arange(2)
    for k, pid in enumerate(PIDS):
        v = [turn[(turn["portfolio"] == pid) & (turn["split"] == s)]["annualized_turnover"].sum() for s in ("train", "validation")]
        ax.bar(x + (k - 1) * 0.27, v, width=0.27, color=PCOL[pid], label=pid)
        for xx, vv in zip(x + (k - 1) * 0.27, v):
            ax.text(xx, vv, f"{vv:.2f}", ha="center", va="bottom", fontsize=7, color=INK_2)
    ax.set_xticks(x, ["Train", "Validation"], color=INK_2)
    ax.set_ylabel("annualized one-way turnover (ex-ante proxy)", fontsize=8, color=INK_2)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Turnover per year", fontsize=10, color=INK, loc="left")
    ax = axes[1]
    _style(ax)
    sc = [c for c in turn.columns if c.startswith("annual_cost_")]
    labels = []
    vals = {pid: [] for pid in PIDS}
    for s in ("train", "validation"):
        for c in sc:
            labels.append(f"{s[:5]} {c.replace('annual_cost_', '')}")
            for pid in PIDS:
                vals[pid].append(1e4 * turn[(turn["portfolio"] == pid) & (turn["split"] == s)][c].sum())
    x = np.arange(len(labels))
    for k, pid in enumerate(PIDS):
        ax.bar(x + (k - 1) * 0.27, vals[pid], width=0.27, color=PCOL[pid], label=pid)
    ax.set_xticks(x, labels, fontsize=8, color=INK_2)
    ax.set_ylabel("prospective cost, bps of portfolio per year", fontsize=8, color=INK_2)
    ax.set_title("Prospective transaction cost by cost scenario (illustrative bps of traded notional)", fontsize=10, color=INK,
                 loc="left")
    _title(fig, "Prospective turnover and transaction costs", "Turnover vs the drifted pre-trade proxy (through t); exact "
           "realized turnover and after-cost returns belong to Stage 8.")
    _save(fig, path)


def fig_cvar(targets, comparison, path):
    import matplotlib.pyplot as plt
    vs = _val_start(targets)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    _style(ax)
    for pid in PIDS:
        g = targets[targets["portfolio"] == pid].sort_values("signal_date")
        ax.plot(g["signal_date"], 100 * g["cvar95_daily_ex_ante"], color=PCOL[pid], linewidth=1.0, label=pid)
    if vs is not None:
        _vline(ax, vs)
    ax.set_ylabel("ex-ante 95% CVaR, daily loss (%)", fontsize=8, color=INK_2)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Historical-scenario CVaR (trailing 504 sessions through t)", fontsize=10, color=INK, loc="left")
    ax = axes[1]
    _style(ax)
    if len(comparison) and "cvar_lp_cvar95" in comparison:
        for pid in ("P1_regime_aware", "P2_ews_enhanced"):
            g = comparison[comparison["portfolio"] == pid]
            ax.scatter(100 * g["mvo_cvar95"], 100 * g["cvar_lp_cvar95"], s=10, color=PCOL[pid], alpha=0.6, label=pid)
        lim = [100 * comparison[["mvo_cvar95", "cvar_lp_cvar95"]].min().min(), 100 * comparison[["mvo_cvar95", "cvar_lp_cvar95"]].max().max()]
        ax.plot(lim, lim, color=INK, linewidth=0.8, linestyle="--")
    ax.set_xlabel("CVaR95 of the primary MVO target (%)", fontsize=8, color=INK_2)
    ax.set_ylabel("CVaR95 of the CVaR-LP target (%)", fontsize=8, color=INK_2)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Secondary CVaR optimizer vs MVO (Validation)", fontsize=10, color=INK, loc="left")
    _title(fig, "Ex-ante tail risk (CVaR)", "Empirical CVaR from a limited number of historical tail days is unstable; it is not "
           "maximum drawdown and not a realized loss.")
    _save(fig, path)


def fig_stress(scen_tab, path):
    import matplotlib.pyplot as plt
    ports = list(dict.fromkeys(scen_tab["portfolio"]))
    cond = next((c for c in ("P2 RED", "P2 ORANGE", "P2 AMBER") if any(c in p for p in ports)), None)
    keep = [p for p in ports if cond and cond in p] or ports[:3]
    g = scen_tab[scen_tab["portfolio"].isin(keep)]
    scen = list(dict.fromkeys(g["scenario"]))
    fig, ax = plt.subplots(figsize=(14, 6.5), facecolor=SURFACE)
    _style(ax, grid_y=False)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    h = 0.8 / max(len(keep), 1)
    cmap = {p: PCOL[next(x for x in PIDS if p.startswith(x))] for p in keep}
    alphas = {}
    for p in keep:
        alphas[p] = 1.0
    for k, p in enumerate(keep):
        v = [100 * g[(g["portfolio"] == p) & (g["scenario"] == s)]["scenario_loss"].sum() for s in scen]
        ax.barh(np.arange(len(scen)) + k * h - 0.4 + h / 2, v, height=h, color=cmap[p], alpha=alphas[p], label=p)
    ax.set_yticks(range(len(scen)), scen, fontsize=8, color=INK_2)
    ax.set_xlabel("ex-ante scenario loss (% of portfolio; positive = loss)", fontsize=8, color=INK_2)
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=3)
    _title(fig, "Stress-scenario exposure of target allocations (most defensive representative P2 date)",
           "Historical scenarios = actual 21-session Train windows selected by fixed rules; two rows are explicit hypothetical "
           "assumptions. Losses are scenario arithmetic on target weights, not realized outcomes.")
    _save(fig, path)


def fig_constraints(con_checks, targets, path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.4, 1, 1]})
    ax = axes[0]
    _style(ax, grid_y=False)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    dyn = con_checks[con_checks["portfolio"] != "P0_strategic"]
    dyn = dyn[dyn["constraint"] != "sum_weights"]                   # an equality binds by definition
    act = dyn.groupby(["portfolio", "constraint"])["active"].mean().unstack(0).fillna(0)
    act = act[(act > 0.02).any(axis=1)].sort_values(act.columns[0])
    y = np.arange(len(act))
    for k, pid in enumerate(act.columns):
        ax.barh(y + (k - 0.5) * 0.4, act[pid], height=0.4, color=PCOL[pid], label=pid)
    ax.set_yticks(y, act.index, fontsize=7, color=INK_2)
    ax.set_xlabel("share of decisions where the constraint binds", fontsize=8, color=INK_2)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Active constraints", fontsize=10, color=INK, loc="left")
    ax = axes[1]
    _style(ax)
    for pid in PIDS:
        g = targets[targets["portfolio"] == pid].sort_values("signal_date")
        ax.plot(g["signal_date"], g["effective_n"], color=PCOL[pid], linewidth=1.0, label=pid)
    ax.set_ylabel("effective number of holdings (1 / HHI)", fontsize=8, color=INK_2)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Concentration", fontsize=10, color=INK, loc="left")
    ax = axes[2]
    _style(ax)
    st = targets[targets["portfolio"] != "P0_strategic"].groupby(["portfolio", "solver_status"]).size().unstack(0).fillna(0)
    x = np.arange(len(st))
    for k, pid in enumerate(st.columns):
        ax.bar(x + (k - 0.5) * 0.4, st[pid], width=0.4, color=PCOL[pid], label=pid)
        for xx, v in zip(x + (k - 0.5) * 0.4, st[pid]):
            ax.text(xx, v, f"{int(v)}", ha="center", va="bottom", fontsize=7, color=INK_2)
    ax.set_xticks(x, st.index, fontsize=8, color=INK_2, rotation=15)
    ax.set_title("Solver outcome (feasibility)", fontsize=10, color=INK, loc="left")
    _title(fig, "Constraint and feasibility diagnostics", "hold_within_no_trade_band = posture within 1% one-way; trade_to_target "
           "= posture reached; turnover_limited_transition = cap prevented reaching it (unmet budget reported); hold fallbacks "
           "would appear as hold_fallback_*.")
    _save(fig, path)


FIGURES = ("strategic_asset_allocation", "portfolio_weight_heatmap", "portfolio_asset_class_allocation",
           "regime_vs_ews_allocation", "risk_budget_timeline", "portfolio_risk_contribution",
           "efficient_frontier_comparison", "portfolio_turnover_comparison", "cvar_risk_comparison",
           "stress_scenario_allocation", "portfolio_constraint_diagnostics")


def save_figures(directory: Path, c: dict) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    pol, tg = c["policy"], c["targets"]
    calls = [lambda p: fig_strategic(pol, p), lambda p: fig_weight_heatmap(tg, pol.assets, p),
             lambda p: fig_asset_class(tg, pol, p), lambda p: fig_regime_vs_ews(tg, p), lambda p: fig_risk_budget(tg, p),
             lambda p: fig_risk_contribution(c["contribs"], pol.assets, p),
             lambda p: fig_frontier(c["frontier"], tg, c["inputs_last"], p), lambda p: fig_turnover(c["turnover"], tg, p),
             lambda p: fig_cvar(tg, c["comparison"], p), lambda p: fig_stress(c["scenarios"], p),
             lambda p: fig_constraints(c["con_checks"], tg, p)]
    out = []
    for name, f in zip(FIGURES, calls):
        path = directory / f"{name}.png"
        f(path)
        out.append(str(path))
    return out


# --------------------------------------------------------------------------------------
# Console, report and Stage 8 manifest
# --------------------------------------------------------------------------------------
def _summary_by(targets: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return targets.groupby(["portfolio", "split"])[cols].mean()


def print_console(c: dict) -> None:
    tg, pol = c["targets"], c["policy"]
    s = c["saturation"]
    print(f"\nP(non-normal) saturation (Train): share == 1.0 {s['train_share_p_eq_1']:.1%}, >= 0.999 "
          f"{s['train_share_p_ge_0_999']:.1%}, <= 0.001 {s['train_share_p_le_0_001']:.1%}; P1 parity scale a = "
          f"{s['parity_scale_a']:.4f}; mean d Train P1 {s['train_mean_d1']:.3f} / P2 {s['train_mean_d2']:.3f}, Validation "
          f"P1 {s['validation_mean_d1']:.3f} / P2 {s['validation_mean_d2']:.3f}")
    est = c["est"]
    print("\nCovariance estimators (mean over decisions):")
    print(est.groupby(["split", "estimator"])[["benchmark_vol_ex_ante", "condition_number", "min_eigenvalue", "vol_BIL",
                                               "evaluation_only_subsequent_realized_benchmark_vol_21d"]]
          .mean().round(4).to_string())
    cols = ["vol_ex_ante", "cvar95_daily_ex_ante", "exposure_equity", "exposure_duration", "exposure_credit",
            "exposure_real_assets", "exposure_liquidity", "effective_n", "turnover_ex_ante_proxy", "tracking_error_ex_ante"]
    print("\nMean ex-ante characteristics by portfolio and split:")
    print(_summary_by(tg, cols).round(4).to_string())
    print("\nRepresentative Validation conditions (target weights):")
    rt = c["rep_tab"]
    if len(rt):
        print(rt[["condition", "signal_date", "portfolio", "signal"] + pol.assets + ["vol_ex_ante", "cvar95_daily"]]
              .round(3).to_string(index=False))
    print("\nTurnover and prospective costs:")
    print(c["turnover"].round(4).to_string(index=False))
    print("\nWeight stability / P1 vs P2 differences:")
    print(c["stability"][["portfolio", "split", "decisions", "mean_abs_weekly_weight_change", "mean_vol", "mean_equity",
                          "mean_bil"]].round(4).to_string(index=False))
    dyn = tg[tg["portfolio"] != "P0_strategic"]
    print("\nOptimization outcomes: " + ", ".join(f"{k}: {v}" for k, v in dyn.groupby(["portfolio", "solver_status"]).size().items()))


def render_report(c: dict) -> str:
    tg, pol, pcfg, cfg = c["targets"], c["policy"], c["pcfg"], c["cfg"]
    data, checks = c["data"], c["checks"]
    counts = pd.Series([x.status for x in checks]).value_counts().to_dict()
    tr, va = data.train_features.index, data.validation_features.index
    s = c["saturation"]
    dyn = tg[tg["portfolio"] != "P0_strategic"]
    stab, turn = c["stability"], c["turnover"]
    L = ["# Stage 7 - Robust portfolio optimization and dynamic risk budgeting", "",
         f"## STAGE 8 HANDOFF STATUS: {c['status']}", "", c["reason"], "",
         f"Checks: {counts.get('PASS', 0)} PASS, {counts.get('WARN', 0)} WARN, {counts.get('FAIL', 0)} FAIL. "
         f"Train {tr.min().date()}..{tr.max().date()} ({len(tr)} sessions); Validation {va.min().date()}..{va.max().date()} "
         f"({len(va)} sessions). Test not read. Everything below is EX ANTE or a retrospective diagnostic; no realized "
         "returns, Sharpe ratios, drawdowns or wealth curves are computed in Stage 7.", ""]
    v = _summary_by(tg, ["vol_ex_ante", "cvar95_daily_ex_ante", "exposure_equity", "w_BIL", "turnover_ex_ante_proxy",
                         "effective_n"]).reset_index()
    vv = v[v["split"] == "validation"].set_index("portfolio")
    L += ["## 1. Executive summary", "",
          f"- A feasible, interpretable weekly allocation system was implemented for P0 (strategic), P1 (HMM P(non-normal)) "
          f"and P2 (Full EWS) with one shared optimizer, risk model, expected-return model, constraint set, cost model, "
          f"calendar and the frozen execution convention. {len(tg[tg['portfolio'] == 'P0_strategic'])} weekly decisions per "
          "portfolio.",
          "- Optimizer outcomes for P1/P2: " + ", ".join(f"{k[0]} {k[1]} {v_}" for k, v_ in
                                                           dyn.groupby(["portfolio", "solver_status"]).size().items()) + ".",
          "- Validation mean ex-ante volatility: " + ", ".join(f"{p} {vv.at[p, 'vol_ex_ante']:.2%}" for p in vv.index) +
          "; mean equity: " + ", ".join(f"{p} {vv.at[p, 'exposure_equity']:.1%}" for p in vv.index) + "; mean BIL: " +
          ", ".join(f"{p} {vv.at[p, 'w_BIL']:.1%}" for p in vv.index) + ".",
          f"- P(non-normal) is saturated: on Train {s['train_share_p_eq_1']:.1%} of days equal 1.0 exactly and "
          f"{s['train_share_p_le_0_001']:.1%} are <= 0.001, so P1's continuous budget behaves almost like a two-state "
          f"switch (mean Train d: P1 {s['train_mean_d1']:.3f} vs P2 {s['train_mean_d2']:.3f}, equal by construction).",
          "- Predicted risk differences are not realized investment improvements. Whether either signal adds value after "
          "costs must be decided by the Stage 8 chronological backtest.", ""]
    L += ["## 2. Investment mandate", "",
          "Moderate-risk multi-asset client with a long-term strategic allocation; nine US-listed ETFs; long only, no "
          "leverage or derivatives; at most one rebalance per week. Benchmark (frozen): " +
          ", ".join(f"{a} {w:.0%}" for a, w in zip(pol.assets, pol.benchmark)) + ".", "",
          "Proposed constraints (research assumptions, not an approved mandate):", "",
          _md_table(pd.DataFrame({"benchmark": pol.benchmark, "max weight": pol.max_weight},
                                 index=pd.Index(pol.assets, name="ETF")).map(lambda x: f"{x:.0%}")), "",
          f"Maximum active weight per ETF: {pol.max_active:.0%}. Group limits: " +
          ", ".join(f"{g} {pol.group_min[g]:.0%}-{pol.group_max[g]:.0%}" for g in pol.groups) +
          f". Turnover cap {pol.turnover_cap:.0%} one-way per rebalance (rising with defensiveness, see section 8).", ""]
    L += ["## 3. Data and temporal protocol", "",
          f"- Returns: the split files store daily LOG returns of adjusted closes ({data.manifest.get('return_definition')}); "
          "no missing values in the nine-ETF common sample. They are converted exactly to SIMPLE returns, R = exp(r) - 1; "
          "portfolio returns, covariances, CVaR scenarios and expected returns are all in simple-return units, annualized "
          "with 252 sessions.",
          "- Decisions: last trading session of every calendar week; execution at the close of t+1; first holding return "
          "t+2 (frozen conservative_close_t_plus_1). Exactly one execution per calendar week. The last Train decision "
          "(2015-12-31) executes on the first Validation session and is flagged split-crossing. The final Validation week "
          "is incomplete without the Test calendar and has no decision.",
          f"- Warm-up: a decision needs >= {cfg.min_obs} past sessions; the first decision is "
          f"{tg['signal_date'].min().date()}.",
          "- Train decisions use frozen HMM/AR/EWS models fitted on all of Train: they are retrospective research "
          "reconstructions, not live or deployable historical decisions. Covariances and expected returns are always "
          "as-of t.", ""]
    ers = c["er_summary"]
    L += ["## 4. Expected-return model", "",
          "Primary (low alpha): benchmark-implied equilibrium returns mu_t = rf_t + (SR_b / sigma_b,t) Sigma_t w_b with a "
          f"PROPOSED benchmark Sharpe ratio SR_b = {cfg.sharpe}. With gamma_t = SR_b / sigma_b,t the unconstrained optimum "
          "is the benchmark, so every deviation is driven by the risk budget and the constraints, not by noisy means.", "",
          _md_table(_fmt(ers.pivot(index="asset", columns="split", values="mu_equilibrium_mean"), 4)
                    .rename(columns=lambda x: f"mean mu_eq {x}")), "",
          "Historical trailing means and their standard errors (sensitivity only):", "",
          _md_table(_fmt(ers[ers["split"] == "validation"].set_index("asset")[["hist_mean_mean", "hist_se_mean"]], 4)), "",
          "A one-standard-error band of a 2-year mean is several percentage points per year for equities - larger than the "
          "equilibrium premia. This is why raw historical means are not used (section 12 reports the weight sensitivity).", ""]
    est = c["est"]
    e = est.groupby(["split", "estimator"])[["benchmark_vol_ex_ante", "evaluation_only_subsequent_realized_benchmark_vol_21d",
                                            "condition_number", "min_eigenvalue", "vol_BIL", "mean_pairwise_correlation",
                                            "mean_abs_weekly_change_asset_vol", "lw_shrinkage", "effective_n"]].mean()
    err = est.dropna(subset=["evaluation_only_subsequent_realized_benchmark_vol_21d"]).copy()
    err["abs_log_ratio"] = np.abs(np.log(err["benchmark_vol_ex_ante"] / err["evaluation_only_subsequent_realized_benchmark_vol_21d"]))
    acc = err.groupby(["split", "estimator"])["abs_log_ratio"].mean().rename("mean |log(predicted / subsequent)|")
    L += ["## 5. Covariance and risk estimation", "",
          f"Primary: Ledoit-Wolf shrinkage (to a scaled identity, analytic intensity; identical to sklearn's estimator) on "
          f"the trailing {cfg.lookback} daily simple returns through t, annualized x252; PSD floor at "
          f"{cfg.psd_floor:g} x mean variance (never needed for Ledoit-Wolf). Alternatives (robustness only): sample 504, "
          "EWMA half-life 63 (effective sample size reported), Ledoit-Wolf 252 and expanding.", "",
          _md_table(_fmt(e, 4)), "",
          "Predicted-versus-subsequent risk (DEVELOPMENT DIAGNOSTIC ONLY - uses returns after t, never used for a decision): "
          "benchmark volatility predicted at t versus realized over the 21 sessions after the first holding date:", "",
          _md_table(_fmt(acc.to_frame(), 3)), "",
          "BIL's estimated volatility is a fraction of a percent; without the BIL cap and the active-weight limit a "
          "risk-averse optimizer would load into it. BIL is treated as the liquidity sleeve with an explicit cap.", ""]
    L += ["## 6. Optimization mathematics", "",
          "Step 1 - posture target (path independent, depends only on information at t and the role's signal):", "",
          "maximize  w'mu - (gamma/2) w'Sigma w - (lambda_TE/2) (w - w_b)'Sigma(w - w_b)", "",
          "subject to sum w = 1, 0 <= w <= cap, |w - w_b| <= active limit, static group limits, risk-budget group limits "
          "and w'Sigma w <= (vol_multiplier(d) sigma_b)^2.", "",
          "Step 2 - trade rule (identical for P0, P1, P2; P0's posture is the benchmark): hold if the one-way distance "
          f"to the posture is <= the no-trade band ({cfg.no_trade_band:.0%}); else trade to the posture if within the "
          "turnover cap; else move along the straight line toward it as far as the cap allows (a convex combination "
          "of feasible points, so every linear hard constraint holds) and report the unmet budget.", "",
          f"- Units: annualized simple returns and covariance. gamma_t = SR_b / sigma_b,t; lambda_TE = {cfg.lambda_tracking}. "
          "With equilibrium returns and gamma = delta, the d = 0 posture is exactly the benchmark.",
          "- Why two steps: a single problem with a linear turnover penalty was tested and is kept as a sensitivity case. "
          "It is path dependent - de-risking is forced by the binding budget, but re-risking is only rewarded by the very "
          "flat objective near the benchmark, so the penalty's no-trade region traps the portfolio in defensive positions "
          "after an alert (a ratchet). The two-step rule removes this while the hard turnover cap and the no-trade band "
          "still control trading.",
          f"- Solver: SLSQP (scipy {c['versions']['scipy']}) with analytic gradients from two starting points (benchmark and "
          f"an LP-feasible point); the strictly concave objective has a unique optimum (max start disagreement "
          f"{dyn['start_disagreement'].max():.1e}).",
          "- Deterministic fallback: if the rule's result violated a hard constraint (only possible if drift pushed the "
          "pre-trade weights outside the policy), a slack formulation keeps every hard constraint and penalizes the unmet "
          "budget; if that fails, hold pre-trade / previous target / benchmark if it satisfies every hard constraint. Never "
          "equal weights.", ""]
    cmp_ = c["comparison"]
    L += ["## 7. CVaR model", "",
          "Ex-ante 95% CVaR from the trailing 504 daily simple-return scenarios through t (empirical, equally weighted); "
          "VaR = 95% loss quantile, CVaR = Rockafellar-Uryasev value (exact for the empirical distribution). Secondary "
          "optimizer: an LP that maximizes expected return net of L1 turnover and tracking penalties subject to CVaR95 <= "
          "vol_multiplier(d) x benchmark CVaR95 and the same linear constraints.", ""]
    if len(cmp_) and "cvar_lp_vol" in cmp_:
        g = cmp_.groupby("portfolio")[["mvo_vol", "cvar_lp_vol", "mvo_cvar95", "cvar_lp_cvar95", "max_abs_weight_diff_vs_mvo"]].mean()
        L += [_md_table(_fmt(g, 4)), "",
              "LP status: " + ", ".join(f"{k} {v_}" for k, v_ in cmp_["cvar_lp_status"].str.split(":").str[0].value_counts().items())
              + ". Tail estimates rest on ~25 days per 504-session window; they are unstable and are not drawdowns.", ""]
    coef = pcfg["risk_budget"]["coefficients"]
    L += ["## 8. Dynamic risk budgeting", "",
          "| d | vol multiplier | equity max | credit max | BIL min | turnover cap |", "|---|---|---|---|---|---|"]
    for d in (0, 1 / 3, 2 / 3, 1):
        L.append(f"| {d:.2f} | {1 - coef['vol_slope'] * d:.3f} | {coef['equity_max0'] - coef['equity_slope'] * d:.3f} | "
                 f"{coef['credit_max0'] - coef['credit_slope'] * d:.3f} | {coef['liquidity_min0'] + coef['liquidity_slope'] * d:.3f}"
                 f" | {coef['turnover_cap0'] + coef['turnover_slope'] * d:.3f} |")
    L += ["", "### Regime-aware allocation (P1)", "",
          f"d1 = min(1, a x P(non-normal)) with a = {s['parity_scale_a']:.4f}, calibrated on Train so that mean Train d1 = "
          "mean Train d2. The map is Lipschitz (no jump near p = 1). Because P(non-normal) is saturated "
          f"(Train: {s['train_share_p_eq_1']:.1%} exactly 1.0, {s['train_share_p_ge_0_999']:.1%} >= 0.999, "
          f"{s['train_share_p_le_0_001']:.1%} <= 0.001), P1 alternates mostly between d = 0 and d = a: a regime switch "
          "rather than a graded signal. P(non-normal) also pools S2 (neutral) with S3 (stress), so P1 is moderately "
          "defensive in neutral regimes too.", "",
          "### EWS-enhanced allocation (P2)", "",
          "d2 = GREEN 0, AMBER 1/3, ORANGE 2/3, RED 1, RECOVERY 1/3 (AMBER posture held until the frozen state machine "
          "returns to GREEN). The level at the decision date is used; intra-week escalations are recorded "
          "(signal_week_max) and acted upon only at the next decision.", ""]
    p2 = tg[tg["portfolio"] == "P2_ews_enhanced"]
    lvl = p2.groupby(["split", "signal_value"])[["budget_d", "budget_vol_multiplier", "vol_ex_ante", "exposure_equity",
                                                 "exposure_duration", "exposure_credit", "w_GLD", "w_DBC", "w_BIL"]].mean()
    p1 = tg[tg["portfolio"] == "P1_regime_aware"].copy()
    p1["d_bucket"] = pd.cut(p1["budget_d"], [-1e-9, 1e-3, s["parity_scale_a"] * 0.5, 1.0], labels=["d~0", "low d", "high d"])
    l1 = p1.groupby(["split", "d_bucket"], observed=True)[["budget_d", "vol_ex_ante", "exposure_equity", "exposure_duration",
                                                           "exposure_credit", "w_GLD", "w_DBC", "w_BIL"]].mean()
    L += ["Do higher warning levels produce more defensive targets? Mean ex-ante characteristics by P2 level:", "",
          _md_table(_fmt(lvl, 3)), "", "P1 by defensiveness bucket:", "", _md_table(_fmt(l1, 3)), ""]
    L += ["## 9. Investment constraints", "",
          "Binding frequency of each constraint (share of P1/P2 decisions):", ""]
    cc = c["con_checks"]
    act = cc[cc["portfolio"] != "P0_strategic"].groupby(["constraint", "portfolio"])["active"].mean().unstack().fillna(0)
    act = act[(act > 0).any(axis=1)]
    L += [_md_table(_fmt(act, 3)), "",
          f"Hard-constraint violations in published targets: {int(tg['hard_violations'].fillna('').ne('').sum())}. "
          f"Turnover-limited transitions (risk budget not fully reached within one week): "
          f"{int((dyn['solver_stage'] == 2).sum())}.", ""]
    rt = c["rep_tab"]
    L += ["## 10. Portfolio results", "", "Representative Validation conditions (dates chosen by fixed rules, not by "
          "performance):", "", _md_table(_fmt(rt.set_index(["condition", "portfolio"]), 3)) if len(rt) else "n/a", "",
          "Mean ex-ante characteristics:", "",
          _md_table(_fmt(_summary_by(tg, ["vol_ex_ante", "cvar95_daily_ex_ante", "expected_return_ex_ante",
                                          "tracking_error_ex_ante", "effective_n", "herfindahl", "active_share"]), 4)), "",
          "Turnover (ex-ante proxy against drifted pre-trade weights) and prospective costs (fraction of portfolio per year):",
          "", _md_table(_fmt(turn.set_index(["portfolio", "split"]), 4)), ""]
    contr = c["contribs"]
    pct = contr.groupby(["portfolio", "split"])[[f"pct_risk_{a}" for a in pol.assets]].mean()
    L += ["Mean share of ex-ante volatility by ETF:", "", _md_table(_fmt(pct.rename(columns=lambda x: x.replace("pct_risk_", "")), 3)), ""]
    L += ["## 11. Train versus Validation diagnostics", "",
          "Train rows are retrospective reconstructions (signals from models fitted on all of Train).", "",
          _md_table(_fmt(stab.set_index(["portfolio", "split"])[["decisions", "mean_abs_weekly_weight_change", "mean_vol",
                                                                  "mean_equity", "mean_bil", "mean_effective_n"]], 4)), ""]
    pv = stab[stab["portfolio"] == "P1_vs_P2"]
    if len(pv):
        L += ["P1 versus P2 target differences:", "",
              _md_table(_fmt(pv.set_index("split")[["decisions", "share_dates_identical", "mean_active_share_between",
                                                    "mean_vol", "mean_equity", "mean_bil"]], 4)), ""]
    sens = c["sensitivity"]
    L += ["## 12. Sensitivity analysis", "",
          "Validation decisions, every variant restarted from the benchmark at the first Validation decision; changes are "
          "measured against the identically restarted reference. None of these variants was used to choose the primary "
          "policy.", "", _md_table(_fmt(sens.set_index(["variant", "portfolio"]), 4)) if len(sens) else "Not run.", ""]
    fails = tg[(tg["portfolio"] != "P0_strategic") & (tg["solver_stage"] > 1)]
    bil_cap = (tg["w_BIL"] >= pol.max_weight[pol.assets.index("BIL")] - 1e-6).groupby(tg["portfolio"]).mean()
    L += ["## 13. Failure cases", "",
          f"- Turnover-limited transitions: {len(fails)} decisions (" + ", ".join(
              f"{p} {n}" for p, n in fails.groupby("portfolio").size().items()) + "); the unmet budget is recorded per decision.",
          "- BIL at its cap: " + ", ".join(f"{p} {v_:.1%}" for p, v_ in bil_cap.items()) + " of decisions.",
          f"- Saturation: P1's budget is effectively binary (Train distinct d values: {s['share_d1_distinct_values_train']}; "
          "most mass at the two extremes).",
          "- Excessive defensiveness is limited by design: the equity floor (20%), the active-weight limit and the BIL cap "
          "bound how far any posture can move from the benchmark.",
          "- Hold fallbacks or hard-constraint violations: see the checks in section 16 (any occurrence fails the stage).", ""]
    L += ["## 14. Research limitations", "",
          "- Validation (2016-2019) was reused in Stages 5-6.1; all Validation conclusions are development-sample evidence.",
          "- HMM/AR/EWS parameters, Train-quantile thresholds and the P1 parity scale use all of Train; Train allocations "
          "are retrospective.",
          "- Point-in-time macro handling is inherited from Stage 2 (ALFRED first releases where available).",
          "- Gaussian HMM probabilities are saturated near 0 and 1; P(non-normal) is a regime score, not a calibrated loss "
          "probability, and pools S2 with S3.",
          "- Few independent stress events (7 Validation SPY -5% drawdowns); risk-budget responses cannot be validated "
          "statistically on so few events.",
          "- Expected returns are equilibrium assumptions; historical means are far too noisy to be forecasts.",
          "- Transaction costs are illustrative scenarios (5/10/25 bps of traded notional, x2 in stress), not observed "
          "costs; adjusted closes are not guaranteed executable prices.",
          "- Turnover is an ex-ante proxy (drift through t); exact execution-day turnover, after-cost returns and all "
          "realized performance belong to the Stage 8 walk-forward backtest.", ""]
    L += ["## 15. Stage 8 handoff", "",
          "`data/processed/portfolio_optimization/stage8_handoff_manifest.json` freezes: universe, benchmark, constraints, "
          "expected-return rule, covariance method, optimizer objective and penalties, risk-budget mappings (incl. the "
          "Train-calibrated P1 scale), CVaR parameters, weekly calendar, cost scenarios, execution timing, role bindings, "
          "solver and fallback policy, and every relevant hash. Primary policies to evaluate: P0_strategic, "
          "P1_regime_aware, P2_ews_enhanced (primary specification); listed sensitivity cases only.", ""]
    L += ["## 16. Checks", "", _md_table(pd.DataFrame([vars(x) for x in checks]).set_index("name")), "",
          f"**Final: {counts.get('PASS', 0)} PASS, {counts.get('WARN', 0)} WARN, {counts.get('FAIL', 0)} FAIL - Stage 8 handoff "
          f"{c['status']}.**", ""]
    ts = c["test_summary"]
    if ts.get("ran"):
        L += [f"Automated real-data tests: {ts['tests']} run, {ts['failures']} failures, {ts['errors']} errors, "
              f"{ts['skipped']} skipped.", ""]
        for f in ts.get("failed_tests", []):
            L += [f"**FAILED: {f['test']}**", "", "```text", f["traceback"].rstrip(), "```", ""]
    L += ["Figures: " + ", ".join(f"`figures/portfolio_optimization/{Path(p).name}`" for p in c["figures"]), ""]
    return "\n".join(L)


def stage8_manifest(policy, pcfg, cfg, scale, stress_thr, h_man, h_path, out_hashes, code_hashes, versions, status,
                    reason, policy_sha, project_sha) -> dict:
    from .ar_hmm_diagnostics import file_sha256
    return {
        "stage": "7 -> 8 handoff: frozen portfolio policies for chronological walk-forward evaluation",
        "code_version": STAGE7_CODE_VERSION, "created_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "handoff_status": status, "handoff_status_reason": reason,
        "portfolios": {
            "P0_strategic": {"signal_role": None, "rule": "posture = strategic weights; same target_then_trade rule "
                                                          "(no-trade band, turnover cap) as P1 and P2"},
            "P1_regime_aware": {"signal_role": "baseline_regime", "signal_column": "baseline_p_non_normal",
                                "defensiveness": f"min(1, {scale!r} * baseline_p_non_normal)"},
            "P2_ews_enhanced": {"signal_role": "challenger_ews", "signal_column": "challenger_alert_level",
                                "defensiveness": pcfg["risk_budget"]["P2_ews_levels"]}},
        "universe": policy.assets, "benchmark_weights": dict(zip(policy.assets, policy.benchmark.tolist())),
        "constraints": pcfg["constraints"], "risk_budget": pcfg["risk_budget"],
        "train_calibrated": {"P1_parity_scale_a": scale, "stress_vol_threshold_spy_21d_annualized": stress_thr},
        "expected_returns": pcfg["expected_returns"], "risk_estimation": pcfg["risk_estimation"],
        "optimizer": {**pcfg["optimizer"], "lambda_turnover": cfg.lambda_turnover, "lambda_tracking": cfg.lambda_tracking,
                      "execution_mode": cfg.execution_mode, "no_trade_band": cfg.no_trade_band},
        "cvar": pcfg["cvar"], "transaction_costs": pcfg["transaction_costs"], "rebalancing": pcfg["rebalancing"],
        "execution_convention": h_man.get("execution_convention"),
        "walk_forward_protocol": [
            "Load signals only with regime_radar.stage7_handoff.load_stage7_inputs(role, split) for the declared role.",
            "At each weekly decision date t use returns with date <= t only (portfolio_engine.prepare_inputs / run_portfolio).",
            "Execute at the close of t+1; credit holding returns from t+2 onward; drift weights daily between executions.",
            "Recompute the pre-trade weights at the execution close (drift through t+1) for realized turnover; the "
            "decision itself must keep using the ex-ante proxy (drift through t).",
            "Charge transaction costs on realized execution-day turnover (pre-trade weights drifted through t+1).",
            "Apply the identical calendar, costs and constraints to P0, P1 and P2; report each against P0.",
            "Do not change any frozen parameter after seeing Test results; sensitivity cases are limited to those listed.",
            "Stage 8 must extend the frozen signals to the Test period with the frozen Stage 5-6 models (no refit) "
            "before evaluation; Stage 7 did not read any Test file."],
        "sensitivity_cases_permitted": ["cov_sample_504", "cov_ewma_hl63", "cov_lw_252", "er_shrunk_historical",
                                        "no_trade_band_0.02", "turnover_cap_halved", "single_step_turnover_penalized",
                                        "mapping_P1_full_range_P2_continuous", "cost 5/25 bps"],
        "role_bindings": {"P1_regime_aware": "baseline_regime", "P2_ews_enhanced": "challenger_ews"},
        "hashes": {"stage7_handoff_manifest": file_sha256(h_path), "portfolio_policy.json": policy_sha,
                   "project.json": project_sha, "stage7_outputs": out_hashes, "stage7_code": code_hashes,
                   "frozen_models": h_man["input_hashes"]["frozen_models"],
                   "split_files": h_man["input_hashes"]["split_files"]},
        "software_versions": versions, "test_data_loaded": False}
