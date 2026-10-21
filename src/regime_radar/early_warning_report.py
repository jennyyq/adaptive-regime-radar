"""Stage 6: figures and research report for the dual-speed early warning system (Train/Validation only).

Every number in the report is read from the objects computed by scripts/08_build_early_warning.py; conclusions are
generated from pre-specified comparisons (block-bootstrap intervals, event counts), not written by hand.
"""

from __future__ import annotations

STAGE6_CODE_VERSION = "stage6-2026-10-21"  # must match scripts/08_build_early_warning.py

from pathlib import Path

import numpy as np
import pandas as pd

from . import alert_state_machine as asm
from . import bocpd as bo
from . import early_warning as ew
from . import early_warning_validation as ev
from .quality import _md_table
from .regime_report import AXIS, GRID, INK, INK_2, MUTED, SHADE, SURFACE, _fmt, _save, _style, _title

STATE_COLORS = {"p_s1": "#1baf7a", "p_s2": "#eda100", "p_s3": "#c93131"}
COMPONENT_COLORS = {"regime": "#2a78d6", "confirmation": "#c93131", "innovation": "#eb6834", "bocpd": "#8a5cd6",
                    "velocity": "#eda100", "macro": "#898781", "entropy": "#1baf7a"}
SYSTEM_LABELS = {"A_hmm_hard_stress": "A hard HMM stress state", "B_p_stress": "B P(stress)",
                 "C_p_non_normal": "C P(non-normal)", "D_volatility_level": "D volatility level",
                 "E_ar_innovation": "E AR(1) innovation", "F_bocpd": "F BOCPD only",
                 "G_hmm_bocpd": "G HMM + BOCPD", "H_hmm_innovation_bocpd": "H HMM + innov. + BOCPD",
                 "I_full_ews_score": "I full EWS (score, 20% budget)",
                 "I_full_ews_state_amber": "I full EWS state >= AMBER",
                 "I_full_ews_state_orange": "I full EWS state >= ORANGE"}
STATUS_COLORS = {"warned": "#1baf7a", "detected_late": "#eda100", "missed": "#c93131"}
FIGURE_NAMES = ("bocpd_change_point_diagnostics", "factor_level_vs_innovation_bocpd", "hmm_regime_uncertainty",
                "probability_velocity", "early_warning_score_timeline", "alert_state_timeline",
                "warning_lead_time_analysis", "false_alarm_analysis", "early_warning_benchmark_comparison",
                "early_warning_ablation")


def _label(name: str) -> str:
    return SYSTEM_LABELS.get(name, name)


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def entropy_around_switches(daily: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Mean normalized entropy at offsets -window..+window around every change of the filtered argmax state."""
    s = daily["hmm_state"].to_numpy()
    e = daily["normalized_entropy"].to_numpy(float)
    sw = np.flatnonzero(s[1:] != s[:-1]) + 1
    rows = []
    for split in ("train", "validation"):
        sel = [t for t in sw if daily["split"].iloc[t] == split and window <= t < len(s) - window]
        for off in range(-window, window + 1):
            vals = e[[t + off for t in sel]] if sel else np.array([])
            rows.append({"split": split, "offset": off, "mean_entropy": float(np.nanmean(vals)) if len(vals) else np.nan,
                         "n_switches": len(sel)})
    return pd.DataFrame(rows)


def spy_series(data) -> pd.Series:
    return pd.concat([data.train_returns["SPY"], data.validation_returns["SPY"]])


def drawdown_by_split(r: pd.Series, split: pd.Series) -> pd.Series:
    out = []
    for name in ("train", "validation"):
        lvl = r[split == name].cumsum()
        out.append(np.expm1(lvl - lvl.cummax()))
    return pd.concat(out)


def _shade_validation(ax, va_index) -> None:
    ax.axvspan(va_index.min(), va_index.max(), color=SHADE, zorder=0, linewidth=0)


def _onsets(ax, events: dict, color: str = INK, alpha: float = 0.35) -> None:
    for e in events.values():
        for d in (e["onset_date"] if not e.empty else []):
            ax.axvline(d, color=color, linewidth=0.7, alpha=alpha, zorder=1)


def _level_spans(ax, levels: pd.Series) -> None:
    v = levels.to_numpy()
    idx = levels.index
    start = 0
    for i in range(1, len(v) + 1):
        if i == len(v) or v[i] != v[start]:
            if v[start] != "GREEN":
                end = idx[i] if i < len(v) else idx[-1]
                ax.axvspan(idx[start], end, color=asm.ALERT_COLORS[v[start]], alpha=0.28, linewidth=0, zorder=0)
            start = i


def _event_tables(ctx: dict) -> dict:
    vm = ctx["val_mask"]
    out = {}
    for name, a in ctx["alerts"].items():
        _, t = ev.event_metrics(a[vm], ctx["events"]["validation"])
        out[name] = t
    return out


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
def fig_bocpd(ctx: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    d, va_idx = ctx["daily"], ctx["va_index"]
    pre = f"bocpd_B2_innovation_h{bo.PRIMARY_RUN_LENGTH}"
    fig, axes = plt.subplots(3, 1, figsize=(13, 8.6), sharex=True, facecolor=SURFACE)
    ax = axes[0]
    _style(ax)
    _shade_validation(ax, va_idx)
    ax.plot(d.index, d["market_innovation"], color=AXIS, linewidth=0.5, label="daily market innovation")
    ax.plot(d.index, d["market_innovation_5d"], color="#eb6834", linewidth=0.9, label="5-session sum (Train sd = 1)")
    ax.axhline(0, color=MUTED, linewidth=0.7)
    ax.set_ylabel("adverse innovation (sd)", fontsize=8, color=INK_2)
    ax.set_title("Primary BOCPD input: AR(1) market innovation (-equity + volatility + credit), positive = adverse",
                 fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper left", ncol=2)
    ax = axes[1]
    _style(ax)
    _shade_validation(ax, va_idx)
    ax.plot(d.index, d[f"{pre}_p_change_within_{bo.PRIMARY_WINDOW}d"], color="#8a5cd6", linewidth=0.8,
            label=f"P(change within last {bo.PRIMARY_WINDOW} sessions | data to t)")
    prior = bo.prior_recent_probability(bo.PRIMARY_RUN_LENGTH, bo.PRIMARY_WINDOW)
    ax.axhline(prior, color=INK, linestyle="--", linewidth=0.8, label=f"prior value {prior:.3f}")
    _onsets(ax, ctx["events"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("posterior probability", fontsize=8, color=INK_2)
    ax.set_title(f"BOCPD posterior, constant hazard 1/{bo.PRIMARY_RUN_LENGTH}; vertical lines = SPY -5% drawdown onsets",
                 fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper left", ncol=2)
    ax = axes[2]
    _style(ax)
    _shade_validation(ax, va_idx)
    ax.plot(d.index, d[f"{pre}_expected_run_length"], color="#2a78d6", linewidth=0.8, label="expected run length")
    ax.plot(d.index, d[f"{pre}_map_run_length"], color=MUTED, linewidth=0.5, label="MAP run length")
    ax.set_yscale("log")
    ax.set_ylabel("sessions since last change", fontsize=8, color=INK_2)
    ax.legend(fontsize=8, frameon=False, loc="upper left", ncol=2)
    _title(fig, "BOCPD change-point diagnostics (Train shaded white, Validation grey)",
           "Normal-Inverse-Gamma model with Train-fitted prior; exact recursion (no truncation). P(r_t = 0) equals the "
           "hazard under a constant hazard, so the recent-change probability is reported. Train values are in-sample.")
    _save(fig, path)


def fig_bocpd_inputs(ctx: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    d, va_idx, cmp_ = ctx["daily"], ctx["va_index"], ctx["bocpd_cmp"]
    names = {"B1_level": "B1 factor LEVEL stress index", "B2_innovation": "B2 AR(1) market innovation",
             "B3_arhmm_surprise": "B3 AR-HMM predictive surprise (probit)"}
    colors = {"B1_level": "#c93131", "B2_innovation": "#8a5cd6", "B3_arhmm_surprise": "#2a78d6",
              "B2_volatility": "#eda100", "B2_credit": "#1baf7a"}
    fig = plt.figure(figsize=(13, 10), facecolor=SURFACE)
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 1.2])
    first = None
    for i, (k, lab) in enumerate(names.items()):
        ax = fig.add_subplot(gs[i], sharex=first)
        first = first or ax
        _style(ax)
        _shade_validation(ax, va_idx)
        ax.plot(d.index, d[f"bocpd_{k}_h{bo.PRIMARY_RUN_LENGTH}_p_change_within_{bo.PRIMARY_WINDOW}d"],
                color=colors[k], linewidth=0.7)
        _onsets(ax, ctx["events"])
        ax.set_ylim(0, 1)
        ax.set_title(f"{lab}: P(change within {bo.PRIMARY_WINDOW} sessions), hazard 1/{bo.PRIMARY_RUN_LENGTH}",
                     fontsize=10, color=INK, loc="left")
    ax = fig.add_subplot(gs[3])
    _style(ax)
    v = cmp_[cmp_["split"] == "validation"].copy()
    labels = [f"{r.input}\nh{r.expected_run_length}" for r in v.itertuples()]
    x = np.arange(len(v))
    ax.bar(x, v["pr_auc"], color=[colors.get(i, MUTED) for i in v["input"]], width=0.7)
    if len(v):
        ax.axhline(v["base_rate"].iloc[0], color=INK, linestyle="--", linewidth=0.8, label="base rate (no skill)")
    for xx, val in zip(x, v["pr_auc"]):
        if pd.notna(val):
            ax.text(xx, val, f"{val:.2f}", ha="center", va="bottom", fontsize=7, color=INK_2)
    ax.set_xticks(x, labels, fontsize=7, color=INK_2)
    ax.set_ylabel("Validation PR-AUC", fontsize=8, color=INK_2)
    ax.set_title("Which BOCPD input carries information about 21-session SPY losses >= 5%? (Validation)",
                 fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False)
    _title(fig, "BOCPD on factor levels vs innovations vs AR-HMM surprise",
           "Levels are persistent and trend, so a level BOCPD re-detects slow drifts; innovations remove the AR(1) "
           "persistence. Hazard grid fixed in advance {63, 126, 252}. Validation = development sample.")
    _save(fig, path)


def fig_hmm_uncertainty(ctx: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    d, va_idx = ctx["daily"], ctx["va_index"]
    fig = plt.figure(figsize=(13, 9), facecolor=SURFACE)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 0.9, 1.1])
    ax = fig.add_subplot(gs[0, :])
    _style(ax)
    ax.stackplot(d.index, d["p_s1"], d["p_s2"], d["p_s3"], colors=list(STATE_COLORS.values()), alpha=0.85,
                 labels=["S1 growth / risk-on", "S2 neutral / transition", "S3 stress"], linewidth=0)
    ax.axvline(va_idx.min(), color=INK, linewidth=0.9)
    ax.set_ylim(0, 1)
    ax.set_title("Filtered (causal) HMM state probabilities; line = start of Validation", fontsize=10, color=INK,
                 loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper left", ncol=3)
    ax2 = fig.add_subplot(gs[1, :], sharex=ax)
    _style(ax2)
    _shade_validation(ax2, va_idx)
    ax2.plot(d.index, d["normalized_entropy"], color="#1baf7a", linewidth=0.6)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("H / log 3", fontsize=8, color=INK_2)
    ax2.set_title("Normalized filtered entropy (0 = certain, 1 = uniform)", fontsize=10, color=INK, loc="left")
    ax3 = fig.add_subplot(gs[2, 0])
    _style(ax3)
    se = ctx["switch_entropy"]
    for split, c in (("train", "#2a78d6"), ("validation", "#eb6834")):
        s = se[se["split"] == split]
        if len(s) and s["n_switches"].iloc[0]:
            ax3.plot(s["offset"], s["mean_entropy"], color=c, marker="o", markersize=3,
                     label=f"{split} ({int(s['n_switches'].iloc[0])} switches)")
    ax3.axvline(0, color=MUTED, linewidth=0.8)
    ax3.set_xlabel("sessions relative to a change of the argmax state", fontsize=8, color=INK_2)
    ax3.set_ylabel("mean entropy", fontsize=8, color=INK_2)
    ax3.set_title("Does entropy rise before regime switches?", fontsize=10, color=INK, loc="left")
    ax3.legend(fontsize=8, frameon=False)
    ax4 = fig.add_subplot(gs[2, 1])
    _style(ax4)
    c = ctx["entropy_cond"]
    if "tercile" in c and len(c):
        lab = [f"{r.regime_bucket.replace('p_non_normal ', 'P(nn) ')}\n{r.tercile}" for r in c.itertuples()]
        ax4.bar(np.arange(len(c)), c["event_rate"], color=["#2a78d6" if "<" in b else "#c93131"
                                                          for b in c["regime_bucket"]], width=0.7)
        ax4.set_xticks(np.arange(len(c)), lab, fontsize=7, color=INK_2)
    ax4.set_ylabel("share of days followed by a >= 5% SPY loss", fontsize=8, color=INK_2)
    ax4.set_title("Validation loss rate by entropy tercile within regime bucket", fontsize=10, color=INK, loc="left")
    _title(fig, "HMM regime uncertainty",
           "Frozen Stage 5 HMM (hmm_std_diag_k3), filtered probabilities only. Train values use parameters estimated on "
           "Train (retrospective). Entropy has weight 0 in the primary EWS and is tested as an addition.")
    _save(fig, path)


def fig_velocity(ctx: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    d = ctx["daily"]
    vd = d[d["split"] == "validation"]
    fig, axes = plt.subplots(2, 1, figsize=(13, 7.4), facecolor=SURFACE, gridspec_kw={"height_ratios": [1.3, 1]})
    ax = axes[0]
    _style(ax)
    ax.plot(vd.index, vd["p_non_normal"], color="#2a78d6", linewidth=0.9, label="P(non-normal) = P(S2) + P(S3)")
    ax.bar(vd.index, vd["non_normal_probability_velocity_positive"], color="#eda100", width=1.5,
           label="positive 5-session change (velocity)")
    sw = ctx["velocity"]
    if len(sw):
        s = sw[sw["split"] == "validation"]
        ax.scatter(s["switch_date"], np.full(len(s), 1.03), marker="v", color="#c93131", s=18,
                   label="argmax switch S1 -> S2/S3", clip_on=False)
    ax.set_ylim(0, 1.06)
    ax.set_title("Validation: regime probability and its velocity", fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper left", ncol=3)
    ax = axes[1]
    _style(ax)
    if len(sw):
        for i, (split, c) in enumerate((("train", "#2a78d6"), ("validation", "#eb6834"))):
            s = sw[sw["split"] == split]
            share = s["velocity_exceeded_before"].mean() if len(s) else np.nan
            ax.bar(i, share, color=c, width=0.6)
            ax.text(i, share if pd.notna(share) else 0, f"{share:.0%} of {len(s)}" if len(s) else "no switches",
                    ha="center", va="bottom", fontsize=8, color=INK_2)
            lead = s["lead_days"].dropna()
            if len(lead):
                ax.scatter(np.full(len(lead), i + 0.4) + np.linspace(-0.05, 0.05, len(lead)), lead / 10,
                           color=c, s=10, alpha=0.7)
        ax.set_xticks([0, 1], ["Train", "Validation"], color=INK_2)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("share of switches preceded by high velocity\n(dots: lead / 10 sessions)", fontsize=8, color=INK_2)
    ax.set_title("Was velocity above its Train 90th percentile in the 10 sessions before the switch?", fontsize=10,
                 color=INK, loc="left")
    _title(fig, "Regime-probability velocity",
           "Velocity = p_t - p_(t-5) of filtered probabilities (past only). A switch is an argmax change of the filtered "
           "state; velocity mechanically rises into a switch, so lead counts only sessions BEFORE the switch day.")
    _save(fig, path)


def fig_ews_timeline(ctx: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    d, va_idx, th = ctx["daily"], ctx["va_index"], ctx["thresholds"]
    contrib = ctx["contrib"]
    dd = drawdown_by_split(spy_series(ctx["data"]), d["split"])
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, facecolor=SURFACE,
                             gridspec_kw={"height_ratios": [1.6, 1]})
    ax = axes[0]
    _style(ax)
    sm = contrib.rolling(5, min_periods=1).mean()
    cols = list(contrib.columns)
    ax.stackplot(sm.index, *[sm[c].fillna(0) for c in cols], colors=[COMPONENT_COLORS[c] for c in cols], alpha=0.8,
                 labels=[ew.COMPONENT_LABELS[c] for c in cols], linewidth=0)
    for k, ls in (("amber_in", "-"), ("orange_in", "-"), ("red_in", "-")):
        ax.axhline(getattr(th, k), color=asm.ALERT_COLORS[k.split("_")[0].upper()], linewidth=1.0, linestyle=ls)
    ax.axvline(va_idx.min(), color=INK, linewidth=0.9)
    ax.set_ylim(0, 1)
    ax.set_ylabel("EWS (5-session mean of contributions)", fontsize=8, color=INK_2)
    ax.set_title("Early Warning Score decomposed by component; horizontal lines = Train AMBER / ORANGE / RED entry",
                 fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="upper left", ncol=3)
    ax = axes[1]
    _style(ax)
    _shade_validation(ax, va_idx)
    ax.fill_between(dd.index, 100 * dd, 0, color="#c93131", alpha=0.5, linewidth=0)
    ax.axhline(-5, color=INK, linestyle="--", linewidth=0.7)
    _onsets(ax, ctx["events"])
    ax.set_ylabel("SPY drawdown from split peak (%)", fontsize=8, color=INK_2)
    ax.set_title("Outcome (not an input): SPY drawdown; lines = event onsets", fontsize=10, color=INK, loc="left")
    _title(fig, "Early Warning Score timeline",
           f"Fixed weights {', '.join(f'{k} {v:.2f}' for k, v in ctx['weights'].items() if v)}; components scaled by "
           "Train ECDFs. The score is a ranking, not a loss probability. Train values are in-sample.")
    _save(fig, path)


def fig_alert_timeline(ctx: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    th, states, score = ctx["thresholds"], ctx["states"], ctx["score"]
    vm = ctx["val_mask"]
    spy = spy_series(ctx["data"])[vm]
    lv = states["alert_level"][vm]
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True, facecolor=SURFACE,
                             gridspec_kw={"height_ratios": [1.3, 1, 0.6]})
    ax = axes[0]
    _style(ax)
    _level_spans(ax, lv)
    ax.plot(spy.index, 100 * np.expm1(spy.cumsum()), color=INK, linewidth=0.9)
    for dte in ctx["events"]["validation"].get("onset_date", []):
        ax.axvline(dte, color=INK, linewidth=0.8, linestyle=":")
    ax.set_ylabel("SPY cumulative return (%)", fontsize=8, color=INK_2)
    ax.set_title("Validation: SPY with alert levels shaded (dotted = -5% drawdown onsets)", fontsize=10, color=INK,
                 loc="left")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=asm.ALERT_COLORS[k], alpha=0.5, label=k) for k in asm.LEVELS if k != "GREEN"],
              fontsize=8, frameon=False, loc="upper left", ncol=4)
    ax = axes[1]
    _style(ax)
    ax.plot(score[vm].index, score[vm], color=INK_2, linewidth=0.8)
    for lvl in ("amber", "orange", "red"):
        c = asm.ALERT_COLORS[lvl.upper()]
        ax.axhline(getattr(th, f"{lvl}_in"), color=c, linewidth=1.0)
        ax.axhline(getattr(th, f"{lvl}_out"), color=c, linewidth=0.8, linestyle="--")
    ax.set_ylabel("EWS", fontsize=8, color=INK_2)
    ax.set_title("EWS with entry (solid) and exit (dashed) thresholds - hysteresis band", fontsize=10, color=INK,
                 loc="left")
    ax = axes[2]
    _style(ax)
    sev = lv.map(lambda s: {"GREEN": 0, "RECOVERY": 0.5, "AMBER": 1, "ORANGE": 2, "RED": 3}[s])
    ax.step(sev.index, sev, where="post", color=INK, linewidth=0.8)
    ax.set_yticks([0, 0.5, 1, 2, 3], ["GREEN", "RECOV.", "AMBER", "ORANGE", "RED"], fontsize=7, color=INK_2)
    share = lv.value_counts(normalize=True)
    _title(fig, "Alert state machine (Validation)",
           "Time share: " + ", ".join(f"{k} {share.get(k, 0):.1%}" for k in asm.LEVELS) +
           f". Escalation needs {asm.CONFIRM_MIN}/{asm.CONFIRM_WINDOW} sessions above entry (or the emergency rule); "
           f"RECOVERY clears after {asm.RECOVERY_CLEAR_DAYS} quiet sessions.")
    _save(fig, path)


def fig_lead_time(ctx: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    tables = _event_tables(ctx)
    evs = ctx["events"]["validation"]
    names = list(tables)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    ax.set_facecolor(SURFACE)
    if len(evs):
        for j, n in enumerate(names):
            t = tables[n]
            for i, r in enumerate(t.itertuples()):
                ax.add_patch(plt.Rectangle((i, len(names) - 1 - j), 0.95, 0.9, color=STATUS_COLORS[r.status],
                                           alpha=0.85))
                txt = f"{r.lead_days:.0f}" if r.status == "warned" else (f"+{r.delay_days:.0f}" if r.status ==
                                                                        "detected_late" else "")
                ax.text(i + 0.47, len(names) - 1 - j + 0.45, txt, ha="center", va="center", fontsize=7, color="white")
        ax.set_xlim(0, len(evs))
        ax.set_ylim(0, len(names))
        ax.set_xticks(np.arange(len(evs)) + 0.47, [pd.Timestamp(x).strftime("%Y-%m-%d") for x in evs["onset_date"]],
                      rotation=45, ha="right", fontsize=7, color=INK_2)
        ax.set_yticks(np.arange(len(names)) + 0.45, [_label(n) for n in names[::-1]], fontsize=7, color=INK_2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Validation events: green = warned (lead days), amber = late (+delay), red = missed", fontsize=10,
                 color=INK, loc="left")
    ax = axes[1]
    _style(ax)
    for j, n in enumerate(names):
        lead = tables[n]["lead_days"].dropna() if len(tables[n]) else pd.Series(dtype=float)
        y = len(names) - 1 - j
        ax.scatter(lead, np.full(len(lead), y), color="#1baf7a", s=16)
        if len(lead):
            ax.plot([lead.median()] * 2, [y - 0.3, y + 0.3], color=INK, linewidth=1.2)
    ax.set_yticks(range(len(names)), [_label(n) for n in names[::-1]], fontsize=7, color=INK_2)
    ax.set_xlabel("lead = onset - start of the last alert episode before it (bar = median)", fontsize=8, color=INK_2)
    ax.set_xlim(0, ev.LEAD_WINDOW + 2)
    ax.set_title("Lead-time distribution", fontsize=10, color=INK, loc="left")
    _title(fig, "Warning lead time (Validation)",
           f"Event = SPY drawdown reaching -5% from its running peak. Warned = any alert in the {ev.LEAD_WINDOW} sessions "
           "before onset. A long-lasting alert can 'warn' mechanically; read lead time together with false alarms.")
    _save(fig, path)


def fig_false_alarms(ctx: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    c = ctx["comparison"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), facecolor=SURFACE)
    ax = axes[0]
    _style(ax)
    v = c[c["split"] == "validation"]
    def _short(n: str) -> str:
        return n.split("_")[0] + (" " + n.replace("I_full_ews_state_", "") if n.startswith("I_full_ews_state") else "")

    for r in v.itertuples():
        ax.scatter(r.false_alerts_per_year, r.event_recall_warned, s=40 + 400 * r.time_share_alerted,
                   color="#c93131" if r.system.startswith("I_") else "#2a78d6", alpha=0.6)
    groups = v.groupby([v["false_alerts_per_year"].round(3), v["event_recall_warned"].round(3)])["system"]
    for (x, y), names in groups:
        ax.annotate(", ".join(_short(n) for n in names), (x, y), fontsize=7, color=INK_2, xytext=(6, 4),
                    textcoords="offset points")
    ax.set_xlabel("false alert episodes per year", fontsize=8, color=INK_2)
    ax.set_ylabel("share of events warned in advance", fontsize=8, color=INK_2)
    ax.set_title("Validation trade-off (marker size = time share alerted)", fontsize=10, color=INK, loc="left")
    ax = axes[1]
    _style(ax)
    p = c[c["split"].isin(["train", "validation"])].pivot(index="system", columns="split", values="time_share_alerted")
    y = np.arange(len(p))
    ax.barh(y - 0.2, p.get("train", 0), height=0.4, color="#2a78d6", label="Train")
    ax.barh(y + 0.2, p.get("validation", 0), height=0.4, color="#eb6834", label="Validation")
    ax.axvline(ev.TRAIN_ALERT_BUDGET, color=INK, linestyle="--", linewidth=0.8, label="Train budget 20%")
    ax.set_yticks(y, [_label(s) for s in p.index], fontsize=7, color=INK_2)
    ax.set_xlabel("share of sessions on alert", fontsize=8, color=INK_2)
    ax.set_title("Alert frequency: equal Train budget, realized Validation share", fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False)
    _title(fig, "False-alarm analysis",
           f"False episode = alert run with no event onset within {ev.LEAD_WINDOW} sessions of its start and no overlap "
           "with an event period. Continuous baselines use their Train 80th percentile as threshold.")
    _save(fig, path)


def fig_benchmark(ctx: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    c, boot, sel = ctx["comparison"], ctx["boot"], ctx["selection"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), facecolor=SURFACE)
    ax = axes[0]
    _style(ax)
    sy = [s for s in SYSTEM_LABELS if s in set(c["system"]) and not s.startswith("I_full_ews_state")]
    p = c[c["split"].isin(["train", "validation"])].pivot(index="system", columns="split", values="pr_auc").reindex(sy)
    y = np.arange(len(p))[::-1]
    ax.barh(y + 0.2, p.get("train"), height=0.4, color="#2a78d6", label="Train (in-sample)")
    ax.barh(y - 0.2, p.get("validation"), height=0.4, color="#eb6834", label="Validation")
    br = c[c["split"] == "validation"]["base_rate"].iloc[0]
    ax.axvline(br, color=INK, linestyle="--", linewidth=0.8, label=f"Validation base rate {br:.3f}")
    ax.set_yticks(y, [_label(s) for s in p.index], fontsize=7, color=INK_2)
    ax.set_xlabel("PR-AUC, label = SPY loss >= 5% within 21 sessions", fontsize=8, color=INK_2)
    ax.set_title("Daily ranking skill", fontsize=10, color=INK, loc="left")
    ax.set_xlim(0, 1.6 * np.nanmax(p.to_numpy(float)))
    ax.legend(fontsize=8, frameon=False, loc="center right")
    ax = axes[1]
    _style(ax)
    b = boot.sort_values("point_ap_diff")
    y = np.arange(len(b))
    ax.errorbar(b["point_ap_diff"], y, xerr=[(b["point_ap_diff"] - b["ci95_low"]).clip(lower=0),
                                            (b["ci95_high"] - b["point_ap_diff"]).clip(lower=0)],
                fmt="o", color="#2a78d6", ecolor=AXIS, capsize=3)
    ax.axvline(0, color=INK, linewidth=0.8)
    ax.set_yticks(y, [_label(m) + (" (selected)" if m == sel["selected"] else "") for m in b["model"]], fontsize=7,
                  color=INK_2)
    ax.set_xlabel(f"PR-AUC minus best ({_label(sel['best_validation_pr_auc'])}): point, 95% block-bootstrap interval",
                  fontsize=8, color=INK_2)
    ax.set_title("Is the best score significantly better?", fontsize=10, color=INK, loc="left")
    _title(fig, "Benchmark comparison",
           f"Moving-block bootstrap ({ev.BOOTSTRAP_BLOCK}-session blocks). Selected: {_label(sel['selected'])} = least "
           "complex score not significantly worse than the best. Validation is a development sample, not a holdout.")
    _save(fig, path)


def fig_ablation(ctx: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    a = ctx["ablation"]
    v = a[a["split"] == "validation"].set_index("specification")
    t = a[a["split"] == "train"].set_index("specification")
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), facecolor=SURFACE)
    for ax, col, title in zip(axes, ("pr_auc", "false_alerts_per_year", "warned"),
                              ("PR-AUC (score)", "false alert episodes / year (>= AMBER)",
                               "events warned (>= AMBER)")):
        _style(ax, grid_y=False)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        y = np.arange(len(v))[::-1]
        ax.barh(y + 0.2, t[col].reindex(v.index), height=0.4, color="#2a78d6", label="Train")
        ax.barh(y - 0.2, v[col], height=0.4, color=["#c93131" if s == "full" else "#eb6834" for s in v.index],
                label="Validation")
        if "full" in v.index:
            ax.axvline(v.at["full", col], color=INK, linestyle="--", linewidth=0.8)
        ax.set_yticks(y, v.index, fontsize=8, color=INK_2)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
    axes[0].legend(fontsize=8, frameon=False, loc="lower right")
    n_ev = int(v["n_events"].iloc[0]) if len(v) else 0
    _title(fig, "Ablations of the Early Warning Score",
           f"Each variant re-fits only its own Train thresholds; weights are renormalized, never re-optimized. "
           f"Validation has {n_ev} events, so differences of one event are noise.")
    _save(fig, path)


def save_figures(directory: Path, ctx: dict) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    funcs = (fig_bocpd, fig_bocpd_inputs, fig_hmm_uncertainty, fig_velocity, fig_ews_timeline, fig_alert_timeline,
             fig_lead_time, fig_false_alarms, fig_benchmark, fig_ablation)
    out = []
    for name, f in zip(FIGURE_NAMES, funcs):
        path = directory / f"{name}.png"
        f(ctx, path)
        out.append(str(path))
    return out


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------
def _row(c: pd.DataFrame, system: str, split: str) -> pd.Series | None:
    r = c[(c["system"] == system) & (c["split"] == split)]
    return r.iloc[0] if len(r) else None


def _f(v, d: int = 3) -> str:
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{d}f}"


def conclusions(ctx: dict) -> dict:
    """Data-driven statements (no hand-written results)."""
    c, boot, sel = ctx["comparison"], ctx["boot"], ctx["selection"]
    full, best, chosen = "I_full_ews_score", sel["best_validation_pr_auc"], sel["selected"]
    out = {"full_is_best": best == full, "selected": chosen, "best": best}
    b = boot.set_index("model")
    simple = [s for s in ("B_p_stress", "C_p_non_normal", "D_volatility_level", "E_ar_innovation", "F_bocpd")]
    if best == full:
        sig = [s for s in simple if s in b.index and b.at[s, "ci95_high"] < 0]
        out["full_vs_simple"] = (f"The full EWS has the highest Validation PR-AUC; it is significantly better than "
                                 f"{len(sig)} of {len(simple)} single-signal baselines ({', '.join(sig) or 'none'}).")
    else:
        d = b.at[full, "point_ap_diff"] if full in b.index else np.nan
        lo, hi = (b.at[full, "ci95_low"], b.at[full, "ci95_high"]) if full in b.index else (np.nan, np.nan)
        out["full_vs_simple"] = (f"The full EWS is NOT the best ranking score on Validation: {_label(best)} is, and the "
                                 f"full EWS differs by {_f(d)} (95% interval {_f(lo)} to {_f(hi)}).")
    out["retain"] = (f"Retained specification: **{_label(chosen)}**" +
                     (" - the full system is not demonstrably better than a simpler method, so the simpler "
                      "specification is retained." if chosen != full else
                      " - no simpler score is statistically indistinguishable from it."))
    am = _row(c, "I_full_ews_state_amber", "validation")
    out["state_machine"] = (f"State machine >= AMBER on Validation: {int(am['warned'])}/{int(am['n_events'])} events "
                            f"warned, {int(am['detected_late'])} late, {int(am['missed'])} missed; median lead "
                            f"{_f(am['median_lead_days'], 0)} sessions; {_f(am['false_alerts_per_year'], 2)} false "
                            f"episodes/year; {am['time_share_alerted']:.1%} of sessions on alert.") if am is not None else ""
    return out


def render_report(ctx: dict) -> str:
    c, a, bc = ctx["comparison"], ctx["ablation"], ctx["bocpd_cmp"]
    data, sel, checks, th = ctx["data"], ctx["selection"], ctx["checks"], ctx["thresholds"]
    meta, weights = ctx["meta"], ctx["weights"]
    concl = conclusions(ctx)
    status = pd.Series([x.status for x in checks]).value_counts().to_dict()
    tr, va = data.train_features.index, data.validation_features.index
    ev_tr, ev_va = ctx["events"]["train"], ctx["events"]["validation"]
    L = ["# Stage 6 - Dual-speed market early warning system", "",
         f"Train {tr.min().date()}..{tr.max().date()} ({len(tr)} sessions); Validation {va.min().date()}.."
         f"{va.max().date()} ({len(va)} sessions). Test not read. Checks: "
         + ", ".join(f"{v} {k}" for k, v in status.items()) + ".", ""]

    # 1
    L += ["## 1. Executive summary", "", f"- {concl['full_vs_simple']}", f"- {concl['retain']}",
          f"- {concl['state_machine']}",
          f"- Events (SPY drawdown reaching -5%): Train {len(ev_tr)}, Validation {len(ev_va)}. With this few Validation "
          "events, event-level differences between systems are not statistically meaningful on their own.",
          "- No portfolio optimization, dynamic allocation or investment backtest was run (Stage 7 scope).", ""]
    # 2
    L += ["## 2. Status of the evidence", "",
          "- Validation (2016-2019) was used to select the Stage 5 HMM and to decide Stage 5.5. Every Validation number "
          "here is **development-sample evidence**, not an independent out-of-sample test.",
          "- Train-period HMM probabilities use parameters estimated on all of Train: they are in-sample "
          "reconstructions, not what a live system would have shown.",
          "- To limit tuning on the reused Validation sample, all weights, quantiles, hazards and windows were fixed "
          "in advance and every threshold is a Train quantile. No parameter was changed after seeing Validation.",
          "- The equity_trend factor contains SPY momentum, so the SPY outcomes are not fully independent of the "
          "signals; outcomes use FORWARD returns only.", ""]
    # 3
    L += ["## 3. Frozen inputs and integrity", "", _md_table(pd.DataFrame([vars(x) for x in checks]).set_index("name")),
          "", "The Stage 5.5 models were verified to come from the corrected exact variance-floor M-step "
          "(variance_rule = floor, model_version stage5.5-2026-10-21b) and their saved Train log-likelihoods were "
          "recomputed from the saved parameters before any signal was built.", ""]
    # 4
    L += ["## 4. Architecture: two speeds", "",
          "- **Slow layer** (persistent regimes): frozen HMM K=3 filtered probabilities, entropy, probability velocity.",
          "- **Fast layer** (abrupt shocks): AR(1) one-step innovations (what the factors did that their own "
          "persistence did not predict) and BOCPD on those innovations.",
          "- **Confirmation layer**: factor LEVELS of equity weakness, volatility and credit stress, plus "
          "rates/inflation pressure; a fast signal without level confirmation is weighted down.",
          "- **Decision layer**: a fixed-weight Early Warning Score and a 5-level alert state machine with "
          "confirmation and hysteresis.", "",
          "Timing: every date-t value uses factors through t and models estimated on Train. Investment returns are "
          "not an argument of the signal builder; they are used only for forward outcomes.", ""]
    # 5
    es = ctx["entropy_summary"]
    L += ["## 5. Slow layer: regime probabilities, entropy and velocity", "",
          "- p_non_normal = P(S2) + P(S3); p_stress = P(S3); regime component = 0.5 p_non_normal + 0.5 p_stress.",
          "- Normalized entropy = -sum p log p / log 3.",
          "- Velocity = p_t - p_(t-5); only the positive part (deterioration) enters the EWS.", "",
          _md_table(_fmt(es.set_index("split"), 3)), ""]
    # 6
    nm = meta["normalization"].to_dict()
    L += ["## 6. Fast layer: AR(1) innovations", "",
          "- z_t,d = (x_t,d - c_d - phi_d x_t-1,d) / sigma_d from the frozen Stage 5.5 AR(1) (corrected EM).",
          "- Adverse direction: equity -1, rates +1, inflation +1, volatility +1, credit +1.",
          "- market_innovation = (-z_equity + z_volatility + z_credit) / k_market; macro_innovation = (z_rate + "
          "z_inflation) / k_macro; k = Train standard deviation, so the correlated market factors are counted once "
          "as a channel, not three times.",
          "- 5-session versions are rolling sums rescaled to unit Train variance; ar_innovation_stress = "
          "max(market_innovation_5d, 0).", "",
          "Train normalization constants: " + ", ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}"
                                                         for k, v in nm.items()) + ".", ""]
    # 7
    L += ["## 7. BOCPD method", "",
          "- Model: y_t | segment ~ N(mu, sigma2), (mu, sigma2) ~ Normal-Inverse-Gamma; predictive is Student-t.",
          f"- Prior: mu0 = Train mean, kappa0 = {bo.PRIOR_KAPPA0}, alpha0 = {bo.PRIOR_ALPHA0}, beta0 set so the prior "
          "predictive variance equals the Train variance.",
          f"- Constant hazard 1/lambda with lambda in {list(bo.HAZARD_RUN_LENGTHS)} (primary {bo.PRIMARY_RUN_LENGTH}).",
          "- Exact recursion, no run-length truncation; verified against brute-force enumeration in tests.",
          "- Under a constant hazard P(r_t = 0 | y_1..t) = h identically, so the reported evidence is "
          f"P(r_t < {bo.PRIMARY_WINDOW} | y_1..t) (a change within the last {bo.PRIMARY_WINDOW} sessions), compared "
          f"with its prior value {bo.prior_recent_probability(bo.PRIMARY_RUN_LENGTH, bo.PRIMARY_WINDOW):.3f}.",
          "- In the EWS the BOCPD component is gated: it counts only when the 5-session market innovation is adverse "
          "(> 0), because a structural change toward calm is not a warning.", "",
          f"Synthetic scenario study ({int(ctx['scenarios']['runs'].iloc[0])} seeds each; change at session "
          f"{bo.SCENARIO_CHANGE}; prior from the first {bo.SCENARIO_PRIOR_ROWS} sessions; alarm = recent-change "
          f"probability > {bo.SCENARIO_ALARM}). It shows why BOCPD runs on innovations: a persistent AR(1) level "
          "with no change produces many spurious change points, its innovations do not; slow drifts are detected "
          "late or not at all.", "",
          _md_table(_fmt(ctx["scenarios"].set_index("scenario"), 3)), "",
          "Priors (fitted on Train):", "",
          _md_table(pd.DataFrame(meta["bocpd_priors"]).T.round(4)), ""]
    # 8
    show = bc[["input", "expected_run_length", "split", "prior_p_change_within_10d", "mean_posterior", "pr_auc",
               "warned", "missed", "false_alerts_per_year", "time_share_alerted"]]
    L += ["## 8. BOCPD input and hazard comparison", "",
          "B1 = factor LEVEL stress index (persistent; expected to re-detect slow trends), B2 = AR(1) market "
          "innovation (primary), B3 = AR-HMM predictive surprise (probit of the Train ECDF), plus single-channel B2 "
          "runs. Each uses its own Train 80th percentile as alert threshold.", "",
          _md_table(_fmt(show.set_index("input"), 3)), ""]
    # 9
    L += ["## 9. Economic confirmation layer", "",
          "- confirmation = Train ECDF of the factor-level stress index (-equity_trend + volatility + "
          "liquidity_credit, standardized on Train).",
          "- Channel tags for the alert explanation: equity weakness, volatility stress, credit stress (level "
          "percentile >= 0.80) and rates/inflation pressure (macro component >= 0.90).",
          "- Ablation `no_credit_confirmation` replaces the index with equity + volatility only.", ""]
    # 10
    L += ["## 10. Early Warning Score", "",
          "EWS_t = sum_i w_i * component_i,t; each component in [0, 1] via its Train ECDF (share of Train values "
          "strictly below the current value). Weights fixed in advance, never optimized:", "",
          _md_table(pd.DataFrame({"weight": weights, "meaning": ew.COMPONENT_LABELS}).rename_axis("component")), "",
          "The EWS is a ranking score, not a probability of loss.", ""]
    # 11
    share = {s: ctx["states"]["alert_level"][m].value_counts(normalize=True)
             for s, m in (("train", ctx["train_mask"]), ("validation", ctx["val_mask"]))}
    L += ["## 11. Alert state machine", "",
          "| level | entry (Train quantile) | exit (Train quantile) |", "|---|---|---|",
          f"| AMBER | {th.amber_in:.4f} (q80) | {th.amber_out:.4f} (q70) |",
          f"| ORANGE | {th.orange_in:.4f} (q90) | {th.orange_out:.4f} (q85) |",
          f"| RED | {th.red_in:.4f} (q97) | {th.red_out:.4f} (q93) |", "",
          f"- Escalation needs EWS >= entry on {asm.CONFIRM_MIN} of the last {asm.CONFIRM_WINDOW} sessions; levels "
          "may be skipped.",
          f"- Emergency RED: {asm.EMERGENCY}.",
          f"- De-escalation needs EWS < exit on {asm.CONFIRM_MIN} of {asm.CONFIRM_WINDOW}; ORANGE/RED fall to "
          f"RECOVERY, which clears after {asm.RECOVERY_CLEAR_DAYS} consecutive sessions below the AMBER exit.",
          "- The machine never reads the HMM argmax state: alert levels and regimes are different objects.", "",
          _md_table(pd.DataFrame(share).reindex(list(asm.LEVELS)).fillna(0).map(lambda v: f"{v:.1%}")
                    .rename_axis("level")), ""]
    # 12
    L += ["## 12. Outcome definitions", "",
          "- Daily label: SPY worst cumulative log return over the next 21 sessions <= log(0.95). Secondary: the "
          "strategic 9-ETF reference portfolio (configs/project.json weights) <= log(0.97).",
          "- Forward windows never cross a split boundary (last 21 sessions of each split unlabeled) or the "
          "2017/2018 block boundary in block-level results.",
          "- Event: SPY drawdown from its running peak reaches -5% (onset); re-arms when it recovers above -2.5%.", "",
          "Validation events:", "",
          _md_table(_fmt(ev_va[["peak_date", "onset_date", "end_date", "trough_loss_pct", "censored"]]
                         .assign(peak_date=lambda x: x["peak_date"].dt.date, onset_date=lambda x: x["onset_date"].dt.date,
                                 end_date=lambda x: x["end_date"].dt.date), 1)) if len(ev_va) else "No events.", ""]
    # 13
    L += ["## 13. Evaluation protocol", "",
          f"- Warned: alert on any of the {ev.LEAD_WINDOW} sessions before onset; late: first alert within "
          f"{ev.LATE_WINDOW} sessions after onset; otherwise missed.",
          f"- False alert episode: an alert run with no onset within {ev.LEAD_WINDOW} sessions of its start and no "
          "overlap with an event period.",
          f"- Equal alert budget: every continuous baseline alerts when its score exceeds its Train "
          f"{1 - ev.TRAIN_ALERT_BUDGET:.0%} quantile. A (hard HMM state) is binary; I is also evaluated through the "
          "state machine.",
          f"- Daily ranking: PR-AUC; uncertainty: moving-block bootstrap ({ev.BOOTSTRAP_BLOCK}-session blocks, "
          f"{int(ctx['boot']['valid_reps'].max()) if len(ctx['boot']) else 0} valid replications) of PR-AUC "
          "differences versus the best.",
          f"- Selection rule (fixed in advance): {sel['rule']}.", ""]
    # 14
    cols = ["system", "split", "pr_auc", "base_rate", "daily_lift", "portfolio_pr_auc", "time_share_alerted"]
    L += ["## 14. Baseline comparison (systems A-I)", "",
          _md_table(_fmt(c[c["split"].isin(["train", "validation"])][cols].set_index("system"), 3)), "",
          "Validation by block (stability):", "",
          _md_table(_fmt(c[c["split"].str.startswith("validation ")][["system", "split", "pr_auc", "base_rate",
                                                                       "n_events", "warned"]].set_index("system"), 3)),
          ""]
    # 15
    cols = ["system", "split", "n_events", "warned", "detected_late", "missed", "median_lead_days",
            "median_gap_to_onset", "median_delay_days"]
    L += ["## 15. Lead time and missed events", "",
          _md_table(_fmt(c[c["split"].isin(["train", "validation"])][cols].set_index("system"), 1)), "",
          "Event detail for the state machine (>= AMBER), Validation:", "",
          _md_table(_fmt(ctx["ev_full"].assign(onset_date=lambda x: pd.to_datetime(x["onset_date"]).dt.date,
                                                peak_date=lambda x: pd.to_datetime(x["peak_date"]).dt.date)
                         .set_index("onset_date"), 1)) if len(ctx["ev_full"]) else "No events.", ""]
    # 16
    cols = ["system", "split", "alert_episodes", "false_alert_episodes", "false_alerts_per_year", "episode_precision",
            "mean_alert_episode_days", "alert_transitions_per_year", "time_share_alerted"]
    L += ["## 16. False alarms and alert stability", "",
          _md_table(_fmt(c[c["split"].isin(["train", "validation"])][cols].set_index("system"), 2)), "",
          "Validation alert episodes of the state machine:", "",
          _md_table(pd.DataFrame(ctx["episodes"]).set_index("start")) if ctx["episodes"] else "None.", ""]
    # 17
    cols = ["specification", "split", "pr_auc", "warned", "missed", "false_alerts_per_year", "time_share_alerted",
            "alert_transitions_per_year"]
    av = a[a["split"] == "validation"].set_index("specification")
    full_ap = av.at["full", "pr_auc"] if "full" in av.index else np.nan
    deltas = (av["pr_auc"] - full_ap).drop("full", errors="ignore")
    L += ["## 17. Ablations", "",
          _md_table(_fmt(a[a["split"].isin(["train", "validation"])][cols].set_index("specification"), 3)), "",
          "Validation PR-AUC change versus the full specification (positive = removing/adding the item helps): "
          + ", ".join(f"{k} {v:+.3f}" for k, v in deltas.items()) + ".",
          "No ablation result was used to change the primary weights.", ""]
    # 18
    inc = ctx["incremental"].pivot(index="signal", columns="split", values="pr_auc")
    L += ["## 18. Entropy and velocity: do they add information?", "",
          _md_table(_fmt(inc, 3)), "", "Loss rate by entropy tercile within regime bucket (Validation):", "",
          _md_table(_fmt(ctx["entropy_cond"].set_index("regime_bucket"), 3)), ""]
    vel = ctx["velocity"]
    if len(vel):
        vs = vel.groupby("split").agg(switches=("switch_date", "size"), share_preceded=("velocity_exceeded_before",
                                                                                       "mean"),
                                      median_lead=("lead_days", "median"))
        L += ["Velocity before argmax switches S1 -> S2/S3 (threshold = Train 90th percentile of positive velocity, "
              "10-session look-back):", "", _md_table(_fmt(vs, 2)), ""]
    se = ctx["switch_entropy"]
    pre = se[se["offset"].between(-10, -6)].groupby("split")["mean_entropy"].mean()
    at = se[se["offset"].between(-2, 0)].groupby("split")["mean_entropy"].mean()
    L += ["Mean entropy 10-6 sessions before a switch vs 2-0 sessions before: " +
          ", ".join(f"{s} {pre.get(s, np.nan):.3f} -> {at.get(s, np.nan):.3f}" for s in ("train", "validation")) +
          ". Entropy rising only at the switch is a coincident, not a leading, signal.", ""]
    # 19
    L += ["## 19. AR-HMM surprise versus AR(1) innovation", "",
          "The frozen AR-HMM K=3 predictive surprise (-log p(x_t | past)) is direction-free; the AR(1) market "
          "innovation is directional. Compare the rows `arhmm_surprise`, `ar_surprise_chi2` (direction-free AR(1)) "
          "and `ar_innovation_stress` (directional) in section 18, and B2 vs B3 in section 8. The AR-HMM is used "
          "only as a supplementary signal; it was not refitted.", ""]
    # 20
    L += ["## 20. Specification selection", "",
          _md_table(_fmt(ctx["boot"].set_index("model"), 4)), "",
          f"- Best Validation PR-AUC: {_label(sel['best_validation_pr_auc'])}.",
          f"- Not significantly worse than the best: {', '.join(_label(s) for s in sel['not_significantly_worse'])}.",
          f"- {concl['retain']}", ""]
    # 21
    L += ["## 21. Causality, leakage controls and limitations", "",
          "- Causality test on the real data: the second half of the Validation factors was replaced by a distorted "
          "copy and every signal, score and alert before the cut was recomputed and compared (section 3).",
          "- Thresholds, ECDF scales, BOCPD priors and channel normalizations are estimated on Train rows only.",
          "- Filtered (not smoothed) HMM probabilities; Validation filtering continues from the last Train filtered "
          "distribution.",
          "- Limitations: few Validation events; Validation reused from Stages 5/5.5; Train signals are in-sample; "
          "the -5% event definition and the 63-session lead window are conventions; long alert episodes can "
          "warn mechanically, so lead time must be read with false alarms and time on alert; the outcome shares "
          "SPY momentum with the equity factor.", ""]
    # 22
    L += ["## 22. Conclusions and handoff", "", f"- {concl['full_vs_simple']}", f"- {concl['retain']}",
          "- Stage 6 output for later use: daily early_warning_{train,validation}.csv (score, level, reason, "
          "drivers). How alerts should affect allocation is a Stage 7 question and was not studied here.",
          "- Test (2020 onward) remains untouched.", "",
          "Figures: " + ", ".join(f"`figures/early_warning/{Path(p).name}`" for p in ctx.get("figures", [])), ""]
    return "\n".join(L)
