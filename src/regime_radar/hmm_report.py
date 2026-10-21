"""Stage 5: figures, selection manifest and the research report for the HMM stage (Train/Validation only)."""

from __future__ import annotations

STAGE5_CODE_VERSION = "stage5-2026-10-21c"  # must match scripts/07_fit_hmm.py

from pathlib import Path

import numpy as np
import pandas as pd

from .dataset import FEATURE_COLUMNS
from .hmm_calibration import CALIBRATION_RULES
from .hmm_robustness import PHASE_RULES
from .hmm_diagnostics import EVENT_RULES, PROTOCOL_AMENDMENTS, SELECTION_PROCEDURE, SELECTION_RULES, Candidate
from .hmm_models import (HMM_K, HMM_MAX_ITER, HMM_REG_COVAR, HMM_TOL, STANDARD_COVARIANCE_TYPES,
                         STICKY_COVARIANCE_TYPES, STICKY_PRIOR_WEIGHTS, expected_durations)
from .quality import _md_table
from .regime_report import (AXIS, GRID, INK, INK_2, MUTED, SHADE, STATE_COLORS, SURFACE, _fmt,
                            _heatmap, _save, _style, _title)

STD_COLOR = "#2a78d6"
FULL_COLOR = "#1baf7a"
STICKY_COLORS = {0.10: "#f2a37a", 0.30: "#eb6834", 1.00: "#a8401a"}
GMM_COLOR = "#52514e"


def _color(row) -> str:
    if row["family"] == "standard":
        return STD_COLOR if row["covariance_type"] == "diag" else FULL_COLOR
    return STICKY_COLORS.get(round(float(row["prior_weight"]), 2), "#eb6834")


def _short(model_id: str) -> str:
    return model_id.replace("hmm_", "").replace("_diag", " diag").replace("_full", " full").replace("_k", " K") \
        .replace("_w", " w").replace("std", "Std").replace("sticky", "Sticky")


def _split_line(ax, boundary) -> None:
    ax.axvline(boundary, color=INK_2, linewidth=0.9, linestyle="--", zorder=4)


def _shade_validation(ax, start, end) -> None:
    ax.axvspan(start, end, color=SHADE, zorder=0, linewidth=0)


def _best_sticky(comp: pd.DataFrame, k: int, cov: str = "diag") -> str | None:
    rows = comp[(comp["family"] == "sticky") & (comp["k"] == k) & (comp["covariance_type"] == cov)]
    return None if rows.empty else rows.loc[rows["validation_avg_loglik"].idxmax(), "model_id"]


def display_pair(sel: dict, comp: pd.DataFrame) -> tuple[str, str]:
    """(Standard, Sticky) pair shown in the detailed figures: the selected K and covariance."""
    k = sel.get("selected_k") or 2
    cov = sel.get("selected_covariance") or "diag"
    std = sel.get("standard_counterpart") or f"hmm_std_{cov}_k{k}"
    selected = sel.get("selected")
    sticky = selected if selected and selected.startswith("hmm_sticky") else _best_sticky(comp, k, "diag")
    return std, sticky


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
def fig_model_selection(comp: pd.DataFrame, sel: dict, path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    t = comp.sort_values(["k", "covariance_type", "family", "prior_weight"], ascending=[False, False, False, False])
    t = t.reset_index(drop=True)
    y = np.arange(len(t))
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.42 * len(t) + 2.4), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    _style(ax, grid_y=False)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    for i, r in t.iterrows():
        c = _color(r)
        short = r.get("mean_daily_shortfall_vs_best", np.nan)
        se = r.get("nw_t_vs_best", np.nan)
        err = abs(short / se) * 1.96 if pd.notna(se) and se not in (0, np.inf) and pd.notna(short) else 0
        ax.errorbar(-short if pd.notna(short) else np.nan, i, xerr=err, fmt="none", ecolor=AXIS, elinewidth=1.2, zorder=2)
        ax.scatter(-short if pd.notna(short) else np.nan, i, s=60, color=c if r["eligible"] else SURFACE,
                   edgecolors=c, linewidths=1.6, zorder=3)
        if r["model_id"] == sel.get("selected"):
            ax.scatter(-short, i, s=260, facecolors="none", edgecolors=INK, linewidths=1.4, zorder=4)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_yticks(y, [_short(m) for m in t["model_id"]], fontsize=8, color=INK_2)
    ax.set_xlabel("Validation log score minus best model (per day; 95% Newey-West interval)", fontsize=8, color=INK_2)
    ax.set_title("Validation predictive log score relative to the best model", fontsize=10, color=INK, loc="left")
    for kk in sorted(t["k"].unique())[:-1]:
        edge = t.index[t["k"] == kk].min() - 0.5
        ax.axhline(edge, color=GRID, linewidth=1)
    handles = [Line2D([], [], marker="o", linestyle="", color=STD_COLOR, label="Standard diag"),
               Line2D([], [], marker="o", linestyle="", color=FULL_COLOR, label="Standard full")]
    handles += [Line2D([], [], marker="o", linestyle="", color=c, label=f"Sticky diag w={w:.2f}")
                for w, c in STICKY_COLORS.items()]
    handles += [Line2D([], [], marker="o", linestyle="", markerfacecolor=SURFACE, markeredgecolor=MUTED,
                       label="ineligible (hollow)"),
                Line2D([], [], marker="o", linestyle="", markersize=12, markerfacecolor="none", markeredgecolor=INK,
                       label="selected")]
    ax.legend(handles=handles, fontsize=7.5, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.09), ncol=4)
    ax = axes[1]
    _style(ax, grid_y=False)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.barh(y, t["train_bic"] / 1e3, color=[_color(r) for _, r in t.iterrows()], height=0.6)
    ax.set_yticks(y, [""] * len(t))
    ax.set_xlabel("Train BIC (thousands; lower = better in-sample)", fontsize=8, color=INK_2)
    ax.set_title("Train BIC (reference only)", fontsize=10, color=INK, loc="left")
    lo = (t["train_bic"] / 1e3).min()
    ax.set_xlim(lo - 0.02 * abs(lo), None)
    _title(fig, "HMM model selection: Validation evidence first, Train fit for reference",
           "Parameters estimated on Train only; Validation 2016-2019 scored with fixed parameters by the causal forward "
           "filter. Sticky BIC uses the likelihood at the MAP estimate. Test period not used.")
    _save(fig, path)


def fig_transition_matrices(cands: dict[str, Candidate], comp: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    rows = [("Standard diag", [f"hmm_std_diag_k{k}" for k in HMM_K]),
            ("Sticky diag (best w per K)", [_best_sticky(comp, k) for k in HMM_K])]
    fig, axes = plt.subplots(2, len(HMM_K), figsize=(4.2 * len(HMM_K), 8), facecolor=SURFACE)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", ["#f0efec", "#9ec5f0", "#2a78d6", "#123f75"])
    for r, (label, ids) in enumerate(rows):
        for j, m in enumerate(ids):
            ax = axes[r, j]
            if m is None or m not in cands:
                ax.axis("off")
                continue
            a = cands[m].reference.params.transmat
            k = a.shape[0]
            ax.imshow(a, cmap=cmap, vmin=0, vmax=1)
            dur = expected_durations(a)
            for i in range(k):
                for jj in range(k):
                    ax.text(jj, i, f"{a[i, jj]:.3f}", ha="center", va="center", fontsize=8,
                            color="#ffffff" if a[i, jj] > 0.6 else INK)
            ax.set_xticks(range(k), [f"to S{i + 1}" for i in range(k)], fontsize=8, color=INK_2)
            ax.set_yticks(range(k), [f"S{i + 1} ({'inf' if not np.isfinite(d) else f'{d:.0f}'}d)" for i, d in
                                     enumerate(dur)], fontsize=8, color=INK_2)
            ax.tick_params(length=0)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_title(_short(m), fontsize=10, color=INK, loc="left")
    _title(fig, "Estimated transition matrices A_ij = P(Z_t = j | Z_t-1 = i)",
           "Row labels give the implied expected duration 1 / (1 - A_ii) in trading days. States ordered by stress "
           "score (S1 = lowest). Estimated on Train only.")
    _save(fig, path)


def fig_filtered_probabilities(cands, comp, sel, gmm, tr_index, va_index, path: Path) -> None:
    import matplotlib.pyplot as plt

    std, sticky = display_pair(sel, comp)
    panels = [(m, _short(m) + (" (selected)" if m == sel.get("selected") else "")) for m in (std, sticky) if m in cands]
    n = len(panels) + (1 if gmm is not None else 0)
    fig, axes = plt.subplots(n, 1, figsize=(13, 2.4 * n + 1), facecolor=SURFACE, sharex=True)
    axes = np.atleast_1d(axes)
    idx = tr_index.append(va_index)
    for ax, (m, title) in zip(axes, panels):
        _style(ax)
        c = cands[m]
        p = np.r_[c.stress_probability("train"), c.stress_probability("validation")]
        _shade_validation(ax, va_index[0], va_index[-1])
        ax.fill_between(idx, 0, p, color="#e66767", alpha=0.35, linewidth=0, step="post")
        ax.plot(idx, p, color="#a72f2f", linewidth=0.7, drawstyle="steps-post")
        ax.axhline(0.5, color=MUTED, linewidth=0.7, linestyle=":")
        _split_line(ax, va_index[0])
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("P(stress states)", fontsize=8, color=INK_2)
        ax.set_title(f"{title}: filtered P(Z_t in {{{', '.join(f'S{j + 1}' for j in c.stress_states)}}} | x_1..x_t)",
                     fontsize=10, color=INK, loc="left")
    if gmm is not None:
        ax = axes[-1]
        _style(ax)
        p = np.r_[gmm["train"]["p_stress"], gmm["validation"]["p_stress"]]
        _shade_validation(ax, va_index[0], va_index[-1])
        ax.fill_between(idx, 0, p, color=GMM_COLOR, alpha=0.3, linewidth=0, step="post")
        ax.plot(idx, p, color=GMM_COLOR, linewidth=0.6, drawstyle="steps-post")
        ax.axhline(0.5, color=MUTED, linewidth=0.7, linestyle=":")
        _split_line(ax, va_index[0])
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("P(S2 | x_t)", fontsize=8, color=INK_2)
        ax.set_title("Stage 4 GMM diag K=2: static posterior of S2 (no persistence)", fontsize=10, color=INK, loc="left")
    _title(fig, "Filtered stress-state probabilities (causal; not smoothed)",
           "Shaded = Validation (fixed Train parameters; each value uses data up to that date only). Train-period values use "
           "parameters estimated with data through 2015 and are NOT live historical predictions. Test not shown.")
    _save(fig, path)


def _state_band(ax, index, labels, k) -> None:
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(STATE_COLORS[:k])
    x = np.r_[index.values, index.values[-1] + np.timedelta64(1, "D")]
    ax.pcolormesh(x, [0, 1], np.asarray(labels)[None, :], cmap=cmap, vmin=-0.5, vmax=k - 0.5, shading="flat")
    ax.set_yticks([])


def fig_regime_timeline(cands, comp, tr_index, va_index, path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    ids = []
    for k in HMM_K:
        ids += [f"hmm_std_diag_k{k}", _best_sticky(comp, k)]
    ids = [m for m in ids if m in cands]
    idx = tr_index.append(va_index)
    fig, axes = plt.subplots(len(ids), 1, figsize=(13, 0.75 * len(ids) + 1.6), facecolor=SURFACE, sharex=True)
    for ax, m in zip(np.atleast_1d(axes), ids):
        c = cands[m]
        _state_band(ax, idx, np.r_[c.train_labels, c.validation_labels], c.spec.k)
        _split_line(ax, va_index[0])
        ax.set_ylabel(_short(m), rotation=0, ha="right", va="center", fontsize=8, color=INK_2)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8, length=0)
    handles = [Patch(color=STATE_COLORS[i], label=f"S{i + 1}") for i in range(max(HMM_K))]
    np.atleast_1d(axes)[0].legend(handles=handles, ncol=4, fontsize=8, frameon=False, loc="lower right",
                                  bbox_to_anchor=(1, 1.05))
    np.atleast_1d(axes)[0].text(va_index[0], 1.25, "  Validation", fontsize=8, color=INK_2,
                                transform=np.atleast_1d(axes)[0].get_xaxis_transform())
    _title(fig, "State sequences: argmax of the filtered probabilities",
           "S1..S4 are neutral identifiers ordered by stress score within each model; colours are comparable only "
           "within a row. Dashed line = start of Validation. Test period not shown.")
    _save(fig, path)


def fig_factor_profiles(profiles: pd.DataFrame, comp: pd.DataFrame, sel: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    fam_sticky = bool(sel.get("selected") and sel["selected"].startswith("hmm_sticky"))
    ids = [(_best_sticky(comp, k) if fam_sticky else f"hmm_std_diag_k{k}") for k in HMM_K]
    fig, axes = plt.subplots(1, len(ids), figsize=(4.6 * len(ids), 4.2), facecolor=SURFACE)
    vmax = 2.0
    for ax, m in zip(axes, ids):
        p = profiles[(profiles["model_id"] == m) & (profiles["basis"] == "filtered argmax")]
        tr = p[p["split"] == "train"].set_index("state")
        va = p[p["split"] == "validation"].set_index("state")
        frame = tr[FEATURE_COLUMNS].copy()
        frame.index = [f"{s}  tr {tr.loc[s, 'share']:.0%} / va {va.loc[s, 'share']:.0%}" for s in tr.index]
        _heatmap(ax, frame, vmax)
        ax.set_title(_short(m), fontsize=10, color=INK, loc="left")
    _title(fig, "Train factor profiles of the HMM states (mean robust z-score, filtered argmax)",
           "Row labels: share of Train / Validation days. Profiles are descriptive; economic names require the "
           "full evidence in the report.")
    _save(fig, path)


def fig_durations(comp: pd.DataFrame, cands, path: Path) -> None:
    import matplotlib.pyplot as plt

    t = comp[comp["covariance_type"] == "diag"].sort_values(["k", "family", "prior_weight"]).reset_index(drop=True)
    x = np.arange(len(t))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), facecolor=SURFACE)
    ax = axes[0]
    _style(ax)
    w = 0.38
    ax.bar(x - w / 2, t["train_switches_per_year"], width=w, color=[_color(r) for _, r in t.iterrows()], alpha=0.45,
           label="Train")
    ax.bar(x + w / 2, t["validation_switches_per_year"], width=w, color=[_color(r) for _, r in t.iterrows()],
           label="Validation")
    ax.set_xticks(x, [_short(m) for m in t["model_id"]], rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("state switches per year (filtered argmax)", fontsize=8, color=INK_2)
    ax.set_title("Switching frequency (light = Train, solid = Validation)", fontsize=10, color=INK, loc="left")
    ax = axes[1]
    _style(ax)
    for i, r in t.iterrows():
        dur = expected_durations(cands[r["model_id"]].reference.params.transmat)
        for s, d in enumerate(dur):
            ax.scatter(i, min(d, 1e4), color=STATE_COLORS[s], s=34, zorder=3)
    ax.set_yscale("log")
    from matplotlib.ticker import FuncFormatter, LogLocator
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.set_xticks(x, [_short(m) for m in t["model_id"]], rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("expected duration 1/(1-A_ii), trading days (log)", fontsize=8, color=INK_2)
    ax.set_title("Implied expected state durations (dot colour = state S1..S4)", fontsize=10, color=INK, loc="left")
    _title(fig, "Persistence: switching and implied durations (diagonal-covariance models)",
           "Expected durations come from the Train transition matrix; switching is measured on filtered argmax states.")
    _save(fig, path)


def fig_stability_delay(comp, det, study, sel, path: Path) -> None:
    import matplotlib.pyplot as plt

    d = det[(det["reference"] == "factor_stress_score") & (det["split"] == "pooled")]
    m = comp.merge(d[["model_id", "mean_lag_p50", "undetected_share_p50"]], on="model_id", how="left")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=SURFACE)
    ax = axes[0]
    _style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    markers = {2: "o", 3: "s", 4: "^"}
    for _, r in m.iterrows():
        ax.scatter(r["validation_switches_per_year"], r["mean_lag_p50"], s=70, color=_color(r),
                   marker=markers.get(r["k"], "o"), edgecolors=INK if r["model_id"] == sel.get("selected") else "none",
                   linewidths=1.5, zorder=3)
    for k, mk in markers.items():
        ax.scatter([], [], marker=mk, color=MUTED, label=f"K={k}")
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    vals = m["mean_lag_p50"].dropna()
    lo, hi = (float(vals.min()), float(vals.max())) if len(vals) else (0.0, 1.0)
    pad = max(hi - lo, 2.0) * 0.12
    ax.set_ylim(min(lo, 0.0) - pad, max(hi, 1.0) + pad)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xlabel("Validation switches per year (lower = more stable)", fontsize=8, color=INK_2)
    ax.set_ylabel("mean signed lag to P(stress) >= 0.5 (days; < 0 = model led the onset)", fontsize=8, color=INK_2)
    ax.set_title("Stability vs detection delay (factor-score onsets, Train+Validation)", fontsize=10, color=INK, loc="left")
    ax = axes[1]
    _style(ax)
    std, sticky = display_pair(sel, comp)
    for model, col in ((std, STD_COLOR), (sticky, "#eb6834")):
        for split, ls in (("train", "-"), ("validation", "--")):
            s = study[(study["model_id"] == model) & (study["split"] == split)]
            if s.empty or s["mean_p_stress"].isna().all():
                continue
            ax.plot(s["offset"], s["mean_p_stress"], color=col, linestyle=ls, linewidth=1.8,
                    label=f"{_short(model)} - {split} (n={int(s['n'].max())})")
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.axhline(0.5, color=MUTED, linewidth=0.7, linestyle=":")
    ax.set_ylim(0, 1)
    ax.set_xlabel("trading days relative to factor stress-score onset", fontsize=8, color=INK_2)
    ax.set_ylabel("mean filtered P(stress states)", fontsize=8, color=INK_2)
    ax.set_title("Event study around reference onsets", fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    _title(fig, "Does persistence delay detection?",
           f"Onset = factor stress score (vol + credit - equity trend) above its Train {EVENT_RULES['stress_score_quantile']:.0%} "
           "quantile for >= 10 days (rules fixed in advance). Reference events are descriptive, not ground-truth regimes.")
    _save(fig, path)


def fig_vs_gmm(cands, sel, gmm, scores, va_index, path: Path) -> None:
    import matplotlib.pyplot as plt

    m = sel.get("selected")
    if gmm is None or m is None or "gmm_diag_k2" not in scores:
        return
    std = sel.get("standard_counterpart")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), facecolor=SURFACE, gridspec_kw={"width_ratios": [1, 1.3]})
    ax = axes[0]
    _style(ax)
    for model in dict.fromkeys([m, std]):
        if model is None:
            continue
        ax.plot(va_index, (scores[model] - scores["gmm_diag_k2"]).cumsum(), linewidth=1.8, label=_short(model),
                color="#eb6834" if model.startswith("hmm_sticky") else STD_COLOR)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_ylabel("cumulative log-score difference vs GMM", fontsize=8, color=INK_2)
    ax.set_title("Validation: HMM minus GMM predictive log score (cumulative)", fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False)
    ax = axes[1]
    _style(ax)
    c = cands[m]
    ax.plot(va_index, gmm["validation"]["p_stress"], color=GMM_COLOR, linewidth=0.7, label="GMM diag K=2 posterior S2 (static)")
    ax.plot(va_index, c.stress_probability("validation"), color="#a72f2f", linewidth=1.2,
            label=f"{_short(m)} filtered P(stress)")
    ax.set_ylim(0, 1.02)
    ax.set_title("Validation stress probabilities", fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    _title(fig, "Selected HMM vs frozen Stage 4 GMM reference (Validation only)",
           "Both models scored as one-step-ahead predictive densities of x_t with Train parameters; the GMM ignores "
           "the past, the HMM conditions on x_1..x_t-1. Agreement of labels is not forecasting evidence.")
    _save(fig, path)


def fig_generalization(tv: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    t = tv.sort_values(["k", "covariance_type", "family", "prior_weight"]).reset_index(drop=True)
    y = np.arange(len(t))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.4 * len(t) + 2.2), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axes[0]
    _style(ax, grid_y=False)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    for yy, (_, r) in zip(y, t.iterrows()):
        ax.plot([r["validation_avg_loglik"], r["train_avg_loglik"]], [yy, yy], color=AXIS, linewidth=1.5, zorder=1)
        ax.scatter(r["train_avg_loglik"], yy, color=STD_COLOR, s=40, zorder=3)
        ax.scatter(r["validation_avg_loglik"], yy, color="#eb6834", s=40, zorder=3)
    ax.scatter([], [], color=STD_COLOR, label="Train (in-sample)")
    ax.scatter([], [], color="#eb6834", label="Validation (out-of-sample)")
    ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.08), ncol=2)
    ax.set_yticks(y, [_short(m) for m in t["model_id"]], fontsize=8, color=INK_2)
    ax.set_xlabel("average log score per day", fontsize=8, color=INK_2)
    ax.set_title("Train vs Validation average log score", fontsize=10, color=INK, loc="left")
    ax = axes[1]
    _style(ax, grid_y=False)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.barh(y, t["train_validation_gap"], color=[_color(r) for _, r in t.iterrows()], height=0.6)
    ax.set_yticks(y, [""] * len(t))
    ax.set_xlabel("Train minus Validation (per day)", fontsize=8, color=INK_2)
    ax.set_title("Generalization gap", fontsize=10, color=INK, loc="left")
    _title(fig, "Train vs Validation generalization (same Train parameters and scaler for both partitions)",
           "Validation continues the forward recursion from the last Train filtered distribution. A gap alone is not "
           "proof of overfitting: 2016-2019 was calmer than 2008-2015.")
    _save(fig, path)


def fig_calibration(cands, sel, extras, va_index, path: Path) -> None:
    import matplotlib.pyplot as plt

    cal = extras["calibration"]
    m = sel.get("selected") or extras["calibration_models"][0]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.6), facecolor=SURFACE)
    # PIT histogram (all factors pooled)
    ax = axes[0, 0]
    _style(ax)
    pit = cal["pit_values"][m]
    ax.hist(pit.ravel(), bins=20, range=(0, 1), density=True, color=STD_COLOR, alpha=0.75, edgecolor=SURFACE)
    ax.axhline(1.0, color=INK, linewidth=1, linestyle="--")
    summ = cal["pit"][(cal["pit"]["model_id"] == m) & (cal["pit"]["factor"] == "all factors")].iloc[0]
    ax.set_title(f"PIT of one-step predictive density - {_short(m)}", fontsize=10, color=INK, loc="left")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
    ax.text(0.02, 0.97, f"90% interval coverage {summ['coverage_90']:.0%} (nominal 90%)\n"
                        f"mean lag-1 autocorrelation {summ['lag1_autocorrelation']:.2f} (nominal 0)",
            transform=ax.transAxes, fontsize=8, color=INK_2, va="top")
    ax.set_xlabel("PIT value (uniform if calibrated)", fontsize=8, color=INK_2)
    # reliability
    ax = axes[0, 1]
    _style(ax)
    rel = cal["reliability"][cal["reliability"]["model_id"] == m].dropna(subset=["mean_predicted"])
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=0.8, linestyle="--")
    ax.scatter(rel["mean_predicted"], rel["mean_filtered"], s=np.clip(rel["n"], 10, 400), color="#a72f2f", alpha=0.75)
    for r in rel.itertuples():
        ax.annotate(f"n={r.n}", (r.mean_predicted, r.mean_filtered), xytext=(6, -10), textcoords="offset points",
                    fontsize=7, color=MUTED)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("predicted P(stress) at t (from x_1..x_t-1)", fontsize=8, color=INK_2)
    ax.set_ylabel("mean filtered P(stress) at t", fontsize=8, color=INK_2)
    ax.set_title("Reliability: E[filtered | predicted] should equal predicted", fontsize=10, color=INK, loc="left")
    # inflation curve
    ax = axes[1, 0]
    _style(ax)
    cur = cal["inflation_curve"][cal["inflation_curve"]["model_id"] == m]
    ax.plot(cur["inflation"], cur["val_a_mean_log_score"], marker="o", color=STD_COLOR, label="2016-2017 (choose c)")
    ax.plot(cur["inflation"], cur["val_b_mean_log_score"], marker="o", color="#eb6834", label="2018-2019 (test)")
    dec = cal["decisions"][m]
    ax.axvline(dec["chosen_inflation_on_val_a"], color=INK, linewidth=0.9, linestyle=":")
    ax.set_xscale("log")
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xticks(list(cur["inflation"]))
    ax.set_xlabel("emission covariance inflation c (filter only)", fontsize=8, color=INK_2)
    ax.set_ylabel("mean predictive log score per day", fontsize=8, color=INK_2)
    ax.set_title(f"Recalibration test: c* = {dec['chosen_inflation_on_val_a']:g}, 2018-19 gain "
                 f"{dec['val_b_mean_gain']:+.3f} (t {dec['val_b_nw_t']:.1f}), adopted {dec['adopted']}",
                 fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False)
    # raw vs recalibrated probabilities
    ax = axes[1, 1]
    _style(ax)
    ax.plot(va_index, dec["_p_stress_raw"], color="#a72f2f", linewidth=1.0, label="c = 1 (Train estimate)")
    if dec["chosen_inflation_on_val_a"] != 1.0:
        ax.plot(va_index, dec["_p_stress_chosen"], color=STD_COLOR, linewidth=1.0,
                label=f"c = {dec['chosen_inflation_on_val_a']:g}")
    ax.axvline(pd.Timestamp("2018-01-01"), color=MUTED, linewidth=0.8, linestyle="--")
    ax.set_ylim(0, 1.02)
    ax.set_title("Validation filtered P(stress): raw vs recalibrated", fontsize=10, color=INK, loc="left")
    ax.set_ylim(0, 1.25)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(fontsize=8, frameon=False, loc="upper left", ncol=2)
    _title(fig, "Are the filtered probabilities calibrated? (Validation only; Train parameters fixed)",
           "PIT and reliability use observed data only. The recalibration multiplies emission covariances by c inside "
           "the filter; c is chosen on 2016-17 and judged on 2018-19. Nothing is smoothed. Test period not used.")
    _save(fig, path)


def fig_robustness(comp, sel, extras, stab, path: Path) -> None:
    import matplotlib.pyplot as plt

    win, ph = extras["windows"], extras["phases"]
    palette = ["#2a78d6", "#1baf7a", "#eda100", "#8a5cd1", "#e87ba4", "#0a7d8c"]
    models = list(dict.fromkeys(win["model_id"]))
    colors = {m: palette[i % len(palette)] for i, m in enumerate(models)}
    colors["gmm_diag_k2"] = GMM_COLOR
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.9), facecolor=SURFACE)
    ax = axes[0]
    _style(ax)
    order = list(extras["window_definitions"].keys())
    for m in models:
        g = win[win["model_id"] == m].set_index("window").reindex(order)
        ax.plot(range(len(order)), g["validation_avg_loglik"], marker="o", color=colors[m], linewidth=1.8,
                linestyle="--" if m.startswith("hmm_sticky") else "-", label=_short(m))
    ax.set_xticks(range(len(order)), [o.replace("_", " ") for o in order], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Validation log score per day", fontsize=8, color=INK_2)
    ax.set_title("Training-window sensitivity", fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=7.5, frameon=False)
    ax = axes[1]
    _style(ax)
    v = ph[(ph["split"] == "validation") & ph["phase"].str.fullmatch(r"\d{4}")]
    piv = v.pivot_table(index="phase", columns="model_id", values="mean_log_score")
    base = "hmm_std_diag_k2" if "hmm_std_diag_k2" in piv else piv.columns[0]
    others = [c for c in piv.columns if c != base]
    w = 0.8 / max(len(others), 1)
    for i, m in enumerate(others):
        d = piv[m] - piv[base]
        ax.bar(np.arange(len(piv)) + (i - (len(others) - 1) / 2) * w, d, width=w, color=colors.get(m, MUTED),
               label=_short(m))
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xticks(range(len(piv)), piv.index, fontsize=8)
    ax.set_ylabel(f"mean daily log score minus {_short(base)}", fontsize=8, color=INK_2)
    ax.set_title("Market phases: Validation advantage by year", fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=7.5, frameon=False)
    ax = axes[2]
    _style(ax)
    t = stab.merge(comp[["model_id", "k", "family", "covariance_type", "prior_weight"]], on="model_id")
    t = t[t["covariance_type"] == "diag"].sort_values(["k", "family", "prior_weight"]).reset_index(drop=True)
    x = np.arange(len(t))
    ax.bar(x - 0.2, t["single_start_best_basin_share"], width=0.4, color=AXIS, label="single EM start")
    ax.bar(x + 0.2, t["train_mean_pairwise_ari"], width=0.4, color=STD_COLOR, label="seed ARI (best of n_init)")
    ax.axhline(SELECTION_RULES["min_mean_pairwise_ari_train"], color=INK, linewidth=0.8, linestyle=":")
    ax.set_xticks(x, [_short(m) for m in t["model_id"]], rotation=45, ha="right", fontsize=7.5)
    ax.set_ylim(0, 1.3)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title("Initialization: local optima vs estimator stability", fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left", ncol=2)
    _title(fig, "Robustness of the HMM comparison (Train fits, Validation scores)",
           "Windows and the yearly split were fixed before these results were seen. Dotted line = ARI threshold 0.80. "
           "Test period not used.")
    _save(fig, path)


def save_figures(directory: Path, cands, comp, tv, det, study, sel, gmm_cmp, scores, profiles, durations,
                 tr_index, va_index, gmm, extras=None) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    directory.mkdir(parents=True, exist_ok=True)
    jobs = [
        ("hmm_model_selection.png", lambda p: fig_model_selection(comp, sel, p)),
        ("hmm_transition_matrices.png", lambda p: fig_transition_matrices(cands, comp, p)),
        ("hmm_filtered_probabilities.png", lambda p: fig_filtered_probabilities(cands, comp, sel, gmm, tr_index, va_index, p)),
        ("hmm_regime_timeline.png", lambda p: fig_regime_timeline(cands, comp, tr_index, va_index, p)),
        ("hmm_regime_factor_profiles.png", lambda p: fig_factor_profiles(profiles, comp, sel, p)),
        ("hmm_state_duration_comparison.png", lambda p: fig_durations(comp, cands, p)),
        ("hmm_stability_vs_detection_delay.png", lambda p: fig_stability_delay(comp, det, study, sel, p)),
        ("hmm_vs_gmm_comparison.png", lambda p: fig_vs_gmm(cands, sel, gmm, scores, va_index, p)),
        ("hmm_generalization_gap.png", lambda p: fig_generalization(tv, p)),
    ]
    if extras:
        stab = extras.get("stability")
        jobs += [("hmm_probability_calibration.png", lambda p: fig_calibration(cands, sel, extras, va_index, p)),
                 ("hmm_robustness.png", lambda p: fig_robustness(comp, sel, extras, stab, p))]
    out = []
    for name, job in jobs:
        job(directory / name)
        if (directory / name).exists():
            out.append(name)
    return out


# --------------------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------------------
def selection_manifest(data, scaler_dict, reason, seeds, sel, comp, checks, events, late_note, gmm_status, outputs,
                       version, extras=None) -> dict:
    selected = sel.get("selected")
    row = comp[comp["model_id"] == selected].iloc[0].to_dict() if selected else {}
    return {
        "stage": "5 - Standard and Sticky Gaussian HMM", "model_version": version,
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "stage3_protocol_version": data.manifest.get("protocol_version"),
        "files_read": data.files_read, "test_data_loaded": any(f.startswith("test_") for f in data.files_read),
        "input_sha256": {n: data.manifest.get("split_files", {}).get(n) for n in data.files_read if n.endswith(".csv")},
        "train_period": {"first": str(data.train_features.index.min().date()),
                         "last": str(data.train_features.index.max().date()), "rows": len(data.train_features)},
        "validation_period": {"first": str(data.validation_features.index.min().date()),
                              "last": str(data.validation_features.index.max().date()),
                              "rows": len(data.validation_features)},
        "factor_columns": FEATURE_COLUMNS, "preprocessing": scaler_dict, "scaling_decision": reason,
        "candidates": {"k": list(HMM_K), "standard_covariance": list(STANDARD_COVARIANCE_TYPES),
                       "sticky_covariance": list(STICKY_COVARIANCE_TYPES),
                       "sticky_prior_weights": list(STICKY_PRIOR_WEIGHTS),
                       "kappa_definition": "kappa = w * (T_train - 1) / K; Dirichlet alpha_ij = 1 + kappa * 1[i=j]"},
        "estimation": {"algorithm": "EM (Baum-Welch); MAP-EM for Sticky", "seeds": list(seeds),
                       "initialization": "per seed: n_init EM starts, each from one K-Means run (n_init=1); the "
                                         "seed's estimate is the best Train objective; identical starts for Standard "
                                         "and Sticky",
                       "reference_fit": "highest Train objective among converged seeds",
                       "reg_covar": HMM_REG_COVAR, "max_iter": HMM_MAX_ITER, "tol_per_obs": HMM_TOL},
        "filtering": {"decision_outputs": "forward-filtered and one-step predicted probabilities only",
                      "validation_initialization": "last Train filtered distribution propagated by A",
                      "train_period_caveat": "Train filtered probabilities use parameters estimated through 2015 "
                                             "(retrospective in-sample analysis, not live predictions)"},
        "selection_rules": SELECTION_RULES, "selection_procedure": SELECTION_PROCEDURE, "event_rules": EVENT_RULES,
        "protocol_amendments": PROTOCOL_AMENDMENTS,
        "event_stress_threshold": events.get("_threshold"),
        "selection": {"selected": selected, "reason": sel.get("reason"),
                      "best_validation_model": sel.get("best_validation_model"),
                      "competitive_set": sel.get("competitive_set"),
                      "standard_counterpart": sel.get("standard_counterpart"),
                      "selected_row": row},
        "gmm_reference": {"model": "gmm_diag_k2 (Stage 4, frozen)", "status": gmm_status},
        "late_train_sensitivity": late_note,
        "phase_consistency": sel.get("phase_consistency"), "phase_fallback": sel.get("phase_fallback"),
        "calibration": None if not extras else {
            "rules": CALIBRATION_RULES,
            "decisions": {m: {k: v for k, v in d.items() if not k.startswith("_")}
                          for m, d in extras["calibration"]["decisions"].items()}},
        "training_windows": None if not extras else extras["window_definitions"],
        "gmm_rebuild": None if not extras else extras["gmm_info"],
        "n_init_per_seed": None if not extras else extras["n_init"],
        "software": None if not extras else extras["software"],
        "checks": [vars(c) for c in checks], "outputs": outputs,
    }


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------
def _num(v, fmt="{:.3f}"):
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else fmt.format(v)


def generalization_statements(comp: pd.DataFrame) -> list[str]:
    out = []
    for (fam, cov), g in comp[comp["family"] == "standard"].groupby(["family", "covariance_type"]):
        g = g.sort_values("k")
        tr_up = bool((g["train_avg_loglik"].diff().dropna() > 0).all())
        va_best = int(g.loc[g["validation_avg_loglik"].idxmax(), "k"])
        out.append(f"Standard {cov}: Train log score {'rises' if tr_up else 'does not rise monotonically'} with K "
                   f"({', '.join(f'K{k} {v:.3f}' for k, v in zip(g['k'], g['train_avg_loglik']))}); Validation is "
                   f"highest at K={va_best} ({', '.join(f'K{k} {v:.3f}' for k, v in zip(g['k'], g['validation_avg_loglik']))}).")
    for k in sorted(comp["k"].unique()):
        d = comp[(comp["k"] == k) & (comp["family"] == "standard")].set_index("covariance_type")
        if {"diag", "full"} <= set(d.index):
            out.append(f"K={k} full vs diag: Train {d.loc['full', 'train_avg_loglik'] - d.loc['diag', 'train_avg_loglik']:+.3f}"
                       f"/day, Validation {d.loc['full', 'validation_avg_loglik'] - d.loc['diag', 'validation_avg_loglik']:+.3f}"
                       "/day" + (" -> full covariance fits Train better but generalizes worse (overfitting signal)."
                                 if d.loc['full', 'train_avg_loglik'] > d.loc['diag', 'train_avg_loglik']
                                 and d.loc['full', 'validation_avg_loglik'] < d.loc['diag', 'validation_avg_loglik'] else "."))
    big = comp.loc[comp["train_validation_gap"].idxmax()]
    small = comp.loc[comp["train_validation_gap"].idxmin()]
    out.append(f"Largest Train-Validation gap: {big['model_id']} ({big['train_validation_gap']:.3f} per day); smallest: "
               f"{small['model_id']} ({small['train_validation_gap']:.3f}).")
    for k in sorted(comp["k"].unique()):
        std = comp[(comp["k"] == k) & (comp["family"] == "standard") & (comp["covariance_type"] == "diag")]
        st = comp[(comp["k"] == k) & (comp["family"] == "sticky")]
        if std.empty or st.empty:
            continue
        s0 = std.iloc[0]
        parts = [f"w={r.prior_weight:.2f}: val {r.validation_avg_loglik - s0['validation_avg_loglik']:+.3f}/day, "
                 f"switches {r.validation_switches_per_year:.1f} vs {s0['validation_switches_per_year']:.1f}/yr"
                 for r in st.sort_values("prior_weight").itertuples()]
        out.append(f"K={k} Sticky vs Standard diag (Validation): " + "; ".join(parts) + ".")
    return out


def key_findings(comp: pd.DataFrame, sel: dict, profiles: pd.DataFrame, assets: pd.DataFrame) -> list[str]:
    """Data-driven findings; every number is computed from this run."""
    out = []
    best_all = sel.get("best_overall_validation_model")
    best_el = sel.get("best_validation_model")
    if best_all and best_all != best_el:
        r = comp[comp["model_id"] == best_all].iloc[0]
        out.append(f"The highest Validation log score overall is {best_all} ({r['validation_avg_loglik']:.3f}); it is "
                   f"not eligible ({r['ineligible_reasons']}), so the benchmark is {best_el}.")
    # Sticky effect at each K (diag)
    for k in sorted(comp["k"].unique()):
        std = comp[(comp["k"] == k) & (comp["family"] == "standard") & (comp["covariance_type"] == "diag")]
        st = comp[(comp["k"] == k) & (comp["family"] == "sticky")]
        if std.empty or st.empty:
            continue
        s0 = std.iloc[0]
        dur0 = s0["expected_durations_days"]
        top = st.sort_values("prior_weight").iloc[-1]
        out.append(f"K={k}: the strongest prior (w={top['prior_weight']:.2f}) changes implied durations from {dur0} to "
                   f"{top['expected_durations_days']} days, but Validation switching from "
                   f"{s0['validation_switches_per_year']:.1f} to {top['validation_switches_per_year']:.1f} per year and "
                   f"the Validation log score by {top['validation_avg_loglik'] - s0['validation_avg_loglik']:+.3f} per day.")
    if "validation_median_emission_log_lr" in comp:
        r = comp[comp["model_id"] == (sel.get("selected") or best_el)].iloc[0]
        out.append(f"Why the prior matters little here: on a typical Validation day the emission log-likelihood ratio "
                   f"between the two most likely states is {r['validation_median_emission_log_lr']:.1f} nats (median), "
                   f"while the transition log-odds log(A_ii / (1 - A_ii)) average {r['mean_transition_log_odds']:.1f}. "
                   "Daily evidence from five persistent factors dominates the transition prior, so the filtered state "
                   "is mostly decided by the emissions.")
        out.append(f"Filtered probabilities are very confident (mean max probability {r['validation_mean_max_filtered_prob']:.3f}, "
                   f"uncertain days {r['validation_share_uncertain_days']:.1%} on Validation). They should not be read "
                   "as calibrated crisis probabilities without a calibration check in the Early Warning stage.")
    sel_id = sel.get("selected")
    if sel_id:
        p = profiles[(profiles["model_id"] == sel_id) & (profiles["basis"] == "filtered argmax") & (profiles["split"] == "train")]
        shifted = p[p["train_validation_profile_distance"] > 1.0]
        if not shifted.empty:
            out.append("Profile shift > 1.0 between Train and Validation (selected model): "
                       + ", ".join(f"{r.state} {r.train_validation_profile_distance:.2f}" for r in shifted.itertuples())
                       + ". The state's composition changes out of sample; read its label with care.")
        a = assets[(assets["model_id"] == sel_id) & (assets["horizon"] == "next_session") & (assets["etf"] == "SPY")]
        for split in ("train", "validation"):
            g = a[a["split"] == split].set_index("state")["annualized_mean_pct"]
            if len(g):
                out.append(f"SPY next-session annualized mean by state ({split}): "
                           + ", ".join(f"{st} " + ("n/a (no days)" if pd.isna(v) else f"{v:+.1f}%") for st, v in g.items())
                           + " (descriptive).")
    return out


def calibration_section(extras: dict, selected: str | None) -> list[str]:
    cal = extras["calibration"]
    L = ["## Probability calibration", "",
         "Question: are filtered probabilities near 0/1 a property of well separated regimes, or a symptom of "
         "misspecification? All tests use Validation with Train parameters fixed; nothing is smoothed.", "",
         "- PIT: u = predictive CDF of each factor under the one-step-ahead mixture. Calibrated -> uniform, 90% "
         "interval coverage 0.90, no autocorrelation.",
         "- Reliability (martingale property): under a correct model E[filtered P(stress)_t | x_1..x_t-1] = predicted "
         "P(stress)_t; the realized switching rate should match the model-implied rate sum_i P(Z_t=i)(1-A_ii).",
         "- Residual autocorrelation: lag-1 autocorrelation of standardized residuals on consecutive same-state days. "
         "The HMM assumes zero (x_t independent given Z_t).",
         f"- Recalibration test: emission covariances x c inside the filter, c in {list(CALIBRATION_RULES['inflation_grid'])}, "
         f"chosen on {CALIBRATION_RULES['selection_period'][0][:4]}-{CALIBRATION_RULES['selection_period'][1][:4]}, judged on "
         f"{CALIBRATION_RULES['evaluation_period'][0][:4]}-{CALIBRATION_RULES['evaluation_period'][1][:4]}; adopted only if "
         f"the out-of-sample gain has NW t >= {CALIBRATION_RULES['adopt_min_nw_t']}.", ""]
    pit = cal["pit"][cal["pit"]["factor"] == "all factors"].drop(columns=["factor", "uniform_std"])
    L += ["PIT summary (all five factors pooled):", "", _md_table(_fmt(pit.set_index("model_id"), 3)), ""]
    L += ["Reliability and switching:", "", _md_table(_fmt(cal["reliability_summary"].set_index("model_id"), 3)), ""]
    ac = cal["residual_autocorrelation"]
    L += ["Within-state residual lag-1 autocorrelation:", "", _md_table(_fmt(ac.set_index(["model_id"]), 2)), ""]
    dec = cal["inflation_decision"].drop(columns=["rule"])
    L += ["Covariance-inflation test:", "", _md_table(_fmt(dec.set_index("model_id"), 3)), ""]
    if selected and selected in cal["decisions"]:
        p = cal["pit"][(cal["pit"]["model_id"] == selected) & (cal["pit"]["factor"] == "all factors")].iloc[0]
        r = cal["reliability_summary"].set_index("model_id").loc[selected]
        a = ac[(ac["model_id"] == selected) & (ac["split"] == "validation")]["mean"].iloc[0]
        d = cal["decisions"][selected]
        L += ["**Reading for the selected model (computed):**", ""]
        L.append(f"- Predictive intervals: 90% coverage {p['coverage_90']:.0%} (nominal 90%); PIT std {p['std']:.3f} "
                 f"(uniform 0.289); " + ("intervals too narrow -> overconfident predictive density."
                                         if p["coverage_90"] < 0.85 else
                                         "intervals too wide." if p["coverage_90"] > 0.95 else "coverage acceptable."))
        L.append(f"- Temporal dependence: mean within-state residual autocorrelation {a:.2f} on Validation and mean PIT "
                 f"autocorrelation {p['lag1_autocorrelation']:.2f}. " +
                 ("The conditional-independence assumption is strongly violated: consecutive days are treated as "
                  "independent evidence, which pushes filtered probabilities toward 0/1."
                  if a > 0.5 else "Moderate dependence." if a > 0.2 else "Little dependence."))
        L.append(f"- Reliability: mean filtered minus predicted P(stress) {r['mean_filtered_minus_predicted']:+.3f} "
                 f"(NW t {r['nw_t']:.2f}); realized / model-implied switches {r['realized_to_implied_switch_ratio']:.2f}.")
        L.append(f"- Recalibration: c* = {d['chosen_inflation_on_val_a']:g} on 2016-17; 2018-19 gain "
                 f"{d['val_b_mean_gain']:+.3f} per day (NW t {d['val_b_nw_t']:.2f}); adopted: {d['adopted']}. "
                 + ("The decision filter for later stages uses this inflation." if d["adopted"] else
                    "The Train estimate (c = 1) stays the decision filter; the confidence of the probabilities must be "
                    "read with the diagnostics above."))
        L.append("- Structural remedy (not implemented here, it changes the model class): an autoregressive-emission "
                 "HMM (Markov-switching VAR) models x_t given x_t-1 and the state, which removes most of the temporal "
                 "dependence instead of rescaling it.")
    return L + [""]


def robustness_section(extras: dict, sel: dict, stab: pd.DataFrame) -> list[str]:
    win, ph = extras["windows"], extras["phases"]
    L = ["## Robustness: training windows, initialization, market phases", "",
         "Training windows (fixed in advance; parameters estimated on the window only, the filter then runs to the "
         "end of Train and into Validation): " + ", ".join(f"{k} {v[0]}..{v[1]}" for k, v in
                                                          extras["window_definitions"].items()) + ".", ""]
    cols = ["window", "model_id", "fit_rows", "converged", "validation_avg_loglik", "rank_in_window",
            "validation_ari_vs_full", "max_emission_mean_shift_original_units", "validation_switches_per_year",
            "validation_shares"]
    L += [_md_table(_fmt(win[cols].set_index("window"), 3)), ""]
    best_per = win.loc[win.groupby("window")["validation_avg_loglik"].idxmax(), ["window", "model_id"]]
    L.append("- Best Validation model per window: " + ", ".join(f"{r.window}: {r.model_id}" for r in best_per.itertuples())
             + (" (same in every window)." if best_per["model_id"].nunique() == 1 else " (changes with the window)."))
    L += ["", "Initialization (EM local optima vs estimator stability):", ""]
    sc = ["model_id", "n_init_per_seed", "single_start_best_basin_share", "best_basin_share", "train_mean_pairwise_ari",
          "validation_mean_pairwise_ari", "best_basin_max_mean_distance"]
    L += [_md_table(_fmt(stab[sc].set_index("model_id"), 3)), ""]
    v = ph[ph["split"] == "validation"].pivot_table(index="phase", columns="model_id", values="mean_log_score")
    t = ph[ph["split"] == "train"].pivot_table(index="phase", columns="model_id", values="mean_log_score")
    L += ["Mean daily log score by market phase - Validation (out-of-sample):", "", _md_table(_fmt(v, 3)), "",
          "Train (in-sample, for context only):", "", _md_table(_fmt(t, 3)), ""]
    pc = sel.get("phase_consistency")
    if pc:
        L.append(f"- Phase consistency (rule: larger K must win in >= {PHASE_RULES['min_positive_years']} of "
                 f"{PHASE_RULES['n_years']} Validation years): {pc['larger']} vs {pc['smaller']}: "
                 + ", ".join(f"{y} {a:+.3f}" for y, a in pc["yearly_advantage"].items())
                 + f" -> positive in {pc['positive_years']} of {pc['n_years']}; "
                 + ("consistent." if pc["consistent"] else "NOT consistent, smaller K kept."))
    else:
        L.append("- Phase consistency: not triggered (the selected K is the smallest eligible K).")
    g = extras["gmm_info"]
    L += [f"- Stage 4 GMM reference: rebuilt on Train with scikit-learn {g['installed_sklearn']} (the pickle was written "
          f"with {g['pickle_sklearn']}); " + ("compared with the pickle: max |log-density difference| "
                                               f"{g['max_abs_score_diff']:.1e}." if g["max_abs_score_diff"] is not None
                                               else "pickle not unpickled across versions."), ""]
    return L


def render_report(data, scaler_dict, reason, seeds, cands, comp, tv, stab, trans, durations, profiles, assets, det,
                  onsets, events, sel, gmm_cmp, late, late_note, checks, figures, paired=None, extras=None) -> str:
    selected = sel.get("selected")
    status = pd.Series([c.status for c in checks]).value_counts().to_dict()
    tr, va = data.train_features, data.validation_features
    L = ["# Stage 5 - Standard and Sticky Gaussian HMM regime detection", "",
         f"Train {tr.index.min().date()}..{tr.index.max().date()} ({len(tr)} rows); Validation "
         f"{va.index.min().date()}..{va.index.max().date()} ({len(va)} rows). Stage 3 protocol "
         f"{data.manifest.get('protocol_version')}. Test split not read. Checks: "
         + ", ".join(f"{v} {k}" for k, v in status.items()) + ".", ""]
    if selected:
        r = comp[comp["model_id"] == selected].iloc[0]
        L += ["## Summary", "",
              f"- Selected HMM: **{selected}** ({r['family']}, K={r['k']}, {r['covariance_type']} covariance"
              + (f", prior weight w={r['prior_weight']:.2f}, kappa={r['kappa']:.1f}" if r["family"] == "sticky" else "")
              + ").",
              f"- Rule outcome: {sel.get('reason')}",
              f"- Best Validation log score among eligible models: {sel.get('best_validation_model')}; best overall "
              f"(any eligibility): {sel.get('best_overall_validation_model')}; competitive set (NW t < "
              f"{SELECTION_RULES['nw_t_threshold']}): {', '.join(sel.get('competitive_set', []))}.",
              f"- Selected model: Validation log score {r['validation_avg_loglik']:.3f} per day, Train {r['train_avg_loglik']:.3f}; "
              f"Validation switches {r['validation_switches_per_year']:.1f}/year; self-transitions {r['self_transitions']}.",
              ""]
        if not gmm_cmp.empty:
            g = gmm_cmp[(gmm_cmp["model_id"] == selected) & (gmm_cmp["split"] == "validation")].iloc[0]
            L += [f"- Versus the frozen Stage 4 GMM (diag, K=2) on Validation: mean daily log-score difference "
                  f"{g['mean_daily_difference']:+.3f} (Newey-West t {g['nw_t']:.2f}); HMM better on "
                  f"{g['share_days_hmm_better']:.0%} of days; state ARI {g['ari_hmm_vs_gmm_states']:.2f}.", ""]
        L += ["**Key findings (computed from this run)**", ""] + [f"- {x}" for x in key_findings(comp, sel, profiles, assets)] + [""]
    L += ["## 1. Research question", "",
          "Does explicitly modelling regime persistence (Standard vs Sticky Gaussian HMM) give more stable and more "
          "informative causal state probabilities than static clustering, without unacceptable detection delay? "
          "Sticky is not assumed to be better; the comparison uses Validation evidence fixed in advance.", "",
          "## 2. Data and temporal protocol", "",
          f"- Factors (frozen order): {', '.join(FEATURE_COLUMNS)}. Factor definitions unchanged (factors.py hash checked).",
          f"- Preprocessing: {scaler_dict['method']}, fitted on Train only, identical to Stage 4. {reason}",
          "- Train: all HMM parameters, the scaler and the reference seed. Validation: fixed parameters, causal "
          "filtering, model selection. Test (2020 onward): not loaded.",
          "- No shuffling; one continuous Train sequence; no refit on Train + Validation in this stage.", "",
          "## 3. Standard Gaussian HMM", "",
          "Z_t in {1..K} follows a first-order Markov chain with P(Z_1 = k) = pi_k and A_ij = P(Z_t = j | Z_t-1 = i). "
          "Emissions X_t | Z_t = k ~ N(mu_k, Sigma_k), with Sigma_k diagonal (primary baseline) or full (controlled "
          "challenger). The likelihood is log p(x_1:T) = sum_t log sum_k P(Z_t = k | x_1:t-1) N(x_t; mu_k, Sigma_k). "
          f"Covariances get reg_covar = {HMM_REG_COVAR} on the diagonal (as in Stage 4). Assumptions: time-homogeneous "
          "transitions, conditionally independent Gaussian emissions given the state, geometric state durations.", "",
          "## 4. Sticky HMM", "",
          "Each row of A has a Dirichlet prior with alpha_ij = 1 + kappa 1[i = j] (finite-K analogue of the sticky "
          "HDP-HMM, Fox et al. 2011). MAP-EM maximizes log p(x | theta) + kappa sum_i log A_ii; the M-step is "
          "A_ij = (n_ij + kappa 1[i=j]) / (sum_j n_ij + kappa), with n_ij the expected transition counts. The prior acts "
          "inside every EM iteration and therefore changes the states themselves, not just their labels. "
          f"kappa = w (T_train - 1) / K with the predetermined grid w in {list(STICKY_PRIOR_WEIGHTS)} (prior pseudo-"
          "transitions as a share of the average observed transitions per row). Train AIC/BIC for Sticky models use the "
          "likelihood at the MAP estimate with the nominal parameter count; the prior reduces effective complexity, "
          "so these numbers are indicative only.", "",
          "## 5. Estimation and initialization", "",
          f"- {len(seeds)} seeds ({seeds[0]}..{seeds[-1]}). For each seed the estimator runs several EM starts "
          "(n_init_per_seed in the stability table; protocol 5, as for the Stage 4 GMM) and keeps the start with the "
          "highest Train objective. Each start begins from one K-Means run (n_init=1): means, within-cluster "
          "covariances and a smoothed K-Means transition matrix. Standard and Sticky fits with the same seed use "
          "identical starting points.",
          f"- EM until the objective changes by < {HMM_TOL} per observation (max {HMM_MAX_ITER} iterations); monotonicity "
          "of the objective is checked every iteration. The reference fit is the converged seed with the highest Train "
          "objective; the others are aligned to it by Hungarian matching of emission means.",
          "- States are ordered by the stress score (volatility + liquidity_credit - equity_trend) of their emission "
          "means in original units; S1..SK are neutral identifiers. 'Stress states' = states whose emission stress score "
          "exceeds the Train average (fixed rule); P(stress) is the sum of their filtered probabilities.", "",
          "## 6. Filtering versus smoothing", "",
          "- Decision quantities: predicted P(Z_t | x_1:t-1) and filtered P(Z_t | x_1:t), computed in log space with "
          "log-sum-exp, saved for Validation (predicted and filtered) and Train (filtered).",
          "- Validation starts from the last Train filtered distribution propagated by A; this equals filtering the "
          "concatenated sequence and uses no later Validation data (checked below).",
          "- Smoothed probabilities and Viterbi paths use the whole sequence. They appear only as a diagnostic "
          "('hindsight revision share' = Validation days whose smoothed argmax differs from the filtered argmax).",
          "- **Train-period filtered states use parameters estimated with data through 2015. They are retrospective "
          "in-sample analysis, not live predictions available in 2008.**", "",
          "## 7. Model-selection criteria (fixed before results)", ""]
    L += [f"- {p}" for p in SELECTION_PROCEDURE]
    L += ["", "Thresholds: " + ", ".join(f"{k} = {v}" for k, v in SELECTION_RULES.items()) + ".", "",
          "**Protocol amendments after the first real-data run** (the first-run outcome is recorded; nothing is "
          "deleted):", ""]
    for a in PROTOCOL_AMENDMENTS:
        L.append(f"- {a['version']} ({a['date']}): {a['change']} Reason: {a['reason']}"
                 + (f" First run: {a['first_run_outcome']}" if a.get("first_run_outcome") else ""))
    L += ["", "## 8. K = 2 / 3 / 4 comparison", "",
          "nw_t_vs_best = Newey-West t-statistic of the mean daily shortfall versus the best ELIGIBLE model; a negative "
          "value means the model scores higher than that benchmark (possible only for ineligible models).", ""]
    cols = ["model_id", "n_parameters", "converged", "train_avg_loglik", "validation_avg_loglik", "nw_t_vs_best",
            "competitive", "eligible", "train_bic", "train_shares", "validation_shares", "flags"]
    L.append(_md_table(_fmt(comp[cols].set_index("model_id"))))
    L += ["", "Initialization stability:", ""]
    L.append("Each seed's estimate is the best of its EM starts (n_init_per_seed). best_basin_share = share of seeds "
             "whose estimate reaches the best Train objective; single_start_best_basin_share = the same for individual "
             "EM starts (local-optimum frequency).")
    L.append("")
    scol = ["model_id", "n_init_per_seed", "converged_share", "best_basin_share", "single_start_best_basin_share",
            "objective_per_obs_range", "train_mean_pairwise_ari", "validation_mean_pairwise_ari",
            "best_basin_max_mean_distance", "mean_iterations"]
    L.append(_md_table(_fmt(stab[scol].set_index("model_id"))))
    sa = sel.get("sticky_assessment")
    if sa is not None and not sa.empty:
        L += ["", "Sticky assessment at the selected K and covariance (rule 4):", "", _md_table(_fmt(sa.set_index("model_id")))]
    L += ["", "## 9. Transition matrices", ""]
    tcols = ["model_id", "self_transitions", "expected_durations_days", "stress_states"]
    L.append(_md_table(comp[tcols].set_index("model_id")))
    L += ["", "Full matrices: `data/processed/hmm/hmm_transition_matrices.csv`; figure `hmm_transition_matrices.png`.", "",
          "## 10. State durations and switching", ""]
    dcols = ["model_id", "train_switches_per_year", "validation_switches_per_year", "train_median_duration_days",
             "validation_median_duration_days", "validation_hindsight_revision_share"]
    L.append(_md_table(_fmt(comp[dcols].set_index("model_id"), 2)))
    rare = durations[durations["rare_state_warning"]]
    L += ["", "Rare-state warnings (< 5% of a split): " + ("none." if rare.empty else "; ".join(
        f"{r.model_id} {r.state} {r.split} {r.share:.1%}" for r in rare.itertuples()))]
    L += ["", "## 11. Filtered probability behaviour", ""]
    pcols = ["model_id", "train_mean_max_filtered_prob", "validation_mean_max_filtered_prob",
             "train_mean_normalized_entropy", "validation_mean_normalized_entropy",
             "validation_mean_abs_daily_change_p_stress", "validation_p_stress_crossings_per_year",
             "validation_share_uncertain_days", "validation_mean_uncertain_spell_days",
             "validation_median_emission_log_lr", "mean_transition_log_odds"]
    L.append(_md_table(_fmt(comp[pcols].set_index("model_id"))))
    L += ["", "validation_median_emission_log_lr = median daily log-density gap between the two most likely states; "
          "mean_transition_log_odds = mean of log(A_ii / (1 - A_ii)). When the first is much larger than the second, "
          "the observation of the day decides the filtered state and a persistence prior has little room to act."]
    L += ["", "## 12. Economic interpretation (descriptive)", ""]
    if selected:
        p = profiles[(profiles["model_id"] == selected) & (profiles["basis"] == "filtered argmax")]
        show = p[["split", "state", "share", *FEATURE_COLUMNS, "is_stress_state", "hypothesis_label",
                  "train_validation_profile_distance"]].set_index(["split", "state"])
        show.index = [f"{a} {b}" for a, b in show.index]
        L.append(f"Selected model {selected} - mean factor values by filtered state:")
        L += ["", _md_table(_fmt(show, 2)), ""]
        a = assets[(assets["model_id"] == selected) & (assets["etf"].isin(["SPY", "IEF", "HYG", "GLD", "DBC", "TIP"]))]
        piv = a.pivot_table(index=["split", "horizon", "state"], columns="etf", values="annualized_mean_pct")
        piv.index = [f"{x} / {y} / {z}" for x, y, z in piv.index]
        L += ["Annualized mean ETF log return by state (%). same_day pairs state_t with return_t and is NOT tradable "
              "(the state is known only at the close of t); next_session pairs state_t with return_t+1. Both are "
              "descriptive and in-sample conditional on Train parameters:", "", _md_table(_fmt(piv, 1)), ""]
        k3 = profiles[(profiles["k"] == 3) & (profiles["basis"] == "filtered argmax") & (profiles["split"] == "train")]
        rows = []
        for m in k3["model_id"].unique():
            g = k3[k3["model_id"] == m].set_index("state")
            top = g[["interest_rate", "inflation"]].max(axis=1)
            cand = top[~g["is_stress_state"]]
            if len(cand):
                s = cand.idxmax()
                rows.append(f"{m}: {s} rates {g.loc[s, 'interest_rate']:+.2f}, inflation {g.loc[s, 'inflation']:+.2f}, "
                            f"share {g.loc[s, 'share']:.0%}")
        L += ["K=3 check for a rates/inflation state (non-stress state with the largest rates or inflation mean, "
              "Train): " + ("; ".join(rows) if rows else "none") + ". Stage 4 criterion: >= +0.5.", ""]
    L += ["## 13. Validation generalization", "",
          "See the dedicated section 'Train vs Validation Generalization' below.", "",
          "## 14. Comparison with K-Means / GMM", ""]
    if gmm_cmp.empty:
        L.append("Stage 4 GMM reference not available; comparison not run.")
    else:
        L.append(_md_table(_fmt(gmm_cmp.set_index(["model_id", "split"]).reset_index(level=1), 3)))
        L += ["", "The daily difference compares one-step-ahead predictive densities of the same standardized "
              "observation: the GMM density ignores the past, the HMM density conditions on x_1..x_t-1 through the "
              "filtered state. This is a valid proper-scoring-rule comparison of two forecasting models; label "
              "agreement (ARI) is descriptive and is not forecasting evidence. Stage 4 K-Means has no density and is "
              "compared through the Stage 4 report only."]
    L += ["", "## 15. Stability versus detection delay", "",
          f"Reference onsets (rules fixed in advance): factor stress score above its Train "
          f"{EVENT_RULES['stress_score_quantile']:.0%} quantile ({_num(events.get('_threshold'), '{:.2f}')}), gaps < "
          f"{EVENT_RULES['close_gap_days']} days closed, episodes >= {EVENT_RULES['min_episode_days']} days, episodes "
          f"< {EVENT_RULES['merge_gap_days']} days apart merged; and Stage 4 GMM stress-state runs with the same rules. "
          f"Lag = trading days from onset until P(stress) >= threshold; if the model was already above the threshold "
          f"on the previous day the lag is negative (minus the length of its ongoing stress run, capped at "
          f"-{EVENT_RULES['max_lag_days']}): the model led the reference. Undetected = not reached within "
          f"{EVENT_RULES['max_lag_days']} days. These are descriptive references, "
          "not ground-truth regimes, and no model was tuned on them.", ""]
    n_on = onsets.groupby(["reference", "split"]).size().to_dict() if not onsets.empty else {}
    L.append("Onsets: " + (", ".join(f"{a} {b}: {v}" for (a, b), v in n_on.items()) or "none") + ".")
    dd = det[det["split"] == "pooled"][["model_id", "reference", "n_onsets", "already_in_stress_share_p50",
                                         "median_lag_p50", "mean_lag_p50", "undetected_share_p50", "median_lag_p80",
                                         "missed_share_in_episodes", "stress_share_outside_episodes"]]
    L += ["", _md_table(_fmt(dd.set_index("model_id"), 2)), "",
          "already_in_stress_share = share of onsets at which the model was already in stress (negative lags). The "
          "same-sequence comparison below is the direct Standard-vs-Sticky delay test.", ""]
    if paired is not None and not paired.empty:
        L += ["**Same-sequence comparison (Sticky vs its Standard counterpart).** Episodes are the Standard model's "
              "runs of P(stress) >= 0.5 (same event rules). Entry lag = days until the Sticky P(stress) reaches 0.5, "
              "searched from 10 days before the Standard onset (negative = Sticky earlier); exit lag = days until the "
              "Sticky P(stress) falls below 0.5 relative to the Standard exit (positive = Sticky stays in stress longer).",
              "", _md_table(_fmt(paired.set_index("model_id"), 2)), ""]
    L += ["## 16. Limitations", "",
          "- Some historical NFCI values are unavailable before the 2011 real-time vintages; liquidity_credit starts "
          "2007-10-30 and its composition changes inside Train. Daily FRED series use conservative availability lags; "
          "macro release timestamps are not fully verified (Stage 3 PIT categories).",
          "- Factors are correlated (volatility, liquidity_credit, equity_trend), so states are driven mainly by a "
          "common stress dimension; rolling robust z-scores measure stress relative to the trailing window, not "
          "permanent economic levels.",
          "- Gaussian emissions understate fat tails; geometric durations are a strong assumption; parameters are "
          "time-homogeneous over 2007-2015, which includes the GFC.",
          "- Validation (2016-2019) contains no crisis comparable to 2008, so stress-state calibration and detection "
          "delay are judged largely on Train, in-sample.",
          f"- Late-Train composition sensitivity: {late_note}."]
    if not late.empty:
        L += ["", _md_table(_fmt(late.set_index("model_id"), 3))]
    L += ["", "## 17. Final selected HMM configuration", ""]
    if selected:
        c = cands[selected]
        L += [f"- Model: **{selected}**; seed {c.reference.seed}; kappa {c.reference.kappa:.2f}; converged "
              f"{c.reference.converged} after {c.reference.n_iter} iterations.",
              "- Transition matrix (Train): " + "; ".join(
                  f"S{i + 1}: " + ", ".join(f"{v:.4f}" for v in row) for i, row in enumerate(c.reference.params.transmat)),
              f"- Stress states: {', '.join(f'S{j + 1}' for j in c.stress_states)}.",
              ("- Decision filter: emission covariance inflation c = "
               f"{extras['calibration']['decisions'][selected]['decision_inflation']:g} (see 'Probability calibration')."
               if extras and selected in extras["calibration"]["decisions"] else "- Decision filter: Train estimate."),
              f"- Parameters, preprocessing, seeds and input hashes: `data/processed/hmm/models/{selected}.json`; "
              "selection evidence: `hmm_selection_manifest.json`.", ""]
    else:
        L += ["No model met the eligibility rules; no configuration is carried forward.", ""]
    L += ["## 18. Readiness for the Early Warning stage", "",
          "- Available for later stages: Train-estimated parameters, the causal filter, and Validation predicted / "
          "filtered probabilities. Later stages must keep the same filter and must not use smoothed or Viterbi output.",
          "- Better regime classification does not by itself imply better portfolio returns; that requires the later "
          "out-of-sample evaluation with t+1 execution.", "",
          "## Train vs Validation Generalization", "",
          "Same Train parameters and scaler for both partitions; Validation continues the forward recursion from the "
          "last Train filtered distribution; both averages are one-step-ahead predictive log densities per day "
          "(the Train value is in-sample).", ""]
    L.append(_md_table(_fmt(tv.set_index("model_id"), 3)))
    L += [""] + [f"- {s}" for s in generalization_statements(comp)]
    pr = profiles[(profiles["basis"] == "filtered argmax") & (profiles["split"] == "train")]
    shift = pr.groupby("model_id")["train_validation_profile_distance"].max()
    L += ["- Largest Train-to-Validation factor-profile shift of any state: "
          + ", ".join(f"{m} {v:.2f}" for m, v in shift.items()) + " (Euclidean, factor units; > 1.0 flagged in Stage 4).",
          "- A worse Validation score is not by itself rejection: 2016-2019 had lower and less varied stress than "
          "2008-2015, so all models are expected to score differently out of sample. Overfitting is indicated when "
          "Train improves with complexity while Validation deteriorates.", ""]
    if extras:
        L += calibration_section(extras, selected)
        L += robustness_section(extras, sel, stab)
    L += ["## Checks", ""]
    L.append(_md_table(pd.DataFrame([vars(c) for c in checks]).set_index("name")))
    L += ["", "## Figures", ""] + [f"- `reports/figures/hmm/{f}`" for f in figures]
    return "\n".join(L) + "\n"
