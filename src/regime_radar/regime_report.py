"""Figures and Markdown report for stage 4 (Train and Validation only; no test data is ever passed in)."""

from __future__ import annotations

STAGE4_CODE_VERSION = "stage4-2026-10-21b"  # must match scripts/06_explore_regimes.py

from pathlib import Path

import numpy as np
import pandas as pd

from . import regime_diagnostics as rd
from .dataset import FEATURE_COLUMNS
from .factors import FACTORS
from .quality import _md_table
from .regime_exploration import (
    ELBOW_PRIMARY_MAX_K, GMM_COVARIANCE_TYPES, GMM_N_INIT, GMM_REG_COVAR, KMEANS_N_INIT, SEEDS, SELECTION_PROCEDURE, SELECTION_RULES,
    ExplorationResult, Stage4Data,
)

STATE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SERIES = {"train": "#2a78d6", "validation": "#eb6834", "kmeans": "#2a78d6", "gmm_full": "#eb6834",
          "gmm_diag": "#1baf7a"}
INK, INK_2, MUTED, GRID, AXIS, SURFACE, SHADE = ("#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7",
                                                 "#fcfcfb", "#ecebe6")
FACTOR_LABELS = {f: FACTORS[f] for f in FEATURE_COLUMNS}


def display_candidates(result: ExplorationResult) -> tuple[str, str, str]:
    """Models used for the detailed figures: the rule-based preferences, else documented fallbacks."""
    from .regime_decision import STAGE4_DECISION
    if STAGE4_DECISION["reference_model"] in result.reference and STAGE4_DECISION["baseline_model"] in result.reference:
        return STAGE4_DECISION["reference_model"], STAGE4_DECISION["baseline_model"], "stage-4 decision (K=2)"
    sel = result.selection
    comp = result.comparison
    gmm = sel.get("gmm_preferred") or sel.get("gmm_best_validation") or \
        comp[comp["family"] == "gmm"].sort_values("validation_avg_loglik").iloc[-1]["model_id"]
    km_rows = comp[comp["family"] == "kmeans"]
    kmeans = sel.get("kmeans_preferred") or km_rows.sort_values("validation_silhouette").iloc[-1]["model_id"]
    note = ("rule-based preferred candidates" if sel.get("gmm_preferred") and sel.get("kmeans_preferred")
            else "no fully eligible candidate in at least one family; figures show the best validation candidate")
    return gmm, kmeans, note


# --------------------------------------------------------------------------------------
# Figure helpers
# --------------------------------------------------------------------------------------
def _style(ax, grid_y: bool = True) -> None:
    ax.set_facecolor(SURFACE)
    if grid_y:
        ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.grid(axis="x", visible=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)


def _title(fig, text: str, sub: str | None = None) -> None:
    fig.suptitle(text, x=0.01, ha="left", fontsize=13, color=INK, fontweight="bold")
    if sub:
        fig.text(0.01, 0.005, sub, fontsize=8, color=MUTED)


def _save(fig, path: Path) -> None:
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    import matplotlib.pyplot as plt
    plt.close(fig)


def _diverging():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("div", ["#184f95", "#5598e7", "#f0efec", "#e66767", "#a72f2f"])


def _heatmap(ax, frame: pd.DataFrame, vmax: float, fmt: str = "{:.2f}", cmap=None):
    cmap = cmap or _diverging()
    image = ax.imshow(frame.to_numpy(dtype=float), cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(frame.shape[1]), frame.columns, rotation=30, ha="right", fontsize=8, color=INK_2)
    ax.set_yticks(range(frame.shape[0]), frame.index, fontsize=8, color=INK_2)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            v = frame.iat[i, j]
            if pd.notna(v):
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7.5,
                        color="#ffffff" if abs(v) > 0.6 * vmax else INK)
    return image


def _mark_k(ax, k: int, label: str) -> None:
    ax.axvline(k, color=MUTED, linestyle="--", linewidth=0.8, zorder=1)
    ax.text(k, 1.0, f" {label}", transform=ax.get_xaxis_transform(), fontsize=7, color=MUTED, va="top")


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
def fig_kmeans(result: ExplorationResult, path: Path) -> None:
    import matplotlib.pyplot as plt

    km = result.comparison[result.comparison["family"] == "kmeans"].sort_values("k")
    stab = result.stability[result.stability["family"] == "kmeans"].sort_values("k")
    panels = [
        ("Inertia per observation (lower = tighter)", "train_inertia_per_obs", "validation_inertia_per_obs", km),
        ("Silhouette (higher = better separated)", "train_silhouette", "validation_silhouette", km),
        ("Davies-Bouldin (lower = better separated)", "train_davies_bouldin", "validation_davies_bouldin", km),
        ("Calinski-Harabasz (train only, higher = better)", "train_calinski_harabasz", None, km),
        ("Smallest state share", "train_min_state_share", "validation_min_state_share", km),
        ("Seed stability: mean pairwise ARI (10 seeds)", "train_mean_ari", "validation_mean_ari", stab),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), facecolor=SURFACE)
    preferred = result.selection.get("kmeans_preferred")
    for ax, (title, tr_col, va_col, frame) in zip(axes.ravel(), panels):
        _style(ax)
        ax.plot(frame["k"], frame[tr_col], marker="o", color=SERIES["train"], linewidth=2, markersize=6, label="Train")
        if va_col:
            ax.plot(frame["k"], frame[va_col], marker="o", color=SERIES["validation"], linewidth=2, markersize=6,
                    label="Validation (centroids fixed from Train)")
        if "share" in tr_col:
            ax.axhline(SELECTION_RULES["min_train_state_share"], color=AXIS, linewidth=0.8, linestyle=":")
            ax.axhline(SELECTION_RULES["min_validation_state_share"], color=SERIES["validation"], linewidth=0.8,
                       linestyle=":")
        if "ari" in tr_col:
            ax.axhline(SELECTION_RULES["min_mean_ari_train"], color=AXIS, linewidth=0.8, linestyle=":")
        if preferred:
            _mark_k(ax, int(preferred.split("_k")[-1]), "rule-based preference")
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")
        ax.set_xlabel("Number of states K", fontsize=8, color=INK_2)
        ax.set_xticks(frame["k"])
    axes[0, 0].legend(fontsize=7.5, frameon=False, loc="lower left")
    _title(fig, "K-Means selection diagnostics (fitted on Train; Validation assigned with fixed centroids)",
           "Dotted lines: eligibility thresholds. Test period not used.")
    _save(fig, path)


def fig_gmm_ic(result: ExplorationResult, path: Path) -> None:
    import matplotlib.pyplot as plt

    gmm = result.comparison[result.comparison["family"] == "gmm"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURFACE)
    for ax, metric, name in ((axes[0], "train_bic", "BIC"), (axes[1], "train_aic", "AIC")):
        _style(ax)
        for cov in GMM_COVARIANCE_TYPES:
            rows = gmm[gmm["covariance_type"] == cov].sort_values("k")
            ax.plot(rows["k"], rows[metric], marker="o", linewidth=2, markersize=6, color=SERIES[f"gmm_{cov}"],
                    label=f"{cov} covariance")
            best = rows.loc[rows[metric].idxmin()]
            ax.scatter([best["k"]], [best[metric]], s=110, facecolors="none", edgecolors=SERIES[f"gmm_{cov}"],
                       linewidths=1.5, zorder=5)
        ax.set_title(f"Train {name} (lower is better; circle = minimum)", fontsize=10, color=INK, loc="left")
        ax.set_xlabel("Number of states K", fontsize=8, color=INK_2)
        ax.set_xticks(sorted(gmm["k"].unique()))
    axes[0].legend(fontsize=8, frameon=False)
    _title(fig, "Gaussian mixture information criteria (Train only)",
           "In-sample criteria; validation likelihood is shown separately. Test period not used.")
    _save(fig, path)


def fig_gmm_loglik(result: ExplorationResult, path: Path) -> None:
    import matplotlib.pyplot as plt

    gmm = result.selection["gmm_table"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), facecolor=SURFACE)
    ax = axes[0]
    _style(ax)
    for cov in GMM_COVARIANCE_TYPES:
        rows = gmm[gmm["covariance_type"] == cov].sort_values("k")
        color = SERIES[f"gmm_{cov}"]
        ax.plot(rows["k"], rows["train_avg_loglik"], marker="o", color=color, linewidth=1.2, linestyle="--",
                markersize=5, label=f"{cov}: Train")
        ax.plot(rows["k"], rows["validation_avg_loglik"], marker="o", color=color, linewidth=2.2, markersize=6,
                label=f"{cov}: Validation")
    ax.set_title("Average log-likelihood per observation", fontsize=10, color=INK, loc="left")
    ax.set_xlabel("Number of states K", fontsize=8, color=INK_2)
    ax.set_xticks(sorted(gmm["k"].unique()))
    ax.legend(fontsize=7.5, frameon=False, ncol=2)

    ax = axes[1]
    _style(ax)
    offsets = {"full": -0.12, "diag": 0.12}
    for cov in GMM_COVARIANCE_TYPES:
        rows = gmm[gmm["covariance_type"] == cov].sort_values("k")
        color = SERIES[f"gmm_{cov}"]
        t = rows["nw_t_vs_best"].clip(upper=15)
        ax.scatter(rows["k"] + offsets[cov], t, s=55, color=[color if e else "none" for e in rows["eligible"]],
                   edgecolors=color, linewidths=1.5, label=f"{cov} (hollow = ineligible)")
    ax.axhline(SELECTION_RULES["nw_t_threshold"], color=MUTED, linestyle=":", linewidth=1)
    ax.text(0.01, SELECTION_RULES["nw_t_threshold"], " significance threshold (t = 2)", fontsize=7, color=MUTED,
            va="bottom", transform=ax.get_yaxis_transform())
    ax.set_title("Validation shortfall vs best candidate: Newey-West t (capped at 15)", fontsize=10, color=INK,
                 loc="left")
    ax.set_xlabel("Number of states K", fontsize=8, color=INK_2)
    ax.set_xticks(sorted(gmm["k"].unique()))
    ax.legend(fontsize=7.5, frameon=False)
    pref = result.selection.get("gmm_preferred")
    _title(fig, f"Gaussian mixture out-of-sample fit on Validation (preferred: {pref or 'none'})",
           "Parameters fitted on Train only; Validation used for comparison. Test period not used.")
    _save(fig, path)


def _profile_frame(data_features: pd.DataFrame, labels: np.ndarray, k: int, labels_by_state: dict) -> pd.DataFrame:
    prof = rd.factor_profile(data_features[FEATURE_COLUMNS], labels, k)
    frame = prof[FEATURE_COLUMNS].rename(columns=FACTOR_LABELS)
    frame.index = [f"{s} {labels_by_state.get(s, '')} ({prof.loc[s, 'share']:.0%})".replace("  ", " ")
                   for s in prof.index]
    return frame


def state_labels(profiles: pd.DataFrame, model_id: str) -> dict:
    rows = profiles[(profiles["model_id"] == model_id) & (profiles["split"] == "train")]
    return {r.state: ("" if r.hypothesis_label == "Neutral" else f"- {r.hypothesis_label.replace(' (hypothesis)', '')}")
            for r in rows.itertuples()}


def fig_factor_heatmap(result, data, profiles, gmm_id, km_id, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5), facecolor=SURFACE, constrained_layout=True)
    image = None
    for row, model_id in enumerate((gmm_id, km_id)):
        fit = result.reference[model_id]
        names = state_labels(profiles, model_id)
        for col, (split, feats, labels) in enumerate((("Train", data.train_features, fit.train_labels),
                                                      ("Validation", data.validation_features,
                                                       fit.validation_labels))):
            frame = _profile_frame(feats, labels, fit.spec.k, names if col == 0 else {})
            ax = axes[row, col]
            image = _heatmap(ax, frame, vmax=2.0)
            for i, state_row in enumerate(frame.index):
                if frame.iloc[i].isna().all():
                    ax.text((frame.shape[1] - 1) / 2, i, "state absent in this period", ha="center", va="center",
                            fontsize=8, color=MUTED)
            if row == 0:
                ax.set_xticklabels([])
            ax.set_title(f"{model_id} (K={fit.spec.k}) - {split} mean factor values", fontsize=10, color=INK,
                         loc="left")
    bar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.01)
    bar.set_label("factor value (robust z units; colour capped at +/-2)", fontsize=8, color=INK_2)
    bar.outline.set_visible(False)
    fig.suptitle("Regime factor profiles: preferred GMM (top) and K-Means (bottom)", x=0.01, ha="left",
                 fontsize=13, color=INK, fontweight="bold")
    fig.text(0.01, -0.01, "Hypothesis labels (left panels) come from fixed rules on Train profiles; 'Neutral' states "
                          "keep their id. Percentages are the share of days in each period. Validation is assigned "
                          "with Train-fitted models. Test period not used.", fontsize=8, color=MUTED)
    fig.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def _state_band(ax, dates: pd.DatetimeIndex, labels: np.ndarray, k: int) -> None:
    import matplotlib.dates as mdates
    from matplotlib.colors import BoundaryNorm, ListedColormap

    nums = mdates.date2num(dates.to_pydatetime())
    step = np.diff(nums)
    edges = np.r_[nums, nums[-1] + (np.median(step) if len(step) else 1)]
    cmap = ListedColormap(STATE_COLORS[:k])
    norm = BoundaryNorm(np.arange(-0.5, k + 0.5), k)
    ax.pcolormesh(edges, [0, 1], labels[np.newaxis, :].astype(float), cmap=cmap, norm=norm, shading="flat")
    ax.set_yticks([])
    ax.xaxis_date()


def fig_timeline(result, data, profiles, gmm_id, km_id, path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    g, km = result.reference[gmm_id], result.reference[km_id]
    dates = data.train_features.index.append(data.validation_features.index)
    split_date = data.validation_features.index.min()
    fig, axes = plt.subplots(3, 1, figsize=(14, 7.5), sharex=True, facecolor=SURFACE,
                             gridspec_kw={"height_ratios": [1, 1.6, 1]})
    _state_band(axes[0], dates, np.r_[g.train_labels, g.validation_labels], g.spec.k)
    axes[0].set_title(f"{gmm_id} (K={g.spec.k}): most likely mixture component per day", fontsize=10, color=INK,
                      loc="left")
    ax = axes[1]
    _style(ax)
    max_post = np.r_[g.train_proba.max(axis=1), g.validation_proba.max(axis=1)]
    ax.plot(dates, max_post, color=INK_2, linewidth=0.6)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("max posterior", fontsize=8, color=INK_2)
    ax.set_title("Static GMM membership probability of the assigned state (not an HMM filtered probability)",
                 fontsize=10, color=INK, loc="left")
    _state_band(axes[2], dates, np.r_[km.train_labels, km.validation_labels], km.spec.k)
    axes[2].set_title(f"{km_id} (K={km.spec.k}): nearest centroid per day", fontsize=10, color=INK, loc="left")
    for ax in axes:
        ax.axvline(split_date, color=INK, linewidth=1.2)
    box = {"facecolor": SURFACE, "edgecolor": AXIS, "boxstyle": "round,pad=0.3"}
    axes[1].text(data.train_features.index[len(data.train_features) // 2], 0.12, "TRAIN (models fitted here)",
                 ha="center", fontsize=9, color=INK, fontweight="bold", bbox=box)
    axes[1].text(data.validation_features.index[len(data.validation_features) // 2], 0.12,
                 "VALIDATION (Train-fitted models applied)", ha="center", fontsize=9, color=INK, fontweight="bold",
                 bbox=box)
    k_max = max(g.spec.k, km.spec.k)
    handles = [Patch(color=STATE_COLORS[i], label=f"S{i + 1}") for i in range(k_max)]
    fig.legend(handles=handles, ncol=k_max, fontsize=8, frameon=False, loc="upper right",
               bbox_to_anchor=(0.99, 0.995))
    fig.suptitle("Regime assignments over time - Train and Validation only", x=0.01, ha="left", fontsize=13,
                 color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.text(0.01, 0.005, "Each day is classified from its own factor values with Train-fitted parameters "
                          "(no smoothing, no future data). State ids are ordered by a stress score, not by meaning.",
             fontsize=8, color=MUTED)
    fig.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def fig_durations(result, gmm_id, km_id, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), facecolor=SURFACE)
    for ax, model_id in zip(axes, (gmm_id, km_id)):
        _style(ax)
        fit = result.reference[model_id]
        positions, data_, colors, ticks = [], [], [], []
        for s in range(fit.spec.k):
            for j, (split, labels) in enumerate((("Train", fit.train_labels), ("Validation", fit.validation_labels))):
                runs = rd.run_lengths(labels)
                lengths = runs.loc[runs["state"] == s, "length"].to_numpy()
                positions.append(s * 3 + j)
                data_.append(lengths if len(lengths) else np.array([np.nan]))
                colors.append(STATE_COLORS[s])
            ticks.append(s * 3 + 0.5)
        box = ax.boxplot(data_, positions=positions, widths=0.75, patch_artist=True, showfliers=True,
                         flierprops={"markersize": 2, "markerfacecolor": MUTED, "markeredgecolor": MUTED},
                         medianprops={"color": INK})
        for i, patch in enumerate(box["boxes"]):
            patch.set_facecolor(colors[i])
            patch.set_alpha(1.0 if i % 2 == 0 else 0.45)
            patch.set_edgecolor(SURFACE)
        ax.set_yscale("log")
        ax.set_xticks(ticks, [f"S{s + 1}" for s in range(fit.spec.k)])
        ax.set_ylabel("consecutive days in state (log scale)", fontsize=8, color=INK_2)
        stats_tr = rd.switching_stats(fit.train_labels)
        ax.set_title(f"{model_id}: run lengths (solid = Train, light = Validation); train median "
                     f"{stats_tr['median_duration_days']:.0f} d, {stats_tr['switches_per_year']:.0f} switches/yr",
                     fontsize=9, color=INK, loc="left")
    _title(fig, "Regime duration distributions (no persistence imposed)",
           "First and last runs are censored by the sample boundaries. Test period not used.")
    _save(fig, path)


MIN_DAYS_FOR_ASSET_STATS = 20


def fig_assets(assets: pd.DataFrame, profiles: pd.DataFrame, gmm_id: str, path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    names = state_labels(profiles, gmm_id)
    sub = assets[(assets["model_id"] == gmm_id) & (assets["horizon"] == "same_day")]
    etfs = [c for c in ["SPY", "EFA", "EEM", "IEF", "TIP", "HYG", "GLD", "DBC", "BIL"] if c in set(sub["etf"])]
    panels = [("Train: annualized mean return (%)", "train", "annualized_mean_pct"),
              ("Train: annualized volatility (%)", "train", "annualized_vol_pct"),
              ("Validation: annualized mean return (%)", "validation", "annualized_mean_pct")]
    seq = LinearSegmentedColormap.from_list("seq", ["#f0efec", "#86b6ef", "#256abf", "#0d366b"])
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), facecolor=SURFACE, constrained_layout=True)
    for i, (ax, (title, split, metric)) in enumerate(zip(axes, panels)):
        part = sub[sub["split"] == split]
        frame = part.pivot(index="state", columns="etf", values=metric)[etfs]
        n = part[part["etf"] == "SPY"].set_index("state")["n_days"].reindex(frame.index).fillna(0).astype(int)
        thin = n < MIN_DAYS_FOR_ASSET_STATS
        shown = frame.mask(thin, np.nan)
        prefix = names if i == 0 else {}
        shown.index = [f"{s} {prefix.get(s, '')} (n={n[s]})".replace("  ", " ") for s in frame.index]
        vol = "vol" in metric
        ax.imshow(shown.to_numpy(float), cmap=seq if vol else _diverging(), vmin=0 if vol else -60, vmax=60,
                  aspect="auto")
        ax.set_xticks(range(len(etfs)), etfs, fontsize=8, color=INK_2)
        ax.set_yticks(range(len(shown)), shown.index, fontsize=8, color=INK_2)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for r in range(shown.shape[0]):
            if thin.iloc[r]:
                ax.text((len(etfs) - 1) / 2, r, "state absent" if n.iloc[r] == 0 else
                        f"n < {MIN_DAYS_FOR_ASSET_STATS} days: not shown", ha="center", va="center", fontsize=8,
                        color=MUTED)
                continue
            for c in range(shown.shape[1]):
                v = shown.iat[r, c]
                if pd.notna(v):
                    dark = v > 36 if vol else abs(v) > 36
                    ax.text(c, r, f"{v:.0f}", ha="center", va="center", fontsize=7.5,
                            color="#ffffff" if dark else INK)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
    fig.suptitle(f"ETF behaviour by regime - {gmm_id} (same-day returns, descriptive only)", x=0.01, ha="left",
                 fontsize=13, color=INK, fontweight="bold")
    fig.text(0.01, -0.02, "return_t is grouped by the state of the same day t: both are known only at the close of t, "
                          "so these are not tradable returns. Colour capped at +/-60%. States with fewer than "
                          f"{MIN_DAYS_FOR_ASSET_STATS} days are not shown. Test period not used.", fontsize=8,
             color=MUTED)
    fig.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def fig_stability(result: ExplorationResult, path: Path) -> None:
    import matplotlib.pyplot as plt

    stab, comp = result.stability, result.comparison
    groups = [("kmeans", "-", "K-Means"), ("gmm", "full", "GMM full"), ("gmm", "diag", "GMM diag")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), facecolor=SURFACE)
    for (family, cov, label), key in zip(groups, ("kmeans", "gmm_full", "gmm_diag")):
        s = stab[(stab["family"] == family) & (stab["covariance_type"] == cov)].sort_values("k")
        c = comp[(comp["family"] == family) & (comp["covariance_type"] == cov)].sort_values("k")
        color = SERIES[key]
        axes[0].plot(s["k"], s["train_mean_ari"], marker="o", color=color, linewidth=2, label=label)
        axes[0].vlines(s["k"], s["train_min_ari"], s["train_mean_ari"], color=color, linewidth=1)
        axes[1].plot(c["k"], c["train_switches_per_year"], marker="o", color=color, linewidth=2, label=label)
        axes[2].plot(c["k"], c["validation_min_state_share"], marker="o", color=color, linewidth=2, label=label)
    titles = ("Seed stability on Train: mean ARI (line to min)", "Train switches per year",
              "Smallest state share in Validation (below 1% = flagged)")
    for ax, title in zip(axes, titles):
        _style(ax)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("Number of states K", fontsize=8, color=INK_2)
        ax.set_xticks(sorted(comp["k"].unique()))
    axes[0].axhline(SELECTION_RULES["min_mean_ari_train"], color=MUTED, linestyle=":", linewidth=1)
    axes[2].axhline(SELECTION_RULES["min_validation_state_share"], color=MUTED, linestyle=":", linewidth=1)
    axes[0].legend(fontsize=8, frameon=False)
    _title(fig, f"Regime stability comparison ({len(SEEDS)} seeds per candidate)",
           "Dotted lines: eligibility thresholds. ARI is invariant to label permutation. Test period not used.")
    _save(fig, path)


def fig_elbow(result: ExplorationResult, path: Path) -> None:
    import matplotlib.pyplot as plt

    e = result.elbow
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), facecolor=SURFACE)
    ax = axes[0]
    _style(ax)
    lo, hi = min(e.loc[e["stage4_candidate"], "k"]), max(e.loc[e["stage4_candidate"], "k"])
    ax.axvspan(lo - 0.4, hi + 0.4, color=SHADE, zorder=0, linewidth=0)
    ax.text((lo + hi) / 2, 0.98, f"stage-4 candidates K={lo}-{hi}", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=7.5, color=MUTED)
    ax.plot(e["k"], e["train_inertia_per_obs"], marker="o", color=SERIES["train"], linewidth=2, markersize=6,
            label="Train (fit)")
    ax.plot(e["k"], e["validation_inertia_per_obs"], marker="o", color=SERIES["validation"], linewidth=2,
            markersize=6, label="Validation (Train centroids)")
    sens = result.elbow_sensitivity
    primary_hi = int(sens.loc[sens["primary_range"], "max_k"].iloc[0])
    full_hi = int(e["k"].max())
    full_col = f"kneedle_elbow_range_1_{full_hi}"
    chord = e[e["k"].isin([1, primary_hi])]
    ax.plot(chord["k"], chord["train_inertia_per_obs"], linestyle=":", color=MUTED, linewidth=1, zorder=1)
    row = e[e["kneedle_elbow"]].iloc[0]
    ax.scatter([row["k"]], [row["train_inertia_per_obs"]], s=240, facecolors="none", edgecolors=INK,
               linewidths=1.8, zorder=5)
    ax.annotate(f"Kneedle elbow (K=1-{primary_hi}, candidate range): K={int(row['k'])}",
                (row["k"], row["train_inertia_per_obs"]), xytext=(22, 22), textcoords="offset points",
                fontsize=8.5, color=INK, fontweight="bold",
                arrowprops={"arrowstyle": "-", "color": INK_2, "linewidth": 0.8})
    alt = e[e[full_col]].iloc[0]
    if int(alt["k"]) != int(row["k"]):
        ax.scatter([alt["k"]], [alt["train_inertia_per_obs"]], s=240, facecolors="none", edgecolors=MUTED,
                   linewidths=1.2, linestyles="--", zorder=5)
        ax.annotate(f"K=1-{full_hi} range: K={int(alt['k'])} (sensitivity)",
                    (alt["k"], alt["train_inertia_per_obs"]), xytext=(-30, -42), textcoords="offset points",
                    fontsize=7.5, color=MUTED, arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.6})
    ax.set_xticks(e["k"])
    ax.set_xlabel("Number of clusters K", fontsize=8, color=INK_2)
    ax.set_ylabel("within-cluster sum of squares per observation", fontsize=8, color=INK_2)
    ax.set_title("Inertia per observation (standardized factors)", fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper right", bbox_to_anchor=(1, 0.9))

    ax = axes[1]
    _style(ax)
    sub = e[e["k"] > 1]
    ax.bar(sub["k"], sub["pct_reduction_vs_previous_k"], color=SERIES["train"], width=0.6, label="Train")
    ax.scatter(sub["k"], sub["validation_pct_reduction_vs_previous_k"], color=SERIES["validation"], s=45,
               zorder=5, label="Validation")
    for k, v, w in zip(sub["k"], sub["pct_reduction_vs_previous_k"], sub["validation_pct_reduction_vs_previous_k"]):
        top = max(v, w if pd.notna(w) else v)
        ax.text(k, top + 0.8, f"{v:.0f}%", ha="center", va="bottom", fontsize=7.5, color=INK_2)
    ax.set_xticks(sub["k"])
    ax.set_xlabel("K (reduction from K-1 to K clusters)", fontsize=8, color=INK_2)
    ax.set_ylabel("% reduction in inertia", fontsize=8, color=INK_2)
    ax.set_title("Marginal gain from one more cluster", fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False)
    by_range = ", ".join(f"{r.k_range}: K={r.kneedle_elbow}" for r in sens.itertuples())
    _title(fig, "K-Means elbow method (diagnostic; fitted on Train only)",
           f"Primary elbow on K=1-{primary_hi} (the stage-4 candidate range; dotted chord). Range sensitivity - "
           f"{by_range}. Heuristic, not a selection rule. Test period not used.")
    _save(fig, path)


def fig_selected_stress(states: pd.DataFrame, episodes: pd.DataFrame, decision: dict, path: Path) -> None:
    """Static membership probability of the stress state, Train and Validation only."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    names = decision["state_names"]
    fig, axes = plt.subplots(2, 1, figsize=(14, 6.4), sharex=True, facecolor=SURFACE,
                             gridspec_kw={"height_ratios": [2.4, 1]})
    ax = axes[0]
    _style(ax)
    for e in episodes.itertuples():
        ax.axvspan(pd.Timestamp(e.start), pd.Timestamp(e.end), color=SHADE, zorder=0, linewidth=0)
    ax.fill_between(states.index, states["p_S2"], color=STATE_COLORS[1], alpha=0.25, linewidth=0)
    ax.plot(states.index, states["p_S2"], color=STATE_COLORS[1], linewidth=0.7)
    ax.axhline(0.5, color=MUTED, linestyle=":", linewidth=0.8)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("p(S2 | x_t)", fontsize=8, color=INK_2)
    ax.set_title(f"{decision['reference_model']}: static membership probability of S2 '{names['S2']}' "
                 f"(shaded: S2 runs of >= 10 days)", fontsize=10, color=INK, loc="left")
    split_date = states.index[states["split"] == "validation"].min()
    box = {"facecolor": SURFACE, "edgecolor": AXIS, "boxstyle": "round,pad=0.3"}
    for split, label in (("train", "TRAIN (fit)"), ("validation", "VALIDATION (Train-fitted model)")):
        idx = states.index[states["split"] == split]
        ax.text(idx[len(idx) // 2], 0.92, label, ha="center", fontsize=9, color=INK, fontweight="bold", bbox=box)

    ax = axes[1]
    nums = mdates.date2num(states.index.to_pydatetime())
    edges = np.r_[nums, nums[-1] + 1]
    grid = np.vstack([states["baseline_state"].str[1:].astype(int) - 1, states["state"].str[1:].astype(int) - 1])
    ax.pcolormesh(edges, [0, 1, 2], grid.astype(float), cmap=ListedColormap(STATE_COLORS[:2]),
                  norm=BoundaryNorm([-0.5, 0.5, 1.5], 2), shading="flat")
    ax.set_yticks([0.5, 1.5], [decision["baseline_model"], decision["reference_model"]], fontsize=8, color=INK_2)
    ax.tick_params(length=0)
    ax.xaxis_date()
    ax.set_title("Assigned state per day (reference GMM vs K-Means baseline)", fontsize=10, color=INK, loc="left")
    for a in axes:
        a.axvline(split_date, color=INK, linewidth=1.2)
    handles = [Patch(color=STATE_COLORS[i], label=f"S{i + 1} {names.get(f'S{i + 1}', '')}") for i in range(2)]
    fig.legend(handles=handles, ncol=2, fontsize=8, frameon=False, loc="upper right", bbox_to_anchor=(0.99, 0.995))
    fig.suptitle("Selected stage-4 model (K=2) - Train and Validation only", x=0.01, ha="left", fontsize=13,
                 color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.text(0.01, 0.005, "Static GMM membership p(state | x_t) from day t only: not an HMM filtered probability, "
                          "no persistence. Test period not used.", fontsize=8, color=MUTED)
    fig.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def save_figures(result, data, profiles, assets, figure_dir: Path, states: pd.DataFrame | None = None,
                 episodes: pd.DataFrame | None = None) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    figure_dir.mkdir(parents=True, exist_ok=True)
    gmm_id, km_id, _ = display_candidates(result)
    paths = {name: figure_dir / name for name in (
        "kmeans_selection_diagnostics.png", "gmm_aic_bic_comparison.png", "gmm_validation_loglikelihood.png",
        "regime_factor_heatmap.png", "regime_timeline_train_validation.png", "regime_duration_distribution.png",
        "regime_asset_behavior.png", "regime_stability_comparison.png", "kmeans_elbow_method.png")}
    fig_kmeans(result, paths["kmeans_selection_diagnostics.png"])
    fig_gmm_ic(result, paths["gmm_aic_bic_comparison.png"])
    fig_gmm_loglik(result, paths["gmm_validation_loglikelihood.png"])
    fig_factor_heatmap(result, data, profiles, gmm_id, km_id, paths["regime_factor_heatmap.png"])
    fig_timeline(result, data, profiles, gmm_id, km_id, paths["regime_timeline_train_validation.png"])
    fig_durations(result, gmm_id, km_id, paths["regime_duration_distribution.png"])
    fig_assets(assets, profiles, gmm_id, paths["regime_asset_behavior.png"])
    fig_stability(result, paths["regime_stability_comparison.png"])
    fig_elbow(result, paths["kmeans_elbow_method.png"])
    if states is not None and episodes is not None:
        from .regime_decision import STAGE4_DECISION
        extra = figure_dir / "selected_model_stress_probability.png"
        fig_selected_stress(states, episodes, STAGE4_DECISION, extra)
        paths[extra.name] = extra
    return list(paths.values())


# --------------------------------------------------------------------------------------
# Markdown report
# --------------------------------------------------------------------------------------
def _fmt(frame: pd.DataFrame, digits: int = 3) -> pd.DataFrame:
    out = frame.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].map(lambda v: "" if pd.isna(v) else f"{v:.{digits}f}")
    return out


def elbow_statement(elbow: pd.DataFrame, sensitivity: pd.DataFrame) -> str:
    """One data-driven sentence on the elbow: primary range, largest drop and range sensitivity."""
    primary = sensitivity[sensitivity["primary_range"]].iloc[0]
    full = sensitivity.iloc[-1]
    drops = elbow.dropna(subset=["pct_reduction_vs_previous_k"])
    top = drops.loc[drops["pct_reduction_vs_previous_k"].idxmax()]
    text = (f"Elbow (supporting, not decisive): the largest inertia reduction is from K={int(top['k']) - 1} to "
            f"K={int(top['k'])} ({top['pct_reduction_vs_previous_k']:.0f}%); the Kneedle elbow over the candidate "
            f"range {primary['k_range']} is K={primary['kneedle_elbow']}")
    if sensitivity["kneedle_elbow"].nunique() == 1:
        return text + " and does not change with the range."
    return (text + f", and over {full['k_range']} it is K={full['kneedle_elbow']} (margin "
            f"{full['margin_vs_runner_up']:.3f} over K={full['runner_up']}), so the curve has no sharp knee.")


def decision_section(d: dict, data: Stage4Data, elbow: pd.DataFrame | None = None,
                     elbow_sensitivity: pd.DataFrame | None = None) -> list[str]:
    """Executive summary of the recorded stage-4 decision for the selected K=2 model."""
    rec = d["record"]
    summary, tm, eps, assets, checks = d["summary"], d["transitions"], d["episodes"], d["assets"], d["checks"]
    lines = ["", "## Stage 4 decision (executive summary)", ""]
    lines += [f"- Decision date: {rec['decision_date']}; primary K = **{rec['primary_k']}**; reference model "
              f"**{rec['reference_model']}** (probabilistic), baseline **{rec['baseline_model']}** (hard clustering); "
              f"alternatives carried into the HMM search: K = {', '.join(str(k) for k in rec['alternatives_for_hmm'])}.",
              f"- Data used: {rec['data_used']}.", "", "Basis:", ""]
    lines += [f"- {b}" for b in rec["basis"]]
    lines += ["", "Consistency of the recorded decision with this run:", ""]
    lines += [f"- [{c.status}] {c.name}: {c.detail}" for c in checks]
    lines += ["", f"State naming: {rec['state_naming_rule']}", "",
              "**State profiles, shares and persistence (reference model):**", "",
              _md_table(_fmt(summary.set_index("state"), 2)), "",
              "**Empirical daily transition probabilities** (implied expected duration = 1 / (1 - p_stay)):", "",
              _md_table(_fmt(tm.set_index("model_id"), 3)), "",
              f"**S2 runs of at least 10 trading days** ({len(eps)} episodes; Train and Validation):", "",
              _md_table(eps.set_index("split")) if not eps.empty else "none.", ""]
    key = ["SPY", "EFA", "EEM", "HYG", "IEF", "TIP", "GLD", "DBC", "BIL"]
    sub = assets[assets["model_id"] == rec["reference_model"]]
    if not sub.empty:
        piv = sub.pivot_table(index=["split", "horizon", "state"], columns="etf", values="annualized_mean_pct")
        piv = piv[[c for c in key if c in piv.columns]]
        piv.index = [f"{a} / {b} / {c}" for a, b, c in piv.index]
        piv.index.name = "split / horizon / state"
        lines += ["**Annualized mean ETF return by state (%; descriptive, not tradable for same_day):**", "",
                  _md_table(_fmt(piv, 1)), ""]
    from .regime_decision import evidence_statements
    lines += ["Reading of the evidence (computed from this run):", ""]
    lines += [f"- {x}" for x in evidence_statements(summary, assets, rec["reference_model"])]
    if elbow is not None and elbow_sensitivity is not None:
        lines.append(f"- {elbow_statement(elbow, elbow_sensitivity)}")
    lines += [f"- Known limitation: {rec['known_limitation']}", ""]
    return lines


def render_report(result: ExplorationResult, data: Stage4Data, checks, profiles: pd.DataFrame,
                  assets: pd.DataFrame, decision: dict | None = None) -> str:
    sel = result.selection
    gmm_id, km_id, display_note = display_candidates(result)
    tr, va = data.train_features, data.validation_features
    comp = result.comparison
    lines = ["# Stage 4: exploratory regime models (K-Means and Gaussian mixtures)", ""]
    status = pd.Series([c.status for c in checks]).value_counts().to_dict()
    lines.append(f"Generated {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC from the local Train and Validation "
                 f"splits (stage-3 protocol {data.manifest.get('protocol_version')}). Checks: "
                 + ", ".join(f"{v} {k}" for k, v in status.items()) + ". The test split was not read.")

    if decision is not None:
        lines += decision_section(decision, data, result.elbow, result.elbow_sensitivity)
    lines += ["", "## 1. Research objectives", "",
              "1. How many statistically distinguishable regimes exist in the five-factor space?",
              "2. Are the regimes economically interpretable and stable (across seeds and into Validation)?",
              "3. Do probabilistic memberships (GMM) add information beyond hard clustering (K-Means)?",
              "", "Candidate state counts K = " + ", ".join(str(k) for k in sorted(comp['k'].unique()))
              + ". No state count was fixed in advance and no historical crisis labels were used."]

    lines += ["", "## 2. Dataset and temporal split", "",
              "| split | first date | last date | rows | role in stage 4 |", "|---|---|---|---|---|",
              f"| Train | {tr.index.min().date()} | {tr.index.max().date()} | {len(tr)} | fitting (scaler, centroids, "
              "mixture parameters) |",
              f"| Validation | {va.index.min().date()} | {va.index.max().date()} | {len(va)} | comparison and "
              "selection only (predict / score) |",
              "| Test | 2020-01-02 onwards | - | - | not read in stage 4 |",
              "", "Split files are verified against the SHA-256 hashes in `data/splits/split_manifest.json`."]

    lines += ["", "## 3. Input factors", ""]
    lines.append(_md_table(pd.DataFrame({"factor": FEATURE_COLUMNS, "name": [FACTORS[f] for f in FEATURE_COLUMNS],
                                         "direction": ["higher = healthier equity trend"] + ["higher = more stress"] * 4})
                           .set_index("factor")))
    lines.append("")
    lines.append("Definitions are frozen (stage 2); the factors.py hash is checked against the stage-3 manifest.")

    lines += ["", "## 4. Pre-modelling checks and preprocessing", ""]
    lines.append(_md_table(pd.DataFrame([vars(c) for c in checks]).set_index("name")))
    lines += ["", "Train-only distribution of the factors:", "", _md_table(_fmt(result.train_stats, 3))]
    lines += ["", f"**Scaling decision (rule fixed before results):** {result.scaling_reason}",
              "", "Scaler parameters (Train only): " + ", ".join(
                  f"{c}: mean {result.scaler.mean_[c]:.3f}, std {result.scaler.scale_[c]:.3f}" for c in FEATURE_COLUMNS),
              "", "GMM log-likelihoods are computed in the scaled space; the Jacobian constant is identical for all "
                  "candidates, so likelihood comparisons are unaffected."]

    lines += ["", "## 5. K-Means methodology", "",
              f"- Fitted on Train for each K with {KMEANS_N_INIT} k-means++ initializations and seeds "
              f"{SEEDS[0]}..{SEEDS[-1]}; the first seed is the reference fit.",
              "- Validation days are assigned with `predict()` to the fixed Train centroids; nothing is refitted.",
              "- Metrics: inertia, silhouette, Calinski-Harabasz, Davies-Bouldin, state populations, switching and "
              "run lengths. Distances are not probabilities; AIC/BIC are not defined for K-Means and are not used.",
              "- Elbow method: K-Means inertia for K = 1-10 on Train (K=1 = total sum of squares) and the validation "
              "inertia of the same Train centroids; diagnostic only. The primary Kneedle elbow is computed over "
              f"K = 1-{ELBOW_PRIMARY_MAX_K}, the same range as the candidate comparison; the elbow for every other "
              "range is reported as a sensitivity check."]

    lines += ["", "## 6. Gaussian mixture methodology", "",
              f"- Covariance types: {', '.join(GMM_COVARIANCE_TYPES)}; {GMM_N_INIT} initializations per seed; "
              f"reg_covar = {GMM_REG_COVAR}; convergence warnings captured and recorded.",
              "- Validation: `predict_proba` and `score_samples` with Train parameters. Posteriors sum to one (checked).",
              "- The posterior p(state | x_t) is a static mixture membership computed from day t alone. It is not an "
              "HMM filtered probability p(state_t | x_1..x_t) and carries no persistence."]

    lines += ["", "## 7. Selection metrics and predefined rules", ""]
    lines += [f"- {p}" for p in SELECTION_PROCEDURE]
    lines += ["", "Thresholds: " + ", ".join(f"{k} = {v}" for k, v in SELECTION_RULES.items()) + "."]

    lines += ["", "## 8-9. Candidate state counts: Train and Validation results", "", "**K-Means**", ""]
    km_cols = ["model_id", "train_inertia_per_obs", "train_silhouette", "train_calinski_harabasz",
               "train_davies_bouldin", "validation_silhouette", "validation_davies_bouldin",
               "validation_to_train_distortion", "train_shares", "validation_shares"]
    lines.append(_md_table(_fmt(comp[comp["family"] == "kmeans"][km_cols].set_index("model_id"))))
    if result.elbow is not None:
        e = result.elbow
        lines += ["", f"**K-Means elbow method (diagnostic, K = 1-{int(e['k'].max())}, Train fit)**", ""]
        show = e[["k", "train_inertia", "train_inertia_per_obs", "validation_inertia_per_obs", "train_explained_share",
                  "pct_reduction_vs_previous_k", "validation_pct_reduction_vs_previous_k", "kneedle_elbow"]]
        lines.append(_md_table(_fmt(show.set_index("k"), 3)))
        sens = result.elbow_sensitivity
        full = sens.iloc[-1]
        lines += ["", f"Primary Kneedle elbow (maximum distance from the chord, K = 1-{ELBOW_PRIMARY_MAX_K}): "
                      f"**K = {result.elbow_k}**. The range matches the stage-4 candidate set (K = 2-"
                      f"{ELBOW_PRIMARY_MAX_K}); K = {ELBOW_PRIMARY_MAX_K + 1}-{int(e['k'].max())} were never candidates "
                      "but would still move the chord, so they are shown as sensitivity rather than used for the "
                      "primary reading.", "", "*Kneedle range sensitivity*", ""]
        lines.append(_md_table(_fmt(sens.set_index("k_range"), 3)))
        verdict = ("is the same in every range" if sens["kneedle_elbow"].nunique() == 1 else
                   f"depends on the range: K = {full['kneedle_elbow']} over {full['k_range']} (margin "
                   f"{full['margin_vs_runner_up']:.3f} over K = {full['runner_up']}). The inertia curve is smooth "
                   "with no sharp knee, so the elbow alone does not separate the leading candidates")
        lines += ["", f"The elbow {verdict}. The elbow is a visual heuristic for the point of diminishing returns; "
                      "it is reported as supporting evidence and does not override the predefined selection rules."]
    lines += ["", "**Gaussian mixtures**", ""]
    g_cols = ["model_id", "n_parameters", "converged", "n_iter", "train_bic", "train_aic", "train_avg_loglik",
              "validation_avg_loglik", "validation_mean_max_posterior", "validation_mean_entropy",
              "validation_share_max_posterior_below_0.6", "train_shares", "validation_shares"]
    lines.append(_md_table(_fmt(comp[comp["family"] == "gmm"][g_cols].set_index("model_id"))))
    warn = comp[comp["convergence_warnings"] != "-"]
    lines += ["", "Convergence warnings: " + ("none." if warn.empty else "; ".join(
        f"{r.model_id}: {r.convergence_warnings}" for r in warn.itertuples()))]

    lines += ["", "## 10. Regime stability", ""]
    lines.append("Switching and durations (reference seed; durations in trading days):")
    sw_cols = ["model_id", "train_switches_per_year", "train_median_duration_days", "train_mean_duration_days",
               "train_share_single_day_runs", "validation_switches_per_year", "validation_median_duration_days"]
    lines += ["", _md_table(_fmt(comp[sw_cols].set_index("model_id"), 2))]
    lines += ["", f"Initialization sensitivity across {len(SEEDS)} seeds (ARI and label-aligned agreement are "
                  "permutation invariant):", ""]
    lines.append(_md_table(_fmt(result.stability.set_index("model_id"), 3)))

    lines += ["", "## 11. Economic interpretation", ""]
    lines.append("Hypothesis labels come from fixed rules applied to Train factor profiles:")
    lines += [""] + [f"- **{name}**: {desc}" for name, desc, _ in rd.HYPOTHESIS_RULES]
    lines.append("- Otherwise, or if several rules match, the state stays **Neutral** and keeps its S# id.")
    for model_id in dict.fromkeys([gmm_id, km_id]):
        sub = profiles[profiles["model_id"] == model_id]
        show = sub[["state", "split", "n_obs", "share"] + FEATURE_COLUMNS
                   + ["hypothesis_label", "label_rule", "train_validation_profile_distance", "generalization_flags"]]
        lines += ["", f"**{model_id}**", "", _md_table(_fmt(show.set_index("state"), 2))]
    lines.append("")
    lines.append("All candidates' profiles are in `data/processed/regime_exploration/regime_factor_profiles.csv`.")

    lines += ["", "## 12. Conditional ETF behaviour (descriptive)", ""]
    lines.append("`same_day` groups return_t by the state of day t (both known only at the close of t: not "
                 "tradable). `next_session` groups return_(t+1) by the state of day t within the same split; it is "
                 "closer to an implementable timing but remains in-sample for Train. Neither is evidence of "
                 "predictive power.")
    for horizon in ("same_day", "next_session"):
        sub = assets[(assets["model_id"] == gmm_id) & (assets["horizon"] == horizon)]
        for split in ("train", "validation"):
            piv = sub[sub["split"] == split].pivot(index="state", columns="etf", values="annualized_mean_pct")
            vol = sub[sub["split"] == split].pivot(index="state", columns="etf", values="annualized_vol_pct")
            n = sub[(sub["split"] == split) & (sub["etf"] == "SPY")].set_index("state")["n_days"]
            piv.insert(0, "n_days", n)
            lines += ["", f"{gmm_id}, {split}, {horizon}: annualized mean return (%)", "", _md_table(_fmt(piv, 1))]
            if horizon == "same_day":
                lines += ["", f"{gmm_id}, {split}, same_day: annualized volatility (%)", "", _md_table(_fmt(vol, 1))]

    lines += ["", "## 13. Model comparison", ""]
    lines.append(_md_table(_fmt(sel["gmm_table"].set_index("model_id"), 4)))
    lines += ["", sel.get("gmm_note", ""), "",
              "`train_minus_validation_loglik` is the in-sample minus out-of-sample average log-likelihood; a gap "
              "that widens with K indicates components fitted to train-specific structure.", "",
              _md_table(_fmt(sel["kmeans_table"].set_index("model_id"), 3)), "",
              sel.get("kmeans_note", ""), "",
              f"Train BIC minimum (reference only): {sel.get('gmm_train_bic_best')}.",
              sel.get("bic_validation_conflict", "")]
    agree = sel.get("preferred_agreement")
    if agree:
        lines.append(f"Agreement between the preferred GMM and K-Means assignments: ARI {agree['train_ari']:.2f} "
                     f"(Train), {agree['validation_ari']:.2f} (Validation); same K: {agree['same_k']}.")
    lines += ["", "Eligibility of every candidate:", "",
              _md_table(sel["eligibility"].set_index("model_id"))]

    lines += ["", "## 14. Preferred exploratory candidate", ""]
    if sel.get("gmm_preferred"):
        g = comp.set_index("model_id").loc[sel["gmm_preferred"]]
        lines.append(f"- **GMM: {sel['gmm_preferred']}** under the predefined rules (validation average "
                     f"log-likelihood {g['validation_avg_loglik']:.3f}; {int(g['n_parameters'])} parameters).")
    else:
        lines.append(f"- GMM: no candidate met every eligibility rule. {sel.get('gmm_note', '')}")
    lines.append(f"- K-Means: **{sel['kmeans_preferred']}**." if sel.get("kmeans_preferred")
                 else f"- K-Means: {sel.get('kmeans_note')}")
    if sel.get("candidate_k_range"):
        lines.append(f"- Candidates not significantly worse than the best on Validation span K = "
                     f"{', '.join(str(k) for k in sel['candidate_k_range'])}. The rule picks the most parsimonious "
                     "one, but the evidence does not reject the larger models in this range; they are the leading "
                     "alternatives, not discarded models.")
    from .regime_exploration import ALTERNATIVE_STATE_CRITERIA, alternative_state_assessment
    alt = alternative_state_assessment(profiles, result.stability)
    if not alt.empty:
        c = ALTERNATIVE_STATE_CRITERIA
        lines += ["", "**K=3 alternative: is there a distinct rates / inflation state?** Criteria fixed on 2026-10-20, "
                      "before the K=3 profiles were inspected: (c1) a state other than the most stressed one has train "
                      f"interest_rate or inflation >= {c['min_rates_or_inflation']}; (c2) its validation share >= "
                      f"{c['min_validation_share']:.0%}; (c3) train-validation profile distance < "
                      f"{c['max_profile_distance']}; (c4) seed mean ARI >= {c['min_mean_ari_train']}.", "",
                  _md_table(_fmt(alt.set_index("model_id"), 2)), ""]
        accepted = alt[alt["accepted"]]["model_id"].tolist()
        lines.append("- Result: " + (f"accepted for {', '.join(accepted)}." if accepted else
                                     "no K=3 candidate meets all four criteria; K=3 remains an alternative inside the "
                                     "HMM search range, not the primary hypothesis."))
    lines.append("- Validation (2016-2019) contains no stress episode comparable to 2008, so the validation "
                 "likelihood cannot reward a separate crisis component. A small preferred K therefore partly "
                 "reflects the calm validation period and should not be read as proof that only that many regimes "
                 "exist.")
    lines.append(f"- Figures show {gmm_id} and {km_id} ({display_note}). The recorded decision is in the "
                 "executive summary at the top of this report.")
    lines.append("- This is an exploratory baseline. The final state count is decided in the Sticky HMM stage, "
                 "using Train/Validation only, with the candidates above as the starting range.")

    lines += ["", "## 15. Limitations and risks", "",
              "- K-Means and GMM treat days as exchangeable: they ignore the time order, so regime persistence is "
              "neither modelled nor imposed, and day-to-day flicker is expected near state boundaries.",
              "- GMM memberships are static; they are not forward-filtered state probabilities and must not be "
              "described as such.",
              "- Train (2007-10 to 2015) contains one major crisis; rare stress states are estimated from few "
              "episodes, so their parameters are uncertain.",
              "- Factors are rolling 252-day z-scores: persistent stress decays toward zero inside the window, which "
              "can split a long crisis into several states.",
              "- PIT limitations from stage 3 remain: NFCI enters Liquidity / Credit only from 2011-11; daily FRED "
              "series rely on a one-day publication lag; the ETF universe was chosen ex post.",
              "- Conditional ETF statistics are descriptive and in-sample; a statistically good classification does "
              "not imply better portfolio performance.",
              "- Validation (2016-2019) is a relatively calm period; states that are rare in Validation are hard to "
              "assess out of sample."]

    lines += ["", "## 16. Implications for the Sticky HMM stage", "",
              "- HMM search range: K = " + (", ".join(str(k) for k in sel.get("candidate_k_range", []))
              or "see section 14") + " (the candidates not significantly worse on Validation), fixed now, before "
              "any HMM is fitted; the HMM is not restricted to the static models' preferred K.",
              "- The observed switching frequency and median run lengths of the static models are the benchmark that "
              "a sticky transition matrix should improve on without losing validation likelihood.",
              "- The HMM must output forward-filtered probabilities p(state_t | x_1..x_t) for decisions; smoothed "
              "probabilities may only be used for description.",
              "- Keep the Train-fitted scaler (or refit it inside each walk-forward window with the same rule), the "
              "seed policy and the label-ordering convention, and compare HMM states with the GMM states via ARI.",
              "- Carry the PIT and composition sensitivity analyses committed in the stage-3 protocol."]
    return "\n".join(lines) + "\n"
