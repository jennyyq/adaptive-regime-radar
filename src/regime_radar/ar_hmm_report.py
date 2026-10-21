"""Stage 5.5: figures and research report for the AR-HMM robustness experiment (Train/Validation only)."""

from __future__ import annotations

STAGE55_CODE_VERSION = "stage5.5-2026-10-21b"  # must match scripts/07b_compare_ar_hmm.py

from pathlib import Path

import numpy as np
import pandas as pd

from .ar_hmm import AR_MAX_ITER, AR_N_INIT, AR_TOL, PHI_BOUND, REG_VAR
from .ar_hmm_diagnostics import (ACF_LAGS, ADOPTION_PROCEDURE, ADOPTION_RULES, COVERAGE_LEVELS, ESTIMATION_AMENDMENTS,
                                 ModelResult, acf,
                                 pit_histogram, quantile_residuals)
from .quality import _md_table
from .regime_report import AXIS, GRID, INK, INK_2, MUTED, SHADE, STATE_COLORS, SURFACE, _fmt, _heatmap, _save, _style, _title

MODEL_COLORS = {"frozen_hmm_k3": "#2a78d6", "ar_hmm_k3": "#eb6834", "ar1": "#1baf7a", "ar_hmm_k2": "#eda100",
                "iid_gaussian": "#898781"}
MODEL_NAMES = {"frozen_hmm_k3": "A  frozen HMM K=3", "ar_hmm_k3": "B  AR-HMM K=3", "ar1": "C  AR(1)",
               "ar_hmm_k2": "D  AR-HMM K=2", "iid_gaussian": "ref  i.i.d. Gaussian"}


def _name(m: str) -> str:
    return MODEL_NAMES.get(m, m)


def _col(m: str) -> str:
    return MODEL_COLORS.get(m, MUTED)


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
def fig_model_comparison(comp: pd.DataFrame, daily_va: pd.DataFrame, frozen: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    t = comp.reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), facecolor=SURFACE, gridspec_kw={"width_ratios": [1, 1.3]})
    ax = axes[0]
    _style(ax, grid_y=False)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    y = np.arange(len(t))[::-1]
    ax.barh(y, t["validation_avg_log_score"], color=[_col(m) for m in t["model_id"]], height=0.6)
    for yy, (_, r) in zip(y, t.iterrows()):
        neg = r["validation_avg_log_score"] < 0
        ax.text(r["validation_avg_log_score"], yy, f"  {r['validation_avg_log_score']:.3f}  ", va="center",
                ha="left", fontsize=8, color="#ffffff" if neg else INK_2)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_yticks(y, [_name(m) for m in t["model_id"]], fontsize=8, color=INK_2)
    ax.set_xlabel("Validation average one-step predictive log score (per day)", fontsize=8, color=INK_2)
    ax.set_title("Validation log score (higher = better density forecast)", fontsize=10, color=INK, loc="left")
    ax = axes[1]
    _style(ax)
    base = "ar1" if "ar1" in daily_va.columns else frozen
    for m in daily_va.columns:
        if m in (base, frozen, "iid_gaussian"):
            continue
        ax.plot(daily_va.index, (daily_va[m] - daily_va[base]).cumsum(), color=_col(m), linewidth=1.6,
                label=_name(m))
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_ylabel(f"cumulative log-score difference vs {_name(base).strip()}", fontsize=8, color=INK_2)
    ax.set_title("Validation: what latent states add beyond AR(1) (cumulative)", fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False)
    _title(fig, "Predictive density comparison (parameters fixed at Train estimates)",
           "Identical scored dates for all models; each score uses only information through t-1. Validation was "
           "already used in Stage 5, so this is a robustness check, not a new holdout. Test not used.")
    _save(fig, path)


def fig_coverage(calib: pd.DataFrame, columns, path: Path) -> None:
    import matplotlib.pyplot as plt

    v = calib[calib["split"] == "validation"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), facecolor=SURFACE)
    ax = axes[0]
    _style(ax)
    levels = list(COVERAGE_LEVELS)
    ax.plot(levels, levels, color=INK, linestyle="--", linewidth=0.9, label="nominal")
    for m, g in v[v["factor"] == "all factors"].groupby("model_id"):
        ax.plot(levels, [g[f"coverage_{int(l * 100)}"].iloc[0] for l in levels], marker="o", color=_col(m),
                linewidth=1.8, label=_name(m))
    ax.set_xlabel("nominal central interval", fontsize=8, color=INK_2)
    ax.set_ylabel("empirical coverage (Validation, all factors)", fontsize=8, color=INK_2)
    ax.set_title("Predictive interval coverage (mixture quantiles)", fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False)
    ax = axes[1]
    _style(ax)
    models = list(dict.fromkeys(v["model_id"]))
    w = 0.8 / len(models)
    for i, m in enumerate(models):
        g = v[(v["model_id"] == m) & (v["factor"].isin(columns))].set_index("factor").reindex(columns)
        ax.bar(np.arange(len(columns)) + (i - (len(models) - 1) / 2) * w, g["coverage_90"], width=w, color=_col(m),
               label=_name(m))
    ax.axhline(0.9, color=INK, linestyle="--", linewidth=0.9)
    ax.set_xticks(range(len(columns)), columns, fontsize=8, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_title("90% interval coverage by factor (Validation)", fontsize=10, color=INK, loc="left")
    _title(fig, "Calibration of the factor predictive distributions",
           "Intervals come from the full state-weighted Gaussian mixture. This is calibration of observed factor "
           "forecasts, not of latent regime probabilities.")
    _save(fig, path)


def fig_residual_acf(models: dict[str, ModelResult], resid: pd.DataFrame, columns, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), facecolor=SURFACE)
    ax = axes[0]
    _style(ax)
    lags = range(1, 22)
    for m, r in models.items():
        q = quantile_residuals(r.validation)
        ax.plot(list(lags), [np.mean([acf(q[:, j], L) for j in range(q.shape[1])]) for L in lags], marker="o",
                markersize=3, color=_col(m), linewidth=1.6, label=_name(m))
    n = len(next(iter(models.values())).validation.x)
    ax.axhspan(-1.96 / np.sqrt(n), 1.96 / np.sqrt(n), color=SHADE, zorder=0)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xlabel("lag (trading days)", fontsize=8, color=INK_2)
    ax.set_ylabel("mean ACF of quantile residuals (5 factors)", fontsize=8, color=INK_2)
    ax.set_title("Serial dependence of one-step predictive residuals (Validation)", fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False)
    ax = axes[1]
    _style(ax)
    v = resid[resid["split"] == "validation"]
    ms = list(models)
    w = 0.8 / len(ms)
    for i, m in enumerate(ms):
        g = v[v["model_id"] == m].set_index("factor").reindex(columns)
        ax.bar(np.arange(len(columns)) + (i - (len(ms) - 1) / 2) * w, g["qres_acf_lag1"], width=w, color=_col(m),
               label=_name(m))
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xticks(range(len(columns)), columns, fontsize=8, rotation=15)
    ax.set_title("Lag-1 autocorrelation by factor (Validation)", fontsize=10, color=INK, loc="left")
    _title(fig, "Residual temporal dependence",
           "Quantile residuals = Phi^-1(PIT) of the one-step predictive mixture; shaded band = +/-1.96/sqrt(n). "
           "No hindsight-smoothed states are used.")
    _save(fig, path)


def fig_pit(models: dict[str, ModelResult], path: Path) -> None:
    import matplotlib.pyplot as plt

    ms = list(models)
    fig, axes = plt.subplots(1, len(ms), figsize=(3.3 * len(ms), 3.8), facecolor=SURFACE, sharey=True)
    for ax, m in zip(np.atleast_1d(axes), ms):
        _style(ax)
        h = pit_histogram(models[m], "validation", 10)
        ax.bar(np.arange(10) / 10 + 0.05, h, width=0.095, color=_col(m))
        ax.axhline(1.0, color=INK, linestyle="--", linewidth=0.9)
        ax.set_title(_name(m), fontsize=9.5, color=INK, loc="left")
        ax.set_xlim(0, 1)
        ax.set_xlabel("PIT", fontsize=8, color=INK_2)
    np.atleast_1d(axes)[0].set_ylabel("density (uniform = 1)", fontsize=8, color=INK_2)
    _title(fig, "PIT histograms of the one-step predictive mixtures (Validation, 5 factors pooled)",
           "U-shape = predictive intervals too narrow (overconfident); hump = too wide.")
    _save(fig, path)


def fig_probabilities(models: dict[str, ModelResult], frozen: str, cands: list[str], tr_index, va_index, path: Path
                      ) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    ids = [frozen] + [c for c in cands if c in models]
    fig = plt.figure(figsize=(13, 2.0 + 1.2 * len(ids) + 2.6), facecolor=SURFACE)
    gs = fig.add_gridspec(len(ids) + 1, 1, height_ratios=[0.6] * len(ids) + [2.2], hspace=0.6)
    idx = tr_index.append(va_index)
    for i, m in enumerate(ids):
        ax = fig.add_subplot(gs[i])
        r = models[m]
        lab = np.r_[r.train_labels, r.validation_labels]
        x = np.r_[idx.values, idx.values[-1] + np.timedelta64(1, "D")]
        ax.pcolormesh(x, [0, 1], lab[None, :], cmap=ListedColormap(STATE_COLORS[:r.k]), vmin=-0.5, vmax=r.k - 0.5,
                      shading="flat")
        ax.axvline(va_index[0], color=INK, linewidth=0.9, linestyle="--")
        ax.set_yticks([])
        ax.set_ylabel(_name(m), rotation=0, ha="right", va="center", fontsize=8, color=INK_2)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=7, length=0)
    ax = fig.add_subplot(gs[-1])
    _style(ax)
    for m in ids:
        ax.plot(va_index, models[m].p_stress("validation"), color=_col(m), linewidth=1.1, label=_name(m))
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("filtered P(stress states)", fontsize=8, color=INK_2)
    ax.set_title("Validation filtered stress probability (causal)", fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    _title(fig, "Frozen HMM vs AR-HMM: filtered states and stress probabilities",
           "Bands: argmax of filtered probabilities, Train (left of dashed line) and Validation. State ids are model-"
           "specific (ordered by Train stress score); compare with the profile figure, not by colour alone.")
    fig.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def fig_profiles(profiles: pd.DataFrame, ids: list[str], columns, path: Path) -> None:
    import matplotlib.pyplot as plt

    ar = [m for m in ids if f"phi_{columns[0]}" in profiles.columns
          and profiles.loc[profiles["model_id"] == m, f"phi_{columns[0]}"].notna().any()]
    n = len(ids) + len(ar)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 4.2), facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, ids):
        p = profiles[(profiles["model_id"] == m)]
        tr = p[p["split"] == "train"].set_index("state")
        va = p[p["split"] == "validation"].set_index("state")
        frame = tr[list(columns)].copy()
        frame.index = [f"{s} tr {tr.loc[s, 'share']:.0%}/va {va.loc[s, 'share']:.0%}" for s in tr.index]
        _heatmap(ax, frame, 2.0)
        ax.set_title(f"{_name(m)}: Train profile", fontsize=9.5, color=INK, loc="left")
    for ax, m in zip(axes[len(ids):], ar):
        p = profiles[(profiles["model_id"] == m) & (profiles["split"] == "train")].set_index("state")
        frame = p[[f"phi_{c}" for c in columns]].copy()
        frame.columns = list(columns)
        _heatmap(ax, frame, 1.0)
        ax.set_title(f"{_name(m)}: AR coefficients phi", fontsize=9.5, color=INK, loc="left")
    _title(fig, "State profiles (mean robust z-scores by filtered state) and AR coefficients",
           f"phi bounded to |phi| <= {PHI_BOUND}. Profiles are descriptive; labels are hypotheses.")
    _save(fig, path)


def fig_generalization(comp: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    t = comp.reset_index(drop=True)
    y = np.arange(len(t))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.5 * len(t) + 2.6), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axes[0]
    _style(ax, grid_y=False)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    for yy, (_, r) in zip(y, t.iterrows()):
        ax.plot([r["validation_avg_log_score"], r["train_avg_log_score"]], [yy, yy], color=AXIS, linewidth=1.5)
        ax.scatter(r["train_avg_log_score"], yy, color="#2a78d6", s=40, zorder=3)
        ax.scatter(r["validation_avg_log_score"], yy, color="#eb6834", s=40, zorder=3)
    ax.scatter([], [], color="#2a78d6", label="Train (in-sample)")
    ax.scatter([], [], color="#eb6834", label="Validation")
    ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.12), ncol=2)
    ax.set_yticks(y, [_name(m) for m in t["model_id"]], fontsize=8, color=INK_2)
    ax.set_title("Train vs Validation average log score", fontsize=10, color=INK, loc="left")
    ax = axes[1]
    _style(ax, grid_y=False)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.barh(y, t["train_validation_gap"], color=[_col(m) for m in t["model_id"]], height=0.6)
    ax.set_yticks(y, [""] * len(t))
    ax.set_title("Gap (Train - Validation, per day)", fontsize=10, color=INK, loc="left")
    _title(fig, "Train vs Validation generalization", "Same Train parameters and scaler; identical scored dates.")
    _save(fig, path)


def save_figures(directory: Path, models, comp, calib, resid, profiles, daily_va, frozen, cands, columns, tr_index,
                 va_index) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    directory.mkdir(parents=True, exist_ok=True)
    main = {m: models[m] for m in models if m != "iid_gaussian"}
    jobs = [("ar_hmm_model_comparison.png", lambda p: fig_model_comparison(comp, daily_va, frozen, p)),
            ("predictive_interval_coverage.png", lambda p: fig_coverage(calib, columns, p)),
            ("residual_autocorrelation_comparison.png", lambda p: fig_residual_acf(main, resid, columns, p)),
            ("predictive_pit_diagnostics.png", lambda p: fig_pit(main, p)),
            ("hmm_vs_ar_hmm_probabilities.png", lambda p: fig_probabilities(models, frozen, cands, tr_index, va_index, p)),
            ("ar_hmm_regime_factor_profiles.png", lambda p: fig_profiles(profiles, [frozen] + cands, columns, p)),
            ("train_validation_generalization.png", lambda p: fig_generalization(comp, p))]
    out = []
    for name, job in jobs:
        job(directory / name)
        out.append(name)
    return out


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------
def _pick(comp, m, col):
    r = comp[comp["model_id"] == m]
    return float(r[col].iloc[0]) if len(r) and col in r else np.nan


def render_report(ctx: dict) -> str:
    comp, calib, resid, init, regimes = ctx["comparison"], ctx["calibration"], ctx["residuals"], ctx["init"], ctx["regimes"]
    profiles, matches, decision, checks = ctx["profiles"], ctx["matches"], ctx["decision"], ctx["checks"]
    frozen, cand, ar1 = ctx["frozen_id"], ctx["candidate_id"], ctx["ar1_id"]
    data = ctx["data"]
    status = pd.Series([c.status for c in checks]).value_counts().to_dict()
    L = ["# Stage 5.5 - Autoregressive HMM robustness and predictive calibration", "",
         f"Train {data.train_features.index.min().date()}..{data.train_features.index.max().date()} "
         f"({len(data.train_features)} rows; scored from row 2); Validation {data.validation_features.index.min().date()}"
         f"..{data.validation_features.index.max().date()} ({len(data.validation_features)} rows). Test not read. "
         "Checks: " + ", ".join(f"{v} {k}" for k, v in status.items()) + ".", "",
         "**Status of the evidence:** Validation was already used for Stage 5 model selection. This experiment is a "
         "robustness check on the development sample, not an independent holdout. The Stage 5 model is frozen and "
         "was not refitted.", "", "## Decision", ""]
    L += [f"- Verdict: **{decision['verdict']}** -> model for Stage 6: **{decision['stage6_model']}**.",
          "- Pre-specified criteria: " + "; ".join(f"{k} {'PASS' if v else 'FAIL'}" for k, v in decision["checks"].items())
          + ".", ""]
    va = lambda m: _pick(comp, m, "validation_avg_log_score")  # noqa: E731
    L += ["## 1. Research question and design", "",
          "Can autoregressive emissions fix the temporal-dependence misspecification of the frozen HMM (within-state "
          "residual autocorrelation ~0.97, 90% interval coverage ~81%) while keeping stable, interpretable regimes? "
          "The comparison separates three sources of predictive information:", "",
          "- autoregression alone: C (AR(1)) vs the i.i.d. Gaussian reference;",
          "- latent states alone: A (frozen HMM) vs the i.i.d. Gaussian reference;",
          "- states on top of autoregression: B (AR-HMM) vs C. Only this last comparison measures what the latent "
          "regimes add once the obvious persistence of the factors is modelled.", "",
          "## 2. Models and likelihood", "",
          "- A: frozen Stage 5 `hmm_std_diag_k3` (parameters read from its JSON, no refit).",
          "- B: AR-HMM K=3: X_t = mu_k + Phi_k (X_t-1 - mu_k) + e_t, e_t ~ N(0, Sigma_k), Phi_k and Sigma_k diagonal, "
          "state-specific mu_k, Phi_k, Sigma_k, transition matrix A.",
          "- C: single-regime Gaussian AR(1) with diagonal Phi and Sigma (B with K=1; closed-form least squares).",
          "- D: AR-HMM K=2 (robustness only). Reference: i.i.d. Gaussian with Train mean and variance.",
          "- Conditional likelihood: the first Train row is used only as a lag; the chain starts at t=2 with pi. "
          "p(x_t | x_1..x_t-1) = sum_k P(Z_t=k | x_1..x_t-1) N(x_t; c_k + Phi_k x_t-1, Sigma_k).",
          f"- Estimation: EM on Train; M-step = per-state weighted least squares of x_t on (1, x_t-1) with |phi| <= "
          f"{PHI_BOUND} and innovation variance max(s2, {REG_VAR}) (exact constrained optimum, so EM is monotone; "
          "checked for every start and by a Q-function optimality test); "
          f"{len(ctx['seeds'])} seeds x {AR_N_INIT} starts, tol {AR_TOL}/obs, max {AR_MAX_ITER} iterations.",
          "- Filtering: log space; the predicted distribution and the score of day t are computed before x_t is used. "
          "Validation continues from the last Train filtered distribution with the last Train row as first lag.",
          "- Scoring: identical dates for every model (Train rows 2..T, all Validation rows); A's Train day 1 is "
          "filtered but not scored.", "",
          "## 3. Adoption criteria (fixed before results)", ""] + [f"- {p}" for p in ADOPTION_PROCEDURE]
    L += ["", "Thresholds: " + ", ".join(f"{k} = {v}" for k, v in ADOPTION_RULES.items()) + ".", "",
          "## 4. Convergence and initialization", ""]
    L += [_md_table(_fmt(init.set_index("model_id"), 3)), ""]
    L += ["Estimation amendments (the first-run outcome is recorded, not deleted):", ""]
    L += [f"- {a['version']} ({a['date']}): {a['finding']} Fix: {a['fix']} First run: {a['first_run_outcome']}"
          for a in ESTIMATION_AMENDMENTS] + [""]
    L += ["## 5. Predictive log scores", ""]
    cols = ["model_id", "k", "n_parameters", "train_avg_log_score", "validation_avg_log_score", "train_validation_gap",
            "converged", "phi_bound_share"]
    L += [_md_table(_fmt(comp[cols].set_index("model_id"), 3)), "", "Paired daily differences on Validation "
          "(Newey-West, 21 lags):", ""]
    pt = []
    for m in comp["model_id"]:
        for tag, ref in (("vs_frozen_hmm", frozen), ("vs_ar1", ar1)):
            col = f"validation_mean_diff_{tag}"
            if col in comp and pd.notna(_pick(comp, m, col)) and m != ref:
                pt.append({"model": m, "versus": ref, "mean_diff": _pick(comp, m, col),
                           "nw_t": _pick(comp, m, f"validation_nw_t_{tag}"),
                           "ci95_low": _pick(comp, m, f"validation_ci95_low_{tag}"),
                           "ci95_high": _pick(comp, m, f"validation_ci95_high_{tag}"),
                           "share_days_better": _pick(comp, m, f"validation_share_days_better_{tag}")})
    L += [_md_table(_fmt(pd.DataFrame(pt).set_index("model"), 3)), ""]
    iid = va("iid_gaussian")
    L += ["Decomposition of the Validation log score (per day, relative to the i.i.d. Gaussian reference):", "",
          f"- latent states only (A - ref): {va(frozen) - iid:+.3f}",
          f"- autoregression only (C - ref): {va(ar1) - iid:+.3f}",
          f"- autoregression + states (B - ref): {va(cand) - iid:+.3f}",
          f"- states beyond autoregression (B - C): {va(cand) - va(ar1):+.3f}", ""]
    L += ["## 6. Residual temporal dependence", "",
          "Quantile residuals r = Phi^-1(PIT) of the one-step predictive mixture (no hindsight states). Under a correct "
          "model they are i.i.d. N(0,1). within_state_resid_acf1 repeats the Stage 5 metric (filtered argmax state).", ""]
    rv = resid.groupby(["model_id", "split"]).agg(**{f"mean_acf_lag{L_}": (f"qres_acf_lag{L_}", "mean") for L_ in ACF_LAGS},
                                                  mean_sq_acf_lag1=("qres_sq_acf_lag1", "mean"),
                                                  max_ljung_box_p21=("ljung_box_p21", "max"),
                                                  mean_qres_std=("qres_std", "mean"),
                                                  mean_within_state_acf1=("within_state_resid_acf1", "mean"))
    L += [_md_table(_fmt(rv.reset_index().set_index("model_id"), 3)), "", "By factor (Validation):", ""]
    L += [_md_table(_fmt(resid[resid["split"] == "validation"].drop(columns=["split"]).set_index("model_id"), 3)), ""]
    L += ["## 7. Predictive calibration", "",
          "Coverage of central intervals of the full mixture predictive distribution (equivalent to PIT inside "
          "[(1-p)/2, (1+p)/2]); widths in standardized units (sharpness); CRPS per factor (lower = better).", ""]
    cv = calib[calib["factor"] == "all factors"].drop(columns=["factor"])
    L += [_md_table(_fmt(cv.set_index("model_id"), 3)), "", "By factor (Validation, 90%):", ""]
    pf = calib[(calib["split"] == "validation") & (calib["factor"] != "all factors")].pivot_table(
        index="model_id", columns="factor", values="coverage_90")
    L += [_md_table(_fmt(pf, 3)), "",
          "This is calibration of observed factor forecasts. Without observed regime labels, no model's P(stress) can "
          "be called a calibrated crisis probability.", ""]
    L += ["## 8. Probability extremeness and regime quality", ""]
    L += [_md_table(_fmt(regimes.set_index("model_id"), 3)), ""]
    L += ["## 9. Economic interpretation of the AR-HMM states", ""]
    show = profiles[profiles["model_id"].isin([frozen, cand, ctx.get("k2_id")])]
    base_cols = ["model_id", "state", "split", "share", *ctx["columns"], "is_stress_state", "hypothesis_label",
                 "train_validation_profile_distance"]
    L += [_md_table(_fmt(show[base_cols].set_index("model_id"), 2)), "", "AR coefficients and innovation s.d. "
          "(original units) of the AR models (Train estimates):", ""]
    arcols = ["model_id", "state"] + [c for c in profiles.columns if c.startswith(("phi_", "innov_sd_"))]
    ar_rows = profiles[(profiles["split"] == "train") & profiles["model_id"].isin([cand, ctx.get("k2_id")])]
    L += [_md_table(_fmt(ar_rows[arcols].set_index("model_id"), 3)), ""]
    L += ["State matching with the frozen HMM (label-invariant ARI; economic matching by Train profile distance):", ""]
    mt = pd.DataFrame([{k: v for k, v in m.items() if k != "crosstab_train"} for m in matches])
    L += [_md_table(_fmt(mt.set_index("model_id"), 3)), ""]
    L += ["## 10. Causality and leakage checks", "", _md_table(pd.DataFrame([vars(c) for c in checks]).set_index("name")), ""]
    L += ["## 11. Limitations", "",
          "- Validation reused from Stage 5 (development sample); conclusions are robustness evidence, not a holdout test.",
          "- The factors are rolling-window statistics (21-252 day windows). Overlapping windows induce moving-average "
          "dependence at many lags that an AR(1) cannot fully remove; residual autocorrelation at lags 5 and 21 shows how "
          "much remains.",
          "- Diagonal Phi and Sigma ignore cross-factor dynamics and contemporaneous correlation.",
          f"- |phi| <= {PHI_BOUND} keeps state means identified; bound hits are reported (phi_bound_share).",
          "- Better density forecasts of the factors do not imply better crisis prediction or portfolio returns.", ""]
    L += ["## 12. Recommendation for Stage 6", "", f"- {decision['stage6_model']} (verdict {decision['verdict']}).",
          "- Stage 5 selection results and files are unchanged (hash check above).", ""]
    L += ["## Figures", ""] + [f"- `reports/figures/ar_hmm/{f}`" for f in ctx["figures"]]
    return "\n".join(L) + "\n"
