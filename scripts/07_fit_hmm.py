"""Stage 5: Standard and Sticky Gaussian HMMs on Train, causal filtering and model selection on Validation.

Reads ONLY data/splits/{train,validation}_{features,returns}.csv and split_manifest.json, the Stage 4 GMM
reference pickle (predictions only) and, for the late-Train composition sensitivity, the rows of
data/processed/factor_components.csv up to the last Train date. The test split is never opened. The Stage 4
GMM reference is rebuilt from Train (the pickle is only compared when written by the same scikit-learn).

Outputs
  data/processed/hmm/
      hmm_model_comparison.csv, hmm_train_validation_comparison.csv, hmm_validation_loglikelihood.csv,
      hmm_transition_matrices.csv, hmm_state_durations.csv, hmm_state_profiles.csv,
      hmm_initialization_stability.csv, hmm_seed_fits.csv, hmm_regime_asset_statistics.csv,
      hmm_eligibility.csv, hmm_selection_tests.csv, hmm_sticky_assessment.csv,
      hmm_detection_delay.csv, hmm_detection_event_lags.csv, hmm_reference_episodes.csv, hmm_event_study.csv,
      hmm_paired_delay.csv,
      hmm_vs_gmm_comparison.csv, hmm_late_train_sensitivity.csv, hmm_checks.csv,
      hmm_window_sensitivity.csv, hmm_phase_log_scores.csv, hmm_calibration_{pit,reliability,summary,decision}.csv,
      hmm_residual_autocorrelation.csv, hmm_covariance_inflation.csv, hmm_warnings.csv,
      validation_hmm_decision_probabilities.csv (only if the recalibration is adopted),
      train_hmm_filtered_probabilities.csv, validation_hmm_filtered_probabilities.csv,
      validation_hmm_predicted_probabilities.csv, train_hmm_state_assignments.csv,
      validation_hmm_state_assignments.csv, hmm_selection_manifest.json, models/<model_id>.json
  reports/hmm_model_selection.md
  reports/figures/hmm/*.png (11 figures)
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regime_radar.paths import PROCESSED, REPORTS  # noqa: E402

SPLITS_DIR = ROOT / "data" / "splits"
STAGE5_CODE_VERSION = "stage5-2026-10-21c"   # all src/regime_radar/hmm_*.py modules must carry the same value
OUT_DIR = PROCESSED / "hmm"
FIGURE_DIR = REPORTS / "figures" / "hmm"
STAGE4_DIR = PROCESSED / "regime_exploration"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-figures", action="store_true", help="skip matplotlib figures")
    parser.add_argument("--n-seeds", type=int, default=10, help="number of protocol seeds (protocol: 10)")
    parser.add_argument("--n-init", type=int, default=None, help="EM starts per seed (protocol: 5)")
    args = parser.parse_args()

    missing = []
    for module, package in (("numpy", "numpy"), ("pandas", "pandas"), ("scipy", "scipy"), ("sklearn", "scikit-learn")):
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    if missing:
        print("Missing dependencies. Install them into the interpreter you use for this project:\n"
              f"  {sys.executable} -m pip install {' '.join(missing)}\nNothing was run.")
        return 2
    try:
        import hmmlearn  # noqa: F401
        print("hmmlearn found: tests/test_hmm.py also cross-checks the likelihood against it.")
    except ImportError:
        print("hmmlearn not installed (optional; the HMM is implemented and tested here). For the optional "
              f"cross-check:\n  {sys.executable} -m pip install hmmlearn")

    from regime_radar import hmm_diagnostics as hd
    from regime_radar import hmm_filtering as hf
    from regime_radar import hmm_models as hm
    from regime_radar import hmm_calibration as hcal
    from regime_radar import hmm_report as hr
    from regime_radar import hmm_robustness as hrob
    from regime_radar import regime_exploration as rx
    from regime_radar.dataset import FEATURE_COLUMNS

    stale = [m for m in (hm, hf, hd, hr, hcal, hrob) if getattr(m, "STAGE5_CODE_VERSION", None) != STAGE5_CODE_VERSION]
    if stale:
        print(f"Stage 5 modules do not match this script (expected version {STAGE5_CODE_VERSION}):")
        for m in stale:
            print(f"  {m.__file__}: version {getattr(m, 'STAGE5_CODE_VERSION', 'missing (older file)')}")
        print("Re-extract the latest stage5 zip inside the project folder, delete __pycache__ folders, and rerun.")
        return 3
    if not (SPLITS_DIR / "split_manifest.json").exists():
        raise SystemExit("data/splits/split_manifest.json not found. Run scripts/05_prepare_datasets.py first.")

    started = time.perf_counter()
    recorder = warnings.catch_warnings(record=True)
    caught = recorder.__enter__()          # every library warning is recorded, printed once and saved
    warnings.simplefilter("default")
    data = rx.load_stage4_data(SPLITS_DIR)
    tr_feat, va_feat = data.train_features[FEATURE_COLUMNS], data.validation_features[FEATURE_COLUMNS]
    print(f"Loaded {', '.join(data.files_read)}")
    print(f"Train {tr_feat.index.min().date()}..{tr_feat.index.max().date()} ({len(tr_feat)} rows); Validation "
          f"{va_feat.index.min().date()}..{va_feat.index.max().date()} ({len(va_feat)} rows)")

    checks = rx.input_checks(data)
    checks.append(rx.frozen_factor_check(data.manifest, ROOT / "src" / "regime_radar" / "factors.py"))
    bad_hash = [n for n, ok in data.hash_status.items() if not ok]
    checks.append(hd.Check("Split files match split_manifest.json hashes", hd.FAIL if bad_hash else hd.PASS,
                           ", ".join(bad_hash) or "4 files verified"))
    blocking = [c for c in checks if c.status == hd.FAIL]
    if blocking:
        for c in blocking:
            print(f"  [FAIL] {c.name}: {c.detail}")
        print("Input checks failed; no model was fitted.")
        return 1

    # ---------------- Train-only preprocessing, identical to Stage 4 ----------------
    apply, reason = rx.scaling_decision(rx.describe_train(tr_feat))
    scaler = rx.TrainScaler(apply).fit(tr_feat)
    scaler_dict = scaler.to_dict()
    s4_manifest = STAGE4_DIR / "regime_exploration_manifest.json"
    if s4_manifest.exists():
        s4 = pd.read_json(s4_manifest, typ="series")["preprocessing"]
        same = s4["applied"] == scaler_dict["applied"] and all(
            abs(s4["mean"][c] - scaler_dict["mean"][c]) < 1e-8 and abs(s4["scale"][c] - scaler_dict["scale"][c]) < 1e-8
            for c in FEATURE_COLUMNS)
        checks.append(hd.Check("Preprocessing identical to the Stage 4 Train scaler", hd.PASS if same else hd.FAIL,
                               f"{scaler_dict['method']}; Stage 4 manifest {'matches' if same else 'differs'}"))
    else:
        checks.append(hd.Check("Preprocessing identical to the Stage 4 Train scaler", hd.NOT_RUN,
                               "regime_exploration_manifest.json not found"))
    print(f"Scaling: {reason}")
    x_tr, x_va = scaler.transform(tr_feat), scaler.transform(va_feat)
    mean, scale = scaler.mean_[FEATURE_COLUMNS].to_numpy(), scaler.scale_[FEATURE_COLUMNS].to_numpy()
    train_original = tr_feat.to_numpy(float)

    # ---------------- Fit every candidate on Train ----------------
    seeds = tuple(rx.SEEDS[: args.n_seeds])
    n_init = args.n_init or hm.HMM_N_INIT
    specs = hm.candidate_specs()
    print(f"Fitting {len(specs)} HMM candidates x {len(seeds)} seeds x {n_init} EM starts on Train "
          "(this takes several minutes) ...")
    done = []

    def progress(c):
        done.append(c.model_id)
        ref = c.reference
        print(f"  [{len(done):>2}/{len(specs)}] {c.model_id:<26} train LL/obs {ref.train_loglik / ref.n_train:7.3f}  "
              f"val {c.validation.log_score.mean():7.3f}  A_ii {' '.join(f'{a:.3f}' for a in ref.params.transmat.diagonal())}"
              f"  {sum(f.converged for f in c.seed_fits)}/{len(seeds)} converged  "
              f"{sum(f.seconds for f in c.seed_fits):5.1f}s")

    cands = hd.run_all(x_tr, x_va, mean, scale, train_original, seeds, specs, progress, n_init=n_init)

    comp = pd.DataFrame([hd.comparison_row(c, len(FEATURE_COLUMNS)) for c in cands.values()])
    stab = pd.DataFrame([hd.stability_row(c) for c in cands.values()])
    seed_tab = hd.seed_table(cands)
    tv = hd.train_validation_table(comp, stab)
    trans = hd.transition_table(cands)
    durations = hd.duration_table(cands, {"train": tr_feat.index, "validation": va_feat.index})
    profiles = hd.profile_table(cands, tr_feat, va_feat, mean, scale)
    assets = hd.asset_table(cands, data.train_returns, data.validation_returns)
    p_tr, p_va, pred_va, lab_tr, lab_va = hd.probability_frames(cands, tr_feat.index, va_feat.index)

    # ---------------- Stage 4 GMM reference, rebuilt from Train (no cross-version unpickling) ----------------
    gmm_ref, gmm_info, gmm_status = hrob.rebuild_gmm_reference(x_tr, x_va, tr_feat, STAGE4_DIR / "models")
    gmm = gmm_ref
    gmm_tr, gmm_va = gmm_ref["train"], gmm_ref["validation"]
    diff = gmm_info.get("max_abs_score_diff")
    checks.append(hd.Check("Stage 4 GMM reference rebuilt from Train (predictions only)",
                           hd.PASS if diff is None or diff < 1e-6 else hd.WARN, gmm_status))

    # ---------------- Detection delay ----------------
    events = hd.reference_events(tr_feat, va_feat, None if gmm is None else
                                 {"train": gmm_tr["labels"], "validation": gmm_va["labels"]})
    lags, det_summary, onsets = hd.detection_tables(cands, events, tr_feat.index, va_feat.index)
    study = hd.event_study(cands, events)
    paired = hd.paired_delay_table(cands)

    # ---------------- Selection ----------------
    scores = hd.daily_score_frame(cands, va_feat.index, None if gmm is None else gmm_va["score"])
    elig = hd.eligibility(comp, stab)
    sel = hd.select_model(comp, elig, scores[[m for m in cands]], det_summary, paired)
    selected = sel.get("selected")
    comp_out = comp.merge(elig, on="model_id").merge(
        sel["table"][["model_id", "mean_daily_shortfall_vs_best", "nw_t_vs_best", "competitive"]]
        if "table" in sel else pd.DataFrame({"model_id": comp["model_id"]}), on="model_id", how="left")
    comp_out["selected"] = comp_out["model_id"] == selected

    gmm_cmp = pd.DataFrame()
    if selected:
        ids = list(dict.fromkeys([selected, sel.get("standard_counterpart"), sel.get("best_validation_model")]))
        gmm_cmp = pd.concat([hd.compare_with_gmm(cands[m], gmm_tr, gmm_va) for m in ids if m], ignore_index=True)

    # ---------------- Robustness: training windows and market phases ----------------
    stable_from, late_note = hd.composition_stable_from(PROCESSED / "factor_components.csv", tr_feat.index.max(),
                                                        tr_feat.index.min())
    windows = hrob.window_definitions(tr_feat.index, stable_from)
    robust_ids = list(dict.fromkeys([m for m in ("hmm_std_diag_k2", "hmm_std_diag_k3", "hmm_std_diag_k4",
                                                 selected, sel.get("best_overall_validation_model")) if m in cands]))
    print(f"Training-window sensitivity: {len(windows) - 1} extra windows x {len(robust_ids)} models "
          f"({hrob.WINDOW_SEEDS} seeds x {n_init} starts) ...")
    window_tab = hrob.window_sensitivity(cands, robust_ids, x_tr, x_va, tr_feat.index, windows, seeds, mean, scale,
                                         train_original, n_init)
    late = window_tab[window_tab["window"] == "post_composition"]
    phases = hrob.phase_table(cands, robust_ids, tr_feat.index, va_feat.index, tr_feat, va_feat, gmm)
    checks.append(hd.Check("Training-window sensitivity", hd.PASS if window_tab["converged"].all() else hd.WARN,
                           f"windows {', '.join(f'{k} {v[0].date()}..{v[1].date()}' for k, v in windows.items())}; "
                           f"{late_note}"))

    # ---------------- Probability calibration ----------------
    calib_ids = list(dict.fromkeys([m for m in (selected, "hmm_std_diag_k2", "hmm_std_diag_k3", "hmm_std_diag_k4",
                                                sel.get("best_overall_validation_model")) if m in cands]))
    calib = hcal.calibration_report(cands, calib_ids, x_tr, x_va, va_feat.index, FEATURE_COLUMNS)
    dec = calib["decisions"].get(selected) if selected else None
    if dec:
        checks.append(hd.Check("Calibration: covariance inflation chosen on 2016-17, tested on 2018-19", hd.PASS,
                               f"{selected}: c* = {dec['chosen_inflation_on_val_a']:g}, 2018-19 gain "
                               f"{dec['val_b_mean_gain']:+.3f}/day (NW t {dec['val_b_nw_t']:.2f}); adopted "
                               f"{dec['adopted']} -> decision inflation {dec['decision_inflation']:g}"))

    checks += hd.protocol_checks(cands, selected, x_tr, x_va, data.files_read)
    warn_df = summarize_warnings(caught)
    checks.append(hd.Check("No library warnings during fitting and scoring", hd.PASS if warn_df.empty else hd.WARN,
                           "none" if warn_df.empty else "; ".join(f"{r.category}: {r.message[:120]} (x{r.count})"
                                                                  for r in warn_df.itertuples())))

    extras = {"calibration": calib, "calibration_models": calib_ids, "windows": window_tab,
              "window_definitions": {k: [str(v[0].date()), str(v[1].date())] for k, v in windows.items()},
              "phases": phases, "gmm_info": gmm_info, "n_init": n_init, "software": rx.software_versions(),
              "stability": stab}

    # ---------------- Outputs ----------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "models").mkdir(exist_ok=True)
    tables = {
        "hmm_model_comparison.csv": comp_out, "hmm_train_validation_comparison.csv": tv,
        "hmm_transition_matrices.csv": trans, "hmm_state_durations.csv": durations,
        "hmm_state_profiles.csv": profiles, "hmm_initialization_stability.csv": stab, "hmm_seed_fits.csv": seed_tab,
        "hmm_regime_asset_statistics.csv": assets, "hmm_eligibility.csv": elig,
        "hmm_selection_tests.csv": sel.get("table", pd.DataFrame())[
            ["model_id", "validation_avg_loglik", "mean_daily_shortfall_vs_best", "nw_se", "nw_t_vs_best",
             "eligible", "competitive"]] if "table" in sel else pd.DataFrame(),
        "hmm_sticky_assessment.csv": sel.get("sticky_assessment", pd.DataFrame()),
        "hmm_detection_delay.csv": det_summary, "hmm_detection_event_lags.csv": lags,
        "hmm_reference_episodes.csv": onsets, "hmm_event_study.csv": study, "hmm_paired_delay.csv": paired,
        "hmm_vs_gmm_comparison.csv": gmm_cmp, "hmm_late_train_sensitivity.csv": late,
        "hmm_window_sensitivity.csv": window_tab, "hmm_phase_log_scores.csv": phases,
        "hmm_calibration_pit.csv": calib["pit"], "hmm_calibration_reliability.csv": calib["reliability"],
        "hmm_calibration_summary.csv": calib["reliability_summary"],
        "hmm_residual_autocorrelation.csv": calib["residual_autocorrelation"],
        "hmm_covariance_inflation.csv": calib["inflation_curve"],
        "hmm_calibration_decision.csv": calib["inflation_decision"],
        "hmm_checks.csv": pd.DataFrame([vars(c) for c in checks]),
    }
    for name, frame in tables.items():
        frame.to_csv(OUT_DIR / name, index=False, float_format="%.6g")
    for name, frame in (("train_hmm_filtered_probabilities.csv", p_tr), ("validation_hmm_filtered_probabilities.csv", p_va),
                        ("validation_hmm_predicted_probabilities.csv", pred_va),
                        ("train_hmm_state_assignments.csv", lab_tr), ("validation_hmm_state_assignments.csv", lab_va),
                        ("hmm_validation_loglikelihood.csv", scores)):
        frame.to_csv(OUT_DIR / name, date_format="%Y-%m-%d", float_format="%.8g")
    if dec and dec["adopted"]:
        cal_params = hm.inflate_covariance(cands[selected].reference.params, dec["decision_inflation"])
        _, va_cal = hf.filter_train_validation(cal_params, x_tr, x_va)
        k = cands[selected].spec.k
        pd.DataFrame({**{f"{selected}__filtered_S{j + 1}": va_cal.filtered[:, j] for j in range(k)},
                      **{f"{selected}__predicted_S{j + 1}": va_cal.predicted[:, j] for j in range(k)}},
                     index=va_feat.index).to_csv(OUT_DIR / "validation_hmm_decision_probabilities.csv",
                                                 date_format="%Y-%m-%d", float_format="%.8g")
    hashes = {n: data.manifest.get("split_files", {}).get(n) for n in data.files_read if n.endswith(".csv")}
    for model_id, c in cands.items():
        hd.save_json(hd.model_metadata(c, data.manifest, scaler_dict, tr_feat.index, hashes, STAGE5_CODE_VERSION),
                     OUT_DIR / "models" / f"{model_id}.json")

    figures = []
    if not args.no_figures:
        try:
            figures = hr.save_figures(FIGURE_DIR, cands, comp_out, tv, det_summary, study, sel, gmm_cmp, scores,
                                      profiles, durations, tr_feat.index, va_feat.index,
                                      {"train": gmm_tr, "validation": gmm_va}, extras)
        except ImportError:
            print(f"matplotlib not installed; figures skipped ({sys.executable} -m pip install matplotlib)")
    warn_df = summarize_warnings(caught)
    warn_df.to_csv(OUT_DIR / "hmm_warnings.csv", index=False)
    recorder.__exit__(None, None, None)
    written = sorted(str(p.relative_to(ROOT)) for p in OUT_DIR.rglob("*") if p.is_file())
    manifest = hr.selection_manifest(data, scaler_dict, reason, seeds, sel, comp_out, checks, events, late_note,
                                     gmm_status, written, STAGE5_CODE_VERSION, extras)
    hd.save_json(manifest, OUT_DIR / "hmm_selection_manifest.json")
    report = hr.render_report(data, scaler_dict, reason, seeds, cands, comp_out, tv, stab, trans, durations, profiles,
                              assets, det_summary, onsets, events, sel, gmm_cmp, late, late_note, checks, figures,
                              paired, extras)
    (REPORTS / "hmm_model_selection.md").write_text(report, encoding="utf-8")

    # ---------------- Console summary ----------------
    print("\nValidation average predictive log score (Train parameters, causal filter):")
    show = comp_out.sort_values(["k", "covariance_type", "family", "prior_weight"])
    for r in show.itertuples():
        mark = "  <- selected" if r.model_id == selected else ""
        print(f"  {r.model_id:<26} train {r.train_avg_loglik:7.3f}  val {r.validation_avg_loglik:7.3f}  "
              f"gap {r.train_validation_gap:6.3f}  val switches/yr {r.validation_switches_per_year:5.1f}  "
              f"eligible {str(r.eligible):<5} t vs best eligible {r.nw_t_vs_best:6.2f}{mark}")
    print(f"Best Validation model (eligible): {sel.get('best_validation_model')}; best overall: "
          f"{sel.get('best_overall_validation_model')}; selected: {selected}")
    print(f"  {sel.get('reason')}")
    if not gmm_cmp.empty:
        g = gmm_cmp[(gmm_cmp["model_id"] == selected) & (gmm_cmp["split"] == "validation")].iloc[0]
        print(f"  vs Stage 4 GMM diag K=2 (Validation): mean daily log-score difference {g.mean_daily_difference:+.3f} "
              f"(NW t {g.nw_t:.2f}); state ARI {g.ari_hmm_vs_gmm_states:.2f}")
    if dec:
        print(f"  Calibration ({selected}): covariance inflation chosen on 2016-17 = {dec['chosen_inflation_on_val_a']:g}; "
              f"2018-19 gain {dec['val_b_mean_gain']:+.3f}/day (NW t {dec['val_b_nw_t']:.2f}); adopted {dec['adopted']}")
    ph = sel.get("phase_consistency")
    if ph:
        print(f"  Phase consistency {ph['larger']} vs {ph['smaller']}: positive in {ph['positive_years']}/{ph['n_years']} "
              "Validation years: " + ", ".join(f"{y} {v:+.3f}" for y, v in ph["yearly_advantage"].items()))
    print("Training-window sensitivity (Validation log score per day):")
    piv = window_tab.pivot_table(index="model_id", columns="window", values="validation_avg_loglik")
    for m, r in piv.iterrows():
        print(f"  {m:<26} " + "  ".join(f"{w} {v:7.3f}" for w, v in r.items()))
    if not warn_df.empty:
        print("Library warnings (recorded in hmm_warnings.csv):")
        for r in warn_df.itertuples():
            print(f"  {r.category} x{r.count}: {r.message[:200]}")
    print("Checks:")
    for c in checks:
        print(f"  [{c.status}] {c.name}: {c.detail}")
    counts = pd.Series([c.status for c in checks]).value_counts()
    print("Summary: " + ", ".join(f"{counts.get(s, 0)} {s}" for s in (hd.PASS, hd.WARN, hd.FAIL, hd.NOT_RUN)))
    print(f"Report: {REPORTS / 'hmm_model_selection.md'}; figures: {len(figures)} in {FIGURE_DIR}")
    print(f"Elapsed {time.perf_counter() - started:.0f}s. Stage 5 complete; later stages were not run.")
    return 1 if counts.get(hd.FAIL, 0) else 0


def summarize_warnings(caught) -> pd.DataFrame:
    rows = {}
    for w in caught:
        key = (w.category.__name__, str(w.message), f"{Path(w.filename).name}:{w.lineno}")
        rows[key] = rows.get(key, 0) + 1
    return pd.DataFrame([{"category": c, "message": m, "location": loc, "count": n} for (c, m, loc), n in rows.items()],
                        columns=["category", "message", "location", "count"])


if __name__ == "__main__":
    sys.exit(main())
