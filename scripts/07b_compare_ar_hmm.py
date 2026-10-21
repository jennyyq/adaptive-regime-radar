"""Stage 5.5: AR-HMM robustness experiment against the frozen Stage 5 Gaussian HMM (Train/Validation only).

Reads ONLY data/splits/{train,validation}_{features,returns}.csv and split_manifest.json, plus the frozen Stage 5
model data/processed/hmm/models/<selected>.json and data/processed/hmm/hmm_selection_manifest.json (read-only;
their hashes are checked before and after). The test split is never opened. Nothing in data/processed/hmm/ is
written.

Models: A frozen HMM K=3 (no refit), B AR-HMM K=3, C single-regime AR(1), D AR-HMM K=2, reference i.i.d. Gaussian.

Outputs
  data/processed/ar_hmm/
      ar_hmm_model_comparison.csv, ar_hmm_predictive_calibration.csv, ar_hmm_residual_diagnostics.csv,
      ar_hmm_state_profiles.csv, ar_hmm_transition_matrices.csv, ar_hmm_initialization_stability.csv,
      ar_hmm_regime_summary.csv, ar_hmm_state_matching.csv, ar_hmm_validation_probabilities.csv,
      ar_hmm_daily_log_scores.csv, ar_hmm_checks.csv, ar_hmm_selection_manifest.json, models/<model_id>.json
  reports/ar_hmm_robustness.md
  reports/figures/ar_hmm/*.png (7 figures)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regime_radar.paths import PROCESSED, REPORTS  # noqa: E402

SPLITS_DIR = ROOT / "data" / "splits"
STAGE55_CODE_VERSION = "stage5.5-2026-10-21b"   # src/regime_radar/ar_hmm*.py must carry the same value
STAGE5_DIR = PROCESSED / "hmm"
OUT_DIR = PROCESSED / "ar_hmm"
FIGURE_DIR = REPORTS / "figures" / "ar_hmm"
EXPECTED_FROZEN = "hmm_std_diag_k3"
FROZEN_ID, CAND_ID, AR1_ID, K2_ID = "frozen_hmm_k3", "ar_hmm_k3", "ar1", "ar_hmm_k2"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--n-seeds", type=int, default=10, help="protocol seeds (protocol: 10)")
    parser.add_argument("--n-init", type=int, default=None, help="EM starts per seed (protocol: 5)")
    args = parser.parse_args()

    missing = []
    for module, package in (("numpy", "numpy"), ("pandas", "pandas"), ("scipy", "scipy"), ("sklearn", "scikit-learn")):
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    if missing:
        print(f"Missing dependencies. Install into this interpreter:\n  {sys.executable} -m pip install "
              f"{' '.join(missing)}\nNothing was run.")
        return 2

    from regime_radar import ar_hmm as ah
    from regime_radar import ar_hmm_diagnostics as ad
    from regime_radar import ar_hmm_report as ar
    from regime_radar import regime_exploration as rx
    from regime_radar.dataset import FEATURE_COLUMNS
    from regime_radar.factors import Check
    from regime_radar.hmm_filtering import forward_filter
    from regime_radar.hmm_models import HMMParams

    stale = [m for m in (ah, ad, ar) if getattr(m, "STAGE55_CODE_VERSION", None) != STAGE55_CODE_VERSION]
    if stale:
        print(f"Stage 5.5 modules do not match this script (expected {STAGE55_CODE_VERSION}): "
              + ", ".join(m.__file__ for m in stale))
        print("Re-extract the latest stage5.5 zip, delete __pycache__ folders and rerun.")
        return 3
    PASS, WARN, FAIL, NOT_RUN = "PASS", "WARN", "FAIL", "NOT RUN"

    manifest_path = STAGE5_DIR / "hmm_selection_manifest.json"
    if not manifest_path.exists():
        raise SystemExit("data/processed/hmm/hmm_selection_manifest.json not found. Run scripts/07_fit_hmm.py first.")
    started = time.perf_counter()
    recorder = warnings.catch_warnings(record=True)
    caught = recorder.__enter__()
    warnings.simplefilter("default")
    hashes_before = ad.directory_hashes(STAGE5_DIR)

    stage5 = json.loads(manifest_path.read_text(encoding="utf-8"))
    frozen_name = stage5.get("selection", {}).get("selected")
    checks = [Check("Frozen Stage 5 model is the documented baseline", PASS if frozen_name == EXPECTED_FROZEN else WARN,
                    f"Stage 5 manifest selected {frozen_name}; expected {EXPECTED_FROZEN}")]
    model_path = STAGE5_DIR / "models" / f"{frozen_name}.json"
    frozen_meta = json.loads(model_path.read_text(encoding="utf-8"))

    data = rx.load_stage4_data(SPLITS_DIR)
    tr_feat, va_feat = data.train_features[FEATURE_COLUMNS], data.validation_features[FEATURE_COLUMNS]
    print(f"Loaded {', '.join(data.files_read)}")
    checks += rx.input_checks(data)
    checks.append(rx.frozen_factor_check(data.manifest, ROOT / "src" / "regime_radar" / "factors.py"))
    bad = [n for n, ok in data.hash_status.items() if not ok]
    checks.append(Check("Split files match split_manifest.json hashes", FAIL if bad else PASS, ", ".join(bad) or "4 files"))

    apply, reason = rx.scaling_decision(rx.describe_train(tr_feat))
    scaler = rx.TrainScaler(apply).fit(tr_feat)
    sd, fz = scaler.to_dict(), frozen_meta["preprocessing"]
    same = sd["applied"] == fz["applied"] and all(abs(sd["mean"][c] - fz["mean"][c]) < 1e-8 and
                                                  abs(sd["scale"][c] - fz["scale"][c]) < 1e-8 for c in FEATURE_COLUMNS)
    checks.append(Check("Train-only scaler identical to the frozen model's preprocessing", PASS if same else FAIL,
                        sd["method"]))
    if any(c.status == FAIL for c in checks):
        for c in checks:
            if c.status == FAIL:
                print(f"  [FAIL] {c.name}: {c.detail}")
        print("Input checks failed; nothing fitted.")
        return 1
    x_tr, x_va = scaler.transform(tr_feat), scaler.transform(va_feat)
    mean, scale = scaler.mean_[FEATURE_COLUMNS].to_numpy(), scaler.scale_[FEATURE_COLUMNS].to_numpy()

    # ---------------- Model A: frozen HMM ----------------
    params_a = HMMParams.from_dict(frozen_meta["parameters"])
    stress_a = [int(s[1:]) - 1 for s in frozen_meta.get("stress_states", [])]
    models = {FROZEN_ID: ad.frozen_hmm_result(params_a, x_tr, x_va, stress_a, FROZEN_ID)}
    s5 = pd.read_csv(STAGE5_DIR / "hmm_model_comparison.csv")
    s5_val = float(s5.loc[s5["model_id"] == frozen_name, "validation_avg_loglik"].iloc[0])
    rep = models[FROZEN_ID].info["validation_avg_loglik_all_days"]
    tr_total = float(forward_filter(params_a, x_tr).log_score.sum())
    saved_total = float(frozen_meta.get("train_loglik", np.nan))
    ok = abs(tr_total - saved_total) <= 1e-8 * abs(saved_total) and abs(rep - s5_val) <= 5e-6 * max(1, abs(s5_val))
    checks.append(Check("Frozen HMM reproduced from its saved parameters (no refit)", PASS if ok else FAIL,
                        f"Train log-likelihood {tr_total:.6f} vs saved {saved_total:.6f}; Validation log score "
                        f"{rep:.6f} vs Stage 5 table {s5_val:.6f} (table stored to 6 significant digits)"))

    # ---------------- New models (Train only) ----------------
    seeds = tuple(rx.SEEDS[: args.n_seeds])
    n_init = args.n_init or ah.AR_N_INIT
    models["iid_gaussian"] = ad.iid_gaussian_result(x_tr, x_va)
    for mid, k, desc in ((AR1_ID, 1, "C - single-regime Gaussian AR(1), diagonal"),
                         (CAND_ID, 3, "B - AR-HMM K=3, diagonal Phi and Sigma"),
                         (K2_ID, 2, "D - AR-HMM K=2 (robustness)")):
        t0 = time.perf_counter()
        print(f"Fitting {mid} ({len(seeds) if k > 1 else 1} seeds x {n_init if k > 1 else 1} starts) ...")
        models[mid] = ad.ar_model_result(mid, desc, k, x_tr, x_va, tr_feat, seeds, n_init, FEATURE_COLUMNS)
        ref = models[mid].info["reference_fit"]
        print(f"  {mid}: train LL/obs {ref.train_loglik / ref.n_obs:.3f}, val {models[mid].validation.log_score.mean():.3f}"
              f", converged {sum(f.converged for f in models[mid].fits)}/{len(models[mid].fits)}, phi at bound "
              f"{ref.bound_share:.0%}, {time.perf_counter() - t0:.0f}s")
    order = [FROZEN_ID, CAND_ID, AR1_ID, K2_ID, "iid_gaussian"]
    models = {m: models[m] for m in order}

    # ---------------- Diagnostics ----------------
    comp = ad.comparison_table(models, FROZEN_ID, AR1_ID)
    calib = pd.concat([ad.calibration_table(m, FEATURE_COLUMNS) for m in models.values()], ignore_index=True)
    resid = pd.concat([ad.residual_table(m, FEATURE_COLUMNS) for m in models.values()], ignore_index=True)
    init = pd.DataFrame([ad.initialization_table(models[m]) for m in (CAND_ID, K2_ID)])
    regimes = pd.DataFrame([ad.regime_summary(models[m]) for m in (FROZEN_ID, CAND_ID, K2_ID)])
    trans = pd.DataFrame([r for m in (FROZEN_ID, CAND_ID, K2_ID) for r in ad.transition_rows(models[m])])
    profiles = pd.concat([ad.state_profiles(models[m], tr_feat, va_feat, FEATURE_COLUMNS, scale, mean)
                          for m in (FROZEN_ID, CAND_ID, K2_ID)], ignore_index=True)
    matches = [ad.match_states(models[FROZEN_ID], models[m], profiles, FEATURE_COLUMNS) for m in (CAND_ID, K2_ID)]
    daily_va = ad.daily_scores(models, "validation", va_feat.index)
    daily_tr = ad.daily_scores(models, "train", tr_feat.index[1:])
    decision = ad.adoption_decision(comp, calib, resid, init, regimes, profiles, CAND_ID, FROZEN_ID, AR1_ID)
    decision_k2 = ad.adoption_decision(comp, calib, resid, init, regimes, profiles, K2_ID, FROZEN_ID, AR1_ID)

    # ---------------- Causality / leakage checks on the real run ----------------
    b = models[CAND_ID]
    pb = b.params
    tr_b, va_b = ah.filter_train_validation_ar(pb, x_tr, x_va)
    worst = max(np.abs(r.filtered.sum(1) - 1).max() for m in models.values() for r in (m.train, m.validation))
    checks.append(Check("Filtered probabilities sum to one (all models)", PASS if worst < 1e-9 else FAIL, f"{worst:.1e}"))
    rows = max(np.abs(models[m].params.transmat.sum(1) - 1).max() for m in (CAND_ID, K2_ID))
    checks.append(Check("AR-HMM transition rows sum to one", PASS if rows < 1e-12 else FAIL, f"{rows:.1e}"))
    ll = abs(tr_b.log_score.sum() - b.info["reference_fit"].train_loglik) / abs(b.info["reference_fit"].train_loglik)
    checks.append(Check("Filter log scores reproduce the AR-HMM EM log-likelihood", PASS if ll < 1e-8 else FAIL, f"{ll:.1e}"))
    y_all, lag_all = ah.lagged_pairs(np.vstack([x_tr, x_va]))
    joint = ah.filter_from_log_emission(ah.log_emission_ar(pb, y_all, lag_all), pb.transmat,
                                        initial_predicted=pb.startprob)
    diff = np.abs(joint.filtered[len(x_tr) - 1:] - va_b.filtered).max()
    checks.append(Check("Validation continues the Train recursion (lag and state) without reset",
                        PASS if diff < 1e-10 else FAIL, f"max difference vs filtering the joined sequence {diff:.1e}"))
    cut = len(x_va) // 2
    shocked = x_va.copy()
    shocked[cut:] = shocked[cut:] * -5 + 10
    _, va_s = ah.filter_train_validation_ar(pb, x_tr, shocked)
    before = np.abs(va_s.filtered[:cut] - va_b.filtered[:cut]).max()
    pred_cut = np.abs(va_s.predicted[cut] - va_b.predicted[cut]).max()
    checks.append(Check("Future Validation changes leave earlier filtered probabilities unchanged",
                        PASS if before == 0 else FAIL, f"max change before cut {before:.1e}"))
    checks.append(Check("Predicted state probability and score of day t use only data through t-1",
                        PASS if pred_cut == 0 else FAIL,
                        f"changing x_t leaves P(Z_t | x_1..x_t-1) unchanged (max change {pred_cut:.1e})"))
    refit = ah.fit_ar_hmm_best(x_tr, 3, b.info["reference_fit"].seed, n_init)
    rdiff = abs(refit.train_loglik - b.info["reference_fit"].train_loglik)
    checks.append(Check("AR-HMM estimates depend on Train only and are reproducible", PASS if rdiff < 1e-6 else WARN,
                        f"refit of the reference seed: |log-likelihood difference| {rdiff:.1e}"))
    checks.append(Check("No test file read", FAIL if any(f.startswith("test_") for f in data.files_read) else PASS,
                        ", ".join(data.files_read)))
    conv = [m for m in (AR1_ID, CAND_ID, K2_ID) if not models[m].info["reference_fit"].converged]
    checks.append(Check("AR reference fits converged", WARN if conv else PASS, ", ".join(conv) or "all converged"))
    status, detail = ad.monotonicity_check(models, (AR1_ID, CAND_ID, K2_ID))
    checks.append(Check(f"EM objective monotone for every start (all seeds x starts, tolerance {ah.MONOTONE_ABS_TOL:g} nats)",
                        status, detail))
    opt = {m: ad.mstep_optimality(models[m].params, x_tr) for m in (AR1_ID, CAND_ID, K2_ID)}
    checks.append(Check("M-step is the exact constrained maximizer of the EM Q-function (Train)",
                        PASS if all(o["optimal"] for o in opt.values()) else FAIL,
                        "; ".join(f"{m}: max Q gain of any feasible coordinate move {o['max_q_gain_from_feasible_move']:.1e}"
                                  for m, o in opt.items())))

    # ---------------- Outputs ----------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "models").mkdir(exist_ok=True)
    probs = {}
    for m in (FROZEN_ID, CAND_ID, K2_ID):
        r = models[m]
        for j in range(r.k):
            probs[f"{m}__filtered_S{j + 1}"] = r.validation.filtered[:, j]
            probs[f"{m}__predicted_S{j + 1}"] = r.validation.weights[:, j]
        probs[f"{m}__p_stress"] = r.p_stress("validation")
    tables = {"ar_hmm_model_comparison.csv": comp, "ar_hmm_predictive_calibration.csv": calib,
              "ar_hmm_residual_diagnostics.csv": resid, "ar_hmm_state_profiles.csv": profiles,
              "ar_hmm_transition_matrices.csv": trans, "ar_hmm_initialization_stability.csv": init,
              "ar_hmm_regime_summary.csv": regimes,
              "ar_hmm_state_matching.csv": pd.DataFrame([{k: v for k, v in m.items() if k != "crosstab_train"}
                                                         for m in matches])}
    for name, frame in tables.items():
        frame.to_csv(OUT_DIR / name, index=False, float_format="%.6g")
    pd.DataFrame(probs, index=va_feat.index).to_csv(OUT_DIR / "ar_hmm_validation_probabilities.csv",
                                                    date_format="%Y-%m-%d", float_format="%.8g")
    pd.concat([daily_tr.assign(split="train"), daily_va.assign(split="validation")]).to_csv(
        OUT_DIR / "ar_hmm_daily_log_scores.csv", date_format="%Y-%m-%d", float_format="%.8g")
    hashes = {n: data.manifest.get("split_files", {}).get(n) for n in data.files_read if n.endswith(".csv")}
    for m in (AR1_ID, CAND_ID, K2_ID):
        ref = models[m].info["reference_fit"]
        ad.save_json({"model_id": m, "description": models[m].description, "n_states": models[m].k,
                      "emission": "Gaussian AR(1), diagonal Phi and Sigma, state-specific", "phi_bound": ah.PHI_BOUND,
                      "reg_var": ah.REG_VAR, "variance_rule": ref.variance_rule, "random_seed": ref.seed, "seeds_evaluated": [f.seed for f in models[m].fits],
                      "n_init_per_seed": len(ref.start_objectives), "converged": ref.converged, "n_iter": ref.n_iter,
                      "train_loglik_conditional_on_first_row": ref.train_loglik,
                      "train_first_date": str(tr_feat.index.min().date()), "train_last_date": str(tr_feat.index.max().date()),
                      "factor_columns": FEATURE_COLUMNS, "preprocessing": sd, "input_sha256": hashes,
                      "stress_states": [f"S{j + 1}" for j in models[m].stress_states],
                      "model_version": STAGE55_CODE_VERSION, "parameters": models[m].params.to_dict()},
                     OUT_DIR / "models" / f"{m}.json")

    hashes_after = ad.directory_hashes(STAGE5_DIR)
    checks.append(Check("Stage 5 outputs unchanged (data/processed/hmm hashes before == after)",
                        PASS if hashes_after == hashes_before else FAIL, f"{len(hashes_before)} files compared"))
    warn_list = sorted({f"{w.category.__name__}: {str(w.message)[:160]}" for w in caught})
    checks.append(Check("No library warnings", PASS if not warn_list else WARN, "; ".join(warn_list) or "none"))
    pd.DataFrame([vars(c) for c in checks]).to_csv(OUT_DIR / "ar_hmm_checks.csv", index=False)

    figures = []
    if not args.no_figures:
        try:
            figures = ar.save_figures(FIGURE_DIR, models, comp, calib, resid, profiles, daily_va, FROZEN_ID,
                                      [CAND_ID, K2_ID], FEATURE_COLUMNS, tr_feat.index[1:], va_feat.index)
        except ImportError:
            print(f"matplotlib not installed; figures skipped ({sys.executable} -m pip install matplotlib)")
    recorder.__exit__(None, None, None)
    ad.save_json({"stage": "5.5 - AR-HMM robustness experiment", "model_version": STAGE55_CODE_VERSION,
                  "created_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
                  "frozen_model": {"id": frozen_name, "file": str(model_path.relative_to(ROOT)),
                                   "sha256": ad.file_sha256(model_path)},
                  "files_read": data.files_read + [str(manifest_path.relative_to(ROOT)), str(model_path.relative_to(ROOT)),
                                                   "data/processed/hmm/hmm_model_comparison.csv"],
                  "test_data_loaded": False, "evidence_status": "robustness check on the Stage 5 development sample "
                  "(Validation reused), not an independent holdout",
                  "adoption_rules": ad.ADOPTION_RULES, "adoption_procedure": ad.ADOPTION_PROCEDURE,
                  "estimation_amendments": ad.ESTIMATION_AMENDMENTS,
                  "decision": decision, "decision_k2_for_reference": decision_k2,
                  "specification": {"phi_bound": ah.PHI_BOUND, "reg_var": ah.REG_VAR, "variance_rule": "floor",
                                    "monotone_abs_tol": ah.MONOTONE_ABS_TOL, "seeds": list(seeds),
                                    "n_init_per_seed": n_init, "tol_per_obs": ah.AR_TOL, "max_iter": ah.AR_MAX_ITER},
                  "scoring": {"train_scored_from": str(tr_feat.index[1].date()), "validation_scored_days": len(x_va)},
                  "checks": [vars(c) for c in checks], "software": rx.software_versions(),
                  "outputs": sorted(str(p.relative_to(ROOT)) for p in OUT_DIR.rglob("*") if p.is_file())},
                 OUT_DIR / "ar_hmm_selection_manifest.json")
    ctx = {"comparison": comp, "calibration": calib, "residuals": resid, "init": init, "regimes": regimes,
           "profiles": profiles, "matches": matches, "decision": decision, "checks": checks, "frozen_id": FROZEN_ID,
           "candidate_id": CAND_ID, "ar1_id": AR1_ID, "k2_id": K2_ID, "data": data, "seeds": seeds,
           "columns": FEATURE_COLUMNS, "figures": figures}
    (REPORTS / "ar_hmm_robustness.md").write_text(ar.render_report(ctx), encoding="utf-8")

    # ---------------- Console summary ----------------
    print("\nValidation average one-step predictive log score (identical scored dates):")
    for r in comp.itertuples():
        extra = ""
        if pd.notna(getattr(r, "validation_mean_diff_vs_frozen_hmm", np.nan)):
            extra += f"  vs A {r.validation_mean_diff_vs_frozen_hmm:+.3f} (t {r.validation_nw_t_vs_frozen_hmm:.1f})"
        if pd.notna(getattr(r, "validation_mean_diff_vs_ar1", np.nan)):
            extra += f"  vs AR(1) {r.validation_mean_diff_vs_ar1:+.3f} (t {r.validation_nw_t_vs_ar1:.1f})"
        print(f"  {r.model_id:<14} train {r.train_avg_log_score:8.3f}  val {r.validation_avg_log_score:8.3f}{extra}")
    v = resid[resid["split"] == "validation"].groupby("model_id")
    cv = calib[(calib["split"] == "validation") & (calib["factor"] == "all factors")].set_index("model_id")
    print("Validation residual dependence and calibration:")
    for m in models:
        g = v.get_group(m)
        print(f"  {m:<14} quantile-residual ACF lag1 {g['qres_acf_lag1'].mean():+.3f} lag5 {g['qres_acf_lag5'].mean():+.3f} "
              f"lag21 {g['qres_acf_lag21'].mean():+.3f} | coverage 50/90/95 {cv.loc[m, 'coverage_50']:.2f}/"
              f"{cv.loc[m, 'coverage_90']:.2f}/{cv.loc[m, 'coverage_95']:.2f}")
    print("Regimes (Validation): " + "; ".join(
        f"{r.model_id}: max prob {r.validation_mean_max_filtered_prob:.3f}, switches/yr {r.validation_switches_per_year:.1f}"
        for r in regimes.itertuples()))
    for mt in matches:
        print(f"  {mt['model_id']} vs frozen: ARI train {mt['train_ari']:.2f} / val {mt['validation_ari']:.2f}; "
              f"{mt['profile_matching']}")
    print(f"Decision: {decision['verdict']} -> {decision['stage6_model']}")
    print("  " + "; ".join(f"{k} {'PASS' if v_ else 'FAIL'}" for k, v_ in decision["checks"].items()))
    print("Checks:")
    for c in checks:
        print(f"  [{c.status}] {c.name}: {c.detail}")
    counts = pd.Series([c.status for c in checks]).value_counts()
    print("Summary: " + ", ".join(f"{counts.get(s, 0)} {s}" for s in (PASS, WARN, FAIL, NOT_RUN)))
    print(f"Report: {REPORTS / 'ar_hmm_robustness.md'}; figures: {len(figures)}. Elapsed "
          f"{time.perf_counter() - started:.0f}s. Stage 5.5 complete; Stage 6 not started.")
    return 1 if counts.get(FAIL, 0) else 0


if __name__ == "__main__":
    sys.exit(main())
