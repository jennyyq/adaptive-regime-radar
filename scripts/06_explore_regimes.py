"""Stage 4: exploratory regime models (K-Means, Gaussian mixtures) on Train, compared on Validation.

Reads ONLY data/splits/{train,validation}_{features,returns}.csv and split_manifest.json. The test split is
never opened. Nothing from earlier stages is recomputed.

Outputs
  data/processed/regime_exploration/
      kmeans_model_comparison.csv, gmm_model_comparison.csv, regime_stability.csv, model_eligibility.csv,
      regime_factor_profiles.csv, regime_asset_statistics.csv, train_factor_statistics.csv,
      kmeans_elbow.csv, kmeans_elbow_sensitivity.csv, k3_alternative_assessment.csv,
      train_regime_assignments.csv, validation_regime_assignments.csv,
      train_gmm_probabilities.csv, validation_gmm_probabilities.csv,
      stage4_checks.csv, regime_exploration_manifest.json, models/<model_id>.pkl + .json,
      stage4_decision.json, selected_model_{states,summary,transitions,episodes}.csv  (recorded K=2 decision)
  reports/regime_exploration.md
  reports/figures/regime_exploration/*.png (9 figures, incl. kmeans_elbow_method.png)
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regime_radar.paths import PROCESSED, REPORTS  # noqa: E402

SPLITS_DIR = ROOT / "data" / "splits"
STAGE4_CODE_VERSION = "stage4-2026-10-21b"   # all src/regime_radar/regime_*.py modules must carry the same value
OUT_DIR = PROCESSED / "regime_exploration"
FIGURE_DIR = REPORTS / "figures" / "regime_exploration"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-figures", action="store_true", help="skip matplotlib figures")
    args = parser.parse_args()

    try:
        import scipy  # noqa: F401
        import sklearn  # noqa: F401
    except ImportError as exc:
        print(f"Missing dependency ({exc.name}). Install it into the interpreter you use for this project:\n"
              f"  {sys.executable} -m pip install scikit-learn\nNothing was run.")
        return 2

    from regime_radar import regime_exploration as rx
    from regime_radar import regime_decision as rdec
    from regime_radar import regime_diagnostics as rdiag
    from regime_radar import regime_report as rr

    stale = [m for m in (rx, rdiag, rr, rdec) if getattr(m, "STAGE4_CODE_VERSION", None) != STAGE4_CODE_VERSION]
    if stale:
        print(f"Stage 4 modules do not match this script (expected version {STAGE4_CODE_VERSION}):")
        for m in stale:
            print(f"  {m.__file__}: version {getattr(m, 'STAGE4_CODE_VERSION', 'missing (older file)')}")
        print("Re-extract the latest stage4_code.zip inside the project folder, delete __pycache__ folders, and rerun.")
        return 3

    if not (SPLITS_DIR / "split_manifest.json").exists():
        raise SystemExit("data/splits/split_manifest.json not found. Run scripts/05_prepare_datasets.py first.")

    started = time.perf_counter()
    data = rx.load_stage4_data(SPLITS_DIR)
    print(f"Loaded {', '.join(data.files_read)}")
    print(f"Train {data.train_features.index.min().date()}..{data.train_features.index.max().date()} "
          f"({len(data.train_features)} rows); Validation {data.validation_features.index.min().date()}.."
          f"{data.validation_features.index.max().date()} ({len(data.validation_features)} rows)")

    checks = rx.input_checks(data)
    checks.append(rx.frozen_factor_check(data.manifest, ROOT / "src" / "regime_radar" / "factors.py"))
    blocking = [c for c in checks if c.status == rx.FAIL]
    if blocking:
        for c in blocking:
            print(f"  [FAIL] {c.name}: {c.detail}")
        print("Input checks failed; no model was fitted.")
        return 1

    n_specs = len(rx.candidate_specs())
    print(f"Fitting {n_specs} candidates x {len(rx.SEEDS)} seeds on Train ...")
    done = []

    def progress(model_id, fit):
        done.append(model_id)
        flag = "" if fit.converged else "  (convergence warning)"
        print(f"  [{len(done):>2}/{n_specs}] {model_id:<13} fitted ({fit.seconds:5.2f}s for the reference seed){flag}")

    result = rx.run_exploration(data, progress=progress)
    print(f"Scaling: {result.scaling_reason}")

    gmm_id, km_id, _ = rr.display_candidates(result)
    detailed = list(dict.fromkeys([gmm_id, km_id]))
    checks += rx.leakage_checks(data, result, detailed)
    profiles = rx.profiles_table(result, data)
    assets = rx.asset_table(result, data, detailed)
    tr_assign, va_assign, tr_prob, va_prob = rx.assignments(result, data)

    # ---------------- recorded stage-4 decision (K=2) ----------------
    record = rdec.STAGE4_DECISION
    decision_ready = record["reference_model"] in result.reference and record["baseline_model"] in result.reference
    decision = None
    if decision_ready:
        decision_checks = rdec.decision_checks(result, profiles)
        checks += decision_checks
        states = rdec.selected_states(result, data)
        decision = {"record": record, "checks": decision_checks, "states": states,
                    "summary": rdec.summary_table(result, data, profiles),
                    "transitions": rdec.transition_table(result), "episodes": rdec.episodes(states),
                    "assets": assets}

    # ---------------- outputs ----------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "models").mkdir(exist_ok=True)
    sel = result.selection
    stab = result.stability
    km = result.comparison[result.comparison["family"] == "kmeans"].merge(stab, on=["model_id", "family", "k",
                                                                                     "covariance_type"])
    gmm = result.comparison[result.comparison["family"] == "gmm"].merge(stab, on=["model_id", "family", "k",
                                                                                   "covariance_type"])
    gmm = gmm.merge(sel["gmm_table"][["model_id", "mean_loglik_shortfall", "nw_t_vs_best",
                                      "not_significantly_worse"]], on="model_id")
    outputs = {
        "kmeans_model_comparison.csv": km, "gmm_model_comparison.csv": gmm, "regime_stability.csv": stab,
        "model_eligibility.csv": sel["eligibility"], "regime_factor_profiles.csv": profiles,
        "regime_asset_statistics.csv": assets,
        "train_factor_statistics.csv": result.train_stats.rename_axis("factor").reset_index(),
        "kmeans_elbow.csv": result.elbow, "kmeans_elbow_sensitivity.csv": result.elbow_sensitivity,
        "k3_alternative_assessment.csv": rx.alternative_state_assessment(profiles, stab),
        "stage4_checks.csv": pd.DataFrame([vars(c) for c in checks]),
    }
    for name, frame in outputs.items():
        frame.to_csv(OUT_DIR / name, index=False, float_format="%.6g")
    for name, frame in (("train_regime_assignments.csv", tr_assign), ("validation_regime_assignments.csv", va_assign),
                        ("train_gmm_probabilities.csv", tr_prob), ("validation_gmm_probabilities.csv", va_prob)):
        frame.to_csv(OUT_DIR / name, date_format="%Y-%m-%d", float_format="%.6g")
    if decision:
        decision["states"].to_csv(OUT_DIR / "selected_model_states.csv", date_format="%Y-%m-%d", float_format="%.6g")
        for name, key in (("selected_model_summary.csv", "summary"), ("selected_model_transitions.csv", "transitions"),
                          ("selected_model_episodes.csv", "episodes")):
            decision[key].to_csv(OUT_DIR / name, index=False, float_format="%.6g")
        rx.save_json({**record, "consistency_checks": [vars(c) for c in decision["checks"]]},
                     OUT_DIR / "stage4_decision.json")
    for model_id, fit in result.reference.items():
        meta = rx.model_metadata(fit, data, result.scaler)
        with open(OUT_DIR / "models" / f"{model_id}.pkl", "wb") as handle:
            pickle.dump({"model": fit.model, "metadata": meta}, handle)
        rx.save_json(meta, OUT_DIR / "models" / f"{model_id}.json")
    written = sorted(str(p.relative_to(ROOT)) for p in OUT_DIR.rglob("*") if p.is_file())

    figures = []
    if not args.no_figures:
        try:
            figures = rr.save_figures(result, data, profiles, assets, FIGURE_DIR,
                                      decision["states"] if decision else None,
                                      decision["episodes"] if decision else None)
        except ImportError:
            print(f"matplotlib not installed: figures skipped ({sys.executable} -m pip install matplotlib)")
    report = rr.render_report(result, data, checks, profiles, assets, decision)
    (REPORTS / "regime_exploration.md").write_text(report, encoding="utf-8")
    manifest = rx.stage4_manifest(data, result, checks, written + [str(p.relative_to(ROOT)) for p in figures]
                                  + ["reports/regime_exploration.md"])
    manifest["stage4_decision"] = record if decision else "decision models not present in this run"
    rx.save_json(manifest, OUT_DIR / "regime_exploration_manifest.json")

    # ---------------- console summary ----------------
    comp = result.comparison
    print("\nK-Means (fit on Train, Validation assigned):")
    for r in comp[comp["family"] == "kmeans"].itertuples():
        print(f"  K={r.k}  silhouette train {r.train_silhouette:.3f} / val {r.validation_silhouette:.3f}  "
              f"min share train {r.train_min_state_share:.1%} / val {r.validation_min_state_share:.1%}  "
              f"switches/yr {r.train_switches_per_year:.0f}")
    print("GMM (fit on Train, scored on Validation):")
    for r in comp[comp["family"] == "gmm"].itertuples():
        print(f"  {r.model_id:<13} BIC {r.train_bic:9.1f}  val loglik {r.validation_avg_loglik:7.3f}  "
              f"train-val gap {r.train_minus_validation_loglik:5.2f}  min share val "
              f"{r.validation_min_state_share:.1%}  converged {r.converged}")
    print("Selection (predefined rules):")
    print(f"  GMM:     {sel.get('gmm_note')}")
    print(f"  K-Means: {sel.get('kmeans_note')}")
    if sel.get("bic_validation_conflict"):
        print(f"  BIC vs validation: {sel['bic_validation_conflict']}")
    if sel.get("candidate_k_range"):
        print(f"  Leading K range for the HMM stage: {sel['candidate_k_range']}")
    e = result.elbow
    sens = result.elbow_sensitivity
    print(f"Elbow (K-Means, Train, diagnostic): Kneedle elbow K={result.elbow_k} over the candidate range "
          f"K=1-{rx.ELBOW_PRIMARY_MAX_K}; range sensitivity: "
          + ", ".join(f"{r.k_range} -> K={r.kneedle_elbow}" for r in sens.itertuples()))
    print("  inertia reduction per extra "
          "cluster: " + ", ".join(f"K{int(k)} {v:.0f}%" for k, v in zip(e["k"][1:], e["pct_reduction_vs_previous_k"][1:])))
    alt = outputs["k3_alternative_assessment.csv"]
    for r in alt.itertuples():
        print(f"K=3 rates/inflation state check {r.model_id:<12} state {r.candidate_state}: {r.criteria_met} criteria "
              f"(rates {r.train_interest_rate:+.2f}, val share {r.validation_share:.0%}, profile distance "
              f"{r.profile_distance:.2f}, ARI {r.train_mean_ari:.2f}) -> {'ACCEPTED' if r.accepted else 'not accepted'}")
    if decision:
        print(f"Stage 4 decision ({record['decision_date']}): K={record['primary_k']}, reference "
              f"{record['reference_model']}, baseline {record['baseline_model']}, HMM alternatives "
              f"K={record['alternatives_for_hmm']}")
        for r in decision["summary"].itertuples():
            print(f"  {r.split:<10} {r.state} {r.state_name:<20} share {r.share:5.1%}  equity {r.equity_trend:+.2f}  "
                  f"vol {r.volatility:+.2f}  credit {r.liquidity_credit:+.2f}  median run {r.median_run_days:.0f}d")
        print(f"  S2 episodes of >= {rdec.MIN_EPISODE_DAYS} days: {len(decision['episodes'])}")
    print("Checks:")
    for c in checks:
        print(f"  [{c.status}] {c.name}: {c.detail}")
    counts = pd.Series([c.status for c in checks]).value_counts().to_dict()
    print("Summary: " + ", ".join(f"{counts.get(s, 0)} {s}" for s in (rx.PASS, rx.WARN, rx.FAIL, rx.NOT_RUN)))
    print(f"Saved {len(written)} files under data/processed/regime_exploration/, "
          f"{len(figures)} figures, reports/regime_exploration.md ({time.perf_counter() - started:.0f}s)")
    return 1 if counts.get(rx.FAIL) else 0


if __name__ == "__main__":
    raise SystemExit(main())
