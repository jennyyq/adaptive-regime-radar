"""Stage 5.5 audit: EM monotonicity, M-step optimality and reproducibility of the AR-HMM (Train only).

For every AR model (K = 3 primary, K = 2, K = 1) and both variance rules
    additive_legacy : v1 rule, var = s2 + 1e-4 and pi floored at 1e-8 (as used in the first real run)
    floor           : exact constrained maximizer, var = max(s2, 1e-4), pi = gamma_1 (current rule)
every EM start of every seed (10 seeds x 5 starts) is re-run and its full objective trajectory is stored.
Decreases are classified against a floating-point error estimate of the objective itself.

Reads only data/splits/{train,validation}_*.csv (Validation is loaded by the shared loader but not used), and,
if present, data/processed/ar_hmm/models/*.json from the previous 07b run (read-only, for the reproduction
check). Writes data/processed/ar_hmm/audit/ and reports/ar_hmm_em_audit.md. Test is never read; Stage 0-5
files are not touched.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regime_radar.paths import PROCESSED, REPORTS  # noqa: E402

SPLITS_DIR = ROOT / "data" / "splits"
STAGE55_CODE_VERSION = "stage5.5-2026-10-21b"
AR_DIR = PROCESSED / "ar_hmm"
AUDIT_DIR = AR_DIR / "audit"
EPS = np.finfo(float).eps


def objective_terms(params, y, lag):
    """log c_t (scaled forward recursion) + per-day emission shift: the summands of the Train objective."""
    from regime_radar.ar_hmm import log_emission_ar

    log_b = log_emission_ar(params, y, lag)
    shift = log_b.max(axis=1)
    b = np.exp(log_b - shift[:, None])
    alpha = params.startprob * b[0]
    terms = [math.log(alpha.sum()) + shift[0]]
    alpha /= alpha.sum()
    for t in range(1, len(y)):
        alpha = (alpha @ params.transmat) * b[t]
        s = alpha.sum()
        terms.append(math.log(s) + shift[t])
        alpha /= s
    return np.array(terms)


def fp_error_estimate(params, y, lag) -> float:
    """Rounding error of the objective: |naive sum - exactly rounded sum| of its T summands, plus the
    worst-case bound T * eps * max|term| for the recursion itself."""
    terms = objective_terms(params, y, lag)
    return float(abs(terms.sum() - math.fsum(terms)) + len(terms) * EPS * np.abs(terms).max())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--n-init", type=int, default=5)
    parser.add_argument("--ks", type=str, default="3,2,1")
    args = parser.parse_args()

    from regime_radar import ar_hmm as ah
    from regime_radar import ar_hmm_diagnostics as ad
    from regime_radar import regime_exploration as rx
    from regime_radar.dataset import FEATURE_COLUMNS
    from regime_radar.hmm_models import _forward_backward, start_seeds
    from sklearn.metrics import adjusted_rand_score

    if any(getattr(m, "STAGE55_CODE_VERSION", None) != STAGE55_CODE_VERSION for m in (ah, ad)):
        print("Stage 5.5 modules do not match this script; re-extract the zip and delete __pycache__.")
        return 3
    started = time.perf_counter()
    data = rx.load_stage4_data(SPLITS_DIR)
    tr_feat = data.train_features[FEATURE_COLUMNS]
    apply, _ = rx.scaling_decision(rx.describe_train(tr_feat))
    scaler = rx.TrainScaler(apply).fit(tr_feat)
    x = scaler.transform(tr_feat)
    y, lag = ah.lagged_pairs(x)
    n = len(y)
    seeds = tuple(rx.SEEDS[: args.n_seeds])
    ks = [int(k) for k in args.ks.split(",")]
    print(f"Train only: {len(x)} rows ({n} scored). Rules {ah.VARIANCE_RULES}; K {ks}; {len(seeds)} seeds x "
          f"{args.n_init} starts; tolerance {ah.MONOTONE_ABS_TOL:g} nats.")

    traj, starts, best = [], [], {}
    for k in ks:
        for rule in ah.VARIANCE_RULES:
            t0 = time.perf_counter()
            fits = []
            for seed in seeds:
                for s in start_seeds(seed, args.n_init if k > 1 else 1):
                    f = ah.fit_ar_hmm(x, k, s, variance_rule=rule)
                    h = np.asarray(f.history)
                    d = np.diff(h)
                    fp = fp_error_estimate(f.params, y, lag)
                    neg = np.flatnonzero(d < 0)
                    worst = int(neg[np.argmin(d[neg])]) if len(neg) else -1
                    max_dec = float(-d[worst]) if worst >= 0 else 0.0
                    cls = ("none" if max_dec == 0 else "numerical" if max_dec <= max(100 * fp, 1e-12)
                           else "genuine")
                    starts.append({"k": k, "variance_rule": rule, "seed": seed, "start_seed": s, "n_iter": f.n_iter,
                                   "converged": f.converged, "final_loglik": f.train_loglik,
                                   "final_loglik_per_obs": f.train_loglik / n,
                                   "n_decreases_any": int(len(neg)),
                                   "n_decreases_beyond_tol": int((d < -ah.MONOTONE_ABS_TOL).sum()),
                                   "max_decrease": max_dec, "iteration_of_max_decrease": worst + 2 if worst >= 0 else np.nan,
                                   "relative_max_decrease": max_dec / abs(h[worst]) if worst >= 0 else 0.0,
                                   "fp_error_estimate": fp, "classification": cls,
                                   "phi_bound_share": f.bound_share, "empty_state_events": f.empty_state_events})
                    traj += [{"k": k, "variance_rule": rule, "seed": seed, "start_seed": s, "iteration": i + 1,
                              "objective": v, "delta": (v - h[i - 1]) if i else np.nan} for i, v in enumerate(h)]
                    fits.append((seed, f))
            pool = [f for _, f in fits if f.converged] or [f for _, f in fits]
            best[(k, rule)] = max(pool, key=lambda f: f.train_loglik)
            print(f"  K={k} {rule:<16} {len(fits)} starts, {time.perf_counter() - t0:5.1f}s")

    st = pd.DataFrame(starts)
    summary = st.groupby(["k", "variance_rule"]).agg(
        starts=("start_seed", "size"), converged_share=("converged", "mean"),
        starts_with_any_decrease=("n_decreases_any", lambda s: int((s > 0).sum())),
        starts_with_decrease_beyond_tol=("n_decreases_beyond_tol", lambda s: int((s > 0).sum())),
        total_decreases_beyond_tol=("n_decreases_beyond_tol", "sum"), max_decrease=("max_decrease", "max"),
        max_fp_error_estimate=("fp_error_estimate", "max"),
        genuine=("classification", lambda s: int((s == "genuine").sum())),
        numerical=("classification", lambda s: int((s == "numerical").sum())),
        best_loglik_per_obs=("final_loglik_per_obs", "max")).reset_index()

    # M-step optimality and objective consistency at each rule's best fit
    opt_rows = []
    for (k, rule), f in best.items():
        o = ad.mstep_optimality(f.params, x) if rule == "floor" else None
        g, xi, ll = _forward_backward(f.params, ah.log_emission_ar(f.params, y, lag))
        new_f = ah._m_step(f.params, y, lag, g, xi, "floor")[0]
        new_l = ah._m_step(f.params, y, lag, g, xi, "additive_legacy")[0]
        q_gap = ah.expected_complete_loglik(new_f, y, lag, g, xi) - ah.expected_complete_loglik(new_l, y, lag, g, xi)
        filt = ah.filter_from_log_emission(ah.log_emission_ar(f.params, y, lag), f.params.transmat,
                                           initial_predicted=f.params.startprob).log_score.sum()
        opt_rows.append({"k": k, "variance_rule": rule, "best_seed_start": f.seed,
                         "q_floor_minus_q_legacy_at_same_estep": q_gap,
                         "max_q_gain_feasible_move_floor": o["max_q_gain_from_feasible_move"] if o else np.nan,
                         "mstep_optimal_floor": o["optimal"] if o else np.nan,
                         "em_objective": ll, "filter_score_sum": float(filt),
                         "objective_vs_filter_abs_diff": abs(float(filt) - ll),
                         "min_innovation_var": float(f.params.var.min()),
                         "variance_floor_active_share": float((f.params.var <= ah.REG_VAR * (1 + 1e-12)).mean())})
    opt = pd.DataFrame(opt_rows)

    # Reproducibility and effect of the fix
    rep_rows = []
    for k in ks:
        fl, lg = best[(k, "floor")], best[(k, "additive_legacy")]
        again = ah.fit_ar_hmm(x, k, fl.seed, variance_rule="floor")
        lab = lambda p: ah.filter_from_log_emission(ah.log_emission_ar(p, y, lag), p.transmat,  # noqa: E731
                                                    initial_predicted=p.startprob).filtered.argmax(1)
        row = {"k": k, "refit_abs_loglik_diff": abs(again.train_loglik - fl.train_loglik),
               "refit_max_param_diff": float(max(np.abs(again.params.phi - fl.params.phi).max(),
                                                 np.abs(again.params.var - fl.params.var).max(),
                                                 np.abs(again.params.transmat - fl.params.transmat).max())),
               "floor_minus_legacy_best_loglik_per_obs": (fl.train_loglik - lg.train_loglik) / n,
               "train_ari_floor_vs_legacy": float(adjusted_rand_score(lab(fl.params), lab(lg.params)))}
        saved = AR_DIR / "models" / {3: "ar_hmm_k3.json", 2: "ar_hmm_k2.json", 1: "ar1.json"}[k]
        if saved.exists():
            meta = json.loads(saved.read_text(encoding="utf-8"))
            sp = ah.ARParams.from_dict(meta["parameters"])
            rule = meta.get("variance_rule", "additive_legacy")       # v1 files have no variance_rule key
            ref = best[(k, rule)]
            row.update({"saved_model_version": meta.get("model_version"), "saved_variance_rule": rule,
                        "saved_loglik": meta.get("train_loglik_conditional_on_first_row"),
                        "recomputed_loglik_of_saved_params": float(objective_terms(sp, y, lag).sum()),
                        "audit_best_loglik_same_rule": ref.train_loglik,
                        "abs_diff_saved_vs_audit_best": abs(ref.train_loglik - meta.get(
                            "train_loglik_conditional_on_first_row", np.nan))})
        rep_rows.append(row)
    rep = pd.DataFrame(rep_rows)

    checks = []
    fl_sum = summary[summary["variance_rule"] == "floor"]
    lg_sum = summary[summary["variance_rule"] == "additive_legacy"]
    checks.append(("Exact M-step: no EM start decreases beyond tolerance", "PASS" if fl_sum["starts_with_decrease_beyond_tol"].sum() == 0
                   else "FAIL", f"{int(fl_sum['starts_with_decrease_beyond_tol'].sum())} of {int(fl_sum['starts'].sum())} starts; "
                   f"max decrease {fl_sum['max_decrease'].max():.2e}"))
    checks.append(("Legacy v1 M-step reproduces the reported non-monotonicity (informational)",
                   "INFO", f"{int(lg_sum['starts_with_decrease_beyond_tol'].sum())} of {int(lg_sum['starts'].sum())} starts; "
                   f"max decrease {lg_sum['max_decrease'].max():.2e}; genuine {int(lg_sum['genuine'].sum())}"))
    of = opt[opt["variance_rule"] == "floor"]
    checks.append(("M-step is the exact constrained maximizer of Q (floor rule)",
                   "PASS" if bool(of["mstep_optimal_floor"].all()) else "FAIL",
                   f"max Q gain of any feasible move {of['max_q_gain_feasible_move_floor'].max():.1e}"))
    checks.append(("EM objective = sum of filter log scores (fit/score consistency)",
                   "PASS" if opt["objective_vs_filter_abs_diff"].max() < 1e-6 else "FAIL",
                   f"max |difference| {opt['objective_vs_filter_abs_diff'].max():.1e}"))
    if "abs_diff_saved_vs_audit_best" in rep:
        d = rep["abs_diff_saved_vs_audit_best"].max()
        checks.append(("Saved 07b models reproduced by re-running their rule (same data, seeds, starts)",
                       "PASS" if d <= 1e-6 else "FAIL", f"max |log-likelihood difference| {d:.1e}"))
    checks.append(("Refit reproduces the best fit exactly", "PASS" if rep["refit_abs_loglik_diff"].max() == 0 else "WARN",
                   f"max |log-likelihood difference| {rep['refit_abs_loglik_diff'].max():.1e}"))
    checks.append(("No test file read", "FAIL" if any(f.startswith("test_") for f in data.files_read) else "PASS",
                   ", ".join(data.files_read)))

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(traj).to_csv(AUDIT_DIR / "em_trajectories.csv", index=False, float_format="%.12g")
    st.to_csv(AUDIT_DIR / "em_start_summary.csv", index=False, float_format="%.6g")
    summary.to_csv(AUDIT_DIR / "em_monotonicity_summary.csv", index=False, float_format="%.6g")
    opt.to_csv(AUDIT_DIR / "mstep_optimality.csv", index=False, float_format="%.6g")
    rep.to_csv(AUDIT_DIR / "reproducibility.csv", index=False, float_format="%.10g")
    pd.DataFrame(checks, columns=["name", "status", "detail"]).to_csv(AUDIT_DIR / "audit_checks.csv", index=False)

    from regime_radar.quality import _md_table
    from regime_radar.regime_report import _fmt
    lg_bad = st[(st["variance_rule"] == "additive_legacy") & (st["n_decreases_beyond_tol"] > 0)]
    L = ["# Stage 5.5 audit - AR-HMM EM monotonicity, M-step optimality and reproducibility", "",
         f"Train only ({len(x)} rows, {n} scored); {len(seeds)} seeds x {args.n_init} starts per K; every start's "
         "trajectory is in `data/processed/ar_hmm/audit/em_trajectories.csv`. Test not read; Stage 0-5 untouched.", "",
         "## Question", "", "The first real Stage 5.5 run reported `all_monotone = False` for ar_hmm_k3 while all checks "
         "passed. EM (and constrained generalized EM) cannot decrease the likelihood if the M-step maximizes the "
         "Q-function, so a decrease is either rounding error or an M-step that is not a maximizer.", "",
         "## Monotonicity of every start", "", _md_table(_fmt(summary.set_index("k"), 6)), "",
         f"Classification: a decrease is 'numerical' if it is at most 100 x the floating-point error estimate of the "
         f"objective (|naive sum - exact sum| of the T summands + T x eps x max|summand|), otherwise 'genuine'. "
         f"Tolerance used by the pipeline: {ah.MONOTONE_ABS_TOL:g} nats on the total objective.", ""]
    if len(lg_bad):
        L += ["Legacy starts with decreases beyond tolerance:", "",
              _md_table(_fmt(lg_bad[["k", "seed", "start_seed", "n_iter", "n_decreases_beyond_tol", "max_decrease",
                                     "iteration_of_max_decrease", "fp_error_estimate", "classification"]].set_index("k"), 8)), ""]
    L += ["## M-step optimality and objective consistency", "",
          "q_floor_minus_q_legacy_at_same_estep > 0 means the legacy variance update leaves Q-function value unused at "
          "every iteration; max_q_gain_feasible_move_floor <= 0 means no feasible single-coordinate change of "
          "(intercept, phi, var) improves Q at the exact M-step.", "", _md_table(_fmt(opt.set_index("k"), 8)), "",
          "## Reproducibility and effect of the fix", "", _md_table(_fmt(rep.set_index("k"), 8)), "",
          "## Checks", "", _md_table(pd.DataFrame(checks, columns=["name", "status", "detail"]).set_index("name")), "",
          "## Conclusion (computed)", ""]
    genuine = int(lg_sum["genuine"].sum())
    L.append(f"- Legacy rule: {int(lg_sum['starts_with_decrease_beyond_tol'].sum())} starts decreased beyond tolerance, "
             f"{genuine} classified genuine (max {lg_sum['max_decrease'].max():.2e} nats vs floating-point error "
             f"<= {lg_sum['max_fp_error_estimate'].max():.1e}). " + ("The decreases are not rounding error: the variance "
                                                                    "update s2 + 1e-4 is not a Q maximizer." if genuine
                                                                    else "No genuine decrease on this data."))
    L.append(f"- Exact rule: {int(fl_sum['starts_with_decrease_beyond_tol'].sum())} starts decreased beyond tolerance "
             f"(max decrease {fl_sum['max_decrease'].max():.2e}).")
    L.append("- Stage 5.5 results must be re-run with scripts/07b_compare_ar_hmm.py (exact M-step); the frozen Stage 5 "
             "HMM is not affected.")
    (REPORTS / "ar_hmm_em_audit.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    print("\nMonotonicity summary (all starts):")
    print(summary.to_string(index=False))
    print("\nM-step optimality / consistency:")
    print(opt.to_string(index=False))
    print("\nReproducibility:")
    print(rep.to_string(index=False))
    print("\nChecks:")
    for name, status, detail in checks:
        print(f"  [{status}] {name}: {detail}")
    print(f"Report: {REPORTS / 'ar_hmm_em_audit.md'}. Elapsed {time.perf_counter() - started:.0f}s.")
    return 1 if any(c[1] == "FAIL" for c in checks) else 0


if __name__ == "__main__":
    sys.exit(main())
