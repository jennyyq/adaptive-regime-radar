"""Stage 8: final sequential out-of-sample portfolio backtest (frozen models, realistic execution, costs).

Phase 8A - preflight WITHOUT Test data
  verify the Stage 7 handoff, frozen hashes, policy and roles; rebuild the frozen signals on Train/Validation and
  check them bit-for-bit against the Stage 6.1 handoff; reproduce the published Stage 7 targets; run the daily
  ledger through Train/Validation (accounting verification only, no performance reported); analytic ledger
  examples; Stage 8 unit tests; then write data/processed/backtesting/stage8_pretest_lock.json and
  reports/stage8_pretest_protocol.md. If any preflight check FAILs the script stops before Test is opened.
Phase 8B - one reserved Test evaluation
  open Test only through regime_radar.backtest_integrity.load_test_split (lock verified); continue the frozen
  signals sequentially; simulate P0 / P1 / P2 and the predefined benchmarks with the daily ledger; compute the
  prespecified metrics, statistics, attribution, stress analysis and sensitivities; write outputs, figures and
  reports/final_backtesting_research.md.

Re-running after Test access: an identical lock fingerprint is a deterministic REPRODUCTION run; a changed
fingerprint requires --amendment "reason" (previous outputs are archived and the new result is labelled AMENDMENT).
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
import time
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regime_radar.paths import PROCESSED, REPORTS  # noqa: E402

STAGE8_CODE_VERSION = "stage8-2026-10-21"
SPLITS_DIR = ROOT / "data" / "splits"
EW_DIR = PROCESSED / "early_warning"
S7_DIR = PROCESSED / "portfolio_optimization"
OUT_DIR = PROCESSED / "backtesting"
FIG_DIR = REPORTS / "figures" / "backtesting"
POLICY_PATH = ROOT / "configs" / "portfolio_policy.json"
PROJECT_PATH = ROOT / "configs" / "project.json"
PROTOCOL_PATH = ROOT / "configs" / "stage8_protocol.json"
PRIMARY = ("P0_strategic", "P1_regime_aware", "P2_ews_enhanced")
SUPPLEMENTARY = ("B1_60_40", "B2_constant_defensiveness", "B3_strategic_buy_and_hold")
ROLE_OF = {"P0_strategic": None, "P1_regime_aware": "baseline_regime", "P2_ews_enhanced": "challenger_ews",
           "B1_60_40": None, "B2_constant_defensiveness": None, "B3_strategic_buy_and_hold": None}
PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


def frozen_dir_hashes() -> dict:
    from regime_radar.ar_hmm_diagnostics import directory_hashes
    out = {}
    for d in ("hmm", "ar_hmm", "early_warning", "portfolio_optimization"):
        if (PROCESSED / d).exists():
            out.update({f"{d}/{k}": v for k, v in directory_hashes(PROCESSED / d).items()})
    return out


def run_unit_tests(pattern: str) -> dict:
    class Tee(io.StringIO):
        def write(self, x):
            sys.stdout.write(x)
            return super().write(x)
    stream = Tee()
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern=pattern, top_level_dir=str(ROOT / "tests"))
    res = unittest.TextTestRunner(verbosity=1, stream=stream).run(suite)
    return {"pattern": pattern, "tests": res.testsRun, "failures": len(res.failures), "errors": len(res.errors),
            "skipped": len(res.skipped), "ok": res.wasSuccessful() and res.testsRun > 0, "log": stream.getvalue(),
            "failed_tests": [{"test": str(t), "traceback": tb} for t, tb in res.failures + res.errors]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preflight-only", action="store_true", help="run Phase 8A, write the lock, do not open Test")
    parser.add_argument("--amendment", default=None, help="reason for re-running after Test access with changed code")
    parser.add_argument("--no-tests", action="store_true", help="skip automated tests (the release is then BLOCKED)")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    import scipy
    import sklearn

    from regime_radar import backtest_accounting as ba
    from regime_radar import backtest_attribution as bat
    from regime_radar import backtest_engine as be
    from regime_radar import backtest_execution as bx
    from regime_radar import backtest_integrity as bi
    from regime_radar import backtest_performance as bp
    from regime_radar import backtest_report as brep
    from regime_radar import backtest_signals as bs
    from regime_radar import backtest_statistics as bst
    from regime_radar import early_warning_validation as ev
    from regime_radar import portfolio_constraints as pc
    from regime_radar import portfolio_engine as eng
    from regime_radar import portfolio_execution as pe
    from regime_radar import portfolio_risk as pr
    from regime_radar import regime_exploration as rx
    from regime_radar import risk_budgeting as rb
    from regime_radar import stage7_handoff as h7
    from regime_radar import stage8_preflight as sp
    from regime_radar.factors import Check

    mods = (ba, bat, be, bx, bi, bp, brep, bs, bst, sp)
    stale = [m.__file__ for m in mods if getattr(m, "STAGE8_CODE_VERSION", None) != STAGE8_CODE_VERSION]
    if stale:
        print("Stage 8 modules do not match this script: " + ", ".join(stale) + ". Re-extract the zip, delete "
              "__pycache__ and rerun.")
        return 3
    started = time.perf_counter()
    recorder = warnings.catch_warnings(record=True)
    caught = recorder.__enter__()
    warnings.simplefilter("default")
    pre: list = []          # Phase 8A checks

    def stop(msg: str, code: int = 1) -> int:
        for c in pre:
            if c.status == FAIL:
                print(f"  [FAIL] {c.name}: {c.detail}")
        print(msg)
        recorder.__exit__(None, None, None)
        return code

    # ================================================================== PHASE 8A.1 Stage 7 handoff
    print("=" * 100 + "\nPHASE 8A - PREFLIGHT (no Test data)\n" + "=" * 100)
    s8_path = S7_DIR / "stage8_handoff_manifest.json"
    h61_path = EW_DIR / h7.HANDOFF_DIRNAME / h7.MANIFEST_NAME
    for p in (s8_path, h61_path, S7_DIR / "portfolio_policy_manifest.json", PROTOCOL_PATH):
        if not p.exists():
            raise SystemExit(f"{p.relative_to(ROOT)} not found. Run scripts 08b and 09 first.")
    s8 = json.loads(s8_path.read_text(encoding="utf-8"))
    h61 = json.loads(h61_path.read_text(encoding="utf-8"))
    pman = json.loads((S7_DIR / "portfolio_policy_manifest.json").read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    policy = pc.load_policy(POLICY_PATH, PROJECT_PATH)
    pcfg = policy.raw
    proj = json.loads(PROJECT_PATH.read_text(encoding="utf-8"))
    assets = policy.assets
    hashes_before = frozen_dir_hashes()
    for name, status, detail in sp.verify_stage7_handoff(ROOT, s8, pcfg, proj, POLICY_PATH, PROJECT_PATH, h61_path):
        pre.append(Check(name, status, detail))
    s7_checks = pd.read_csv(S7_DIR / "stage7_checks.csv") if (S7_DIR / "stage7_checks.csv").exists() else None
    s7_fail = [] if s7_checks is None else s7_checks.loc[s7_checks["status"] == FAIL, "name"].tolist()
    s7_warn = [] if s7_checks is None else s7_checks.loc[s7_checks["status"] == WARN, "detail"].tolist()
    pre.append(Check("Stage 7 release checks: no FAIL", PASS if s7_checks is not None and not s7_fail else FAIL,
                     f"{0 if s7_checks is None else len(s7_checks)} checks; FAIL: {', '.join(s7_fail) or 'none'}; "
                     f"documented WARN: {'; '.join(s7_warn)[:160] or 'none'}"))
    if any(c.status == FAIL for c in pre):
        return stop("Stage 7 handoff verification failed; Stage 8 stopped before Test (nothing written).")
    a_scale = float(s8["train_calibrated"]["P1_parity_scale_a"])
    stress_thr = float(s8["train_calibrated"]["stress_vol_threshold_spy_21d_annualized"])
    d_const = float(pman["saturation"]["train_mean_d2"])
    level_map = pcfg["risk_budget"]["P2_ews_levels"]
    print(f"Stage 7 handoff READY; P1 scale a = {a_scale!r}; cost-stress threshold {stress_thr!r}; B2 constant d = {d_const!r}")

    # ================================================================== 8A.2 Train / Validation data and signals
    data = rx.load_stage4_data(SPLITS_DIR)
    manifest = data.manifest
    tr_feat, va_feat = data.train_features, data.validation_features
    cal_pt = tr_feat.index.append(va_feat.index)
    split_pt = pd.Series(["train"] * len(tr_feat) + ["validation"] * len(va_feat), index=cal_pt)
    last_val = va_feat.index.max()
    pre.append(Check("No Test file read during preflight", PASS if not any(f.startswith("test_") for f in data.files_read)
                     else FAIL, ", ".join(data.files_read)))
    frames61 = {}
    for role in h7.ROLES:
        f = pd.concat([h7.load_stage7_inputs(role, s, EW_DIR) for s in ("train", "validation")])
        h7.assert_single_role(f)
        frames61[role] = f
    models = bs.load_frozen_models(ROOT, h61)
    pre.append(Check("Frozen HMM is hmm_std_diag_k3 (K = 3) and the AR models use the corrected variance-floor M-step",
                     PASS if Path(models.files["hmm"]).stem == "hmm_std_diag_k3" and models.hmm.transmat.shape == (3, 3)
                     and all(models.metas[m].get("variance_rule") == "floor" for m in ("ar1", "ar_hmm_k3")) else FAIL,
                     f"{models.files['hmm']}; AR variance rules "
                     f"{[models.metas[m].get('variance_rule') for m in ('ar1', 'ar_hmm_k3')]}"))
    t0 = time.perf_counter()
    run_pt = bs.sequential_signals(models, tr_feat, va_feat, split_pt)
    cmp = bs.compare_with_handoff(run_pt, frames61)
    worst = max(v[1] for v in cmp.values())
    pre.append(Check("Rebuilt Train/Validation signals identical to the Stage 6.1 handoff (17 digits, bit-for-bit)",
                     PASS if all(v[0] for v in cmp.values()) else FAIL,
                     f"{len(cmp)} columns compared; max |difference| {worst:.1e}"))
    const = bs.compare_frozen_constants(run_pt, models)
    pre.append(Check("Train constants reproduce the frozen values (scaler, innovation normalization, BOCPD priors, "
                     "alert thresholds) - no refit", PASS if max(const.values()) <= 1e-12 else FAIL,
                     ", ".join(f"{k} {v:.1e}" for k, v in const.items())))
    cut_v = va_feat.index[len(va_feat) // 2]
    run_cut = bs.sequential_signals(models, tr_feat, va_feat.loc[:cut_v], split_pt)
    pdiff = bs.prefix_difference(run_pt, run_cut)
    pre.append(Check("Signal prefix invariance on Validation (run through t == full run truncated at t)",
                     PASS if pdiff == 0 else FAIL, f"cut {cut_v.date()}; max difference {pdiff:.1e}"))
    print(f"Signals rebuilt and verified in {time.perf_counter() - t0:.0f}s.")

    # ================================================================== 8A.3 policy, calendar, risk inputs
    log_pt = pd.concat([data.train_returns, data.validation_returns])[assets]
    simple_pt = pr.to_simple(log_pt)
    ocfg = pcfg["optimizer"]
    tp = ocfg["turnover_penalty"]
    cfg = eng.EngineConfig(cov_method=pcfg["risk_estimation"]["primary"]["method"],
                           lookback=pcfg["risk_estimation"]["primary"]["lookback"],
                           min_obs=pcfg["risk_estimation"]["primary"]["min_observations"],
                           psd_floor=pcfg["risk_estimation"]["psd_floor_relative"],
                           sharpe=pcfg["expected_returns"]["equilibrium"]["benchmark_sharpe_assumption"],
                           er_shrinkage=pcfg["expected_returns"]["sensitivity"]["shrunk_historical"]["shrinkage_to_prior"],
                           lambda_turnover=2 * tp["cost_rate"] / tp["holding_horizon_years"],
                           lambda_tracking=ocfg["tracking_penalty"], cvar_alpha=pcfg["cvar"]["confidence"],
                           no_trade_band=ocfg["no_trade_band"], execution_mode="target_then_trade",
                           budget_coef=pcfg["risk_budget"]["coefficients"], solver=ocfg["solver"])
    so = s8["optimizer"]
    same_cfg = (abs(so["lambda_turnover"] - cfg.lambda_turnover) < 1e-15 and so["lambda_tracking"] == cfg.lambda_tracking
                and so["execution_mode"] == cfg.execution_mode and so["no_trade_band"] == cfg.no_trade_band)
    pre.append(Check("Engine configuration equals the frozen Stage 7 optimizer (penalties, band, execution mode)",
                     PASS if same_cfg else FAIL, f"lambda_turnover {cfg.lambda_turnover}, lambda_tracking "
                     f"{cfg.lambda_tracking}, band {cfg.no_trade_band}, {cfg.execution_mode}"))
    cost_cfg = pcfg["transaction_costs"]
    tr_mask = (split_pt == "train").to_numpy()
    spy_vol = simple_pt["SPY"].rolling(21).std() * np.sqrt(pr.ANNUALIZATION)
    thr_re = float(np.nanquantile(spy_vol[tr_mask], 0.90))
    scale_re = rb.parity_scale(frames61["baseline_regime"].loc[tr_mask, "baseline_p_non_normal"],
                               frames61["challenger_ews"].loc[tr_mask, "challenger_alert_level"], level_map)
    d2_tr = frames61["challenger_ews"].loc[tr_mask, "challenger_alert_level"].map(level_map).mean()
    pre.append(Check("Train-calibrated constants reproduced exactly (P1 scale a, cost-stress threshold, B2 constant d)",
                     PASS if scale_re == a_scale and thr_re == stress_thr and abs(d2_tr - d_const) < 1e-15 else FAIL,
                     f"a {scale_re!r}; threshold {thr_re!r}; mean Train d2 {float(d2_tr)!r}"))
    stress_pt = bx.stress_indicator(simple_pt, stress_thr)
    sched_pt = pe.weekly_schedule(cal_pt, split_pt)
    inputs_pt = eng.prepare_inputs(simple_pt, sched_pt, policy.benchmark, cfg)
    t_first = min(inputs_pt)
    roles_pt = {r: bs.role_frame(run_pt, r) for r in h7.ROLES}
    d_pt = bs.defensiveness_series(roles_pt, a_scale, level_map)

    # ================================================================== 8A.4 Stage 7 reproduction (proxy engine)
    print("Reproducing the published Stage 7 targets with the frozen Stage 7 engine ...")
    s7_targets = pd.concat([pd.read_csv(S7_DIR / f"portfolio_targets_{s}.csv", parse_dates=["signal_date"])
                            for s in ("train", "validation")])
    rep_rows = []
    for pid in PRIMARY:
        tg, _, _, _ = eng.run_portfolio(pid, ROLE_OF[pid], simple_pt, sched_pt, inputs_pt, policy, cfg, cost_cfg,
                                        stress_pt, None if pid == "P0_strategic" else d_pt[pid], contributions=False)
        pub = s7_targets[s7_targets["portfolio"] == pid].sort_values("signal_date")
        same_dates = list(pub["signal_date"]) == list(tg["signal_date"])
        dw = float(np.abs(pub[[f"w_{a}" for a in assets]].to_numpy() - tg[[f"w_{a}" for a in assets]].to_numpy()).max()) \
            if same_dates else np.inf
        same_status = same_dates and bool((pub["solver_status"].to_numpy() == tg["solver_status"].to_numpy()).all())
        rep_rows.append({"portfolio": pid, "decisions": len(tg), "max_abs_weight_difference": dw, "same_status": same_status})
    rep_df = pd.DataFrame(rep_rows)
    dmax = rep_df["max_abs_weight_difference"].max()
    pre.append(Check("Frozen Stage 7 engine reproduces the published Stage 7 targets on this machine",
                     PASS if dmax <= 1e-8 and rep_df["same_status"].all() else (WARN if dmax <= 1e-6 else FAIL),
                     "; ".join(f"{r.portfolio}: {r.decisions} decisions, max |dw| {r.max_abs_weight_difference:.1e}"
                               for r in rep_df.itertuples())))

    # ================================================================== 8A.5 ledger through Train / Validation
    print("Running the daily ledger through Train/Validation (accounting verification only) ...")
    w6040 = np.array([protocol["supplementary_benchmarks"]["B1_60_40"]["weights"][a] for a in assets], float)

    def specs(dser: dict, start_w=None) -> dict:
        wb = policy.benchmark
        return {"P0_strategic": be.PortfolioSpec("P0_strategic", "optimizer", None, None, start_weights=wb),
                "P1_regime_aware": be.PortfolioSpec("P1_regime_aware", "optimizer", "baseline_regime",
                                                    dser["P1_regime_aware"], start_weights=wb),
                "P2_ews_enhanced": be.PortfolioSpec("P2_ews_enhanced", "optimizer", "challenger_ews",
                                                    dser["P2_ews_enhanced"], start_weights=wb),
                "B1_60_40": be.PortfolioSpec("B1_60_40", "fixed", None, None, fixed_target=w6040, fixed_cap=1.0,
                                             start_weights=w6040),
                "B2_constant_defensiveness": be.PortfolioSpec("B2_constant_defensiveness", "optimizer", None, d_const,
                                                              start_weights=wb)}

    bps_all = tuple(sorted({*protocol["transaction_costs"]["sensitivity_bps"], protocol["transaction_costs"]["primary_bps"]}))
    prim_bps = protocol["transaction_costs"]["primary_bps"]
    runs_pt = {}
    t0 = time.perf_counter()
    for pid, spec in specs(d_pt).items():
        runs_pt[pid] = be.run_sequential(spec, simple_pt, sched_pt, inputs_pt, policy, cfg, cost_cfg, stress_pt, t_first,
                                         bps_all, prim_bps)
    exe_dates_pt = set(sched_pt["execution_date"])
    rec_pt = []
    for pid, run in runs_pt.items():
        rec_pt += ba.reconcile(run.ledger, simple_pt, exe_dates_pt)
    rec_pt = pd.DataFrame(rec_pt)
    nf = rec_pt[rec_pt["status"] == FAIL]
    pre.append(Check("Ledger reconciliation on Train/Validation (all identities, all portfolios)",
                     PASS if nf.empty else FAIL, f"{len(rec_pt)} identity checks; failed: "
                     f"{', '.join(sorted(set(nf['check'])))[:150] or 'none'}"))
    hv = sum(int(r.decisions["hard_violations"].fillna("").ne("").sum()) for p, r in runs_pt.items() if p != "B1_60_40")
    pre.append(Check("Train/Validation decisions satisfy every hard constraint", PASS if hv == 0 else FAIL,
                     f"{hv} decisions with a hard violation"))
    pend = [p for p, r in runs_pt.items() if r.pending]
    pre.append(Check("No pending order at the close of the last Validation session (boundary handled by the calendar rule)",
                     PASS if not pend else WARN, f"pending: {', '.join(pend) or 'none'}; last Validation session "
                     f"{last_val.date()}; the week of {(last_val - pd.Timedelta(days=last_val.weekday())).date()} is "
                     "completed with Test sessions in Phase 8B"))
    # ledger-mode decisions vs published Stage 7 targets (difference only from realized holdings)
    lm_rows = []
    for pid in PRIMARY:
        dec = runs_pt[pid].decisions.set_index("signal_date")
        pub = s7_targets[s7_targets["portfolio"] == pid].set_index("signal_date")
        common = dec.index.intersection(pub.index)
        dw = np.abs(dec.loc[common, [f"w_{a}" for a in assets]].to_numpy() - pub.loc[common, [f"w_{a}" for a in assets]].to_numpy())
        lm_rows.append({"portfolio": pid, "decisions": len(common), "mean_abs_weight_difference": float(dw.mean()),
                        "max_abs_weight_difference": float(dw.max()),
                        "share_same_status": float((dec.loc[common, "solver_status"] == pub.loc[common, "solver_status"]).mean())})
    lm = pd.DataFrame(lm_rows)
    pre.append(Check("Ledger-holdings decisions close to the Stage 7 proxy decisions (differences only from realized drift)",
                     PASS if lm["mean_abs_weight_difference"].max() <= 0.005 else WARN,
                     "; ".join(f"{r.portfolio}: mean |dw| {r.mean_abs_weight_difference:.1e}, max {r.max_abs_weight_difference:.3f}, "
                               f"same status {r.share_same_status:.1%}" for r in lm.itertuples())))
    # causality on Train/Validation: perturb returns after a Validation decision date
    vdates = sorted(t for t in inputs_pt if split_pt.loc[t] == "validation")
    cut_r = vdates[len(vdates) // 2]
    start_v = va_feat.index[0]
    pert = simple_pt.copy()
    pert.loc[pert.index > cut_r] = pert.loc[pert.index > cut_r] * -3 + 0.01
    sched_v = sched_pt[sched_pt["signal_date"] >= start_v]
    inp_o = {t: inputs_pt[t] for t in sched_v["signal_date"] if t in inputs_pt}
    inp_p = eng.prepare_inputs(pert, sched_v, policy.benchmark, cfg)
    same_c = True
    for pid in ("P1_regime_aware", "P2_ews_enhanced"):
        sp_ = specs(d_pt)[pid]
        o = be.run_sequential(sp_, simple_pt, sched_v, inp_o, policy, cfg, cost_cfg, stress_pt, start_v, bps_all, prim_bps)
        q = be.run_sequential(sp_, pert, sched_v, inp_p, policy, cfg, cost_cfg, bx.stress_indicator(pert, stress_thr),
                              start_v, bps_all, prim_bps)
        a_ = o.decisions[o.decisions["signal_date"] <= cut_r][[f"w_{a}" for a in assets]].to_numpy()
        b_ = q.decisions[q.decisions["signal_date"] <= cut_r][[f"w_{a}" for a in assets]].to_numpy()
        wa_, wb_ = o.ledger.weights(), q.ledger.weights()
        same_c &= np.array_equal(a_, b_) and np.array_equal(wa_.loc[:cut_r].to_numpy(), wb_.loc[:cut_r].to_numpy())
    pre.append(Check("Future returns cannot alter past decisions or holdings (Validation perturbation test)",
                     PASS if same_c else FAIL, f"returns after {cut_r.date()} multiplied by -3; decisions and ledger "
                     "up to that date bit-identical"))
    ex = pd.to_datetime([e["date"] for e in manifest.get("excluded_dates_in_range", [])])
    ex_in = ex[(ex > t_first) & (ex <= last_val)]
    pre.append(Check("No excluded session inside the pre-Test ledger window", PASS if len(ex_in) == 0 else WARN,
                     f"{len(ex_in)} excluded sessions between {t_first.date()} and {last_val.date()}"))
    ex_te = ex[ex > last_val]
    pre.append(Check("Split manifest lists no excluded session after the last Validation date (a missing Test return "
                     "would be an omitted holding return)", PASS if len(ex_te) == 0 else FAIL,
                     f"{len(ex_te)} excluded dates (metadata of split_manifest.json only; no Test value read)"))
    print(f"Pre-Test ledger and causality checks done in {time.perf_counter() - t0:.0f}s.")
    init_rows, init_state = [], {}
    for pid, run in runs_pt.items():
        w = run.ledger.w
        prev = run.decisions[[f"w_{a}" for a in assets]].to_numpy()[-1]
        init_state[pid] = {"holdings": [repr(float(x)) for x in w], "previous_target": [repr(float(x)) for x in prev],
                           "last_decision": str(run.decisions["signal_date"].iloc[-1].date()), "pending": len(run.pending)}
        init_rows.append({"portfolio": pid, **{a: round(float(x), 4) for a, x in zip(assets, w)},
                          "last_pre_test_decision": str(run.decisions["signal_date"].iloc[-1].date())})
    init_rows.append({"portfolio": "B3_strategic_buy_and_hold", **{a: float(x) for a, x in zip(assets, policy.benchmark)},
                      "last_pre_test_decision": "none (bought at the last Validation close)"})
    init_state["B3_strategic_buy_and_hold"] = {"holdings": [repr(float(x)) for x in policy.benchmark]}
    init_sha = bi.fingerprint(init_state)

    # ================================================================== 8A.6 analytic examples and unit tests
    examples = pd.DataFrame(sp.analytic_examples())
    pre.append(Check("Analytic ledger examples (two-asset delayed rebalance, zero trades, drift, cap, t+1, round trip)",
                     PASS if (examples["status"] == PASS).all() else FAIL,
                     f"{len(examples)} quantities; max error {examples['abs_error'].max():.1e}"))
    if args.no_tests:
        unit = {"ok": False, "tests": 0, "failures": 0, "errors": 0, "skipped": 0, "log": "not run (--no-tests)"}
    else:
        print("Running Stage 8 unit tests (synthetic data only) ...")
        unit = run_unit_tests("test_backtest_*.py")
    pre.append(Check("Stage 8 unit tests pass before Test access (synthetic data; leakage-injection cases included)",
                     PASS if unit["ok"] and not unit["skipped"] else FAIL,
                     f"{unit['tests']} tests, {unit['failures']} failures, {unit['errors']} errors, {unit['skipped']} skipped"))

    # ================================================================== 8A.7 lock
    artifacts = {"configs/project.json": PROJECT_PATH, "configs/portfolio_policy.json": POLICY_PATH,
                 "configs/stage8_protocol.json": PROTOCOL_PATH,
                 "portfolio_optimization/stage8_handoff_manifest.json": s8_path,
                 "portfolio_optimization/portfolio_policy_manifest.json": S7_DIR / "portfolio_policy_manifest.json",
                 "early_warning/stage7_handoff/stage7_handoff_manifest.json": h61_path,
                 "splits/split_manifest.json": SPLITS_DIR / "split_manifest.json"}
    artifacts.update({f"splits/{n}": SPLITS_DIR / n for n in bi.MODEL_FILES})
    artifacts.update({f"models/{k}": ROOT / v for k, v in models.files.items()})
    artifacts.update({f"portfolio_optimization/portfolio_targets_{s}.csv": S7_DIR / f"portfolio_targets_{s}.csv"
                      for s in ("train", "validation")})
    for role in h7.ROLES:
        for s in ("train", "validation"):
            artifacts[f"early_warning/stage7_handoff/stage7_{role}_{s}.csv"] = EW_DIR / h7.HANDOFF_DIRNAME / f"stage7_{role}_{s}.csv"
    snap = bi.snapshot(ROOT, artifacts)
    versions = {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
                "scipy": scipy.__version__, "sklearn": sklearn.__version__}
    frozen = {
        "protocol_version": protocol["protocol_version"], "protocol": protocol, "hashes": snap,
        "frozen_policies": {
            "P0_strategic": {"benchmark": dict(zip(assets, policy.benchmark.tolist())), "posture": "benchmark",
                             "no_trade_band": cfg.no_trade_band, "turnover_cap": policy.turnover_cap},
            "P1_regime_aware": {"role": "baseline_regime", "column": "baseline_p_non_normal",
                                "d": f"min(1, {a_scale!r} * p)"},
            "P2_ews_enhanced": {"role": "challenger_ews", "column": "challenger_alert_level", "d": level_map},
            "B1_60_40": protocol["supplementary_benchmarks"]["B1_60_40"],
            "B2_constant_defensiveness": {"d": d_const, **protocol["supplementary_benchmarks"]["B2_constant_defensiveness"]},
            "B3_strategic_buy_and_hold": protocol["supplementary_benchmarks"]["B3_strategic_buy_and_hold"],
            "shared": {"risk_budget": pcfg["risk_budget"], "constraints": pcfg["constraints"],
                       "expected_returns": pcfg["expected_returns"], "risk_estimation": pcfg["risk_estimation"],
                       "optimizer": pcfg["optimizer"]}},
        "frozen_constants": {"P1_parity_scale_a": a_scale, "cost_stress_threshold_spy_21d": stress_thr,
                             "B2_constant_d": d_const, "baseline_q80_threshold": models.baseline_threshold,
                             "first_decision_date": str(t_first.date()), "last_validation_session": str(last_val.date())},
        "engine_config": {k: v for k, v in vars(cfg).items()},
        "model_versions": {"hmm": f"{models.files['hmm']} ({models.metas['hmm'].get('model_version')})",
                           "ar1": f"{models.files['ar1']} ({models.metas['ar1'].get('model_version')})",
                           "ar_hmm_k3": f"{models.files['ar_hmm_k3']} ({models.metas['ar_hmm_k3'].get('model_version')})",
                           "stage6_1_handoff": h61.get("code_version"), "stage7": s8.get("code_version"),
                           "stage7_policy": pcfg.get("policy_version"), "stage8_code": STAGE8_CODE_VERSION},
        "data_definitions": {"returns": manifest.get("return_definition"), "simple_returns": "R = exp(r) - 1",
                             "factors": ", ".join(manifest.get("feature_columns", [])),
                             "test_split": "Stage 3 test split (split_manifest.json); opened only by "
                                           "backtest_integrity.load_test_split after this lock",
                             "test_file_hashes_expected": {n: manifest.get("split_files", {}).get(n) for n in bi.TEST_FILES},
                             "train": f"{tr_feat.index.min().date()}..{tr_feat.index.max().date()} ({len(tr_feat)})",
                             "validation": f"{va_feat.index.min().date()}..{last_val.date()} ({len(va_feat)})",
                             "missing_data": "no filling; any missing Test value fails the Test integrity checks"},
        "initial_state_sha256": init_sha, "software_versions": versions}
    ambiguities = [
        "Pre-trade weights at a decision are the ACTUAL ledger holdings at the close of t (known at t). The Stage 7 "
        "proxy (previous target drifted through t) equals them whenever the previous order was executed in full; after a "
        "hold inside the no-trade band it would ignore the realized drift of the holding period (consecutive holds would "
        "accumulate the error). Train/Validation difference vs the published Stage 7 targets: " +
        "; ".join(f"{r.portfolio} mean |dw| {r.mean_abs_weight_difference:.1e}" for r in lm.itertuples()) + ".",
        "A hold decision is a no-trade order: nothing is executed at t+1 even if drift moves the holdings further; the "
        "next weekly decision re-evaluates them.",
        "Execution-day turnover above the decision's cap (drift between t and t+1) is handled by the Stage 7 straight-line "
        "transition rule (cap-preserving projection) and reported as an exception.",
        "The cost-stress flag is evaluated at the decision date t (Stage 7: 'known at the decision date') and applied to "
        "the t+1 execution; it is the same for every portfolio.",
        "The calendar week containing the last Validation session is completed with Test sessions; its decision (last "
        "session of that week) is the first Test-period decision. No other decision is created.",
        "Costs are deducted pro rata at the execution close; therefore the trading path is identical under 0 / 5 / 10 / "
        "25 bps and every cost scenario is evaluated on the same holdings."]
    preflight = {"checks": [vars(c) for c in pre], "unit_tests": {k: v for k, v in unit.items() if k != "log"},
                 "analytic_examples": examples.to_dict("records"), "stage7_reproduction": rep_df.to_dict("records"),
                 "ledger_vs_stage7": lm.to_dict("records"), "ambiguities": ambiguities}
    fails = [c.name for c in pre if c.status == FAIL]
    authorized = not fails
    reason = "all preflight checks PASS" if authorized else "FAILED: " + "; ".join(fails)
    lock = bi.build_lock(frozen, preflight, authorized, reason)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = OUT_DIR / bi.LOCK_NAME
    log_path = OUT_DIR / bi.ACCESS_LOG_NAME
    access_log = bi.read_access_log(log_path)
    mode = "first_evaluation"
    if access_log:
        first_fp = access_log[0]["lock_fingerprint"]
        if lock["fingerprint"] == first_fp:
            mode = "reproduction"
        elif args.amendment:
            mode = "amendment"
        else:
            changed = bi.compare_snapshot(json.loads(lock_path.read_text(encoding="utf-8"))["frozen"]["hashes"], snap) \
                if lock_path.exists() else []
            print(f"The Test split was already opened under lock {first_fp[:16]}... and the frozen content has changed "
                  f"({', '.join(changed[:6]) or 'protocol / constants'}). Re-run with --amendment \"reason\" to produce "
                  "a labelled amendment (the original outputs are archived), or restore the locked files.")
            recorder.__exit__(None, None, None)
            return 2
    if mode == "amendment":
        arch = OUT_DIR / "archive" / pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
        arch.mkdir(parents=True, exist_ok=True)
        for p in OUT_DIR.glob("*"):
            if p.is_file() and p.name != bi.ACCESS_LOG_NAME:
                shutil.copy2(p, arch / p.name)
        for p in (REPORTS / "final_backtesting_research.md", REPORTS / "stage8_pretest_protocol.md"):
            if p.exists():
                shutil.copy2(p, arch / p.name)
        print(f"Amendment: previous outputs archived to {arch.relative_to(ROOT)}")
    bi.write_json(lock, lock_path)
    init_tab = pd.DataFrame(init_rows).set_index("portfolio")
    (REPORTS / "stage8_pretest_protocol.md").write_text(
        sp.render_protocol(lock, [(c.name, c.status, c.detail) for c in pre], examples, protocol,
                           {"ambiguities": ambiguities, "initial_state": init_tab}), encoding="utf-8")
    print("\nPREFLIGHT SUMMARY")
    for c in pre:
        print(f"  [{c.status}] {c.name}: {c.detail}")
    cnt = pd.Series([c.status for c in pre]).value_counts()
    print(f"Preflight: {cnt.get(PASS, 0)} PASS, {cnt.get(WARN, 0)} WARN, {cnt.get(FAIL, 0)} FAIL. Lock "
          f"{lock['fingerprint'][:16]}... written; Test access authorized: {authorized}. Mode: {mode}.")
    if not authorized:
        return stop("Preflight FAILED: Test NOT opened (STAGE 8 BLOCKED).")
    if args.preflight_only:
        recorder.__exit__(None, None, None)
        print("--preflight-only: Test not opened.")
        return 0

    # ================================================================== PHASE 8B.1 open Test (gated)
    print("=" * 100 + f"\nPHASE 8B - RESERVED TEST EVALUATION ({mode})\n" + "=" * 100)
    checks: list = []
    test = bi.load_test_split(SPLITS_DIR, lock_path, bi.snapshot(ROOT, artifacts))
    bi.append_access_log(log_path, {"utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "mode": mode,
                                    "lock_fingerprint": lock["fingerprint"], "files": test.files_read,
                                    "sha256": test.hashes, "amendment_reason": args.amendment})
    access_log = bi.read_access_log(log_path)
    for name, status, detail in bi.test_data_checks(test, manifest, last_val, list(manifest["feature_columns"]), assets,
                                                    list(tr_feat.columns), list(data.train_returns.columns)):
        checks.append(Check(name, status, detail))
    checks.append(Check("Test opened only after a valid pre-Test lock (gated evaluation-only loader)", PASS,
                        f"lock {lock['fingerprint'][:16]}...; access #{len(access_log)} ({mode})"))
    if any(c.status == FAIL for c in checks):
        for c in checks:
            print(f"  [{c.status}] {c.name}: {c.detail}")
        recorder.__exit__(None, None, None)
        print("Test-data integrity FAILED: evaluation stopped (STAGE 8 BLOCKED).")
        return 1
    te_feat, te_ret = test.features, test.returns
    test_start, test_end = te_feat.index.min(), te_feat.index.max()
    print(f"Test split: {test_start.date()}..{test_end.date()} ({len(te_feat)} sessions).")

    # ================================================================== 8B.2 sequential signals through Test
    t0 = time.perf_counter()
    cal = cal_pt.append(te_feat.index)
    split_all = pd.concat([split_pt, pd.Series("test", index=te_feat.index)])
    cont = pd.concat([va_feat, te_feat])
    run = bs.sequential_signals(models, tr_feat, cont, split_all)
    d0 = bs.prefix_difference(run, run_pt)
    checks.append(Check("Appending Test leaves every Train/Validation signal unchanged (prefix invariance at the boundary)",
                        PASS if d0 == 0 else FAIL, f"max difference {d0:.1e} over {len(run_pt.descriptive)} sessions"))
    cmp2 = bs.compare_with_handoff(run, frames61)
    checks.append(Check("Continued signals still identical to the frozen Stage 6.1 handoff on Train/Validation",
                        PASS if all(v[0] for v in cmp2.values()) else FAIL, f"{len(cmp2)} columns"))
    const2 = bs.compare_frozen_constants(run, models)
    checks.append(Check("No frozen signal constant re-estimated (thresholds, priors, scaler identical)",
                        PASS if max(const2.values()) <= 1e-12 else FAIL, ", ".join(f"{k} {v:.1e}" for k, v in const2.items())))
    cut_t = te_feat.index[len(te_feat) // 2]
    run_c = bs.sequential_signals(models, tr_feat, pd.concat([va_feat, te_feat.loc[:cut_t]]), split_all)
    d1 = bs.prefix_difference(run, run_c)
    checks.append(Check("Test signal prefix invariance: running through t equals the full run truncated at t",
                        PASS if d1 == 0 else FAIL, f"cut {cut_t.date()}; max difference {d1:.1e} (HMM filter, BOCPD, "
                        "EWS score, alert levels)"))
    te_p = te_feat.copy()
    te_p.loc[te_p.index > cut_t, list(manifest["feature_columns"])] = te_p.loc[te_p.index > cut_t,
                                                                               list(manifest["feature_columns"])] * -3 + 2
    run_p = bs.sequential_signals(models, tr_feat, pd.concat([va_feat, te_p]), split_all)
    d2_ = bs.prefix_difference(run, run_p, upto=cut_t)
    after = bs.prefix_difference(run, run_p)
    checks.append(Check("Changing Test factors after t leaves earlier HMM probabilities, BOCPD posteriors, EWS scores "
                        "and alerts unchanged", PASS if d2_ == 0 and after > 0 else FAIL,
                        f"cut {cut_t.date()}; difference before {d2_:.1e}, after {after:.1e} (perturbation effective)"))
    lvl = run.roles["challenger_ews"]["challenger_alert_level"]
    checks.append(Check("Alert state machine and HMM filter continue across the boundary (no reset to GREEN / prior)",
                        PASS, f"EWS level {last_val.date()}: {lvl.loc[last_val]}; {test_start.date()}: "
                        f"{lvl.loc[test_start]}; P(non-normal) {last_val.date()}: "
                        f"{run.descriptive.loc[last_val, 'p_non_normal']:.4f} -> {run.descriptive.loc[test_start, 'p_non_normal']:.4f}"))
    print(f"Sequential Test signals built and verified in {time.perf_counter() - t0:.0f}s.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    roles = {r: bs.role_frame(run, r) for r in h7.ROLES}
    for r, f in roles.items():
        f[f["split"] == "test"].drop(columns="split").to_csv(OUT_DIR / f"signals_{r}_test.csv", index_label="signal_date",
                                                             date_format="%Y-%m-%d", float_format="%.17g")
    dser = bs.defensiveness_series(roles, a_scale, level_map)
    dser_alt = bs.defensiveness_series(roles, a_scale, level_map, "full_range_continuous", run.thresholds)

    # ================================================================== 8B.3 data, schedule, risk inputs
    log_all = pd.concat([data.train_returns, data.validation_returns, te_ret])[assets]
    simple = pr.to_simple(log_all)
    stress = bx.stress_indicator(simple, stress_thr)
    sched = pe.weekly_schedule(cal, split_all)
    same_pre = sched[sched["signal_date"] <= sched_pt["signal_date"].max()].reset_index(drop=True)
    sched_ok = same_pre[["signal_date", "execution_date"]].equals(sched_pt[["signal_date", "execution_date"]].reset_index(drop=True))
    boundary = sched[(sched["signal_date"] > last_val)].iloc[0]
    checks.append(Check("Weekly calendar: pre-Test decisions unchanged; boundary week completed by the frozen rule",
                        PASS if sched_ok else FAIL, f"first Test decision {boundary['signal_date'].date()} executes "
                        f"{boundary['execution_date'].date()} (week of {boundary['week_start'].date()}); "
                        f"{int((sched['split'] == 'test').sum())} Test decisions"))
    t0 = time.perf_counter()
    sched_te = sched[sched["signal_date"] > last_val]
    inputs = dict(inputs_pt)
    inputs.update(eng.prepare_inputs(simple, sched_te, policy.benchmark, cfg))
    print(f"Risk inputs for {len(sched_te)} Test decisions in {time.perf_counter() - t0:.0f}s. Simulating portfolios ...")

    # ================================================================== 8B.4 continuation runs (primary)
    t0 = time.perf_counter()
    runs = {}
    for pid, spec in specs(dser).items():
        runs[pid] = be.run_sequential(spec, simple, sched, inputs, policy, cfg, cost_cfg, stress, t_first, bps_all, prim_bps)
        print(f"  {pid}: {len(runs[pid].decisions)} decisions ({time.perf_counter() - t0:.0f}s)")
    runs["B3_strategic_buy_and_hold"] = be.run_sequential(
        be.PortfolioSpec("B3_strategic_buy_and_hold", "buy_and_hold", start_weights=policy.benchmark), simple, sched,
        inputs, policy, cfg, cost_cfg, stress, last_val, bps_all, prim_bps)
    same_init = True
    for pid, rp in runs_pt.items():
        a_ = rp.decisions[[f"w_{a}" for a in assets]].to_numpy()
        b_ = runs[pid].decisions[runs[pid].decisions["signal_date"] <= last_val][[f"w_{a}" for a in assets]].to_numpy()
        wa_ = rp.ledger.weights().to_numpy()
        wb_ = runs[pid].ledger.weights().loc[:last_val].to_numpy()
        same_init &= a_.shape == b_.shape and np.array_equal(a_, b_) and np.array_equal(wa_, wb_)
    init2 = {pid: {"holdings": [repr(float(x)) for x in runs[pid].ledger.weights().loc[last_val].to_numpy()],
                   "previous_target": [repr(float(x)) for x in runs[pid].decisions[runs[pid].decisions["signal_date"] <= last_val]
                                       [[f"w_{a}" for a in assets]].to_numpy()[-1]],
                   "last_decision": str(runs[pid].decisions[runs[pid].decisions["signal_date"] <= last_val]["signal_date"]
                                        .iloc[-1].date()), "pending": 0} for pid in runs_pt}
    init2["B3_strategic_buy_and_hold"] = {"holdings": [repr(float(x)) for x in policy.benchmark]}
    checks.append(Check("Test initialization depends only on pre-Test information (continuation = locked preflight state)",
                        PASS if same_init and bi.fingerprint(init2) == lock["frozen"]["initial_state_sha256"] else FAIL,
                        f"holdings at {last_val.date()} identical to the lock (sha {init_sha[:12]}...)"))

    # ================================================================== 8B.5 restart and model sensitivity runs
    t0 = time.perf_counter()

    def restart(dmap, cfg_v, inputs_v, only=None):
        out = {}
        for pid, spec in specs(dmap).items():
            if only and pid not in only:
                continue
            out[pid] = be.run_sequential(spec, simple, sched, inputs_v, policy, cfg_v, cost_cfg, stress, last_val,
                                         bps_all, prim_bps)
        return out

    runs_rs = restart(dser, cfg, inputs)
    runs_rs["B3_strategic_buy_and_hold"] = runs["B3_strategic_buy_and_hold"]
    sens_runs = {"restart_reference": runs_rs}
    for name, cv in (("cov_lw_252", cfg.variant(lookback=252)), ("cov_ewma_hl63", cfg.variant(cov_method="ewma"))):
        iv = eng.prepare_inputs(simple, sched_te, policy.benchmark, cv)
        sens_runs[name] = restart(dser, cv, iv, only=("P1_regime_aware", "P2_ews_enhanced"))
    sens_runs["mapping_P1_full_range_P2_continuous"] = restart(dser_alt, cfg, inputs,
                                                               only=("P1_regime_aware", "P2_ews_enhanced"))
    print(f"Restart and sensitivity runs in {time.perf_counter() - t0:.0f}s.")

    # causality on Test: perturb returns after a Test decision date (restart runs)
    tdates = [t for t in sched_te["signal_date"] if t in inputs]
    cut_d = tdates[len(tdates) // 2]
    pert = simple.copy()
    pert.loc[pert.index > cut_d] = pert.loc[pert.index > cut_d] * -3 + 0.01
    inp_pt = dict(inputs_pt)
    inp_pt.update(eng.prepare_inputs(pert, sched_te, policy.benchmark, cfg))
    stress_p = bx.stress_indicator(pert, stress_thr)
    same_d = True
    leak_detail = []
    for pid in ("P0_strategic", "P1_regime_aware", "P2_ews_enhanced"):
        q = be.run_sequential(specs(dser)[pid], pert, sched, inp_pt, policy, cfg, cost_cfg, stress_p, last_val, bps_all,
                              prim_bps)
        o = runs_rs[pid]
        a_ = o.decisions[o.decisions["signal_date"] <= cut_d][[f"w_{a}" for a in assets]].to_numpy()
        b_ = q.decisions[q.decisions["signal_date"] <= cut_d][[f"w_{a}" for a in assets]].to_numpy()
        ok = np.array_equal(a_, b_) and np.array_equal(o.ledger.weights().loc[:cut_d].to_numpy(),
                                                        q.ledger.weights().loc[:cut_d].to_numpy())
        same_d &= ok
        leak_detail.append(f"{pid} {'identical' if ok else 'CHANGED'}")
    checks.append(Check("Changing Test returns after t leaves every decision, covariance input and holding up to t "
                        "unchanged", PASS if same_d else FAIL, f"cut {cut_d.date()}: " + ", ".join(leak_detail)))

    # ================================================================== 8B.6 evaluation-window series
    test_dates = te_feat.index
    port_ids = list(PRIMARY) + list(SUPPLEMENTARY)

    def window_series(runs_: dict, ids):
        gross, net = {}, {b: {} for b in bps_all}
        for pid in ids:
            f = runs_[pid].ledger.frame().set_index("date")
            gross[pid] = f.loc[test_dates, "gross_return"]
            for b in bps_all:
                net[b][pid] = f.loc[test_dates, f"net_return_{b:g}bps"]
        return pd.DataFrame(gross), {b: pd.DataFrame(v) for b, v in net.items()}

    gross, net = window_series(runs, port_ids)
    rf = simple.loc[test_dates, "BIL"]
    netp = net[prim_bps]
    post = {pid: runs[pid].ledger.weights("post") for pid in port_ids}
    held = {pid: post[pid].shift(1).loc[test_dates] for pid in port_ids}
    ledgers_te = {pid: runs[pid].ledger.frame().set_index("date").loc[test_dates].reset_index() for pid in port_ids}
    exe_te = {pid: (runs[pid].executions[runs[pid].executions["execution_date"].isin(test_dates)].reset_index(drop=True)
                    if len(runs[pid].executions) else pd.DataFrame()) for pid in port_ids}
    dec_te = {pid: (runs[pid].decisions[runs[pid].decisions["signal_date"] > last_val].reset_index(drop=True)
                    if len(runs[pid].decisions) else pd.DataFrame()) for pid in port_ids}
    years = len(test_dates) / bp.ANN
    groups = policy.groups

    # ================================================================== 8B.7 metrics
    perf_rows = []
    for basis, frame in [("gross", gross)] + [(f"net_{b:g}bps", net[b]) for b in bps_all]:
        for pid in port_ids:
            m = bp.metrics(frame[pid], rf, {"P0": frame["P0_strategic"], "P1": frame["P1_regime_aware"]})
            impl = bp.implementation_stats(ledgers_te[pid], dec_te[pid], held[pid], groups, years, prim_bps)
            perf_rows.append({"portfolio": pid, "basis": basis, "evaluation_start": test_dates[0].date(),
                              "evaluation_end": test_dates[-1].date(),
                              "date_convention": "daily close-to-close; holdings of the previous close",
                              **m, **impl})
    perf = pd.DataFrame(perf_rows)
    perf_p = perf[perf["basis"] == f"net_{prim_bps:g}bps"].set_index("portfolio")
    gcagr = perf[perf["basis"] == "gross"].set_index("portfolio")["cagr"]
    perf["annualized_cost_drag_cagr"] = perf["portfolio"].map(gcagr) - perf["cagr"]
    annual = []
    for basis, frame in (("gross", gross), (f"net_{prim_bps:g}bps", netp)):
        for pid in port_ids:
            for per, v in bp.period_returns(frame[pid], "Y").items():
                annual.append({"year": str(per), "portfolio": pid, "basis": basis, "return": float(v),
                               "sessions": int((frame.index.to_period("Y") == per).sum())})
    annual = pd.DataFrame(annual)
    monthly = pd.DataFrame([{"month": str(per), "portfolio": pid, "basis": f"net_{prim_bps:g}bps", "return": float(v)}
                            for pid in port_ids for per, v in bp.period_returns(netp[pid], "M").items()])
    dd_events = pd.concat([bp.drawdown_episodes(netp[pid], 0.02).assign(portfolio=pid, basis=f"net_{prim_bps:g}bps")
                           for pid in port_ids], ignore_index=True)

    # ================================================================== 8B.8 stress windows and objective events
    sa = protocol["stress_analysis"]
    sig_ctx = {"p": roles["baseline_regime"]["baseline_p_non_normal"], "level": roles["challenger_ews"]["challenger_alert_level"]}
    win = bat.window_analysis(sa["fixed_windows"], netp, gross, held, post, ledgers_te, groups, sig_ctx, f"{prim_bps:g}bps")
    spy_log = te_ret["SPY"]
    events = ev.drawdown_events(spy_log)
    if not events.empty:
        lvl_c = spy_log.cumsum()
        events["trough_date"] = [lvl_c.loc[r.onset_date:r.end_date].idxmin() for r in events.itertuples()]
        events["major"] = events["trough_loss_pct"] <= -10
    exe_full = {pid: runs[pid].executions for pid in port_ids}
    eq_b = float(policy.group_vector("equity") @ policy.benchmark)
    evt = bat.event_analysis(events, netp, post, exe_full, groups, sig_ctx, cal, strategic_equity=eq_b) \
        if not events.empty else pd.DataFrame()
    crisis = pd.concat([win.assign(analysis="fixed calendar window"),
                        evt.assign(analysis="objective SPY drawdown event (Stage 6 rule)")], ignore_index=True, sort=False)
    n_events = len(events)
    n_major = int(events["major"].sum()) if n_events else 0

    # ================================================================== 8B.9 attribution, risk matching, statistics
    att_rows, att_daily = [], {}
    for a, b in protocol["attribution"]["pairs"]:
        s_, d_ = bat.attribution(a, b, held, simple, gross, netp, protocol["attribution"]["groups"])
        att_rows.append(s_)
        att_daily[(a, b)] = d_
    attribution = pd.concat(att_rows, ignore_index=True)
    pairs_primary = [("P1_regime_aware", "P0_strategic"), ("P2_ews_enhanced", "P0_strategic"),
                     ("P2_ews_enhanced", "P1_regime_aware")]
    pairs_const = [("P1_regime_aware", "B2_constant_defensiveness"), ("P2_ews_enhanced", "B2_constant_defensiveness")]
    st = protocol["statistics"]
    stat_rows = []
    for fam, pairs in (("primary", pairs_primary), ("dynamic_vs_constant", pairs_const)):
        tests = [bst.mean_test((netp[a] - netp[b]).to_numpy()) for a, b in pairs]
        adj = bst.holm([t["p_value"] for t in tests]) if fam == "primary" else [np.nan] * len(tests)
        for (a, b), t, h in zip(pairs, tests, adj):
            stat_rows.append({"family": fam, "portfolio_a": a, "portfolio_b": b, "statistic": "mean_active_return",
                              "method": "Newey-West HAC", "estimate": t["mean_annualized"], "ci_low": t["ci_low"],
                              "ci_high": t["ci_high"], "se": t["se_annualized"], "p_value": t["p_value"],
                              "p_value_holm": h, "n": t["n"], "nw_lag": t["nw_lag"]})
    t0 = time.perf_counter()
    boot_frames = []
    bs_cfg = st["bootstrap"]
    rets_b = netp[["P0_strategic", "P1_regime_aware", "P2_ews_enhanced", "B2_constant_defensiveness"]]
    for L, tag in ((bs_cfg["mean_block_length_primary"], "primary"), (bs_cfg["mean_block_length_secondary"], "secondary")):
        bf = bst.bootstrap_differences(rets_b, rf, pairs_primary + pairs_const, L, bs_cfg["replications"], bs_cfg["seed"])
        boot_frames.append(bf.assign(block_setting=tag))
    boot = pd.concat(boot_frames, ignore_index=True)
    print(f"Stationary bootstrap ({bs_cfg['replications']} x 2 block lengths) in {time.perf_counter() - t0:.0f}s.")
    for r in boot.itertuples():
        stat_rows.append({"family": "primary" if (r.portfolio_a, r.portfolio_b) in pairs_primary else "dynamic_vs_constant",
                          "portfolio_a": r.portfolio_a, "portfolio_b": r.portfolio_b, "statistic": r.statistic,
                          "method": f"stationary bootstrap, mean block {r.mean_block_length:g} ({r.block_setting})",
                          "estimate": r.estimate, "ci_low": r.ci_low, "ci_high": r.ci_high, "se": r.bootstrap_se,
                          "p_value": r.bootstrap_p_value, "p_value_holm": np.nan, "n": r.n, "nw_lag": np.nan})
    reg_rows = []
    for a, b in [("P1_regime_aware", "P0_strategic"), ("P2_ews_enhanced", "P0_strategic"),
                 ("B2_constant_defensiveness", "P0_strategic"), ("P2_ews_enhanced", "P1_regime_aware")]:
        rr = bst.ols_hac((netp[a] - rf).to_numpy(), (netp[b] - rf).to_numpy())
        reg_rows.append({"portfolio": a, "reference": b, **rr})
        stat_rows.append({"family": "risk_matching", "portfolio_a": a, "portfolio_b": b,
                          "statistic": "alpha_on_reference_excess", "method": "OLS, Newey-West HAC",
                          "estimate": rr["alpha_annualized"], "ci_low": rr["alpha_ci_low"], "ci_high": rr["alpha_ci_high"],
                          "se": rr["alpha_se_annualized"], "p_value": rr["alpha_p_value"], "p_value_holm": np.nan,
                          "n": rr["n"], "nw_lag": rr["nw_lag"]})
        stat_rows.append({"family": "risk_matching", "portfolio_a": a, "portfolio_b": b, "statistic": "beta_on_reference_excess",
                          "method": "OLS, Newey-West HAC (p-value vs beta = 1)", "estimate": rr["beta"],
                          "ci_low": rr["beta_ci_low"], "ci_high": rr["beta_ci_high"], "se": rr["beta_se"],
                          "p_value": rr["beta_p_value_vs_1"], "p_value_holm": np.nan, "n": rr["n"], "nw_lag": rr["nw_lag"]})
    stats_tab = pd.DataFrame(stat_rows)
    stats_tab["independent_spy_drawdown_events"] = n_events
    stats_tab["major_events_le_minus_10pct"] = n_major

    p0_vol = float(netp["P0_strategic"].std(ddof=1) * np.sqrt(bp.ANN))
    bench_rows = []
    for a, b in pairs_primary + pairs_const + [("P0_strategic", "B1_60_40"), ("P0_strategic", "B3_strategic_buy_and_hold")]:
        pa, pb_ = perf_p.loc[a], perf_p.loc[b]
        act = netp[a] - netp[b]
        te_ = float(act.std(ddof=1) * np.sqrt(bp.ANN))
        row = {"portfolio_a": a, "portfolio_b": b, "basis": f"net_{prim_bps:g}bps",
               "active_return_annualized": float(act.mean() * bp.ANN), "tracking_error": te_,
               "information_ratio": float(act.mean() * bp.ANN / te_) if te_ > 0 else np.nan,
               "cagr_difference": pa["cagr"] - pb_["cagr"], "volatility_difference": pa["annual_volatility"] - pb_["annual_volatility"],
               "sharpe_difference": pa["sharpe"] - pb_["sharpe"], "sortino_difference": pa["sortino"] - pb_["sortino"],
               "max_drawdown_difference": pa["max_drawdown"] - pb_["max_drawdown"],
               "calmar_difference": pa["calmar"] - pb_["calmar"],
               "cost_drag_difference": float(perf[(perf.portfolio == a) & (perf.basis == f"net_{prim_bps:g}bps")]["annualized_cost_drag_cagr"].iloc[0]
                                             - perf[(perf.portfolio == b) & (perf.basis == f"net_{prim_bps:g}bps")]["annualized_cost_drag_cagr"].iloc[0])}
        row.update({f"{k}_vs_b": v for k, v in bat.capture_ratios(netp[a], netp[b]).items()})
        vm = bat.vol_matched(netp[a], rf, p0_vol)
        row["descriptive_vol_matched_cagr_a"] = float((1 + vm).prod() ** (1 / years) - 1)
        row["descriptive_vol_matched_max_drawdown_a"] = bp.max_drawdown(vm.to_numpy())
        bench_rows.append(row)
    bench = pd.DataFrame(bench_rows)

    # cost sensitivity
    cost_rows = []
    for basis in ["gross"] + [f"net_{b:g}bps" for b in bps_all]:
        sub = perf[perf["basis"] == basis].set_index("portfolio")
        for pid in port_ids:
            cost_rows.append({"basis": basis, "portfolio": pid, "cagr": sub.loc[pid, "cagr"], "sharpe": sub.loc[pid, "sharpe"],
                              "sortino": sub.loc[pid, "sortino"], "max_drawdown": sub.loc[pid, "max_drawdown"],
                              "annual_volatility": sub.loc[pid, "annual_volatility"],
                              "total_cost_fraction": sub.loc[pid].get(f"total_cost_fraction_{basis.replace('net_', '')}", 0.0)
                              if basis != "gross" else 0.0})
        for a, b in pairs_primary + pairs_const:
            cost_rows.append({"basis": basis, "portfolio": f"{a} - {b}", "cagr": sub.loc[a, "cagr"] - sub.loc[b, "cagr"],
                              "sharpe": sub.loc[a, "sharpe"] - sub.loc[b, "sharpe"],
                              "sortino": sub.loc[a, "sortino"] - sub.loc[b, "sortino"],
                              "max_drawdown": sub.loc[a, "max_drawdown"] - sub.loc[b, "max_drawdown"],
                              "annual_volatility": sub.loc[a, "annual_volatility"] - sub.loc[b, "annual_volatility"],
                              "total_cost_fraction": np.nan})
    cost_sens = pd.DataFrame(cost_rows)

    # model / initialization sensitivity (restart protocol)
    ms_rows = []
    ref_w = {pid: sens_runs["restart_reference"][pid].ledger.weights().loc[test_dates] for pid in
             ("P0_strategic", "P1_regime_aware", "P2_ews_enhanced")}
    for variant, rr_ in sens_runs.items():
        g_, n_ = window_series(rr_, [p for p in rr_ if p in port_ids])
        for pid in n_[prim_bps].columns:
            m = bp.metrics(n_[prim_bps][pid], rf)
            lg = rr_[pid].ledger.frame().set_index("date").loc[test_dates]
            wv = rr_[pid].ledger.weights().loc[test_dates]
            ms_rows.append({"variant": variant, "initialization": "restart_benchmark_at_test_start", "portfolio": pid,
                            "basis": f"net_{prim_bps:g}bps", "cagr": m["cagr"], "annual_volatility": m["annual_volatility"],
                            "sharpe": m["sharpe"], "sortino": m["sortino"], "max_drawdown": m["max_drawdown"],
                            "annualized_one_way_turnover": float(lg["turnover"].sum() / years),
                            "mean_equity_weight": float(wv[groups["equity"]].sum(axis=1).mean()),
                            "mean_bil_weight": float(wv["BIL"].mean()),
                            "max_abs_weight_difference_vs_restart_reference": float(np.abs(wv - ref_w[pid]).to_numpy().max())
                            if pid in ref_w else np.nan,
                            "fallback_decisions": int((rr_[pid].decisions.get("solver_stage", pd.Series(dtype=float)) > 1).sum())
                            if len(rr_[pid].decisions) else 0})
    for pid in port_ids:
        m = perf_p.loc[pid]
        ms_rows.append({"variant": "primary", "initialization": "continuation (primary)", "portfolio": pid,
                        "basis": f"net_{prim_bps:g}bps", "cagr": m["cagr"], "annual_volatility": m["annual_volatility"],
                        "sharpe": m["sharpe"], "sortino": m["sortino"], "max_drawdown": m["max_drawdown"],
                        "annualized_one_way_turnover": m["annualized_one_way_turnover"],
                        "mean_equity_weight": m["mean_weight_equity"], "mean_bil_weight": m["mean_weight_BIL"],
                        "max_abs_weight_difference_vs_restart_reference": float(np.abs(post[pid].loc[test_dates] -
                                                                                       ref_w[pid]).to_numpy().max())
                        if pid in ref_w else np.nan, "fallback_decisions": int((dec_te[pid].get("solver_stage", pd.Series(dtype=float)) > 1).sum())
                        if len(dec_te[pid]) else 0})
    model_sens = pd.DataFrame(ms_rows)

    # ================================================================== 8B.10 constraint and execution audits
    con_rows, exc_rows = [], []
    lo, hi = pc.bounds(policy)
    for pid in port_ids:
        dts = dec_te[pid].copy()
        for c in ("hard_violations", "unmet_budget", "solver_warnings"):
            if c in dts:
                dts[c] = dts[c].fillna("")
        if len(dts):
            for r in dts.itertuples():
                st_ = r.solver_status
                cls = ("hold inside the no-trade band" if st_ == "hold_within_no_trade_band" else
                       "turnover-cap-constrained transition" if "transition" in st_ else
                       "trade to target" if st_ == "trade_to_target" else st_)
                if "unattainable" in st_:
                    cls = "risk budget unattainable (closest feasible posture)"
                if getattr(r, "solver_stage", 1) > 2:
                    cls = "numerical solver fallback (hold)"
                con_rows.append({"portfolio": pid, "date": r.signal_date, "stage": "decision-time target",
                                 "classification": cls, "hard_violations": r.hard_violations or "",
                                 "unmet_budget": r.unmet_budget or "", "solver_warnings": r.solver_warnings or ""})
                if pid != "B1_60_40" and r.hard_violations:
                    exc_rows.append({"date": r.signal_date, "portfolio": pid, "constraint": r.hard_violations,
                                     "reason": "decision-time target violates a hard constraint", "target_value": "",
                                     "actual_value": "", "trade_executable": "", "fallback_used": getattr(r, "solver_stage", 1) > 1})
                if r.unmet_budget and pid in ("P1_regime_aware", "P2_ews_enhanced", "B2_constant_defensiveness"):
                    exc_rows.append({"date": r.signal_date, "portfolio": pid, "constraint": r.unmet_budget,
                                     "reason": cls, "target_value": f"d = {r.budget_d_input:.3f}", "actual_value": "",
                                     "trade_executable": "yes" if r.order_kind == "trade" else "hold",
                                     "fallback_used": getattr(r, "solver_stage", 1) > 1})
        ex_ = exe_te[pid]
        for r in ex_.itertuples() if len(ex_) else []:
            con_rows.append({"portfolio": pid, "date": r.execution_date, "stage": "executed weights",
                             "classification": "cap-preserving projection" if r.projected_to_cap else r.order_kind,
                             "hard_violations": r.executed_hard_violations or "", "unmet_budget": "", "solver_warnings": ""})
            if r.projected_to_cap or (pid not in ("B1_60_40",) and r.executed_hard_violations):
                exc_rows.append({"date": r.execution_date, "portfolio": pid,
                                 "constraint": "turnover_cap" if r.projected_to_cap else r.executed_hard_violations,
                                 "reason": r.exceptions or "executed weights outside a hard limit",
                                 "target_value": f"{r.turnover_cap:.4f}", "actual_value": f"{r.execution_day_turnover_to_target:.4f}",
                                 "trade_executable": "partially (projection)" if r.projected_to_cap else "yes",
                                 "fallback_used": False})
        if pid not in ("B1_60_40",):
            h = held[pid]
            for i, a in enumerate(assets):
                for nm, bad in ((f"min_{a}", h[a] < lo[i] - 1e-9), (f"max_{a}", h[a] > hi[i] + 1e-9)):
                    if bad.any():
                        con_rows.append({"portfolio": pid, "date": h.index[bad.to_numpy()][0], "stage": "passive drift",
                                         "classification": f"{int(bad.sum())} sessions outside {nm}",
                                         "hard_violations": nm, "unmet_budget": "", "solver_warnings": ""})
            for g in groups:
                e = h[groups[g]].sum(axis=1)
                for nm, bad in ((f"{g}_max", e > policy.group_max[g] + 1e-9), (f"{g}_min", e < policy.group_min[g] - 1e-9)):
                    if bad.any():
                        con_rows.append({"portfolio": pid, "date": h.index[bad.to_numpy()][0], "stage": "passive drift",
                                         "classification": f"{int(bad.sum())} sessions outside {nm}",
                                         "hard_violations": nm, "unmet_budget": "", "solver_warnings": ""})
    con_audit = pd.DataFrame(con_rows)
    exceptions = pd.DataFrame(exc_rows, columns=["date", "portfolio", "constraint", "reason", "target_value", "actual_value",
                                                 "trade_executable", "fallback_used"])
    pos = {d: i for i, d in enumerate(cal)}
    exa_rows = []
    for pid in port_ids:
        ex_ = exe_te[pid]
        dmap = dec_te[pid].set_index("signal_date") if len(dec_te[pid]) else None
        for r in ex_.itertuples() if len(ex_) else []:
            nxt = pos[r.execution_date] == pos[r.signal_date] + 1
            fh = dmap.loc[r.signal_date, "first_holding_return_date"] if dmap is not None and r.signal_date in dmap.index else pd.NaT
            fh_ok = pd.isna(fh) or pos.get(fh, -1) == pos[r.execution_date] + 1
            exa_rows.append({"portfolio": pid, "signal_date": r.signal_date, "execution_date": r.execution_date,
                             "execution_is_next_session": nxt, "first_holding_return_date": fh, "first_holding_is_t_plus_2": fh_ok,
                             "order_kind": r.order_kind, "decision_status": r.decision_status, "turnover": r.turnover,
                             "gross_notional": r.gross_notional, "projected_to_cap": r.projected_to_cap,
                             "fraction_of_move_executed": r.fraction_of_move_executed, "stress_on": r.stress_on,
                             "executed_hard_violations": r.executed_hard_violations,
                             "solver_warnings": dmap.loc[r.signal_date, "solver_warnings"] if dmap is not None and
                             r.signal_date in dmap.index and "solver_warnings" in dmap else ""})
    exe_audit = pd.DataFrame(exa_rows)
    warn_all = pd.DataFrame([w for pid in port_ids for w in runs[pid].warnings])
    warn_te = warn_all[warn_all["signal_date"] > last_val] if len(warn_all) else warn_all

    # ================================================================== 8B.11 checks
    rec = pd.DataFrame([r for pid in port_ids for r in ba.reconcile(runs[pid].ledger, simple, set(sched["execution_date"]))])
    for name, g in rec.groupby("check", sort=False):
        bad = g[g["status"] == FAIL]
        checks.append(Check("Ledger: " + name, PASS if bad.empty else FAIL,
                            f"{len(g)} portfolios; failed: {', '.join(bad['portfolio']) or 'none'}; max error "
                            f"{g['max_error'].max():.1e}" if g["max_error"].notna().any() else
                            f"{len(g)} portfolios; failed: {', '.join(bad['portfolio']) or 'none'}"))
    first_ok = all(np.isfinite(gross.iloc[0]).tolist())
    checks.append(Check("First Test session return earned by the holdings carried from the last Validation close",
                        PASS if first_ok and abs(float(held["P0_strategic"].iloc[0] @ simple.loc[test_dates[0]]) -
                                                 gross.loc[test_dates[0], "P0_strategic"]) < 1e-14 else FAIL,
                        f"{test_dates[0].date()} included for every portfolio (no split-boundary gap)"))
    t_ok = (exe_audit["execution_is_next_session"].all() and exe_audit["first_holding_is_t_plus_2"].all()) if len(exe_audit) else True
    checks.append(Check("Execution timing: every order executes at the close of the next session (t+1); new holdings first "
                        "earn t+2", PASS if t_ok else FAIL, f"{len(exe_audit)} Test executions audited"))
    viol = []
    for pid in ("P1_regime_aware", "P2_ews_enhanced", "P0_strategic"):
        ex_ = exe_te[pid]
        f = runs[pid].ledger.frame().set_index("date")
        wpost = post[pid]
        for r in ex_[ex_["gross_notional"] > 0].itertuples() if len(ex_) else []:
            g_exec = f.loc[r.execution_date, "gross_return"]
            old = float(wpost.loc[r.signal_date] @ simple.loc[r.execution_date])
            if abs(g_exec - old) > 1e-14:
                viol.append(f"{pid} {r.execution_date.date()}")
            pre_l = runs[pid].ledger.weights("pre").loc[r.execution_date].to_numpy()
            if np.abs(pre_l - np.array([getattr(r, f"pre_{a}") for a in assets])).max() > 1e-15:
                viol.append(f"{pid} {r.execution_date.date()} pre")
    checks.append(Check("Previous holdings earn the t+1 return; new targets never earn it; traded notional uses "
                        "execution-time pre-trade holdings", PASS if not viol else FAIL,
                        f"violations: {', '.join(viol[:4]) or 'none'}"))
    hv_te = sum(int(dec_te[p]["hard_violations"].fillna("").ne("").sum()) for p in port_ids if p != "B1_60_40" and len(dec_te[p]))
    hv_ex = sum(int(exe_te[p]["executed_hard_violations"].fillna("").ne("").sum()) for p in port_ids
                if p not in ("B1_60_40",) and len(exe_te[p]))
    checks.append(Check("Hard constraints: every Test decision-time target and executed portfolio compliant",
                        PASS if hv_te == 0 and hv_ex == 0 else FAIL, f"targets violating {hv_te}; executions violating "
                        f"{hv_ex}; passive-drift episodes reported separately: "
                        f"{int((con_audit['stage'] == 'passive drift').sum()) if len(con_audit) else 0}"))
    proj = int(sum(int(exe_te[p]["projected_to_cap"].sum()) for p in port_ids if len(exe_te[p])))
    checks.append(Check("Execution-day turnover cap: exceptions handled by the frozen projection and reported",
                        PASS, f"{proj} cap-preserving projections in Test (listed in portfolio_constraint_audit.csv)"))
    fb = sum(int((dec_te[p].get("solver_stage", pd.Series(dtype=float)) > 2).sum()) for p in port_ids if len(dec_te[p]))
    checks.append(Check("Optimizer: no hold fallback in Test (numerical failure)", PASS if fb == 0 else FAIL,
                        f"{fb} stage-3 fallbacks"))
    nw = len(warn_te)
    checks.append(Check("Solver warnings documented; every affected final solution independently verified feasible",
                        PASS if nw == 0 else WARN,
                        f"{nw} warning records in Test decisions" + (f": {warn_te['warning'].value_counts().index[0][:110]} "
                                                                     f"(x{int(warn_te['warning'].value_counts().iloc[0])})" if nw else "")
                        + "; hard-constraint check of those decisions: " + ("all compliant" if hv_te == 0 else "violations")))
    p0_ok = (dec_te["P0_strategic"]["budget_d_input"].isna().all() and
             np.allclose(dec_te["P0_strategic"][[f"posture_{a}" for a in assets]].to_numpy(), policy.benchmark))
    roles_ok = (dec_te["P1_regime_aware"]["signal_role"].eq("baseline_regime").all() and
                dec_te["P2_ews_enhanced"]["signal_role"].eq("challenger_ews").all())
    d1_chk = np.allclose(dec_te["P1_regime_aware"].set_index("signal_date")["budget_d_input"],
                         np.minimum(1, a_scale * roles["baseline_regime"]["baseline_p_non_normal"]
                                    .loc[dec_te["P1_regime_aware"]["signal_date"]].to_numpy()), atol=0, rtol=0)
    d2_chk = (dec_te["P2_ews_enhanced"].set_index("signal_date")["budget_d_input"].to_numpy() ==
              roles["challenger_ews"]["challenger_alert_level"].loc[dec_te["P2_ews_enhanced"]["signal_date"]].map(level_map)
              .to_numpy()).all()
    checks.append(Check("Role separation: P1 only baseline_regime, P2 only challenger_ews, P0 no signal",
                        PASS if p0_ok and roles_ok and d1_chk and d2_chk else FAIL,
                        f"P0 posture = benchmark {p0_ok}; roles {roles_ok}; d1 = min(1, a p) {d1_chk}; d2 = level map {d2_chk}"))
    same_sched = all(list(dec_te[p]["signal_date"]) == list(dec_te["P0_strategic"]["signal_date"])
                     for p in ("P1_regime_aware", "P2_ews_enhanced", "B2_constant_defensiveness", "B1_60_40"))
    checks.append(Check("P0, P1, P2 (and B1, B2) share universe, returns, calendar and execution convention",
                        PASS if same_sched else FAIL, f"{len(dec_te['P0_strategic'])} Test decisions each"))
    wk = pd.concat([exe_te[p].assign(portfolio=p) for p in port_ids if len(exe_te[p])])
    wk["week"] = wk["execution_date"] - pd.to_timedelta(wk["execution_date"].dt.weekday, unit="D")
    checks.append(Check("At most one execution per portfolio and calendar week", PASS if not wk.duplicated(["portfolio", "week"]).any()
                        else FAIL, f"{len(wk)} executions"))
    order_ok, gaps = True, []
    for p in port_ids:
        wt = [float((1 + gross[p]).prod())] + [float((1 + net[b][p]).prod()) for b in bps_all]
        order_ok &= all(x >= y - 1e-15 for x, y in zip(wt, wt[1:]))
        tot = ledgers_te[p][f"cost_fraction_{prim_bps:g}bps"].sum()
        gaps.append(tot)
    checks.append(Check("Cost scenarios on one trading path: terminal wealth gross >= 5 >= 10 >= 25 bps for every "
                        "portfolio (costs reduce wealth only, counted once)", PASS if order_ok else FAIL,
                        f"total {prim_bps:g} bps cost fractions: " + ", ".join(f"{p.split('_')[0]} {g:.4f}"
                                                                         for p, g in zip(port_ids, gaps))))
    snap_after = bi.snapshot(ROOT, artifacts)
    lock_still = not bi.compare_snapshot(lock["frozen"]["hashes"], snap_after)
    hashes_after = frozen_dir_hashes()
    changed = sorted(k for k in set(hashes_before) | set(hashes_after) if hashes_before.get(k) != hashes_after.get(k))
    checks.append(Check("No Test result changed a frozen model, signal, policy or configuration (lock hashes after == "
                        "before)", PASS if lock_still and not changed else FAIL,
                        f"{len(snap_after['code'])} code + {len(snap_after['artifacts'])} artifact hashes; frozen "
                        f"Stage 5-7 files changed: {', '.join(changed[:4]) or 'none'}"))
    pre_fail = [c for c in pre if c.status == FAIL]
    checks.insert(0, Check("Pre-Test protocol frozen and Phase 8A passed before Test access", PASS if not pre_fail else FAIL,
                           f"lock {lock['fingerprint'][:16]}...; preflight {cnt.get(PASS, 0)} PASS, {cnt.get(WARN, 0)} WARN"))
    if mode == "amendment":
        checks.append(Check("Evaluation mode", WARN, f"AMENDMENT after Test access: {args.amendment}"))
    elif mode == "reproduction":
        checks.append(Check("Evaluation mode", PASS, f"reproduction of the first evaluation (identical lock, access "
                                                    f"#{len(access_log)})"))

    # ================================================================== 8B.12 outputs
    def write(frame: pd.DataFrame, name: str):
        frame.to_csv(OUT_DIR / name, index=False, float_format="%.10g", date_format="%Y-%m-%d")

    longs = []
    for pid in port_ids:
        f = runs[pid].ledger.frame().set_index("date").loc[test_dates]
        w0 = f.loc[test_dates[0], "wealth_gross"] / (1 + f.loc[test_dates[0], "gross_return"])
        base_w = {c: runs[pid].ledger.frame().set_index("date").loc[last_val, c] for c in
                  ["wealth_gross"] + [f"wealth_net_{b:g}bps" for b in bps_all]}
        g = pd.DataFrame({"date": test_dates, "portfolio": pid,
                          "return_basis_note": "daily simple return of the holdings of the previous close",
                          "gross_return": f["gross_return"].to_numpy()})
        for b in bps_all:
            g[f"net_return_{b:g}bps"] = f[f"net_return_{b:g}bps"].to_numpy()
        g["wealth_gross"] = (f["wealth_gross"] / base_w["wealth_gross"]).to_numpy()
        for b in bps_all:
            g[f"wealth_net_{b:g}bps"] = (f[f"wealth_net_{b:g}bps"] / base_w[f"wealth_net_{b:g}bps"]).to_numpy()
        longs.append(g)
        del w0
    daily = pd.concat(longs, ignore_index=True)
    write(daily[["date", "portfolio", "return_basis_note", "gross_return"]], "portfolio_daily_gross_returns.csv")
    write(daily[["date", "portfolio", "return_basis_note"] + [f"net_return_{b:g}bps" for b in bps_all]],
          "portfolio_daily_net_returns.csv")
    write(daily[["date", "portfolio", "wealth_gross"] + [f"wealth_net_{b:g}bps" for b in bps_all]].assign(
        normalization=f"1.0 at the close of {last_val.date()}"), "portfolio_daily_wealth.csv")
    hold_rows = []
    for pid in port_ids:
        w = post[pid].loc[test_dates]
        f = runs[pid].ledger.frame().set_index("date").loc[test_dates]
        h = w.copy()
        h.insert(0, "portfolio", pid)
        h["weights_basis"] = "end-of-session (post-trade) weights"
        h["wealth_net_primary_normalized"] = (f[f"wealth_net_{prim_bps:g}bps"] /
                                              runs[pid].ledger.frame().set_index("date").loc[last_val, f"wealth_net_{prim_bps:g}bps"]).to_numpy()
        for a in assets:
            h[f"value_{a}"] = h[a] * h["wealth_net_primary_normalized"]
        hold_rows.append(h.reset_index())
    write(pd.concat(hold_rows, ignore_index=True), "portfolio_daily_holdings.csv")
    dec_all = pd.concat([dec_te[p] for p in port_ids if len(dec_te[p])], ignore_index=True)
    sig_val = []
    for r in dec_all.itertuples():
        if r.portfolio == "P1_regime_aware":
            sig_val.append(repr(float(roles["baseline_regime"].at[r.signal_date, "baseline_p_non_normal"])))
        elif r.portfolio == "P2_ews_enhanced":
            sig_val.append(roles["challenger_ews"].at[r.signal_date, "challenger_alert_level"])
        else:
            sig_val.append("")
    dec_all.insert(3, "signal_value", sig_val)
    lead = ["portfolio", "signal_role", "signal_date", "signal_value", "execution_date", "first_holding_return_date", "split",
            "budget_d_input", "order_kind", "solver_status"]
    write(dec_all[lead + [c for c in dec_all.columns if c not in lead]], "portfolio_decision_targets.csv")
    exe_all = pd.concat([exe_te[p] for p in port_ids if len(exe_te[p])], ignore_index=True)
    write(exe_all, "portfolio_executed_weights.csv")
    trades = []
    for r in exe_all.itertuples():
        for a in assets:
            trades.append({"portfolio": r.portfolio, "signal_date": r.signal_date, "execution_date": r.execution_date,
                           "asset": a, "pre_trade_weight": getattr(r, f"pre_{a}"), "post_trade_weight": getattr(r, f"executed_{a}"),
                           "trade_weight_of_pre_trade_wealth": getattr(r, f"executed_{a}") - getattr(r, f"pre_{a}")})
    trades = pd.DataFrame(trades)
    write(trades[trades["trade_weight_of_pre_trade_wealth"].abs() > 0], "portfolio_trades.csv")
    tc_rows = []
    for pid in port_ids:
        f = ledgers_te[pid]
        f = f[f["gross_notional"] > 0]
        for r in f.itertuples():
            row = {"portfolio": pid, "execution_date": r.date, "signal_date": r.executed_signal_date,
                   "gross_notional": r.gross_notional, "one_way_turnover": r.turnover, "stress_on": r.stress_on,
                   "stress_multiplier": cost_cfg["stress_friction_multiplier"] if r.stress_on else 1.0,
                   "cost_amount_primary_normalized_wealth": r.cost_amount_primary /
                   runs[pid].ledger.frame().set_index("date").loc[last_val, f"wealth_net_{prim_bps:g}bps"]}
            for b in bps_all:
                row[f"cost_fraction_{b:g}bps"] = getattr(r, f"cost_fraction_{b:g}bps")
            tc_rows.append(row)
    write(pd.DataFrame(tc_rows), "portfolio_transaction_costs.csv")
    turn_rows = []
    for pid in port_ids:
        f = ledgers_te[pid].set_index("date")
        for yr, g in f.groupby(f.index.year):
            turn_rows.append({"portfolio": pid, "period": str(yr), "sessions": len(g),
                              "executions_with_trades": int((g["gross_notional"] > 0).sum()),
                              "one_way_turnover": float(g["turnover"].sum()), "gross_notional": float(g["gross_notional"].sum()),
                              **{f"cost_fraction_{b:g}bps": float(g[f"cost_fraction_{b:g}bps"].sum()) for b in bps_all}})
        m = perf_p.loc[pid]
        turn_rows.append({"portfolio": pid, "period": "Test (annualized)", "sessions": len(f),
                          "executions_with_trades": int(m["executions_with_trades"]),
                          "one_way_turnover": m["annualized_one_way_turnover"],
                          "gross_notional": m["gross_traded_notional_total"] / years,
                          **{f"cost_fraction_{b:g}bps": m[f"total_cost_fraction_{b:g}bps"] / years for b in bps_all}})
    turnover = pd.DataFrame(turn_rows)
    write(turnover, "portfolio_turnover.csv")
    write(pd.concat([con_audit.assign(record="audit"), exceptions.assign(record="exception")], ignore_index=True, sort=False),
          "portfolio_constraint_audit.csv")
    write(exe_audit, "portfolio_execution_audit.csv")
    write(perf, "portfolio_performance_summary.csv")
    write(annual, "portfolio_annual_returns.csv")
    write(monthly, "portfolio_monthly_returns.csv")
    write(dd_events, "portfolio_drawdown_events.csv")
    write(crisis, "portfolio_crisis_analysis.csv")
    write(attribution, "portfolio_return_attribution.csv")
    write(bench, "portfolio_benchmark_comparison.csv")
    write(stats_tab, "portfolio_statistical_tests.csv")
    write(cost_sens, "portfolio_cost_sensitivity.csv")
    write(model_sens, "portfolio_model_sensitivity.csv")
    write(rec, "portfolio_ledger_reconciliation.csv")
    if len(warn_all):
        write(warn_all, "portfolio_solver_warnings.csv")
    inputs_manifest = {
        "stage": "8 - inputs of the sequential out-of-sample backtest", "code_version": STAGE8_CODE_VERSION,
        "lock_fingerprint": lock["fingerprint"], "evaluation_mode": mode,
        "files_read": data.files_read + test.files_read + [str(p.relative_to(ROOT)) for p in artifacts.values()],
        "input_sha256": {**snap["artifacts"], **{f"splits/{k}": v for k, v in test.hashes.items()}},
        "frozen_constants": lock["frozen"]["frozen_constants"],
        "signals": {"baseline_regime": "data/processed/backtesting/signals_baseline_regime_test.csv",
                    "challenger_ews": "data/processed/backtesting/signals_challenger_ews_test.csv",
                    "generation": "early_warning.build_signals continued Validation -> Test (frozen models, no refit)"},
        "test_period": {"first": str(test_start.date()), "last": str(test_end.date()), "sessions": len(test_dates)}}
    bi.write_json(inputs_manifest, OUT_DIR / "backtest_inputs_manifest.json")

    # ================================================================== 8B.13 conclusion rules (prespecified)
    concl = brep.conclusions(protocol, perf, stats_tab, attribution, cost_sens, reg_rows, evt, win, prim_bps)

    # ================================================================== 8B.14 automated tests on the real outputs
    test_summary = {"ran": False}
    out_hashes = {p.name: bi.sha256(p) for p in sorted(OUT_DIR.glob("*.csv"))}
    test_manifest = {
        "stage": "8 - final sequential out-of-sample backtest", "code_version": STAGE8_CODE_VERSION,
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "evaluation_mode": mode,
        "amendment_reason": args.amendment, "lock_fingerprint": lock["fingerprint"], "test_access_log": access_log,
        "test_period": inputs_manifest["test_period"], "primary_cost_bps": prim_bps, "portfolios": port_ids,
        "conventions": {"returns": "daily simple returns R = exp(r) - 1 of adjusted closes; close-to-close",
                        "wealth": f"1.0 at the close of {last_val.date()}", "execution": "conservative_close_t_plus_1",
                        "gross": "before transaction costs, same holdings path", "net": "after costs (5 / 10 / 25 bps)"},
        "outputs": out_hashes, "conclusions": concl, "release_status": "PENDING"}
    bi.write_json(test_manifest, OUT_DIR / "portfolio_test_manifest.json")
    if not args.no_tests:
        print("Running automated checks on the real Stage 8 outputs ...")
        res = run_unit_tests("test_stage8_real_data.py")
        (OUT_DIR / "real_data_test_results.txt").write_text(res["log"], encoding="utf-8")
        test_summary = {"ran": True, **{k: v for k, v in res.items() if k != "log"}}
        checks.append(Check("Automated tests on the real Stage 8 outputs", PASS if res["ok"] and not res["skipped"] else FAIL,
                            f"{res['tests']} tests, {res['failures']} failures, {res['errors']} errors, {res['skipped']} skipped"))
    else:
        checks.append(Check("Automated tests on the real Stage 8 outputs", FAIL, "not run (--no-tests)"))

    # ================================================================== 8B.15 figures, report, status
    ctx = {"protocol": protocol, "policy": policy, "assets": assets, "lock": lock, "mode": mode, "amendment": args.amendment,
           "access_log": access_log, "perf": perf, "perf_p": perf_p, "gross": gross, "net": net, "netp": netp, "rf": rf,
           "held": held, "post": post, "ledgers": ledgers_te, "decisions": dec_te, "executions": exe_te, "roles": roles,
           "descriptive": run.descriptive, "annual": annual, "monthly": monthly, "dd_events": dd_events, "windows": win,
           "events": evt, "spy_events": events, "attribution": attribution, "att_daily": att_daily, "bench": bench,
           "stats": stats_tab, "regressions": pd.DataFrame(reg_rows), "cost_sens": cost_sens, "model_sens": model_sens,
           "turnover": turnover, "con_audit": con_audit, "exceptions": exceptions, "exe_audit": exe_audit,
           "warnings_te": warn_te, "checks": checks, "pre_checks": pre, "unit": unit, "test_summary": test_summary,
           "test_start": test_start, "test_end": test_end, "last_val": last_val, "t_first": t_first, "years": years,
           "prim_bps": prim_bps, "bps_all": bps_all, "concl": concl, "init_tab": init_tab, "a_scale": a_scale,
           "d_const": d_const, "stress_thr": stress_thr, "lm": lm, "rep_df": rep_df, "boundary": boundary,
           "n_events": n_events, "n_major": n_major, "versions": versions, "ambiguities": ambiguities,
           "sched_te": sched_te, "figures": []}
    if not args.no_figures:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        ctx["figures"] = brep.save_figures(FIG_DIR, ctx)
    recorder.__exit__(None, None, None)
    warn_list = sorted({f"{w.category.__name__}: {str(w.message)[:150]}" for w in caught})
    checks.append(Check("No library warnings outside the documented solver warnings", PASS if not warn_list else WARN,
                        "; ".join(warn_list)[:300] or "none"))
    counts = pd.Series([c.status for c in checks]).value_counts()
    fails = [c.name for c in checks if c.status == FAIL]
    stage8 = FAIL if fails else (WARN if counts.get(WARN, 0) else PASS)
    final = "BLOCKED" if fails else ("COMPLETE WITH LIMITATIONS" if counts.get(WARN, 0) else "COMPLETE")
    ctx.update({"stage8_status": stage8, "final_status": final, "counts": counts})
    pd.DataFrame([vars(c) for c in checks]).to_csv(OUT_DIR / "stage8_checks.csv", index=False)
    test_manifest.update({"release_status": final, "stage8_status": stage8,
                          "checks": [vars(c) for c in checks], "real_data_tests": test_summary,
                          "figures": ctx["figures"], "outputs": {p.name: bi.sha256(p) for p in sorted(OUT_DIR.glob("*.csv"))}})
    bi.write_json(test_manifest, OUT_DIR / "portfolio_test_manifest.json")
    (REPORTS / "final_backtesting_research.md").write_text(brep.render_report(ctx), encoding="utf-8")
    brep.print_console(ctx)
    print("Checks:")
    for c in checks:
        print(f"  [{c.status}] {c.name}: {c.detail}")
    print("Summary: " + ", ".join(f"{counts.get(s_, 0)} {s_}" for s_ in (PASS, WARN, FAIL)))
    print(f"STAGE 8: {stage8}\nFINAL RESEARCH STATUS: {final}")
    print(f"Report: {REPORTS / 'final_backtesting_research.md'}; figures {len(ctx['figures'])}. Elapsed "
          f"{time.perf_counter() - started:.0f}s. Stage 8 complete; no model was re-optimized.")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
