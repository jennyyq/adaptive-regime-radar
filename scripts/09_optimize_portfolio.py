"""Stage 7: robust portfolio optimization, dynamic risk budgeting and allocation-policy design (Train/Validation only).

P0_strategic     frozen strategic weights (no signal)
P1_regime_aware  risk budget from role baseline_regime   (baseline_p_non_normal)
P2_ews_enhanced  risk budget from role challenger_ews    (challenger_alert_level)
All three share the universe, returns, covariance estimator, expected returns, optimizer, constraints, cost model,
weekly calendar and the frozen execution convention (signal t, execution close t+1, first holding return t+2).

Reads ONLY: data/splits/{train,validation}_{features,returns}.csv + split_manifest.json (Test never opened), the
Stage 6.1 handoff through regime_radar.stage7_handoff.load_stage7_inputs, configs/project.json and
configs/portfolio_policy.json. Frozen Stage 5 / 5.5 / 6 / 6.1 artifacts are hashed before and after.
No realized portfolio performance is computed (Stage 8).
"""

from __future__ import annotations

import argparse
import json
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

STAGE7_CODE_VERSION = "stage7-2026-10-21"
SPLITS_DIR = ROOT / "data" / "splits"
EW_DIR = PROCESSED / "early_warning"
OUT_DIR = PROCESSED / "portfolio_optimization"
FIG_DIR = REPORTS / "figures" / "portfolio_optimization"
POLICY_PATH = ROOT / "configs" / "portfolio_policy.json"
PROJECT_PATH = ROOT / "configs" / "project.json"
FROZEN_HMM = "hmm_std_diag_k3"
PORTFOLIOS = {"P0_strategic": None, "P1_regime_aware": "baseline_regime", "P2_ews_enhanced": "challenger_ews"}


def frozen_hashes() -> dict:
    from regime_radar.ar_hmm_diagnostics import directory_hashes
    out = {}
    for d in ("hmm", "ar_hmm", "early_warning"):
        out.update({f"{d}/{k}": v for k, v in directory_hashes(PROCESSED / d).items()})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-tests", action="store_true", help="skip the automated real-data tests")
    parser.add_argument("--no-sensitivity", action="store_true", help="skip the sensitivity runs (debugging only)")
    args = parser.parse_args()

    import scipy
    import sklearn

    from regime_radar import expected_returns as er_mod
    from regime_radar import portfolio_constraints as pc
    from regime_radar import portfolio_diagnostics as pd_
    from regime_radar import portfolio_engine as eng
    from regime_radar import portfolio_execution as pe
    from regime_radar import portfolio_optimizer as po
    from regime_radar import portfolio_report as rep
    from regime_radar import portfolio_risk as pr
    from regime_radar import regime_exploration as rx
    from regime_radar import risk_budgeting as rb
    from regime_radar import stage7_handoff as h7
    from regime_radar.ar_hmm_diagnostics import file_sha256, save_json
    from regime_radar.factors import Check

    mods = (er_mod, pc, pd_, eng, pe, po, rep, pr, rb)
    stale = [m.__file__ for m in mods if getattr(m, "STAGE7_CODE_VERSION", None) != STAGE7_CODE_VERSION]
    if stale:
        print("Stage 7 modules do not match this script: " + ", ".join(stale) + ". Re-extract the zip, delete "
              "__pycache__ and rerun.")
        return 3
    PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
    started = time.perf_counter()
    recorder = warnings.catch_warnings(record=True)
    caught = recorder.__enter__()
    warnings.simplefilter("default")
    checks: list = []

    # ================================================================== 1. Stage 6.1 handoff verification
    hdir = EW_DIR / h7.HANDOFF_DIRNAME
    mpath = hdir / h7.MANIFEST_NAME
    if not mpath.exists():
        raise SystemExit(f"{mpath.relative_to(ROOT)} not found. Run scripts/08b_audit_early_warning.py first.")
    man = json.loads(mpath.read_text(encoding="utf-8"))
    if man.get("handoff_status") != "READY":
        print(f"Stage 6.1 handoff is {man.get('handoff_status')}: {man.get('handoff_status_reason')}. Stage 7 stopped.")
        return 1
    hashes_before = frozen_hashes()
    data = rx.load_stage4_data(SPLITS_DIR)
    checks.append(Check("Stage 6.1 handoff status READY", PASS, man.get("handoff_status_reason", "")[:200]))
    sf = man["input_hashes"]["split_files"]
    mism = [n for n, h in sf.items() if file_sha256(SPLITS_DIR / n) != h]
    checks.append(Check("Split-file hashes match the handoff manifest", PASS if not mism else FAIL,
                        f"{len(sf)} files; mismatched: {', '.join(mism) or 'none'}"))
    s6 = man["input_hashes"]["stage6_outputs"]
    mism6 = [k for k, h in s6.items() if file_sha256(EW_DIR / k.split("/", 1)[1]) != h]
    checks.append(Check("Stage 6 output hashes match the handoff manifest", PASS if not mism6 else FAIL,
                        f"{len(s6)} files; mismatched: {', '.join(mism6[:5]) or 'none'}"))
    fm = man["input_hashes"]["frozen_models"]
    mismm = [k for k, v in fm.items() if file_sha256(ROOT / v["file"]) != v["sha256"]]
    checks.append(Check("Frozen model files (HMM, AR(1), AR-HMM) match the handoff hashes", PASS if not mismm else FAIL,
                        ", ".join(f"{k}: {v['file']}" for k, v in fm.items())))
    code = man["input_hashes"].get("code", {})
    mismc = [k for k, h in code.items() if file_sha256(ROOT / "src" / "regime_radar" / k) != h]
    checks.append(Check("Stage 6 / 6.1 signal code unchanged since the handoff", PASS if not mismc else WARN,
                        f"{len(code)} files; changed: {', '.join(mismc) or 'none'}"))
    ids = {r: man["roles"][r]["frozen_parameters"].get("hmm_model", man["roles"][r]["frozen_parameters"]
                                                       .get("frozen_models", {}).get("hmm", {})).get("id")
           for r in h7.ROLES}
    checks.append(Check("Frozen model ID is hmm_std_diag_k3 for both roles",
                        PASS if all(v == FROZEN_HMM for v in ids.values()) else FAIL, str(ids)))
    frames = {}
    for role in h7.ROLES:
        f = pd.concat([h7.load_stage7_inputs(role, s, EW_DIR) for s in ("train", "validation")])
        h7.assert_single_role(f)
        frames[role] = f
    cal = data.train_features.index.append(data.validation_features.index)
    split = pd.Series(["train"] * len(data.train_features) + ["validation"] * len(data.validation_features), index=cal)
    chron = all(frames[r].index.equals(cal) for r in h7.ROLES) and cal.is_monotonic_increasing and not cal.has_duplicates
    checks.append(Check("Signal dates strictly chronological and identical to the split calendar",
                        PASS if chron else FAIL, f"{len(cal)} sessions {cal.min().date()}..{cal.max().date()}"))
    ts_ok = True
    for r in h7.ROLES:
        f = frames[r]
        nxt = pd.to_datetime(f["execution_date"])
        ts_ok &= bool((nxt.iloc[:-1].to_numpy() == f.index[1:].to_numpy()).all())
        avail = pd.to_datetime(f["signal_available_after"], utc=True).dt.tz_convert(h7.MARKET_TZ)
        ts_ok &= bool((avail.dt.date.to_numpy() == f.index.date).all() and (avail.dt.hour == 16).all())
    checks.append(Check("Decision timestamps valid (available after 16:00 ET on t; execution on t+1)",
                        PASS if ts_ok else FAIL, "both roles"))
    checks.append(Check("No Test file loaded", FAIL if any(n.startswith("test_") for n in data.files_read) else PASS,
                        ", ".join(data.files_read)))
    if any(c.status == FAIL for c in checks):
        for c in checks:
            if c.status == FAIL:
                print(f"  [FAIL] {c.name}: {c.detail}")
        print("Handoff verification failed; Stage 7 stopped (nothing written).")
        return 1
    print(f"Handoff verified: Train {data.train_features.index.min().date()}..{data.train_features.index.max().date()} "
          f"({len(data.train_features)}), Validation {data.validation_features.index.min().date()}.."
          f"{data.validation_features.index.max().date()} ({len(data.validation_features)}).")

    # ================================================================== 2. policy, returns, calendar
    policy = pc.load_policy(POLICY_PATH, PROJECT_PATH)
    pcfg = policy.raw
    proj = json.loads(PROJECT_PATH.read_text(encoding="utf-8"))
    assets = policy.assets
    log_ret = pd.concat([data.train_returns, data.validation_returns])[assets]
    n_missing = int(log_ret.isna().sum().sum())
    simple = pr.to_simple(log_ret)
    checks.append(Check("Return files: nine ETFs, daily LOG returns of adjusted closes, no missing values",
                        PASS if n_missing == 0 and list(log_ret.columns) == assets else FAIL,
                        f"definition from split manifest: {data.manifest.get('return_definition')}; missing {n_missing}"))
    bench_ok = np.allclose(policy.benchmark, [proj["strategic_weights"][a] for a in assets], atol=0) and \
        abs(policy.benchmark.sum() - 1) < 1e-12
    feas, _ = pc.linear_feasible(policy)
    bench_rows = pc.check_weights(policy, policy.benchmark)
    checks.append(Check("Strategic benchmark loaded from configs/project.json, sums to 1, feasible under the policy",
                        PASS if bench_ok and feas and not pc.hard_violations(bench_rows) else FAIL,
                        ", ".join(f"{a} {w:.2f}" for a, w in zip(assets, policy.benchmark))))
    coef = pcfg["risk_budget"]["coefficients"]
    worst = rb.budget_from_d(1.0, coef)
    feas_red, _ = pc.linear_feasible(policy, worst)
    checks.append(Check("Maximum-defensive (d = 1) linear constraints feasible", PASS if feas_red else FAIL,
                        f"equity <= {worst.equity_max:.2f}, credit <= {worst.credit_max:.2f}, BIL >= {worst.liquidity_min:.2f}"))
    schedule = pe.weekly_schedule(cal, split)
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
                           budget_coef=coef, solver=ocfg["solver"])
    cost_cfg = pcfg["transaction_costs"]
    # causal stress indicator for the cost model (identical for all portfolios)
    spy_vol = simple["SPY"].rolling(21).std() * np.sqrt(pr.ANNUALIZATION)
    tr_mask = (split == "train").to_numpy()
    stress_thr = float(np.nanquantile(spy_vol[tr_mask], 0.90))
    stress = spy_vol > stress_thr
    # signals -> defensiveness (one role each)
    level_map = pcfg["risk_budget"]["P2_ews_levels"]
    base, chal = frames["baseline_regime"], frames["challenger_ews"]
    scale = rb.parity_scale(base.loc[tr_mask, "baseline_p_non_normal"], chal.loc[tr_mask, "challenger_alert_level"],
                            level_map)
    d1 = base["baseline_p_non_normal"].map(lambda p: rb.p1_defensiveness(p, scale))
    d2 = chal["challenger_alert_level"].map(lambda lv: rb.p2_defensiveness(lv, level_map))
    p_tr = base.loc[tr_mask, "baseline_p_non_normal"]
    saturation = {"train_share_p_eq_1": float((p_tr == 1.0).mean()), "train_share_p_ge_0_999": float((p_tr >= 0.999).mean()),
                  "train_share_p_le_0_001": float((p_tr <= 0.001).mean()), "train_share_p_ge_0_5": float((p_tr >= 0.5).mean()),
                  "train_mean_p": float(p_tr.mean()), "parity_scale_a": scale,
                  "train_mean_d1": float(d1[tr_mask].mean()), "train_mean_d2": float(d2[tr_mask].mean()),
                  "validation_mean_d1": float(d1[~tr_mask].mean()), "validation_mean_d2": float(d2[~tr_mask].mean()),
                  "share_d1_distinct_values_train": int(d1[tr_mask].round(6).nunique())}
    print(f"Policy {pcfg['policy_version']}: covariance {cfg.cov_method}/{cfg.lookback}, expected returns "
          f"{cfg.er_method} (SR_b {cfg.sharpe}), lambda_turnover {cfg.lambda_turnover:.4f}, lambda_tracking "
          f"{cfg.lambda_tracking}, P1 parity scale a = {scale:.4f}; {len(schedule)} weekly decisions scheduled.")

    # ================================================================== 3. primary runs
    inputs = eng.prepare_inputs(simple, schedule, policy.benchmark, cfg)
    print(f"Risk inputs for {len(inputs)} decisions (warm-up: first {cfg.min_obs} sessions). Optimizing ...")
    signals = {"P0_strategic": None, "P1_regime_aware": d1, "P2_ews_enhanced": d2}
    targets, contribs, attempts, crows = [], [], [], []
    for pid, role in PORTFOLIOS.items():
        t0 = time.perf_counter()
        tg, ct, at, ck = eng.run_portfolio(pid, role, simple, schedule, inputs, policy, cfg, cost_cfg, stress, signals[pid])
        targets.append(tg)
        contribs.append(ct)
        attempts += at
        crows += ck
        print(f"  {pid}: {len(tg)} decisions in {time.perf_counter() - t0:.0f}s; status "
              + ", ".join(f"{k} {v}" for k, v in tg["solver_status"].value_counts().items()))
    targets = pd.concat(targets, ignore_index=True)
    contribs = pd.concat(contribs, ignore_index=True)
    con_checks = pd.DataFrame(crows)
    # signal context columns (each portfolio only gets its own role's columns)
    targets["signal_value"] = pd.Series([None] * len(targets), dtype=object)      # float for P1, level for P2
    targets["signal_week_max"] = pd.Series([None] * len(targets), dtype=object)
    targets["ews_score"] = np.nan
    for pid, role in PORTFOLIOS.items():
        m = targets["portfolio"] == pid
        if role == "baseline_regime":
            targets.loc[m, "signal_value"] = targets.loc[m, "signal_date"].map(base["baseline_p_non_normal"])
            wk_max = base["baseline_p_non_normal"].groupby(cal - pd.to_timedelta(cal.weekday, unit="D")).max()
            targets.loc[m, "signal_week_max"] = targets.loc[m, "signal_date"].map(
                lambda t: wk_max.loc[t - pd.Timedelta(days=t.weekday())])
        elif role == "challenger_ews":
            targets.loc[m, "signal_value"] = targets.loc[m, "signal_date"].map(chal["challenger_alert_level"])
            sev = chal["challenger_alert_level"].map({k: i for i, k in enumerate(rb.LEVEL_ORDER)})
            wk_max = sev.groupby(cal - pd.to_timedelta(cal.weekday, unit="D")).max()
            targets.loc[m, "signal_week_max"] = targets.loc[m, "signal_date"].map(
                lambda t: rb.LEVEL_ORDER[int(wk_max.loc[t - pd.Timedelta(days=t.weekday())])])
            targets.loc[m, "ews_score"] = targets.loc[m, "signal_date"].map(chal["challenger_ews_score"])
    sig_cols = ["signal_value", "signal_week_max"]

    # ================================================================== 4. risk-estimator comparison & ER table
    est_rows, er_rows = [], []
    alt_cfgs = {"ledoit_wolf_504 (primary)": cfg, "sample_504": cfg.variant(cov_method="sample"),
                "ewma_hl63": cfg.variant(cov_method="ewma"), "ledoit_wolf_252": cfg.variant(lookback=252),
                "ledoit_wolf_expanding": cfg.variant(lookback="expanding")}
    alt_inputs = {k: (inputs if v is cfg else eng.prepare_inputs(simple, schedule, policy.benchmark, v))
                  for k, v in alt_cfgs.items()}
    cal_pos = {d: i for i, d in enumerate(cal)}
    sb_r = pd.Series(simple.to_numpy() @ policy.benchmark, index=cal)
    prev = {}
    for t in sorted(inputs):
        i = cal_pos[t]
        fut = sb_r.iloc[i + 2:i + 2 + 21]               # evaluation-only: realized vol after the first holding date
        realized = float(fut.std(ddof=1) * np.sqrt(pr.ANNUALIZATION)) if len(fut) == 21 and \
            split.iloc[min(i + 22, len(cal) - 1)] == split.loc[t] else np.nan
        for name, inp in alt_inputs.items():
            ri = inp[t]
            dg = pr.cov_diagnostics(ri.cov, assets)
            vols = np.sqrt(np.diag(ri.cov))
            chg = float(np.abs(vols - prev[name]).mean()) if name in prev else np.nan
            prev[name] = vols
            est_rows.append({"signal_date": t, "split": split.loc[t], "estimator": name, "n_obs": ri.n_obs,
                             "lw_shrinkage": ri.shrinkage, "effective_n": ri.effective_n,
                             "floored_eigenvalues": ri.floored, "benchmark_vol_ex_ante": ri.sigma_b,
                             "evaluation_only_subsequent_realized_benchmark_vol_21d": realized,
                             "mean_abs_weekly_change_asset_vol": chg, **dg})
        ri = inputs[t]
        for k, a in enumerate(assets):
            er_rows.append({"signal_date": t, "split": split.loc[t], "asset": a, "mu_equilibrium": ri.mu_eq[k],
                            "risk_free": ri.rf, "delta": ri.delta, "hist_mean_trailing": ri.hist_mean[k],
                            "hist_mean_se": ri.hist_se[k],
                            "mu_shrunk_historical": er_mod.shrunk_historical(ri.hist_mean, ri.mu_eq, cfg.er_shrinkage)[k]})
    est = pd.DataFrame(est_rows)
    er_tab = pd.DataFrame(er_rows)
    psd_ok = (est["min_eigenvalue"] > 0).all()
    checks.append(Check("Every covariance estimate positive definite after stabilization", PASS if psd_ok else FAIL,
                        f"min eigenvalue {est['min_eigenvalue'].min():.2e}; floored eigenvalues "
                        f"{int(est['floored_eigenvalues'].sum())} (summed over estimators and dates)"))

    # ================================================================== 5. CVaR secondary optimizer (Validation)
    cmp_rows = []
    val_t = targets[(targets["split"] == "validation")]
    for pid, dser in (("P1_regime_aware", d1), ("P2_ews_enhanced", d2)):
        for _, r in val_t[val_t["portfolio"] == pid].iterrows():
            t = r["signal_date"]
            ri = inputs[t]
            b = rb.budget_from_d(float(dser.loc[t]), coef)
            w_pre = r[[f"pretrade_{a}" for a in assets]].to_numpy(float)
            _, cvar_b = pr.historical_var_cvar(ri.scenarios, policy.benchmark, cfg.cvar_alpha)
            w_c, stat = po.solve_cvar_lp(policy, ri.mu, ri.scenarios, w_pre, b, b.vol_multiplier * cvar_b,
                                         cfg.lambda_turnover, pcfg["cvar"]["tracking_l1_penalty"], cfg.cvar_alpha)
            w_m = r[[f"w_{a}" for a in assets]].to_numpy(float)
            row = {"portfolio": pid, "signal_date": t, "cvar_lp_status": stat, "budget_d": b.d,
                   "mvo_vol": pr.portfolio_vol(w_m, ri.cov), "mvo_cvar95": pr.historical_var_cvar(ri.scenarios, w_m)[1],
                   "cvar_cap": b.vol_multiplier * cvar_b}
            if w_c is not None:
                rws = pc.check_weights(policy, w_c, b, w_pre, b.turnover_cap)
                row.update({"cvar_lp_vol": pr.portfolio_vol(w_c, ri.cov),
                            "cvar_lp_cvar95": pr.historical_var_cvar(ri.scenarios, w_c)[1],
                            "cvar_lp_hard_violations": ";".join(pc.hard_violations(rws)),
                            "max_abs_weight_diff_vs_mvo": float(np.abs(w_c - w_m).max()),
                            **{f"cvar_w_{a}": float(x) for a, x in zip(assets, w_c)}})
            cmp_rows.append(row)
    comparison = pd.DataFrame(cmp_rows)

    # ================================================================== 6. sensitivity (Validation restart)
    sens = pd.DataFrame()
    if not args.no_sensitivity:
        print("Sensitivity runs (Validation decisions, each restarted from the benchmark) ...")
        vsched = schedule[schedule["split"] == "validation"]
        lv = frames["challenger_ews"]
        thr = man["roles"]["challenger_ews"]["frozen_parameters"]["alert_thresholds"]
        d2c = lv["challenger_ews_score"].map(lambda s: rb.p2_defensiveness_continuous(s, thr["amber_out"], thr["red_in"]))
        d1f = base["baseline_p_non_normal"].map(lambda p: rb.p1_defensiveness(p, 1.0))
        steep = dict(coef, vol_slope=0.50)
        variants = {"reference": (cfg, d1, d2), "cov_sample_504": (cfg.variant(cov_method="sample"), d1, d2),
                    "cov_ewma_hl63": (cfg.variant(cov_method="ewma"), d1, d2),
                    "cov_lw_252": (cfg.variant(lookback=252), d1, d2),
                    "cov_lw_expanding": (cfg.variant(lookback="expanding"), d1, d2),
                    "er_shrunk_historical": (cfg.variant(er_method="shrunk_historical"), d1, d2),
                    "er_plus_1se": (cfg.variant(er_method="plus_se"), d1, d2),
                    "er_minus_1se": (cfg.variant(er_method="minus_se"), d1, d2),
                    "gamma_x0.5": (cfg.variant(gamma_multiplier=0.5), d1, d2),
                    "gamma_x2": (cfg.variant(gamma_multiplier=2.0), d1, d2),
                    "single_step_turnover_penalized": (cfg.variant(execution_mode="single_step"), d1, d2),
                    "no_trade_band_0": (cfg.variant(no_trade_band=0.0), d1, d2),
                    "no_trade_band_0.02": (cfg.variant(no_trade_band=0.02), d1, d2),
                    "turnover_cap_halved": (cfg.variant(budget_coef=dict(coef, turnover_cap0=0.10, turnover_slope=0.05)),
                                            d1, d2),
                    "tracking_penalty_0": (cfg.variant(lambda_tracking=0.0), d1, d2),
                    "mapping_P1_full_range_P2_continuous": (cfg, d1f, d2c),
                    "mapping_steeper_volatility": (cfg.variant(budget_coef=steep), d1, d2)}
        cache = {}
        ref = {}
        srows = []
        for name, (vc, s1, s2) in variants.items():
            key = (vc.cov_method, vc.lookback, vc.er_method)
            if key not in cache:
                cache[key] = eng.prepare_inputs(simple, vsched, policy.benchmark, vc)
            for pid, sig in (("P1_regime_aware", s1), ("P2_ews_enhanced", s2)):
                tg, _, _, _ = eng.run_portfolio(pid, PORTFOLIOS[pid], simple, vsched, cache[key], policy, vc, cost_cfg,
                                                stress, sig, contributions=False)
                W = tg[[f"w_{a}" for a in assets]].to_numpy()
                if name == "reference":
                    ref[pid] = W
                srows.append({"variant": name, "portfolio": pid, "decisions": len(tg),
                              "mean_abs_weight_change_vs_reference": float(np.abs(W - ref[pid]).mean()),
                              "max_abs_weight_change_vs_reference": float(np.abs(W - ref[pid]).max()),
                              "mean_vol_ex_ante": tg["vol_ex_ante"].mean(), "mean_cvar95_daily": tg["cvar95_daily_ex_ante"].mean(),
                              "mean_equity": tg["exposure_equity"].mean(), "mean_duration": tg["exposure_duration"].mean(),
                              "mean_bil": tg["w_BIL"].mean(), "mean_turnover": tg["turnover_ex_ante_proxy"].mean(),
                              "mean_budget_d": tg["budget_d"].mean(), "fallbacks": int((tg["solver_stage"] > 1).sum()),
                              "hard_violations": int(tg["hard_violations"].fillna("").ne("").sum())})
            print(f"  {name} done")
        sens = pd.DataFrame(srows)

    # ================================================================== 7. scenarios, frontier, representative dates
    scen = pd.concat([pd_.historical_scenarios(simple[tr_mask], policy.benchmark),
                      pd.DataFrame([{"scenario": k, "type": "hypothetical assumption", "start": None, "end": None, **v}
                                    for k, v in pd_.HYPOTHETICAL.items()])], ignore_index=True)
    vt = targets[targets["split"] == "validation"]
    rep_dates = rep.representative_dates(vt)
    w_at = {}
    for label, t in rep_dates.items():
        for pid in PORTFOLIOS:
            r = targets[(targets["portfolio"] == pid) & (targets["signal_date"] == t)]
            if len(r):
                w_at[f"{pid} @ {label} ({t.date()})"] = r[[f"w_{a}" for a in assets]].to_numpy(float)[0]
    scen_tab = pd_.scenario_losses(w_at, scen, assets)
    for g in policy.groups:
        gv = policy.group_vector(g)
        scen_tab[f"exposure_{g}"] = scen_tab["portfolio"].map(lambda k: float(gv @ w_at[k]))
    scen_tab["herfindahl"] = scen_tab["portfolio"].map(lambda k: pr.herfindahl(w_at[k]))
    # frontier date: the most defensive representative P2 condition (fixed rule), else the last decision
    last_t = next((rep_dates[k] for k in ("P2 RED", "P2 ORANGE", "P2 AMBER") if k in rep_dates), max(inputs))
    ri = inputs[last_t]
    frontier = {"normal (d=0)": po.efficient_frontier(policy, ri.mu, ri.cov, rb.budget_from_d(0.0, coef)),
                "max defensive (d=1)": po.efficient_frontier(policy, ri.mu, ri.cov, rb.budget_from_d(1.0, coef))}

    # ================================================================== 8. checks on the published targets
    hv = targets["hard_violations"].fillna("").ne("")
    checks.append(Check("Every published target satisfies all hard constraints", PASS if not hv.any() else FAIL,
                        f"{int(hv.sum())} of {len(targets)} targets violate a hard constraint"))
    sums = (targets["sum_weights"] - 1).abs().max()
    checks.append(Check("Weights sum to one, no shorts, no leverage", PASS if sums < 1e-8 and targets["min_weight"].min()
                        >= -1e-9 else FAIL, f"max |sum-1| {sums:.1e}; min weight {targets['min_weight'].min():.2e}"))
    dyn = targets[targets["portfolio"] != "P0_strategic"]
    tgt_bad = int(dyn["solver_status"].str.contains("unattainable").sum())
    checks.append(Check("Risk-budget postures attainable under the static policy constraints",
                        PASS if not tgt_bad else WARN,
                        f"{tgt_bad} decisions where even an unconstrained-turnover posture cannot meet the risk budget "
                        "(closest feasible posture used and reported)"))
    checks.append(Check("Optimizer: no hold fallback; transitions reported",
                        PASS if not (dyn["solver_stage"] > 2).any() else FAIL,
                        f"{len(dyn)} decisions: target reached or held in band {int((dyn['solver_stage'] == 1).sum())}, "
                        f"turnover-limited transition {int((dyn['solver_stage'] == 2).sum())}, hold fallback "
                        f"{int((dyn['solver_stage'] == 3).sum())}"))
    dis = dyn["start_disagreement"].max()
    checks.append(Check("Solutions independent of the starting point (convex problem)",
                        PASS if not np.isfinite(dis) or dis < 1e-4 else WARN, f"max weight difference between starts {dis:.1e}"))
    bil_hi = (targets["w_BIL"] >= policy.max_weight[assets.index("BIL")] - 1e-6).mean()
    checks.append(Check("BIL concentration below its cap on most decisions", PASS if bil_hi < 0.25 else WARN,
                        f"BIL at its cap on {bil_hi:.1%} of decisions"))
    red = targets[(targets["portfolio"] == "P2_ews_enhanced")]
    by_lv = red.groupby("signal_value")["budget_vol_target"].mean() / red.groupby("signal_value")["benchmark_vol_ex_ante"].mean()
    mono = all(by_lv.get(a, np.nan) >= by_lv.get(b, np.nan) - 1e-12 for a, b in (("GREEN", "AMBER"), ("AMBER", "ORANGE"),
                                                                                 ("ORANGE", "RED")) if a in by_lv and b in by_lv)
    checks.append(Check("P2 risk budget weakly decreasing GREEN -> AMBER -> ORANGE -> RED", PASS if mono else FAIL,
                        ", ".join(f"{k} {v:.3f}" for k, v in by_lv.items())))
    p0 = targets[targets["portfolio"] == "P0_strategic"]
    p0_post = np.allclose(p0[[f"posture_{a}" for a in assets]].to_numpy(), policy.benchmark)
    p0_band = (0.5 * np.abs(p0[[f"w_{a}" for a in assets]].to_numpy() - policy.benchmark).sum(axis=1)
               <= cfg.no_trade_band + 1e-12).all()
    checks.append(Check("P0 posture equals the strategic benchmark; executed P0 weights within the no-trade band",
                        PASS if p0_post and p0_band else FAIL,
                        f"{len(p0)} decisions; P0 traded back to the benchmark on "
                        f"{int((p0['solver_status'] == 'trade_to_target').sum())}"))
    g0 = targets[(targets["portfolio"] != "P0_strategic") & (targets["budget_d"] == 0)]
    nb = (0.5 * np.abs(g0[[f"posture_{a}" for a in assets]].to_numpy() - policy.benchmark).sum(axis=1)).max() \
        if len(g0) else 0.0
    checks.append(Check("Normal budget (d = 0) posture equals the benchmark for P1 and P2 (no built-in defensiveness)",
                        PASS if nb < 1e-4 else FAIL, f"{len(g0)} decisions with d = 0; max distance {nb:.1e}"))
    same_dates = all(set(targets[targets["portfolio"] == p]["signal_date"]) == set(p0["signal_date"]) for p in PORTFOLIOS)
    checks.append(Check("P0, P1, P2 share identical decision, execution and holding dates", PASS if same_dates else FAIL,
                        f"{len(p0)} decisions each"))
    ex = targets["execution_date"] - pd.to_timedelta(targets["execution_date"].dt.weekday, unit="D")
    one_per_week = not targets.assign(wk=ex).duplicated(["portfolio", "wk"]).any()
    timing = ((targets["execution_date"] > targets["signal_date"]) &
              (targets["first_holding_return_date"].isna() | (targets["first_holding_return_date"] > targets["execution_date"]))).all()
    checks.append(Check("At most one execution per calendar week; execution after the signal; holding after execution",
                        PASS if one_per_week and timing else FAIL,
                        f"{int(targets['split_crossing'].sum() / 3)} decision(s) cross the Train/Validation boundary"))
    roles_ok = (targets.loc[targets["portfolio"] == "P1_regime_aware", "signal_role"].eq("baseline_regime").all() and
                targets.loc[targets["portfolio"] == "P2_ews_enhanced", "signal_role"].eq("challenger_ews").all())
    checks.append(Check("P1 uses only role baseline_regime, P2 only role challenger_ews", PASS if roles_ok else FAIL,
                        "defensiveness series built from one guarded role frame each"))
    # causality spot check on the real data: perturb all returns after a Validation decision date
    vdates = sorted(t for t in inputs if split.loc[t] == "validation")
    cut = vdates[len(vdates) // 2]
    pert = simple.copy()
    after = pert.index > cut
    pert.loc[after] = pert.loc[after] * -3 + 0.01
    sched_cut = schedule[schedule["signal_date"] <= cut]
    inp_p = eng.prepare_inputs(pert, sched_cut, policy.benchmark, cfg)
    same = True
    for pid in ("P1_regime_aware", "P2_ews_enhanced"):
        tg, _, _, _ = eng.run_portfolio(pid, PORTFOLIOS[pid], pert, sched_cut, inp_p, policy, cfg, cost_cfg, stress,
                                        signals[pid], contributions=False)
        orig = targets[(targets["portfolio"] == pid) & (targets["signal_date"] <= cut)]
        same &= np.array_equal(tg[[f"w_{a}" for a in assets]].to_numpy(), orig[[f"w_{a}" for a in assets]].to_numpy())
    checks.append(Check("Changing every return after a decision date leaves all targets up to that date unchanged",
                        PASS if same else FAIL, f"cut {cut.date()}; P1 and P2 recomputed bit-for-bit"))
    bud_viol = con_checks[(con_checks["kind"] == "budget") & con_checks["violated"]]
    viol_keys = set(zip(bud_viol["portfolio"], bud_viol["signal_date"]))
    reported = set(zip(dyn.loc[dyn["unmet_budget"].fillna("") != "", "portfolio"],
                       dyn.loc[dyn["unmet_budget"].fillna("") != "", "signal_date"]))
    unmet_stage1 = dyn[(dyn["solver_stage"] == 1) & (dyn["unmet_budget"].fillna("") != "") &
                       (dyn["solver_status"] != "hold_within_no_trade_band")]
    checks.append(Check("Every unmet risk budget is reported (only transitions or holds inside the no-trade band)",
                        PASS if viol_keys == reported and unmet_stage1.empty else FAIL,
                        f"{len(reported)} decisions with an unmet budget component (all turnover-limited transitions)"))

    # ================================================================== 9. outputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wcols = [f"w_{a}" for a in assets]
    lead = ["portfolio", "signal_role", "signal_date", "execution_date", "first_holding_return_date", "split",
            "split_crossing"] + sig_cols + ["ews_score"]
    lead = [c for c in lead if c in targets.columns]
    tcols = lead + wcols + [c for c in targets.columns if c not in lead + wcols]
    pd.DataFrame({"asset": assets, "weight": policy.benchmark,
                  "group": [next(g for g, m in policy.groups.items() if a in m) for a in assets],
                  "source": "configs/project.json strategic_weights (frozen)"}).to_csv(OUT_DIR / "strategic_benchmark_weights.csv", index=False)
    for s in ("train", "validation"):
        est[est["split"] == s].to_csv(OUT_DIR / f"portfolio_risk_estimates_{s}.csv", index=False, float_format="%.10g")
        targets[targets["split"] == s][tcols].to_csv(OUT_DIR / f"portfolio_targets_{s}.csv", index=False, float_format="%.10g")
    er_summary = er_tab.groupby(["split", "asset"]).agg(mu_equilibrium_mean=("mu_equilibrium", "mean"),
                                                        mu_equilibrium_sd=("mu_equilibrium", "std"),
                                                        hist_mean_mean=("hist_mean_trailing", "mean"),
                                                        hist_se_mean=("hist_mean_se", "mean")).reset_index()
    er_tab.to_csv(OUT_DIR / "expected_return_assumptions.csv", index=False, float_format="%.10g")
    contribs.to_csv(OUT_DIR / "portfolio_risk_contributions.csv", index=False, float_format="%.10g")
    turn = rep.turnover_table(targets, cost_cfg)
    turn.to_csv(OUT_DIR / "portfolio_turnover_diagnostics.csv", index=False, float_format="%.10g")
    con_checks.to_csv(OUT_DIR / "portfolio_constraint_checks.csv", index=False, float_format="%.10g")
    scen_out = scen_tab.merge(scen[["scenario", "start", "end"]], on="scenario", how="left")
    scen_out.to_csv(OUT_DIR / "portfolio_scenario_analysis.csv", index=False, float_format="%.10g")
    scen.to_csv(OUT_DIR / "portfolio_scenario_definitions.csv", index=False, float_format="%.10g")
    comparison.to_csv(OUT_DIR / "portfolio_optimizer_comparison.csv", index=False, float_format="%.10g")
    sens.to_csv(OUT_DIR / "portfolio_model_sensitivity.csv", index=False, float_format="%.10g")
    pd.DataFrame(attempts).to_csv(OUT_DIR / "portfolio_solver_attempts.csv", index=False)
    pd.DataFrame([{"frontier": k, **r} for k, v in frontier.items() for r in v]).to_csv(
        OUT_DIR / "portfolio_efficient_frontier.csv", index=False, float_format="%.10g")
    stab = rep.stability_table(targets, assets)
    stab.to_csv(OUT_DIR / "portfolio_weight_stability.csv", index=False, float_format="%.10g")
    rep_tab = rep.representative_table(targets, rep_dates, assets, policy)
    rep_tab.to_csv(OUT_DIR / "portfolio_representative_targets.csv", index=False, float_format="%.10g")

    hashes_after = frozen_hashes()
    changed = sorted(k for k in set(hashes_before) | set(hashes_after) if hashes_before.get(k) != hashes_after.get(k))
    checks.append(Check("Stage 5, 5.5, 6 and 6.1 artifacts unchanged (hashes before == after)",
                        PASS if not changed else FAIL, f"{len(hashes_before)} files; changed: {', '.join(changed) or 'none'}"))

    # ================================================================== 10. manifests (written before the real-data tests)
    out_hashes = {str(p.relative_to(ROOT)): file_sha256(p) for p in sorted(OUT_DIR.glob("*.csv"))}
    code_files = ["portfolio_risk.py", "expected_returns.py", "portfolio_optimizer.py", "risk_budgeting.py",
                  "portfolio_constraints.py", "portfolio_execution.py", "portfolio_diagnostics.py", "portfolio_engine.py",
                  "portfolio_report.py", "stage7_handoff.py"]
    code_hashes = {f: file_sha256(ROOT / "src" / "regime_radar" / f) for f in code_files}
    versions = {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
                "scipy": scipy.__version__, "sklearn": sklearn.__version__}
    policy_manifest = {
        "stage": "7 - robust portfolio optimization and dynamic risk budgeting", "code_version": STAGE7_CODE_VERSION,
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "policy_version": pcfg["policy_version"], "policy_file_sha256": file_sha256(POLICY_PATH),
        "project_config_sha256": file_sha256(PROJECT_PATH), "software_versions": versions,
        "provenance": {"frozen": ["universe", "benchmark weights", "Stage 6.1 signals and roles", "execution convention",
                                  "weekly rebalancing limit"],
                       "proposed_assumptions": ["constraints", "risk-budget mapping", "benchmark Sharpe 0.30",
                                                "turnover and tracking penalties", "cost scenarios", "covariance lookback 504"],
                       "train_calibrated": {"P1_parity_scale_a": scale, "stress_vol_threshold_spy_21d": stress_thr}},
        "engine_config": {k: (v if not isinstance(v, dict) else v) for k, v in vars(cfg).items()},
        "saturation": saturation, "handoff_manifest_sha256": file_sha256(mpath),
        "test_data_loaded": False, "files_read": data.files_read, "outputs": out_hashes, "code": code_hashes}
    save_json(policy_manifest, OUT_DIR / "portfolio_policy_manifest.json")

    # ================================================================== 11. automated real-data tests
    test_summary = {"ran": False}
    if not args.no_tests:
        print("Running automated tests on the real Train/Validation outputs ...")
        import io

        class Tee(io.StringIO):
            def write(self, x):
                sys.stdout.write(x)
                return super().write(x)

        stream = Tee()
        suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_stage7_real_data.py",
                                                    top_level_dir=str(ROOT / "tests"))
        res = unittest.TextTestRunner(verbosity=2, stream=stream).run(suite)
        (OUT_DIR / "real_data_test_results.txt").write_text(stream.getvalue(), encoding="utf-8")
        test_summary = {"ran": True, "tests": res.testsRun, "failures": len(res.failures), "errors": len(res.errors),
                        "skipped": len(res.skipped),
                        "failed_tests": [{"test": str(t), "traceback": tb} for t, tb in res.failures + res.errors]}
        checks.append(Check("Automated tests on the real Train/Validation outputs",
                            PASS if res.wasSuccessful() and res.testsRun and not res.skipped else FAIL,
                            f"{res.testsRun} tests, {len(res.failures)} failures, {len(res.errors)} errors, "
                            f"{len(res.skipped)} skipped"))
    else:
        checks.append(Check("Automated tests on the real Train/Validation outputs", FAIL, "not run (--no-tests)"))

    figures = rep.save_figures(FIG_DIR, {"targets": targets, "policy": policy, "contribs": contribs, "frontier": frontier,
                                         "turnover": turn, "scenarios": scen_tab, "con_checks": con_checks,
                                         "comparison": comparison, "inputs_last": (last_t, ri), "rep_dates": rep_dates,
                                         "est": est})
    recorder.__exit__(None, None, None)
    warn_list = sorted({f"{w.category.__name__}: {str(w.message)[:150]}" for w in caught})
    checks.append(Check("No library warnings", PASS if not warn_list else WARN, "; ".join(warn_list) or "none"))
    counts = pd.Series([c.status for c in checks]).value_counts()
    fails = [c.name for c in checks if c.status == FAIL]
    status = "BLOCKED" if fails else "READY"
    reason = ("FAILED checks: " + "; ".join(fails)) if fails else ("all blocking checks PASS" + (
        "; WARN: " + "; ".join(f"{c.name}: {c.detail}" for c in checks if c.status == WARN)
        if counts.get(WARN, 0) else ""))
    pd.DataFrame([vars(c) for c in checks]).to_csv(OUT_DIR / "stage7_checks.csv", index=False)

    stage8 = rep.stage8_manifest(policy, pcfg, cfg, scale, stress_thr, man, mpath, out_hashes, code_hashes, versions,
                                 status, reason, file_sha256(POLICY_PATH), file_sha256(PROJECT_PATH))
    save_json(stage8, OUT_DIR / "stage8_handoff_manifest.json")
    ctx = {"data": data, "policy": policy, "pcfg": pcfg, "cfg": cfg, "targets": targets, "contribs": contribs,
           "est": est, "er_tab": er_tab, "er_summary": er_summary, "turnover": turn, "con_checks": con_checks,
           "scenarios": scen, "scen_tab": scen_tab, "comparison": comparison, "sensitivity": sens,
           "frontier": frontier, "stability": stab, "rep_tab": rep_tab, "rep_dates": rep_dates, "checks": checks,
           "status": status, "reason": reason, "saturation": saturation, "stress_thr": stress_thr,
           "figures": figures, "test_summary": test_summary, "schedule": schedule, "versions": versions}
    (REPORTS / "portfolio_optimization_research.md").write_text(rep.render_report(ctx), encoding="utf-8")

    # ================================================================== 12. console
    rep.print_console(ctx)
    print("Checks:")
    for c in checks:
        print(f"  [{c.status}] {c.name}: {c.detail}")
    print("Summary: " + ", ".join(f"{counts.get(s_, 0)} {s_}" for s_ in (PASS, WARN, FAIL)))
    print(f"STAGE 8 HANDOFF STATUS: {status} - {reason}")
    print(f"Report: {REPORTS / 'portfolio_optimization_research.md'}; figures {len(figures)}. Elapsed "
          f"{time.perf_counter() - started:.0f}s. Stage 7 complete; no backtest, no realized performance, Test not read.")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
