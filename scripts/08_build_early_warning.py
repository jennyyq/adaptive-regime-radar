"""Stage 6: dual-speed early warning system (frozen HMM regimes, AR(1) innovations, BOCPD), Train/Validation only.

Reads ONLY: data/splits/{train,validation}_{features,returns}.csv + split_manifest.json; the frozen Stage 5 HMM
(data/processed/hmm/models/<selected>.json, hmm_selection_manifest.json, hmm_model_comparison.csv); the frozen
Stage 5.5 models (data/processed/ar_hmm/models/{ar1,ar_hmm_k3}.json, ar_hmm_selection_manifest.json,
audit/audit_checks.csv); configs/project.json (strategic weights for the reference-portfolio outcome).
The test split is never opened. Stage 5 / 5.5 files are hashed before and after and never written.

Outputs: data/processed/early_warning/*.csv + early_warning_selection_manifest.json,
         reports/early_warning_research.md, reports/figures/early_warning/*.png (10 figures).
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
STAGE6_CODE_VERSION = "stage6-2026-10-21"
STAGE5_DIR, STAGE55_DIR = PROCESSED / "hmm", PROCESSED / "ar_hmm"
OUT_DIR = PROCESSED / "early_warning"
FIGURE_DIR = REPORTS / "figures" / "early_warning"
EXPECTED_HMM = "hmm_std_diag_k3"
EXPECTED_AR_VERSION = "stage5.5-2026-10-21b"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--bootstrap-reps", type=int, default=None)
    args = parser.parse_args()

    from regime_radar import alert_state_machine as asm
    from regime_radar import bocpd as bo
    from regime_radar import early_warning as ew
    from regime_radar import early_warning_report as er
    from regime_radar import early_warning_validation as ev
    from regime_radar import innovation_signals as ins
    from regime_radar import regime_exploration as rx
    from regime_radar.ar_hmm import ARParams, filter_from_log_emission, lagged_pairs, log_emission_ar
    from regime_radar.ar_hmm_diagnostics import directory_hashes, file_sha256, save_json
    from regime_radar.dataset import FEATURE_COLUMNS
    from regime_radar.factors import Check
    from regime_radar.hmm_filtering import forward_filter
    from regime_radar.hmm_models import HMMParams

    mods = (asm, bo, ew, er, ev, ins)
    stale = [m.__file__ for m in mods if getattr(m, "STAGE6_CODE_VERSION", None) != STAGE6_CODE_VERSION]
    if stale:
        print("Stage 6 modules do not match this script: " + ", ".join(stale) + ". Re-extract the zip, delete "
              "__pycache__ and rerun.")
        return 3
    PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
    started = time.perf_counter()
    recorder = warnings.catch_warnings(record=True)
    caught = recorder.__enter__()
    warnings.simplefilter("default")

    # ------------------------------------------------------------------ integrity of frozen artifacts
    for path in (STAGE5_DIR / "hmm_selection_manifest.json", STAGE55_DIR / "ar_hmm_selection_manifest.json",
                 STAGE55_DIR / "models" / "ar1.json", STAGE55_DIR / "models" / "ar_hmm_k3.json"):
        if not path.exists():
            raise SystemExit(f"{path.relative_to(ROOT)} not found. Run scripts 07 and 07b first.")
    hashes_before = {**{f"hmm/{k}": v for k, v in directory_hashes(STAGE5_DIR).items()},
                     **{f"ar_hmm/{k}": v for k, v in directory_hashes(STAGE55_DIR).items()}}
    s5 = json.loads((STAGE5_DIR / "hmm_selection_manifest.json").read_text(encoding="utf-8"))
    s55 = json.loads((STAGE55_DIR / "ar_hmm_selection_manifest.json").read_text(encoding="utf-8"))
    hmm_name = s5.get("selection", {}).get("selected")
    # same rule as 07b: the documented frozen model file is always used; a different manifest choice is flagged
    checks = [Check("Frozen Stage 5 model is the documented hmm_std_diag_k3", PASS if hmm_name == EXPECTED_HMM else WARN,
                    f"Stage 5 manifest selected {hmm_name}; Stage 6 uses {EXPECTED_HMM}.json (as Stage 5.5 did)")]
    hmm_path = STAGE5_DIR / "models" / f"{EXPECTED_HMM}.json"
    if not hmm_path.exists():
        raise SystemExit(f"{hmm_path.relative_to(ROOT)} not found; the frozen Stage 5 model is required.")
    hmm_meta = json.loads(hmm_path.read_text(encoding="utf-8"))
    ar_meta = {m: json.loads((STAGE55_DIR / "models" / f"{m}.json").read_text(encoding="utf-8")) for m in ("ar1", "ar_hmm_k3")}
    for m, meta in ar_meta.items():
        ok = meta.get("variance_rule") == "floor" and meta.get("model_version") == EXPECTED_AR_VERSION
        checks.append(Check(f"Stage 5.5 {m} produced by the corrected exact M-step", PASS if ok else FAIL,
                            f"variance_rule {meta.get('variance_rule')}, model_version {meta.get('model_version')}"))
    s55_fail = [c["name"] for c in s55.get("checks", []) if c.get("status") == "FAIL"]
    checks.append(Check("Stage 5.5 run had no FAILed checks and records the estimation amendment",
                        PASS if not s55_fail and s55.get("estimation_amendments") else FAIL,
                        f"{len(s55.get('checks', []))} checks; FAIL: {', '.join(s55_fail) or 'none'}"))
    audit = STAGE55_DIR / "audit" / "audit_checks.csv"
    if audit.exists():
        a = pd.read_csv(audit)
        ok = (a.loc[a["name"].str.startswith("Exact M-step"), "status"] == "PASS").all() and not (a["status"] == "FAIL").any()
        checks.append(Check("Stage 5.5 EM audit passed", PASS if ok else FAIL, f"{len(a)} audit checks"))
    else:
        checks.append(Check("Stage 5.5 EM audit passed", WARN, "audit/audit_checks.csv not found"))

    data = rx.load_stage4_data(SPLITS_DIR)
    tr_feat, va_feat = data.train_features[FEATURE_COLUMNS], data.validation_features[FEATURE_COLUMNS]
    checks += rx.input_checks(data)
    checks.append(rx.frozen_factor_check(data.manifest, ROOT / "src" / "regime_radar" / "factors.py"))
    apply, _ = rx.scaling_decision(rx.describe_train(tr_feat))
    scaler = rx.TrainScaler(apply).fit(tr_feat)
    sd = scaler.to_dict()
    for name, meta in (("HMM", hmm_meta), ("AR(1)", ar_meta["ar1"]), ("AR-HMM", ar_meta["ar_hmm_k3"])):
        pre = meta["preprocessing"]
        same = pre["applied"] == sd["applied"] and all(abs(pre["mean"][c] - sd["mean"][c]) < 1e-8 and
                                                       abs(pre["scale"][c] - sd["scale"][c]) < 1e-8 for c in FEATURE_COLUMNS)
        checks.append(Check(f"Train scaler identical to the frozen {name} preprocessing", PASS if same else FAIL, sd["method"]))
    x_tr, x_va = scaler.transform(tr_feat), scaler.transform(va_feat)
    hmm_params = HMMParams.from_dict(hmm_meta["parameters"])
    ar1 = ARParams.from_dict(ar_meta["ar1"]["parameters"])
    arhmm = ARParams.from_dict(ar_meta["ar_hmm_k3"]["parameters"])
    tr_ll = float(forward_filter(hmm_params, x_tr).log_score.sum())
    checks.append(Check("Frozen HMM reproduces its saved Train log-likelihood", PASS if abs(tr_ll - hmm_meta["train_loglik"])
                        <= 1e-8 * abs(hmm_meta["train_loglik"]) else FAIL, f"{tr_ll:.6f} vs {hmm_meta['train_loglik']:.6f}"))
    y, lag = lagged_pairs(x_tr)
    for m, p in (("ar1", ar1), ("ar_hmm_k3", arhmm)):
        ll = float(filter_from_log_emission(log_emission_ar(p, y, lag), p.transmat, initial_predicted=p.startprob).log_score.sum())
        saved = ar_meta[m]["train_loglik_conditional_on_first_row"]
        checks.append(Check(f"Frozen {m} reproduces its saved Train log-likelihood",
                            PASS if abs(ll - saved) <= 1e-8 * abs(saved) else FAIL, f"{ll:.6f} vs {saved:.6f}"))
    blocking = [c for c in checks if c.status == FAIL]
    if blocking:
        for c in blocking:
            print(f"  [FAIL] {c.name}: {c.detail}")
        print("Frozen-artifact or input integrity failed; Stage 6 stopped (nothing written).")
        return 1
    print("Frozen artifacts verified. Building causal signals ...")

    # ------------------------------------------------------------------ signals (no returns involved)
    lv_tr = pd.DataFrame(x_tr, index=tr_feat.index, columns=FEATURE_COLUMNS)
    lv_va = pd.DataFrame(x_va, index=va_feat.index, columns=FEATURE_COLUMNS)
    daily, meta = ew.build_signals(hmm_params, ar1, arhmm, x_tr, x_va, tr_feat.index, va_feat.index, lv_tr, lv_va)
    train_mask = (daily["split"] == "train").to_numpy()
    val_mask = ~train_mask
    scaler6 = ew.fit_scaler(daily, train_mask)
    comp = ew.components(daily, scaler6)
    specs = ew.ews_specs()
    weights, subs = specs["full"]
    score, contrib = ew.ews_score(comp, weights, subs)
    th = asm.Thresholds.from_train(score[train_mask])
    states = asm.run_state_machine(score, th, daily["p_stress"], comp["confirmation"], comp["market_innovation_1d"])
    drivers = ew.dominant_drivers(contrib, comp)

    # ------------------------------------------------------------------ outcomes (evaluation only)
    cfg = json.loads((ROOT / "configs" / "project.json").read_text(encoding="utf-8"))
    outcomes, labels, plabels, events = {}, {}, {}, {}
    for split, ret in (("train", data.train_returns), ("validation", data.validation_returns)):
        spy = ret["SPY"]
        port = ev.portfolio_log_returns(ret, cfg["strategic_weights"])
        o = ev.forward_outcomes(spy)
        po = ev.forward_outcomes(port, horizons=(ev.LABEL_HORIZON,)).add_prefix("portfolio_")
        outcomes[split] = pd.concat([o, po], axis=1)
        labels[split] = ev.daily_labels(o)
        plabels[split] = ev.daily_labels(po, f"portfolio_fwd_mdd_{ev.LABEL_HORIZON}", ev.PORTFOLIO_LOSS)
        events[split] = ev.drawdown_events(spy)
    split_data = {"train": (train_mask, labels["train"], events["train"], plabels["train"]),
                  "validation": (val_mask, labels["validation"], events["validation"], plabels["validation"])}
    vidx = daily.index[val_mask]
    for block, (b0, b1) in ev.VALIDATION_BLOCKS.items():
        m = (daily.index >= pd.Timestamp(b0)) & (daily.index <= pd.Timestamp(b1))
        bl = ev.purge_block_boundary(labels["validation"][(vidx >= pd.Timestamp(b0)) & (vidx <= pd.Timestamp(b1))], b1)
        bpl = ev.purge_block_boundary(plabels["validation"][(vidx >= pd.Timestamp(b0)) & (vidx <= pd.Timestamp(b1))], b1)
        e = events["validation"]
        eb = e[(e["onset_date"] >= pd.Timestamp(b0)) & (e["onset_date"] <= pd.Timestamp(b1))].copy()
        offset = int(np.flatnonzero(m)[0] - np.flatnonzero(val_mask)[0])
        for col in ("onset_pos", "peak_pos", "end_pos"):
            eb[col] = eb[col] - offset
        split_data[f"validation {block}"] = (m, bl, eb, bpl)
    # positions in events are relative to each split's own series; masks select that series
    print(f"Events (SPY drawdown >= 5%): Train {len(events['train'])}, Validation {len(events['validation'])}")

    # ------------------------------------------------------------------ systems
    sys_scores = {
        "A_hmm_hard_stress": (daily["hmm_state"] == 3).astype(float),
        "B_p_stress": daily["p_stress"], "C_p_non_normal": daily["p_non_normal"],
        "D_volatility_level": daily["volatility"], "E_ar_innovation": daily["ar_innovation_stress"],
        "F_bocpd": daily["bocpd_primary_recent"],
        "G_hmm_bocpd": ew.equal_weight_score(comp, ["regime", "bocpd"]),
        "H_hmm_innovation_bocpd": ew.equal_weight_score(comp, ["regime", "innovation", "bocpd"]),
        "I_full_ews_score": score}
    rows, alerts = [], {}
    for name, sc in sys_scores.items():
        a = sc.astype(int) if name.startswith("A_") else ev.budget_alert(sc, train_mask)[0]
        alerts[name] = a
        rows += ev.evaluate_system(name, sc, a, {k: (v[0], v[1], v[2], v[3]) for k, v in _masked(split_data, sc, a).items()})
    for lvl in ("AMBER", "ORANGE"):
        a = asm.alert_flag(states["alert_level"], lvl)
        alerts[f"I_full_ews_state_{lvl.lower()}"] = a
        rows += ev.evaluate_system(f"I_full_ews_state_{lvl.lower()}", score, a, _masked(split_data, score, a))
    comparison = pd.DataFrame(rows)
    vscores = pd.DataFrame({k: v for k, v in sys_scores.items() if k != "A_hmm_hard_stress"})[val_mask]
    vap = comparison[(comparison["split"] == "validation") & comparison["system"].isin(vscores.columns)].set_index("system")["pr_auc"]
    best = vap.idxmax()
    reps = args.bootstrap_reps or ev.BOOTSTRAP_REPS
    boot = ev.block_bootstrap_ap(vscores, labels["validation"], best, reps=reps)
    selection = ev.select_specification(boot, comparison, best)

    # ------------------------------------------------------------------ ablations
    abl_rows = []
    for spec, (w, sub) in specs.items():
        s_, _ = ew.ews_score(comp, w, sub)
        t_ = asm.Thresholds.from_train(s_[train_mask])
        st_ = asm.run_state_machine(s_, t_, daily["p_stress"], comp["confirmation"], comp["market_innovation_1d"])
        a_ = asm.alert_flag(st_["alert_level"], "AMBER")
        for r in ev.evaluate_system(spec, s_, a_, _masked(split_data, s_, a_)):
            abl_rows.append({"specification": spec, "weights": json.dumps({k: v for k, v in w.items() if v}),
                             "substitution": json.dumps(sub or {}), **r})
    st_nh = asm.run_state_machine(score, th, hysteresis=False)
    a_nh = asm.alert_flag(st_nh["alert_level"], "AMBER")
    for r in ev.evaluate_system("no_hysteresis", score, a_nh, _masked(split_data, score, a_nh)):
        abl_rows.append({"specification": "no_hysteresis", "weights": json.dumps({k: v for k, v in weights.items() if v}),
                         "substitution": "{}", **r})
    ablation = pd.DataFrame(abl_rows)

    # ------------------------------------------------------------------ BOCPD input / hazard comparison
    bo_rows = []
    for name in list(ew.BOCPD_INPUTS) + list(ew.BOCPD_CHANNELS):
        hz = bo.HAZARD_RUN_LENGTHS if name in ew.BOCPD_INPUTS else (bo.PRIMARY_RUN_LENGTH,)
        for lam in hz:
            col = f"bocpd_{name}_h{lam}_p_change_within_{bo.PRIMARY_WINDOW}d"
            sc = daily[col]
            a, thr = ev.budget_alert(sc, train_mask)
            for r in ev.evaluate_system(f"{name}_h{lam}", sc, a, _masked(split_data, sc, a)):
                if r["split"] in ("train", "validation"):
                    bo_rows.append({"input": name, "expected_run_length": lam, "threshold": thr,
                                    "prior_p_change_within_10d": bo.prior_recent_probability(lam, bo.PRIMARY_WINDOW),
                                    "mean_posterior": float(sc[train_mask if r["split"] == "train" else val_mask].mean()),
                                    **{k: r[k] for k in ("split", "pr_auc", "base_rate", "n_events", "warned", "missed",
                                                         "median_lead_days", "false_alerts_per_year", "time_share_alerted",
                                                         "alert_episodes", "episode_precision")}})
    bocpd_cmp = pd.DataFrame(bo_rows)
    scenarios = bo.synthetic_scenarios()     # synthetic inputs only: what BOCPD detects and how fast

    # ------------------------------------------------------------------ entropy, velocity, AR-HMM surprise
    ent_rows = []
    for split, m in (("train", train_mask), ("validation", val_mask)):
        e = daily.loc[m, "normalized_entropy"]
        ent_rows.append({"split": split, "mean": e.mean(), "median": e.median(), "share_above_0.2": (e > 0.2).mean(),
                         "share_above_0.5": (e > 0.5).mean()})
    entropy_summary = pd.DataFrame(ent_rows)
    entropy_cond = ev.conditional_event_rates(daily[val_mask], labels["validation"], "normalized_entropy")
    vel_thr = scaler6.ecdf["velocity"].quantile(0.90)
    velocity = ev.velocity_lead(daily, vel_thr)
    if not velocity.empty:
        velocity["split"] = np.where(velocity["switch_date"] <= tr_feat.index.max(), "train", "validation")
    # incremental-information diagnostics: equal-weight sums of [0,1] components (Train-fixed scales)
    extra = {"normalized_entropy": daily["normalized_entropy"],
             "p_non_normal_plus_entropy": 0.5 * daily["p_non_normal"] + 0.5 * comp["entropy"],
             "velocity_positive": daily["non_normal_probability_velocity_positive"],
             "p_non_normal_plus_velocity": 0.5 * daily["p_non_normal"] + 0.5 * comp["velocity"],
             "ar_innovation_stress": daily["ar_innovation_stress"], "ar_surprise_chi2": daily["surprise_chi2"],
             "arhmm_surprise": daily["arhmm_surprise"], "p_non_normal": daily["p_non_normal"]}
    inc_rows = []
    for name, sc in extra.items():
        for split, m in (("train", train_mask), ("validation", val_mask)):
            d = ev.daily_metrics(sc[m], sc[m] * 0, labels[split])
            inc_rows.append({"signal": name, "split": split, "pr_auc": d["pr_auc"], "base_rate": d["base_rate"]})
    incremental = pd.DataFrame(inc_rows)
    switch_entropy = er.entropy_around_switches(daily)

    # ------------------------------------------------------------------ causality checks on the real run
    cut = int(val_mask.sum() // 2)
    shocked = x_va.copy()
    shocked[cut:] = shocked[cut:] * -3 + 2
    lv_s = pd.DataFrame(shocked, index=va_feat.index, columns=FEATURE_COLUMNS)
    daily2, _ = ew.build_signals(hmm_params, ar1, arhmm, x_tr, shocked, tr_feat.index, va_feat.index, lv_tr, lv_s)
    comp2 = ew.components(daily2, scaler6)
    score2, _ = ew.ews_score(comp2, weights, subs)
    states2 = asm.run_state_machine(score2, th, daily2["p_stress"], comp2["confirmation"], comp2["market_innovation_1d"])
    n_before = int(np.flatnonzero(val_mask)[0] + cut)
    num_cols = [c for c in daily.columns if pd.api.types.is_numeric_dtype(daily[c])]
    diff = np.nanmax(np.abs(daily2[num_cols].iloc[:n_before].to_numpy(float) - daily[num_cols].iloc[:n_before].to_numpy(float)))
    same_alerts = states2["alert_level"].iloc[:n_before].equals(states["alert_level"].iloc[:n_before])
    sdiff = float(np.nanmax(np.abs(score2.iloc[:n_before] - score.iloc[:n_before])))
    checks.append(Check("Changing later Validation factors leaves every earlier signal, score and alert unchanged",
                        PASS if diff == 0 and sdiff == 0 and same_alerts else FAIL,
                        f"max change before cut: signals {diff:.1e}, EWS {sdiff:.1e}; alerts identical {same_alerts}"))
    import inspect
    checks.append(Check("Investment returns are not an input of signal construction",
                        PASS if "return" not in " ".join(inspect.signature(ew.build_signals).parameters) else FAIL,
                        "build_signals(" + ", ".join(inspect.signature(ew.build_signals).parameters) + ")"))
    psum = float(np.abs(daily[["p_s1", "p_s2", "p_s3"]].sum(axis=1) - 1).max())
    checks.append(Check("Filtered HMM probabilities sum to one (filtered, not smoothed)", PASS if psum < 1e-9 else FAIL, f"{psum:.1e}"))
    fin = bool(np.isfinite(score.iloc[1 + ins.VELOCITY_LAG + ins.ROLLING_DAYS:]).all())
    checks.append(Check("EWS finite after the warm-up sessions", PASS if fin else FAIL, f"{int(score.isna().sum())} warm-up NaN"))
    checks.append(Check("No test file read", FAIL if any(f.startswith("test_") for f in data.files_read) else PASS,
                        ", ".join(data.files_read)))
    spread = comparison.groupby("split")["n_days"].agg(lambda v: int(v.max() - v.min()))
    checks.append(Check("Baselines compared on the same labelled days (up to signal warm-up)",
                        PASS if spread.max() <= 1 + ins.VELOCITY_LAG + ins.ROLLING_DAYS else FAIL,
                        "max difference in eligible days across systems: " +
                        ", ".join(f"{k} {v}" for k, v in spread.items())))

    # ------------------------------------------------------------------ outputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(index=daily.index)
    out["split"] = daily["split"]
    for c in ("p_s1", "p_s2", "p_s3", "p_non_normal", "normalized_entropy", "stress_probability_velocity",
              "non_normal_probability_velocity", "ar_innovation_stress"):
        out[c] = daily[c]
    out["bocpd_change_probability"] = daily["bocpd_primary_recent"]
    out["economic_confirmation_score"] = comp["confirmation"]
    out["early_warning_score"] = score
    out["alert_level"] = states["alert_level"]
    out["alert_reason"] = states["alert_rule"]
    out["dominant_risk_drivers"] = drivers
    for c in contrib.columns:
        out[f"contribution_{c}"] = contrib[c]
    for c in ("regime", "confirmation", "innovation", "bocpd", "velocity", "macro", "entropy"):
        out[f"component_{c}"] = comp[c]
    hmm_cols = ["split", "p_s1", "p_s2", "p_s3", "p_stress", "p_non_normal", "normalized_entropy", "hmm_state",
                "stress_probability_velocity", "stress_probability_velocity_positive", "non_normal_probability_velocity",
                "non_normal_probability_velocity_positive", "hmm_log_predictive"]
    inn_cols = ["split"] + [c for c in daily.columns if c.startswith(("z_", "directional_z_", "abs_z_"))] + [
        "market_innovation", "macro_innovation", "market_innovation_5d", "macro_innovation_5d", "ar_innovation_stress",
        "surprise_chi2", "stress_index_level", "arhmm_surprise", "arhmm_surprise_probit"]
    bo_cols = ["split"] + [c for c in daily.columns if c.startswith("bocpd_")]
    for name, frame, cols in (("bocpd", daily, bo_cols), ("innovation_signals", daily, inn_cols),
                              ("hmm_warning_features", daily, hmm_cols), ("early_warning", out, list(out.columns))):
        for split, m in (("train", train_mask), ("validation", val_mask)):
            frame.loc[m, cols].to_csv(OUT_DIR / f"{name}_{split}.csv", index_label="date", date_format="%Y-%m-%d",
                                      float_format="%.6g")
    m_full, ev_full = ev.event_metrics(alerts["I_full_ews_state_amber"][val_mask], events["validation"])
    ev_full.assign(system="I_full_ews_state_amber").to_csv(OUT_DIR / "alert_events_validation.csv", index=False)
    episodes = [{"start": vidx[s].date(), "end": vidx[t - 1].date(), "days": t - s,
                 "max_level": max(states["alert_level"][val_mask].iloc[s:t], key=asm.SEVERITY.get)}
                for s, t in ev.alert_episodes(alerts["I_full_ews_state_amber"][val_mask].to_numpy())]
    pd.DataFrame(episodes).to_csv(OUT_DIR / "alert_episodes_validation.csv", index=False)
    tables = {"early_warning_model_comparison.csv": comparison, "early_warning_ablation.csv": ablation,
              "bocpd_input_comparison.csv": bocpd_cmp, "bocpd_synthetic_scenarios.csv": scenarios, "pr_auc_bootstrap_validation.csv": boot,
              "entropy_summary.csv": entropy_summary, "entropy_conditional_event_rates.csv": entropy_cond,
              "velocity_lead_before_switches.csv": velocity, "incremental_signal_pr_auc.csv": incremental,
              "entropy_around_switches.csv": switch_entropy,
              "risk_events.csv": pd.concat([events["train"].assign(split="train"), events["validation"].assign(split="validation")]),
              "forward_outcomes.csv": pd.concat([outcomes["train"].assign(split="train"),
                                                 outcomes["validation"].assign(split="validation")]).reset_index()}
    for name, frame in tables.items():
        frame.to_csv(OUT_DIR / name, index=False, float_format="%.6g")
    hashes_after = {**{f"hmm/{k}": v for k, v in directory_hashes(STAGE5_DIR).items()},
                    **{f"ar_hmm/{k}": v for k, v in directory_hashes(STAGE55_DIR).items()}}
    checks.append(Check("Stage 5 and Stage 5.5 artifacts unchanged (hashes before == after)",
                        PASS if hashes_after == hashes_before else FAIL, f"{len(hashes_before)} files compared"))
    warn_list = sorted({f"{w.category.__name__}: {str(w.message)[:150]}" for w in caught})
    checks.append(Check("No library warnings", PASS if not warn_list else WARN, "; ".join(warn_list) or "none"))
    pd.DataFrame([vars(c) for c in checks]).to_csv(OUT_DIR / "early_warning_checks.csv", index=False)

    ctx = {"daily": daily, "comp": comp, "contrib": contrib, "score": score, "states": states, "thresholds": th,
           "events": events, "comparison": comparison, "ablation": ablation, "bocpd_cmp": bocpd_cmp, "boot": boot,
           "selection": selection, "entropy_summary": entropy_summary, "entropy_cond": entropy_cond,
           "velocity": velocity, "incremental": incremental, "switch_entropy": switch_entropy, "checks": checks,
           "weights": weights, "meta": meta, "data": data, "ev_full": ev_full, "m_full": m_full, "episodes": episodes,
           "train_mask": train_mask, "val_mask": val_mask, "labels": labels, "alerts": alerts,
           "va_index": va_feat.index, "scenarios": scenarios, "figures": []}
    if not args.no_figures:
        ctx["figures"] = er.save_figures(FIGURE_DIR, ctx)
    figures = ctx["figures"]
    recorder.__exit__(None, None, None)
    manifest = {
        "stage": "6 - dual-speed early warning system", "model_version": STAGE6_CODE_VERSION,
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "evidence_status": "Validation (2016-2019) was used in Stages 5 and 5.5: all Stage 6 Validation results are "
                           "development-sample evidence, not an independent holdout. Train-period HMM signals are "
                           "in-sample reconstructions.",
        "frozen_inputs": {"hmm": {"file": str(hmm_path.relative_to(ROOT)), "sha256": file_sha256(hmm_path),
                                  "model_version": hmm_meta.get("model_version")},
                          **{m: {"file": f"data/processed/ar_hmm/models/{m}.json",
                                 "sha256": file_sha256(STAGE55_DIR / "models" / f"{m}.json"),
                                 "model_version": ar_meta[m].get("model_version"),
                                 "variance_rule": ar_meta[m].get("variance_rule")} for m in ar_meta}},
        "input_sha256": {n: data.manifest.get("split_files", {}).get(n) for n in data.files_read if n.endswith(".csv")},
        "files_read": data.files_read, "test_data_loaded": False,
        "signal_timing": "every date-t signal uses factors through t and models estimated on Train; no smoothing; "
                         "investment returns are used only to build forward outcomes for evaluation",
        "train_normalization": meta["normalization"].to_dict(), "bocpd_priors": meta["bocpd_priors"],
        "bocpd": {"hazard_grid": list(bo.HAZARD_RUN_LENGTHS), "primary_run_length": bo.PRIMARY_RUN_LENGTH,
                  "primary_input": "B2_innovation (AR(1) market innovation, daily)",
                  "evidence": f"P(r_t < {bo.PRIMARY_WINDOW} | y_1..t)", "kappa0": bo.PRIOR_KAPPA0, "alpha0": bo.PRIOR_ALPHA0,
                  "truncation": "none (exact)"},
        "ews_weights": weights, "component_scales": scaler6.to_dict(), "alert_thresholds": th.to_dict(),
        "alert_quantiles": asm.QUANTILES, "confirmation": f"{asm.CONFIRM_MIN} of {asm.CONFIRM_WINDOW} sessions",
        "emergency_rule": asm.EMERGENCY, "recovery_clear_days": asm.RECOVERY_CLEAR_DAYS,
        "outcomes": {"label": "SPY worst cumulative loss within 21 sessions <= -5%", "portfolio_label":
                     "strategic reference portfolio worst loss within 21 sessions <= -3%",
                     "events": "SPY drawdown from running peak <= -5% (re-arm above -2.5%)", "lead_window": ev.LEAD_WINDOW,
                     "train_alert_budget": ev.TRAIN_ALERT_BUDGET, "validation_blocks": ev.VALIDATION_BLOCKS},
        "selection": selection, "checks": [vars(c) for c in checks],
        "outputs": sorted(str(p.relative_to(ROOT)) for p in OUT_DIR.rglob("*") if p.is_file())}
    save_json(manifest, OUT_DIR / "early_warning_selection_manifest.json")
    (REPORTS / "early_warning_research.md").write_text(er.render_report(ctx), encoding="utf-8")

    # ------------------------------------------------------------------ console
    show = comparison[comparison["split"].isin(["train", "validation"])]
    cols = ["system", "split", "n_events", "warned", "missed", "median_lead_days", "false_alerts_per_year",
            "time_share_alerted", "pr_auc", "base_rate"]
    print("\nSystem comparison (Train = in-sample reconstruction; Validation = development sample):")
    print(show[cols].round(3).to_string(index=False))
    print(f"\nBest Validation PR-AUC: {selection['best_validation_pr_auc']}; selected (least complex not significantly "
          f"worse): {selection['selected']}")
    print("Ablation (Validation, state machine >= AMBER):")
    print(ablation[ablation["split"] == "validation"][["specification", "warned", "missed", "false_alerts_per_year",
                                                       "time_share_alerted", "pr_auc"]].round(3).to_string(index=False))
    print("BOCPD inputs (Validation):")
    print(bocpd_cmp[bocpd_cmp["split"] == "validation"][["input", "expected_run_length", "pr_auc", "warned",
                                                         "false_alerts_per_year", "time_share_alerted"]].round(3).to_string(index=False))
    print("Checks:")
    for c in checks:
        print(f"  [{c.status}] {c.name}: {c.detail}")
    counts = pd.Series([c.status for c in checks]).value_counts()
    print("Summary: " + ", ".join(f"{counts.get(s, 0)} {s}" for s in (PASS, WARN, FAIL)))
    print(f"Report: {REPORTS / 'early_warning_research.md'}; figures {len(figures)}. Elapsed "
          f"{time.perf_counter() - started:.0f}s. Stage 6 complete; no portfolio optimization was run.")
    return 1 if counts.get(FAIL, 0) else 0


def _masked(split_data: dict, score, alerts) -> dict:
    """Restrict score / alerts to each split's rows (labels and events are already per split)."""
    out = {}
    for split, (mask, labels, events, plabels) in split_data.items():
        out[split] = (mask, labels.reindex(score.index[mask]), events, plabels.reindex(score.index[mask]))
    return out


if __name__ == "__main__":
    sys.exit(main())
