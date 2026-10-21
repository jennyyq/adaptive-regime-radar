"""Stage 6.1: non-tuning audit of the Stage 6 early-warning evaluation and Stage 7 handoff (Train/Validation only).

Nothing is fitted, re-optimized or selected. The frozen Stage 6 signals are rebuilt from the frozen Stage 5 / 5.5
models with the Stage 6 code (no refit) and checked against the saved Stage 6 outputs; the frozen thresholds and
alert rules are re-evaluated under 5/10/21/63-session warning windows with the permissive (Stage 6) and strict
one-to-one episode-matching conventions.

Reads ONLY data/splits/{train,validation}_{features,returns}.csv + split_manifest.json, the frozen model JSONs,
data/processed/early_warning/* (Stage 6 outputs) and configs/project.json. The test split is never opened.
Stage 5, 5.5 and 6 artifacts are hashed before and after and never written.

Outputs (new only): data/processed/early_warning/audit/*, data/processed/early_warning/stage7_handoff/*,
                    reports/early_warning_audit.md, reports/figures/early_warning_audit/*.png
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

SPLITS_DIR = ROOT / "data" / "splits"
STAGE61_CODE_VERSION = "stage6.1-2026-10-21b"
STAGE6_CODE_VERSION = "stage6-2026-10-21"
EW_DIR = PROCESSED / "early_warning"
AUDIT_DIR = EW_DIR / "audit"
FIG_DIR = REPORTS / "figures" / "early_warning_audit"
FROZEN_HMM = "hmm_std_diag_k3"
REPRO_RTOL = 2e-5   # Stage 6 CSVs store 6 significant digits


def stage6_hashes() -> dict:
    from regime_radar.ar_hmm_diagnostics import directory_hashes, file_sha256
    h = {f"early_warning/{k}": v for k, v in directory_hashes(EW_DIR).items()
         if not k.startswith(("audit/", "stage7_handoff/"))}
    h.update({f"hmm/{k}": v for k, v in directory_hashes(PROCESSED / "hmm").items()})
    h.update({f"ar_hmm/{k}": v for k, v in directory_hashes(PROCESSED / "ar_hmm").items()})
    h.update({f"figures/early_warning/{k}": v for k, v in directory_hashes(REPORTS / "figures" / "early_warning").items()})
    h["reports/early_warning_research.md"] = file_sha256(REPORTS / "early_warning_research.md")
    return h


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bootstrap-reps", type=int, default=None)
    parser.add_argument("--no-tests", action="store_true", help="skip the automated real-data test run")
    args = parser.parse_args()

    from regime_radar import alert_state_machine as asm
    from regime_radar import early_warning as ew
    from regime_radar import early_warning_audit as au
    from regime_radar import early_warning_audit_report as ar
    from regime_radar import early_warning_validation as ev
    from regime_radar import regime_exploration as rx
    from regime_radar import stage7_handoff as h7
    from regime_radar.ar_hmm import ARParams
    from regime_radar.ar_hmm_diagnostics import file_sha256, save_json
    from regime_radar.dataset import FEATURE_COLUMNS
    from regime_radar.factors import Check
    from regime_radar.hmm_models import HMMParams

    stale = [m.__file__ for m in (au, ar, h7) if getattr(m, "STAGE61_CODE_VERSION", None) != STAGE61_CODE_VERSION]
    stale += [m.__file__ for m in (asm, ew, ev) if getattr(m, "STAGE6_CODE_VERSION", None) != STAGE6_CODE_VERSION]
    if stale:
        print("Module versions do not match this script: " + ", ".join(stale) + ". Re-extract the zip, delete "
              "__pycache__ and rerun.")
        return 3
    PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
    started = time.perf_counter()
    recorder = warnings.catch_warnings(record=True)
    caught = recorder.__enter__()
    warnings.simplefilter("default")

    man6_path = EW_DIR / "early_warning_selection_manifest.json"
    if not man6_path.exists():
        raise SystemExit("Stage 6 outputs not found. Run scripts/08_build_early_warning.py first.")
    man6 = json.loads(man6_path.read_text(encoding="utf-8"))
    hashes_before = stage6_hashes()
    checks = [Check("Stage 6 outputs were produced by the matching Stage 6 code",
                    PASS if man6.get("model_version") == STAGE6_CODE_VERSION else FAIL,
                    f"manifest model_version {man6.get('model_version')}")]
    s6_fail = [c["name"] for c in man6.get("checks", []) if c.get("status") == "FAIL"]
    checks.append(Check("Stage 6 run had no FAILed checks", PASS if not s6_fail else FAIL, ", ".join(s6_fail) or "none"))

    # ------------------------------------------------------------------ rebuild frozen signals (no refit)
    data = rx.load_stage4_data(SPLITS_DIR)
    tr_feat, va_feat = data.train_features[FEATURE_COLUMNS], data.validation_features[FEATURE_COLUMNS]
    apply, _ = rx.scaling_decision(rx.describe_train(tr_feat))
    scaler = rx.TrainScaler(apply).fit(tr_feat)
    x_tr, x_va = scaler.transform(tr_feat), scaler.transform(va_feat)
    frozen = man6["frozen_inputs"]
    for key, meta in frozen.items():
        path = ROOT / meta["file"]
        checks.append(Check(f"Frozen {key} file unchanged since Stage 6", PASS if file_sha256(path) == meta["sha256"]
                            else FAIL, f"{meta['file']} sha256 {str(meta['sha256'])[:12]}..."))
    hmm_meta = json.loads((ROOT / frozen["hmm"]["file"]).read_text(encoding="utf-8"))
    ar1_meta = json.loads((ROOT / frozen["ar1"]["file"]).read_text(encoding="utf-8"))
    arh_meta = json.loads((ROOT / frozen["ar_hmm_k3"]["file"]).read_text(encoding="utf-8"))
    for name, digest in man6.get("input_sha256", {}).items():
        now = file_sha256(SPLITS_DIR / name)
        checks.append(Check(f"Split file {name} unchanged since Stage 6", PASS if now == digest else FAIL,
                            f"sha256 {str(digest)[:12]}..."))
    if any(c.status == FAIL for c in checks):
        for c in checks:
            if c.status == FAIL:
                print(f"  [FAIL] {c.name}: {c.detail}")
        print("Frozen inputs differ from the Stage 6 run; audit stopped (nothing written).")
        return 1
    print("Frozen inputs verified. Rebuilding Stage 6 signals (no refit) ...")
    lv_tr = pd.DataFrame(x_tr, index=tr_feat.index, columns=FEATURE_COLUMNS)
    lv_va = pd.DataFrame(x_va, index=va_feat.index, columns=FEATURE_COLUMNS)
    daily, _ = ew.build_signals(HMMParams.from_dict(hmm_meta["parameters"]), ARParams.from_dict(ar1_meta["parameters"]),
                                ARParams.from_dict(arh_meta["parameters"]), x_tr, x_va, tr_feat.index, va_feat.index,
                                lv_tr, lv_va)
    train_mask = (daily["split"] == "train").to_numpy()
    val_mask = ~train_mask
    comp = ew.components(daily, ew.fit_scaler(daily, train_mask))
    weights, subs = ew.ews_specs()["full"]
    score, _ = ew.ews_score(comp, weights, subs)
    th = asm.Thresholds.from_train(score[train_mask])
    states = asm.run_state_machine(score, th, daily["p_stress"], comp["confirmation"], comp["market_innovation_1d"])
    levels = states["alert_level"]

    # ------------------------------------------------------------------ reproduction of saved Stage 6 outputs
    saved = pd.concat([pd.read_csv(EW_DIR / f"early_warning_{s}.csv", index_col="date", parse_dates=["date"])
                       for s in ("train", "validation")])
    same_idx = saved.index.equals(daily.index)
    rel = {}
    for col, new in (("p_non_normal", daily["p_non_normal"]), ("early_warning_score", score),
                     ("p_s2", daily["p_s2"]), ("p_s3", daily["p_s3"])):
        a, b = saved[col].to_numpy(float), new.to_numpy(float)
        ok = np.isnan(a) == np.isnan(b)
        diff = np.abs(a - b)[~np.isnan(a)] / np.maximum(np.abs(b[~np.isnan(a)]), 1e-12)
        rel[col] = float(np.max(np.where(np.abs(a - b)[~np.isnan(a)] < 1e-9, 0.0, diff))) if ok.all() else np.inf
    same_levels = same_idx and bool((saved["alert_level"].to_numpy() == levels.to_numpy()).all())
    th_saved = man6["alert_thresholds"]
    th_diff = max(abs(th_saved[k] - v) for k, v in th.to_dict().items())
    checks.append(Check("Rebuilt signals reproduce the saved Stage 6 outputs (6 significant digits)",
                        PASS if same_idx and max(rel.values()) <= REPRO_RTOL else FAIL,
                        "max relative difference: " + ", ".join(f"{k} {v:.1e}" for k, v in rel.items())))
    checks.append(Check("Rebuilt alert levels identical to the saved Stage 6 alert levels", PASS if same_levels else FAIL,
                        f"{int((saved['alert_level'].to_numpy() != levels.to_numpy()).sum()) if same_idx else 'index differs'}"
                        " differing days"))
    checks.append(Check("Frozen Stage 6 thresholds reproduced from Train", PASS if th_diff < 1e-12 else FAIL,
                        f"max |difference| {th_diff:.1e}"))
    pnn_def = float(np.max(np.abs(daily["p_non_normal"] - (daily["p_s2"] + daily["p_s3"]))))
    checks.append(Check("P(non-normal) = P(S2) + P(S3) (filtered probabilities)", PASS if pnn_def < 1e-12 else FAIL,
                        f"max |difference| {pnn_def:.1e}"))
    if any(c.status == FAIL for c in checks):
        for c in checks:
            if c.status == FAIL:
                print(f"  [FAIL] {c.name}: {c.detail}")
        print("Rebuilt signals do not match the frozen Stage 6 outputs; audit stopped (nothing written).")
        return 1

    # ------------------------------------------------------------------ outcomes and events (as in Stage 6)
    labels, events, plabels = {}, {}, {}
    cfg = json.loads((ROOT / "configs" / "project.json").read_text(encoding="utf-8"))
    for split, ret in (("train", data.train_returns), ("validation", data.validation_returns)):
        o = ev.forward_outcomes(ret["SPY"])
        labels[split] = ev.daily_labels(o)
        po = ev.forward_outcomes(ev.portfolio_log_returns(ret, cfg["strategic_weights"]), horizons=(ev.LABEL_HORIZON,))
        plabels[split] = ev.daily_labels(po, loss=ev.PORTFOLIO_LOSS)
        events[split] = ev.drawdown_events(ret["SPY"])
    saved_events = pd.read_csv(EW_DIR / "risk_events.csv", parse_dates=["onset_date"])
    ev_now = pd.concat([events["train"].assign(split="train"), events["validation"].assign(split="validation")])
    same_events = (len(saved_events) == len(ev_now) and
                   (saved_events["onset_date"].to_numpy() == ev_now["onset_date"].to_numpy()).all())
    checks.append(Check("Event definition reproduces the Stage 6 risk events", PASS if same_events else FAIL,
                        f"Train {len(events['train'])}, Validation {len(events['validation'])} SPY -5% drawdown onsets"))

    # ------------------------------------------------------------------ frozen specifications and budgets
    masks = {"train": train_mask, "validation": val_mask}
    p = daily["p_non_normal"]
    flags = {lvl: asm.alert_flag(levels, lvl) for lvl in ("AMBER", "ORANGE")}
    share_amber = float(flags["AMBER"][train_mask].mean())
    share_orange = float(flags["ORANGE"][train_mask].mean())
    budgets = {"budget20": ev.TRAIN_ALERT_BUDGET, "amber_exposure": share_amber, "orange_exposure": share_orange}
    specs, spec_meta = {}, []
    for fam, sc, label in (("A", p, "P(non-normal)"), ("B", score, "Full EWS score")):
        for bname, share in budgets.items():
            ex = au.exposure_audit(sc, train_mask, val_mask, share)     # no tie-breaking, no retuning
            thr = ex["threshold"]
            key = f"{fam}_{'p_non_normal' if fam == 'A' else 'ews_score'}@{bname}"
            specs[key] = (sc >= thr).astype(int).where(sc.notna(), 0)
            spec_meta.append({"spec": key, "family": fam, "signal": label, "budget": bname,
                              "rule": f"score >= Train quantile {1 - share!r}", "role": "evaluation", **ex})
    specs["A_p_non_normal@0.5"] = (p >= 0.5).astype(int)
    spec_meta.append({"spec": "A_p_non_normal@0.5", "family": "A", "signal": "P(non-normal)", "budget": "none",
                      "rule": "P(non-normal) >= 0.5 (natural threshold; descriptive)", "threshold": 0.5,
                      "threshold_repr": "0.5", "target_train_share": np.nan, "role": "descriptive",
                      "exposure_matched": np.nan, "exposure_status": "descriptive (no budget)"})
    for lvl in ("AMBER", "ORANGE"):
        key = f"C_ews_state>={lvl}"
        specs[key] = flags[lvl]
        spec_meta.append({"spec": key, "family": "C", "signal": "Full EWS state machine",
                          "budget": "amber_exposure" if lvl == "AMBER" else "orange_exposure",
                          "rule": f"alert level >= {lvl} (RECOVERY counts as no alert)", "threshold": np.nan,
                          "threshold_repr": "frozen state machine", "role": "evaluation",
                          "target_train_share": budgets["amber_exposure" if lvl == "AMBER" else "orange_exposure"],
                          "achieved_train_share": float(flags[lvl][train_mask].mean()), "exposure_matched": True,
                          "exposure_status": "EXPOSURE-MATCHED (defines the budget)"})
    spec_table = pd.DataFrame(spec_meta)
    for split, m in masks.items():
        spec_table[f"{split}_time_share"] = [float(specs[s][m].mean()) for s in spec_table["spec"]]
    spec_table["train_n_exactly_one"] = [int((p[train_mask] == 1.0).sum()) if f == "A" else np.nan
                                         for f in spec_table["family"]]

    # ------------------------------------------------------------------ operational evaluation
    op_rows, ev_tabs, ep_tabs, strict_violations = [], [], [], []
    for name, a in specs.items():
        for split, m in masks.items():
            rows, et, ept = au.evaluate(a[m], events[split])
            dates = daily.index[m]
            r_spy = (data.train_returns if split == "train" else data.validation_returns)["SPY"]
            et = [au.protection_columns(t, r_spy) for t in et]
            for r, t in zip(rows, et):        # rows and tables are produced in the same (window, convention) order
                r["executable_warned"] = int(t["executable_warning"].sum())
                r["median_executable_lead"] = float(t["executable_lead_sessions"].median()) \
                    if t["executable_lead_sessions"].notna().any() else np.nan
                r["median_drawdown_at_execution_pct"] = float(t["drawdown_at_execution_pct"].median()) \
                    if t["drawdown_at_execution_pct"].notna().any() else np.nan
            op_rows += [{"spec": name, "split": split, **r} for r in rows]
            for t in et:
                ev_tabs.append(t.assign(spec=name, split=split))
            for w in au.WINDOWS:                # independent invariant check of every strict table
                tw = [t for t in et if t["window"].iloc[0] == w and t["convention"].iloc[0] == "strict"] if \
                    len(events[split]) else []
                ew_ = [t for t in ept if len(t) and t["window"].iloc[0] == w]
                if tw:
                    viol = au.verify_strict(a[m].fillna(0).astype(int).to_numpy(), events[split], w, tw[0],
                                            ew_[0] if ew_ else pd.DataFrame(columns=["start_pos", "end_pos_exclusive"]))
                    strict_violations += [f"{name}/{split}/W{w}: {x}" for x in viol]
            for t in ept:
                t = t.assign(spec=name, split=split)
                if len(t):
                    t["start_date"] = dates[t["start_pos"].to_numpy(int)]
                    t["last_alert_date"] = dates[t["end_pos_exclusive"].to_numpy(int) - 1]
                ep_tabs.append(t)
    operational = pd.DataFrame(op_rows)
    event_table = pd.concat(ev_tabs, ignore_index=True)
    episode_table = pd.concat(ep_tabs, ignore_index=True)

    # ------------------------------------------------------------------ ranking (statistical) evaluation
    rk_rows = []
    rank_scores = {"A_p_non_normal": p, "B_ews_score": score}
    blocks = {"train": train_mask, "validation": val_mask}
    vidx = daily.index[val_mask]
    for bname, (b0, b1) in ev.VALIDATION_BLOCKS.items():
        blocks[f"validation {bname}"] = (daily.index >= pd.Timestamp(b0)) & (daily.index <= pd.Timestamp(b1))
    for lab_name, labset in (("SPY 21d loss >= 5%", labels), ("portfolio 21d loss >= 3%", plabels)):
        for bname, m in blocks.items():
            if bname.startswith("validation "):
                b0, b1 = ev.VALIDATION_BLOCKS[bname.split(" ", 1)[1]]
                lab = ev.purge_block_boundary(labset["validation"][(vidx >= pd.Timestamp(b0)) &
                                                                   (vidx <= pd.Timestamp(b1))], b1)
            else:
                lab = labset[bname]
            common = lab.notna() & p[m].reindex(lab.index).notna() & score[m].reindex(lab.index).notna()
            for sname, sc in rank_scores.items():
                rk_rows.append({"label": lab_name, "sample": bname, "score": sname,
                                **au.ranking_metrics(sc.reindex(lab.index)[common], lab[common])})
    ranking = pd.DataFrame(rk_rows)
    reps = args.bootstrap_reps or ev.BOOTSTRAP_REPS
    vs = pd.DataFrame({k: v[val_mask] for k, v in rank_scores.items()})
    boot = ev.block_bootstrap_ap(vs, labels["validation"], "A_p_non_normal", reps=reps)
    boot_p = ev.block_bootstrap_ap(vs, plabels["validation"], "A_p_non_normal", reps=reps)
    boot = pd.concat([boot.assign(label="SPY 21d loss >= 5%"), boot_p.assign(label="portfolio 21d loss >= 3%")])

    # ------------------------------------------------------------------ audit checks
    s6cmp = pd.read_csv(EW_DIR / "early_warning_model_comparison.csv")
    mapping = {"A_p_non_normal@budget20": "C_p_non_normal", "B_ews_score@budget20": "I_full_ews_score",
               "C_ews_state>=AMBER": "I_full_ews_state_amber", "C_ews_state>=ORANGE": "I_full_ews_state_orange"}
    mism = []
    for new, old in mapping.items():
        for split in ("train", "validation"):
            r_new = operational[(operational["spec"] == new) & (operational["split"] == split) &
                                (operational["window"] == ev.LEAD_WINDOW) & (operational["convention"] == "permissive")]
            r_old = s6cmp[(s6cmp["system"] == old) & (s6cmp["split"] == split)]
            for k in ("warned", "detected_late", "missed", "alert_episodes", "false_alert_episodes"):
                if r_new.empty or r_old.empty or int(r_new[k].iloc[0]) != int(r_old[k].iloc[0]):
                    mism.append(f"{new}/{split}/{k}")
    checks.append(Check("Permissive convention at W=63 reproduces the Stage 6 event results", PASS if not mism else FAIL,
                        "mismatches: " + ", ".join(mism) if mism else "4 specifications x 2 splits x 5 counts"))
    w_ok = event_table[event_table["status"] == "warned"]
    bad_lead = w_ok[(w_ok["lead_days"] < 1) | (w_ok["lead_days"] > w_ok["window"])]
    onset_only = event_table[event_table["first_alert_on_onset_day"] & (event_table["status"] == "warned")]
    if len(onset_only):
        # warned with an alert starting on onset day is legitimate only if an earlier alert exists in the window
        onset_only = onset_only[onset_only["lead_days"].isna() | (onset_only["lead_days"] < 1)]
    checks.append(Check("No alert first triggered on the onset date is counted as an early warning",
                        PASS if bad_lead.empty and onset_only.empty else FAIL,
                        f"{int(event_table['first_alert_on_onset_day'].sum())} event-window rows with an alert first "
                        f"triggered on onset day; all classified late/missed unless an earlier in-window alert exists"))
    st = event_table[(event_table["convention"] == "strict") & event_table["matched_episode"].notna()]
    dup = st.groupby(["spec", "split", "window"])["matched_episode"].apply(lambda s: s.duplicated().any()).sum()
    bal = []
    for (sp, spl, w), g in operational[operational["convention"] == "strict"].groupby(["spec", "split", "window"]):
        r = g.iloc[0]
        e = episode_table[(episode_table["spec"] == sp) & (episode_table["split"] == spl) & (episode_table["window"] == w)]
        if len(e) != r["alert_episodes"] or (r["warned"] + r["detected_late"]
                                             != e["class"].str.startswith("matched").sum()):
            bal.append(f"{sp}/{spl}/{w}")
    checks.append(Check("Strict matching is one-to-one and episode accounting balances",
                        PASS if dup == 0 and not bal else FAIL,
                        f"duplicate matches {int(dup)}; unbalanced {', '.join(bal) or 'none'}"))
    mono = []
    for (sp, spl, conv), g in operational.groupby(["spec", "split", "convention"]):
        g = g.sort_values("window")
        if (np.diff(g["warned"].to_numpy()) < 0).any():
            mono.append(f"{sp}/{spl}/{conv}")
    checks.append(Check("Warned counts are non-decreasing in the window length", PASS if not mono else WARN,
                        ", ".join(mono) or "all specifications"))
    strict_le = operational.pivot_table(index=["spec", "split", "window"], columns="convention", values="warned")
    checks.append(Check("Strict convention never credits more early warnings than the permissive one",
                        PASS if (strict_le["strict"] <= strict_le["permissive"]).all() else FAIL,
                        f"{int((strict_le['strict'] < strict_le['permissive']).sum())} of {len(strict_le)} "
                        "spec/split/window cells are stricter"))
    budget_rows = spec_table[spec_table["family"].isin(["A", "B"]) & (spec_table["role"] == "evaluation")]
    fam_score = {"A": p, "B": score}

    def recomputed_share(r) -> float:      # same basis as exposure_audit: Train days with a finite score
        sc_ = fam_score[r.family][train_mask]
        sc_ = sc_[np.isfinite(sc_)]
        return float((sc_ >= r.threshold).mean())

    consistent = all(abs(r.achieved_train_share - recomputed_share(r)) < 1e-12 and
                     abs(float(specs[r.spec][train_mask][np.isfinite(fam_score[r.family][train_mask])].mean())
                         - r.achieved_train_share) < 1e-12 and
                     bool(r.exposure_matched) == (abs(r.achieved_train_share - r.target_train_share)
                                                  <= au.EXPOSURE_TOLERANCE) for r in budget_rows.itertuples())
    checks.append(Check("Exposure audit is internally consistent (achieved shares recomputed, no tie-breaking)",
                        PASS if consistent else FAIL,
                        f"{len(budget_rows)} budget rules; tolerance {au.EXPOSURE_TOLERANCE:.1%} of Train sessions"))
    unmatched = budget_rows[~budget_rows["exposure_matched"].astype(bool)]
    checks.append(Check("Every requested exposure budget is attained (else flagged NOT exposure-matched)",
                        PASS if unmatched.empty else WARN,
                        "all budgets attained" if unmatched.empty else
                        "; ".join(f"{r.spec}: target {r.target_train_share:.2%}, achieved {r.achieved_train_share:.2%}, "
                                  f"{int(r.n_train_tied_at_threshold)} Train days tied at {r.threshold_repr}"
                                  for r in unmatched.itertuples()) + " - comparisons at these budgets are reported as "
                        "NOT exposure-matched"))
    checks.append(Check("Strict one-to-one matching verified independently from the raw alert series",
                        PASS if not strict_violations else FAIL,
                        "; ".join(strict_violations[:5]) or
                        f"{len(specs)} specifications x 2 splits x {len(au.WINDOWS)} windows, 0 violations"))
    ex_rows = event_table[(event_table["status"] == "warned")]
    ex_ok = ((ex_rows["executable_lead_sessions"] == ex_rows["signal_lead_sessions"] - 1).all() and
             (ex_rows["executable_warning"] == (ex_rows["signal_lead_sessions"] >= 2)).all())
    checks.append(Check("Executable protection lead = signal lead - 1 (execution at the close of t+1)",
                        PASS if ex_ok else FAIL,
                        f"{int((ex_rows['signal_lead_sessions'] == 1).sum())} warned event rows have signal lead 1 and "
                        "therefore no executable protection before onset"))
    checks.append(Check("No test file read", FAIL if any(f.startswith("test_") for f in data.files_read) else PASS,
                        ", ".join(data.files_read)))

    # ------------------------------------------------------------------ outputs
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    tables = {"audit_specifications.csv": spec_table, "audit_operational_metrics.csv": operational,
              "audit_event_matching.csv": event_table, "audit_alert_episodes.csv": episode_table,
              "audit_ranking_metrics.csv": ranking, "audit_ranking_bootstrap.csv": boot}
    for name, f in tables.items():
        f.to_csv(AUDIT_DIR / name, index=False, float_format="%.17g" if name == "audit_specifications.csv" else "%.6g")

    # Stage 7 handoff
    st_ix = spec_table.set_index("spec")
    base_thr = float(st_ix.at["A_p_non_normal@budget20", "threshold"])
    frames = h7.build_role_frames(daily, score, levels, base_thr)
    hdir = EW_DIR / h7.HANDOFF_DIRNAME
    file_hashes = h7.write_handoff(frames, hdir)
    split_manifest = data.manifest
    code_files = ["bocpd.py", "innovation_signals.py", "early_warning.py", "alert_state_machine.py",
                  "early_warning_validation.py", "early_warning_audit.py", "stage7_handoff.py"]
    code_hashes = {f: file_sha256(ROOT / "src" / "regime_radar" / f) for f in code_files}
    split_rows = {s_["name"]: s_ for s_ in split_manifest.get("splits", []) if s_.get("name") in ("train", "validation")}

    def role_files(role: str) -> dict:
        out = {}
        for split in ("train", "validation"):
            name = f"stage7_{role}_{split}.csv"
            f = frames[role][frames[role]["split"] == split]
            out[name] = {"sha256": file_hashes[name], "split": split, "rows": int(len(f)),
                         "first_signal_date": str(f.index.min().date()), "last_signal_date": str(f.index.max().date()),
                         "first_signal_available_after": f["signal_available_after"].iloc[0],
                         "last_signal_available_after": f["signal_available_after"].iloc[-1],
                         "first_execution_date": f["execution_date"].iloc[0],
                         "last_execution_date": f["execution_date"].dropna().iloc[-1] if f["execution_date"].notna().any()
                         else None,
                         "rows_without_execution_date": int(f["execution_date"].isna().sum())}
        return out

    bx = st_ix.loc["A_p_non_normal@budget20"]
    handoff = {
        "stage": "6.1 - Stage 7 handoff (frozen early-warning inputs)", "code_version": STAGE61_CODE_VERSION,
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "handoff_status": "PENDING", "handoff_status_reason": "audit checks not yet complete",
        "usage": "Stage 7 must read signals only through regime_radar.stage7_handoff.load_stage7_inputs(role, split); "
                 "the loader refuses a handoff that is not READY.",
        "execution_convention": h7.EXECUTION_CONVENTION,
        "roles": {
            "baseline_regime": {
                "name": "Simple regime baseline", "definition": "P(non-normal) = P(S2) + P(S3), causal forward filter",
                "status": "Stage 6 retained specification (pre-specified rule: least complex score not significantly "
                          "worse than the best Validation PR-AUC)",
                "columns": list(h7.ROLE_COLUMNS["baseline_regime"]), "timing_columns": list(h7.TIMING_COLUMNS),
                "files": role_files("baseline_regime"),
                "frozen_parameters": {
                    "hmm_model": {"id": FROZEN_HMM, **frozen["hmm"]},
                    "baseline_alert_ge_q80": {"rule": "baseline_p_non_normal >= threshold",
                                              "threshold": base_thr, "threshold_repr": repr(base_thr),
                                              "threshold_origin": "numpy.nanquantile(Train P(non-normal), 0.8), "
                                                                  "linear interpolation (Stage 6 equal 20% budget)",
                                              "train_days_tied_at_threshold": int(bx["n_train_tied_at_threshold"]),
                                              "train_days_exactly_one": int(bx["n_train_exactly_one"]),
                                              "achieved_train_share": float(bx["achieved_train_share"]),
                                              "validation_share": float(bx["validation_share"]),
                                              "saturation_note": "when the threshold equals 1.0 the rule fires only on "
                                                                 "days where P(S1) underflows below float64 resolution "
                                                                 "(P(non-normal) == 1.0 exactly)"}},
            },
            "challenger_ews": {
                "name": "Full EWS challenger", "status": "challenger only; NOT adopted by the Stage 6 selection rule",
                "columns": list(h7.ROLE_COLUMNS["challenger_ews"]), "timing_columns": list(h7.TIMING_COLUMNS),
                "files": role_files("challenger_ews"),
                "frozen_parameters": {
                    "ews_weights": weights, "alert_thresholds": th.to_dict(),
                    "alert_thresholds_repr": {k: repr(v) for k, v in th.to_dict().items()},
                    "threshold_quantiles": asm.QUANTILES,
                    "alert_rules": {"confirmation": f"{asm.CONFIRM_MIN} of {asm.CONFIRM_WINDOW} sessions above entry",
                                    "emergency_red": asm.EMERGENCY, "recovery_clear_days": asm.RECOVERY_CLEAR_DAYS,
                                    "alert_flag": "RECOVERY counts as no alert"},
                    "frozen_models": {"hmm": {"id": FROZEN_HMM, **frozen["hmm"]}, "ar1": frozen["ar1"],
                                      "ar_hmm_k3": frozen["ar_hmm_k3"]},
                    "bocpd": man6.get("bocpd"), "bocpd_priors": man6.get("bocpd_priors"),
                    "train_normalization": man6.get("train_normalization"),
                    "component_scales": man6.get("component_scales")},
            },
        },
        "files": {name: {"sha256": digest, "role": "baseline_regime" if "baseline" in name else "challenger_ews"}
                  for name, digest in file_hashes.items()},
        "column_dictionary": h7.describe_columns(base_thr, th.to_dict()),
        "permitted_portfolio_inputs": {"baseline_regime": list(h7.ROLE_COLUMNS["baseline_regime"]),
                                       "challenger_ews": list(h7.ROLE_COLUMNS["challenger_ews"]),
                                       "timing (both roles)": list(h7.TIMING_COLUMNS)},
        "descriptive_only": ["alert_reason", "dominant_risk_drivers", "contribution_*", "component_*", "p_s1", "p_s2",
                             "p_s3", "normalized_entropy", "*_velocity", "hmm_state (argmax)", "bocpd_*",
                             "innovation diagnostics", "all Stage 6 / 6.1 tables and figures"],
        "forbidden_inputs": ["forward_outcomes.csv (fwd_ret_*, fwd_mdd_*)", "daily loss labels", "risk_events.csv",
                             "any smoothed or Viterbi HMM output", "any test-period file",
                             "thresholds or weights re-estimated on Validation", "return_t or return_(t+1) for a "
                             "date-t signal"],
        "anti_confusion_controls": [
            "baseline and challenger live in different files with role prefixes 'baseline_' / 'challenger_'",
            "load_stage7_inputs(role, split) verifies READY status, the file hash and the exact column set",
            "requesting a column of the other role raises HandoffError; there is no combined role",
            "assert_single_role(frame) rejects frames that mix both prefixes",
            "results must be labelled with the role name; P(non-normal) is never called an EWS alert"],
        "input_hashes": {"split_files": {n: file_sha256(SPLITS_DIR / n) for n in data.files_read if n.endswith(".csv")},
                         "split_rows": {k: {"rows": v.get("rows"), "first": v.get("actual_first_date"),
                                            "last": v.get("actual_last_date")} for k, v in split_rows.items()},
                         "split_manifest_protocol": split_manifest.get("protocol_version"),
                         "stage6_outputs": {k: v for k, v in hashes_before.items() if k.startswith("early_warning/")},
                         "frozen_models": frozen, "code": code_hashes},
        "remaining_limitations": {
            "point_in_time": [
                "Execution convention frozen as conservative (close t+1, first holding return t+2). The Stage 3 "
                "protocol text credits return_(t+1) and is superseded for Stage 7; Stage 3 files were not modified.",
                "Train-period HMM, AR(1), AR-HMM and EWS values use parameters estimated on all of Train "
                "(in-sample); only Validation values are out-of-sample with respect to parameter estimation.",
                "Train-quantile thresholds and ECDF scales use the full Train period; a live system in, e.g., 2010 "
                "could not have known them.",
                "Macro factor point-in-time handling is inherited from Stage 2 (ALFRED first releases where "
                "available); any revision-sensitive series not covered there remains a residual risk.",
                "Signals are computed from end-of-day data that become final only after the close; intraday "
                "execution on the signal day is not supported.",
                "The last Validation rows have no execution / holding date because the Test calendar is not read."],
            "statistical": [
                "Validation (2016-2019) was used in Stages 5, 5.5 and 6: all Validation results are development-"
                "sample evidence, not independent out-of-sample tests.",
                f"Only {len(events['validation'])} Validation and {len(events['train'])} Train SPY -5% drawdown events; "
                "event-level differences of one or two events are within noise.",
                "Block-bootstrap PR-AUC intervals containing zero do not establish equivalence.",
                "P(non-normal) is saturated at 1.0 on a large share of days; threshold rules on it are step functions "
                "and some exposure budgets are unattainable (see the exposure audit).",
                "The equity factor contains SPY momentum, so signals and SPY outcomes are not fully independent.",
                "Warning windows, event threshold (-5%) and late window (21) are conventions."]},
    }
    h7_manifest_path = hdir / h7.MANIFEST_NAME
    save_json(h7.jsonable(handoff), h7_manifest_path)
    # round-trip through the guarded loader (status still PENDING, so require_ready=False here only)
    rt = []
    for role in h7.ROLES:
        for split in ("train", "validation"):
            f = h7.load_stage7_inputs(role, split, EW_DIR, require_ready=False)
            rt.append(h7.assert_single_role(f) == role)
    try:
        h7.load_stage7_inputs("baseline_regime", "validation", EW_DIR, columns=["challenger_alert_level"],
                              require_ready=False)
        refused = False
    except h7.HandoffError:
        refused = True
    try:
        h7.load_stage7_inputs("baseline_regime", "validation", EW_DIR)
        pending_refused = False
    except h7.HandoffError:
        pending_refused = True
    checks.append(Check("Stage 7 handoff files load through the guarded loader; cross-role and non-READY loads refused",
                        PASS if all(rt) and refused and pending_refused else FAIL,
                        f"{len(rt)} files verified; cross-role refused {refused}; PENDING refused {pending_refused}"))
    exact = True
    for split, m in masks.items():
        rb_ = h7.load_stage7_inputs("baseline_regime", split, EW_DIR, require_ready=False)
        rc_ = h7.load_stage7_inputs("challenger_ews", split, EW_DIR, require_ready=False)
        exact &= bool((rb_["baseline_p_non_normal"].to_numpy() == p[m].to_numpy()).all())
        exact &= bool((rb_["baseline_alert_ge_q80"].to_numpy() == (p[m] >= base_thr).astype(int).to_numpy()).all())
        sc_ = score[m].to_numpy()
        exact &= bool(np.array_equal(rc_["challenger_ews_score"].to_numpy(), sc_, equal_nan=True))
        exact &= bool((rc_["challenger_alert_level"].to_numpy() == levels[m].to_numpy()).all())
    checks.append(Check("Handoff files reproduce the frozen signals and rules bit-for-bit (float64 round trip)",
                        PASS if exact else FAIL, "baseline_p_non_normal, baseline_alert_ge_q80, challenger_ews_score, "
                        "challenger_alert_level; Train and Validation"))
    allh = pd.concat([h7.load_stage7_inputs("baseline_regime", s_, EW_DIR, require_ready=False)
                      for s_ in ("train", "validation")])
    nx = pd.to_datetime(allh["execution_date"])
    nx2 = pd.to_datetime(allh["first_holding_return_date"])
    idx = allh.index
    timing_ok = ((nx.iloc[:-1].to_numpy() == idx[1:].to_numpy()).all() and
                 (nx2.iloc[:-2].to_numpy() == idx[2:].to_numpy()).all() and nx.iloc[-1:].isna().all() and
                 nx2.iloc[-2:].isna().all() and
                 all(str(a_).startswith(str(d_.date()) + "T16:00:00") for a_, d_ in zip(allh["signal_available_after"], idx)))
    checks.append(Check("Execution convention in the files: signal after close t, execute at close t+1, first holding "
                        "return t+2", PASS if timing_ok else FAIL,
                        "execution_date = next session, first_holding_return_date = session after that, "
                        "signal_available_after = t 16:00 America/New_York; last rows blank (Test calendar not read)"))
    test_start = None
    for s_ in split_manifest.get("splits", []):
        if s_.get("name") == "test":
            test_start = s_.get("start")
    max_date = idx.max()
    checks.append(Check("No handoff row on or after the test start", PASS if test_start is None or
                        max_date < pd.Timestamp(test_start) else FAIL, f"last signal {max_date.date()}, test start "
                        f"{test_start}"))

    # ------------------------------------------------------------------ automated real-data tests
    test_summary = {"ran": False}
    if not args.no_tests:
        print("Running automated tests on the real Train/Validation outputs ...")
        import io

        class Tee(io.StringIO):
            def write(self, x):
                sys.stdout.write(x)
                return super().write(x)

        stream = Tee()
        suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_stage61_real_data.py",
                                                    top_level_dir=str(ROOT / "tests"))
        res = unittest.TextTestRunner(verbosity=2, stream=stream).run(suite)
        failed = [{"test": str(t), "kind": kind, "traceback": tb} for kind, items in
                  (("failure", res.failures), ("error", res.errors)) for t, tb in items]
        test_summary = {"ran": True, "tests": res.testsRun, "failures": len(res.failures), "errors": len(res.errors),
                        "skipped": len(res.skipped), "failed_tests": failed}
        (AUDIT_DIR / "real_data_test_results.txt").write_text(stream.getvalue(), encoding="utf-8")
        checks.append(Check("Automated tests on the real Train/Validation outputs",
                            PASS if res.wasSuccessful() and res.testsRun and not res.skipped else FAIL,
                            f"{res.testsRun} tests, {len(res.failures)} failures, {len(res.errors)} errors, "
                            f"{len(res.skipped)} skipped" + (": " + ", ".join(f["test"] for f in failed) if failed
                                                              else "")))
    else:
        checks.append(Check("Automated tests on the real Train/Validation outputs", FAIL,
                            "not run (--no-tests); the handoff cannot be READY without them"))

    hashes_after = stage6_hashes()
    changed = sorted(k for k in set(hashes_before) | set(hashes_after) if hashes_before.get(k) != hashes_after.get(k))
    checks.append(Check("Stage 5, 5.5 and 6 artifacts unchanged (hashes before == after)",
                        PASS if not changed else FAIL, f"{len(hashes_before)} files; changed: {', '.join(changed) or 'none'}"))
    figures = ar.save_figures(FIG_DIR, operational, spec_table, event_table)
    recorder.__exit__(None, None, None)
    warn_list = sorted({f"{w.category.__name__}: {str(w.message)[:150]}" for w in caught})
    checks.append(Check("No library warnings", PASS if not warn_list else WARN, "; ".join(warn_list) or "none"))

    # ------------------------------------------------------------------ final status
    counts = pd.Series([c.status for c in checks]).value_counts()
    fails = [c.name for c in checks if c.status == FAIL]
    warns = [c for c in checks if c.status == WARN]
    if fails:
        status, reason = "BLOCKED", "FAILED checks: " + "; ".join(fails)
    else:
        status = "READY"
        reason = ("all blocking checks PASS" + (". Documented limitations (WARN, not blocking): " +
                                                "; ".join(f"{c.name}: {c.detail}" for c in warns) if warns else ""))
    handoff["handoff_status"], handoff["handoff_status_reason"] = status, reason
    handoff["audit_summary"] = {k: int(counts.get(k, 0)) for k in (PASS, WARN, FAIL)}
    save_json(h7.jsonable(handoff), h7_manifest_path)
    pd.DataFrame([vars(c) for c in checks]).to_csv(AUDIT_DIR / "audit_checks.csv", index=False)
    audit_manifest = {"stage": "6.1 - early-warning evaluation audit (final release audit)",
                      "code_version": STAGE61_CODE_VERSION, "created_utc": handoff["created_utc"],
                      "handoff_status": status, "handoff_status_reason": reason,
                      "summary": handoff["audit_summary"], "no_tuning": True, "no_model_selection": True,
                      "no_tie_breaking": True, "test_data_loaded": False, "windows": list(au.WINDOWS),
                      "late_window": au.LATE_WINDOW, "conventions": list(au.CONVENTIONS), "budgets": budgets,
                      "exposure_tolerance": au.EXPOSURE_TOLERANCE, "bootstrap_reps": reps,
                      "execution_convention": h7.EXECUTION_CONVENTION, "stage6_hashes_verified": not changed,
                      "automated_tests": test_summary, "checks": [vars(c) for c in checks],
                      "outputs": sorted(str(q.relative_to(ROOT)) for q in AUDIT_DIR.rglob("*") if q.is_file())}
    save_json(h7.jsonable(audit_manifest), AUDIT_DIR / "audit_manifest.json")
    ctx = {"data": data, "events": events, "spec_table": spec_table, "operational": operational,
           "event_table": event_table, "episode_table": episode_table, "ranking": ranking, "boot": boot,
           "checks": checks, "figures": figures, "handoff": handoff, "budgets": budgets, "man6": man6,
           "test_summary": test_summary, "thresholds": th, "status": status, "status_reason": reason,
           "saturation": {"n_train": int(train_mask.sum()), "train_exactly_one": int((p[train_mask] == 1.0).sum()),
                          "train_within_1e10_not_one": int(((p[train_mask] > 1 - 1e-10) & (p[train_mask] < 1)).sum()),
                          "baseline_threshold_repr": repr(base_thr)}}
    (REPORTS / "early_warning_audit.md").write_text(ar.render_report(ctx), encoding="utf-8")

    # ------------------------------------------------------------------ console
    print("\nExposure audit (Train thresholds at full precision; no tie-breaking):")
    show = spec_table[spec_table["family"].isin(["A", "B"]) & (spec_table["role"] == "evaluation")]
    for r in show.itertuples():
        print(f"  {r.spec:32s} thr={r.threshold_repr:22s} ties={int(r.n_train_tied_at_threshold):4d} "
              f"target={r.target_train_share:.4f} achieved={r.achieved_train_share:.4f} "
              f"validation={r.validation_share:.4f}  {'matched' if r.exposure_matched else 'NOT EXPOSURE-MATCHED'}")
    v = operational[(operational["split"] == "validation") & operational["spec"].isin(ar.PRIMARY_SPECS) &
                    (operational["convention"] == "strict")]
    print("\nValidation, strict matching: signal-warned / executable-warned events by window:")
    print(v.assign(cell=v["warned"].astype(int).astype(str) + "/" + v["executable_warned"].astype(int).astype(str))
          .pivot(index="spec", columns="window", values="cell").reindex(ar.PRIMARY_SPECS).to_string())
    b = boot[boot["label"] == "SPY 21d loss >= 5%"].iloc[0]
    print("\n" + au.interval_statement(b["ci95_low"], b["ci95_high"], "Validation PR-AUC, full EWS minus P(non-normal)"))
    print("Checks:")
    for c in checks:
        print(f"  [{c.status}] {c.name}: {c.detail}")
    print("Summary: " + ", ".join(f"{counts.get(s_, 0)} {s_}" for s_ in (PASS, WARN, FAIL)))
    print(f"HANDOFF STATUS: {status} - {reason}")
    print(f"Report: {REPORTS / 'early_warning_audit.md'}; handoff: {h7_manifest_path}. Elapsed "
          f"{time.perf_counter() - started:.0f}s. Final audit complete; Stage 7 not started.")
    return 1 if counts.get(FAIL, 0) else 0


if __name__ == "__main__":
    sys.exit(main())
