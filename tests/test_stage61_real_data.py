"""Stage 6.1 automated tests on the REAL Train/Validation outputs.

They run against data/splits (train/validation only), the saved Stage 6 outputs and the Stage 6.1 audit and handoff
files. They are skipped when those files do not exist (e.g. on a fresh clone); scripts/08b_audit_early_warning.py
runs them after writing the audit (while the handoff is still PENDING) and treats any skip as a failure.
All comparisons that involve the frozen handoff are exact (bit-for-bit float64), never tolerance-based.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regime_radar import alert_state_machine as asm  # noqa: E402
from regime_radar import early_warning_audit as au  # noqa: E402
from regime_radar import early_warning_validation as ev  # noqa: E402
from regime_radar import stage7_handoff as h7  # noqa: E402
from regime_radar.ar_hmm_diagnostics import file_sha256  # noqa: E402

SPLITS = ROOT / "data" / "splits"
EW = ROOT / "data" / "processed" / "early_warning"
AUDIT = EW / "audit"
HANDOFF = EW / h7.HANDOFF_DIRNAME
REQUIRED = [SPLITS / "split_manifest.json", EW / "early_warning_train.csv", EW / "early_warning_validation.csv",
            AUDIT / "audit_event_matching.csv", AUDIT / "audit_specifications.csv", HANDOFF / h7.MANIFEST_NAME]


@unittest.skipUnless(all(p.exists() for p in REQUIRED), "real Stage 6 / 6.1 outputs not present")
class TestRealData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from regime_radar import regime_exploration as rx
        cls.data = rx.load_stage4_data(SPLITS)
        cls.saved = {s: pd.read_csv(EW / f"early_warning_{s}.csv", index_col="date", parse_dates=["date"])
                     for s in ("train", "validation")}
        cls.manifest = json.loads((HANDOFF / h7.MANIFEST_NAME).read_text(encoding="utf-8"))
        cls.h = {(r, s): h7.load_stage7_inputs(r, s, EW, require_ready=False)
                 for r in h7.ROLES for s in ("train", "validation")}
        cls.events = {s: ev.drawdown_events(getattr(cls.data, f"{s}_returns")["SPY"]) for s in ("train", "validation")}
        cls.et = pd.read_csv(AUDIT / "audit_event_matching.csv")
        cls.spec = pd.read_csv(AUDIT / "audit_specifications.csv", float_precision="round_trip").set_index("spec")

    def test_only_train_and_validation_read(self):
        self.assertFalse(any(f.startswith("test_") for f in self.data.files_read))
        test = [s for s in self.data.manifest["splits"] if s["name"] == "test"]
        if test:
            last = max(f.index.max() for f in self.h.values())
            self.assertLess(last, pd.Timestamp(test[0]["start"]))

    def test_handoff_dates_match_split_calendar(self):
        for s in ("train", "validation"):
            feats = getattr(self.data, f"{s}_features")
            for r in h7.ROLES:
                self.assertTrue(self.h[(r, s)].index.equals(feats.index), (r, s))

    def test_handoff_hashes_and_columns(self):
        for (r, s), f in self.h.items():
            self.assertEqual(list(f.columns), h7.expected_columns(r))
            self.assertEqual(h7.assert_single_role(f), r)
            name = f"stage7_{r}_{s}.csv"
            self.assertEqual(self.manifest["files"][name]["sha256"], file_sha256(HANDOFF / name))
            self.assertEqual(self.manifest["roles"][r]["files"][name]["sha256"], file_sha256(HANDOFF / name))

    def test_manifest_exposes_roles_separately(self):
        roles = self.manifest["roles"]
        self.assertEqual(set(roles), set(h7.ROLES))
        for r in h7.ROLES:
            self.assertEqual(roles[r]["columns"], list(h7.ROLE_COLUMNS[r]))
            self.assertEqual(roles[r]["timing_columns"], list(h7.TIMING_COLUMNS))
            self.assertTrue(all(c.startswith(h7.ROLE_PREFIX[r]) for c in roles[r]["columns"]))
            self.assertIn("frozen_parameters", roles[r])
        self.assertFalse(set(roles["baseline_regime"]["columns"]) & set(roles["challenger_ews"]["columns"]))
        self.assertEqual(self.manifest["execution_convention"]["name"], "conservative_close_t_plus_1")
        for k in ("split_files", "stage6_outputs", "frozen_models", "code"):
            self.assertTrue(self.manifest["input_hashes"][k], k)

    def test_baseline_is_p_non_normal_not_ews(self):
        for s in ("train", "validation"):
            b, sv = self.h[("baseline_regime", s)], self.saved[s]
            np.testing.assert_allclose(b["baseline_p_non_normal"], sv["p_s2"] + sv["p_s3"], atol=2e-6)
            np.testing.assert_allclose(b["baseline_p_non_normal"], sv["p_non_normal"], rtol=2e-5, atol=1e-9)
            self.assertNotIn("challenger_ews_score", b.columns)

    def test_baseline_rule_threshold_is_train_q80(self):
        """Exact: the frozen rule must be reproducible bit-for-bit from the handoff file (no tolerance)."""
        tr = self.h[("baseline_regime", "train")]["baseline_p_non_normal"]
        thr = float(np.nanquantile(tr.to_numpy(), 1 - ev.TRAIN_ALERT_BUDGET))
        frozen = self.manifest["roles"]["baseline_regime"]["frozen_parameters"]["baseline_alert_ge_q80"]["threshold"]
        self.assertEqual(thr, frozen)
        for s in ("train", "validation"):
            b = self.h[("baseline_regime", s)]
            np.testing.assert_array_equal(b["baseline_alert_ge_q80"].to_numpy(),
                                          (b["baseline_p_non_normal"] >= thr).astype(int).to_numpy())

    def test_exposure_audit_matches_handoff_and_flags_unattainable_budgets(self):
        tr = self.h[("baseline_regime", "train")]["baseline_p_non_normal"]
        va = self.h[("baseline_regime", "validation")]["baseline_p_non_normal"]
        allp = pd.concat([tr, va])
        mtr = np.r_[np.ones(len(tr), bool), np.zeros(len(va), bool)]
        for spec, r in self.spec[self.spec["family"] == "A"].iterrows():
            if r["role"] != "evaluation":
                continue
            ex = au.exposure_audit(allp, mtr, ~mtr, r["target_train_share"])
            self.assertEqual(ex["threshold"], r["threshold"], spec)
            self.assertEqual(ex["n_train_tied_at_threshold"], r["n_train_tied_at_threshold"], spec)
            self.assertAlmostEqual(ex["achieved_train_share"], r["achieved_train_share"], places=12)
            matched = abs(ex["achieved_train_share"] - r["target_train_share"]) <= au.EXPOSURE_TOLERANCE
            self.assertEqual(bool(r["exposure_matched"]), matched, spec)
            if not matched:
                self.assertTrue(str(r["exposure_status"]).startswith("NOT EXPOSURE-MATCHED"), spec)

    def test_challenger_equals_frozen_stage6_alerts(self):
        for s in ("train", "validation"):
            c, sv = self.h[("challenger_ews", s)], self.saved[s]
            np.testing.assert_array_equal(c["challenger_alert_level"].to_numpy(), sv["alert_level"].to_numpy())
            np.testing.assert_allclose(c["challenger_ews_score"], sv["early_warning_score"], rtol=2e-5, atol=1e-9)
            for lvl in ("AMBER", "ORANGE"):
                np.testing.assert_array_equal(c[f"challenger_alert_ge_{lvl.lower()}"].to_numpy(),
                                              asm.alert_flag(c["challenger_alert_level"], lvl).to_numpy())

    def test_challenger_alerts_reproduced_from_handoff_score_and_frozen_thresholds(self):
        sc = pd.concat([self.h[("challenger_ews", s)]["challenger_ews_score"] for s in ("train", "validation")])
        lv = pd.concat([self.h[("challenger_ews", s)]["challenger_alert_level"] for s in ("train", "validation")])
        th = self.manifest["roles"]["challenger_ews"]["frozen_parameters"]["alert_thresholds"]
        # the emergency rule needs inputs that are not in the handoff, so only days without an emergency are checked
        no_emerg = asm.run_state_machine(sc, asm.Thresholds(**th))["alert_level"]
        rules = pd.concat([self.saved[s]["alert_reason"] for s in ("train", "validation")])
        if not rules.str.contains("emergency").any():
            np.testing.assert_array_equal(no_emerg.to_numpy(), lv.to_numpy())
        m6 = json.loads((EW / "early_warning_selection_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(th, m6["alert_thresholds"])

    def test_execution_convention_t_plus_1_close_t_plus_2_return(self):
        for r in h7.ROLES:
            f = pd.concat([self.h[(r, s)] for s in ("train", "validation")])
            idx = f.index
            nxt = pd.to_datetime(f["execution_date"])
            nxt2 = pd.to_datetime(f["first_holding_return_date"])
            self.assertTrue((nxt.iloc[:-1].to_numpy() == idx[1:].to_numpy()).all())
            self.assertTrue((nxt2.iloc[:-2].to_numpy() == idx[2:].to_numpy()).all())
            self.assertTrue(nxt.iloc[-1:].isna().all() and nxt2.iloc[-2:].isna().all())
            avail = pd.to_datetime(f["signal_available_after"], utc=True).dt.tz_convert(h7.MARKET_TZ)
            self.assertTrue((avail.dt.date.to_numpy() == idx.date).all())
            self.assertTrue(((avail.dt.hour == 16) & (avail.dt.minute == 0)).all())
            exe = pd.to_datetime(f["execution_time"].dropna(), utc=True).dt.tz_convert(h7.MARKET_TZ)
            self.assertTrue((exe.dt.hour == 16).all())
            self.assertTrue((exe.dt.date.to_numpy() == nxt.dropna().dt.date.to_numpy()).all())

    def test_stage6_output_hashes_still_match(self):
        for k, v in self.manifest["input_hashes"]["stage6_outputs"].items():
            self.assertEqual(file_sha256(EW / k.split("/", 1)[1]), v, k)

    def test_strict_matching_recomputed_from_handoff_files(self):
        """Independent recomputation of the strict tables for every handoff alert column and window."""
        cols = {("challenger_ews", "challenger_alert_ge_amber"): "C_ews_state>=AMBER",
                ("challenger_ews", "challenger_alert_ge_orange"): "C_ews_state>=ORANGE",
                ("baseline_regime", "baseline_alert_ge_q80"): "A_p_non_normal@budget20"}
        for s in ("train", "validation"):
            for (role, col), spec in cols.items():
                a = self.h[(role, s)][col].to_numpy()
                for w in au.WINDOWS:
                    m, t, e = au.strict_metrics(a, self.events[s], w)
                    self.assertEqual(au.verify_strict(a, self.events[s], w, t, e), [], (s, spec, w))
                    saved = self.et[(self.et["spec"] == spec) & (self.et["split"] == s) &
                                    (self.et["window"] == w) & (self.et["convention"] == "strict")]
                    self.assertEqual(t["status"].tolist(), saved["status"].tolist(), (s, spec, w))
                    np.testing.assert_array_equal(t["lead_days"].to_numpy(float), saved["lead_days"].to_numpy(float))

    def test_executable_lead_is_signal_lead_minus_one(self):
        w = self.et[self.et["status"] == "warned"]
        np.testing.assert_array_equal(w["executable_lead_sessions"].to_numpy(), w["signal_lead_sessions"].to_numpy() - 1)
        self.assertTrue((w["executable_warning"] == (w["signal_lead_sessions"] >= 2)).all())

    def test_onset_day_alerts_are_never_early_warnings(self):
        for s in ("train", "validation"):
            for r, col in (("challenger_ews", "challenger_alert_ge_amber"), ("challenger_ews", "challenger_alert_ge_orange"),
                           ("baseline_regime", "baseline_alert_ge_q80")):
                a = self.h[(r, s)][col].to_numpy()
                for w in au.WINDOWS:
                    for conv in ("permissive", "strict"):
                        t = (au.permissive_metrics(a, self.events[s], w)[1] if conv == "permissive"
                             else au.strict_metrics(a, self.events[s], w)[1])
                        for row in t.itertuples():
                            o = row.onset_pos
                            if not a[max(o - w, 0):o].any():
                                self.assertNotEqual(row.status, "warned", (s, col, w, conv, o))
                            if row.status == "warned":
                                self.assertTrue(1 <= row.lead_days <= w)

    def test_audit_tables_consistent(self):
        op = pd.read_csv(AUDIT / "audit_operational_metrics.csv")
        piv = op.pivot_table(index=["spec", "split", "window"], columns="convention", values="warned")
        self.assertTrue((piv["strict"] <= piv["permissive"]).all())
        self.assertTrue((op["executable_warned"] <= op["warned"]).all())
        n_val = len(self.events["validation"])
        self.assertTrue((op[op["split"] == "validation"]["n_events"] == n_val).all())


if __name__ == "__main__":
    unittest.main()
