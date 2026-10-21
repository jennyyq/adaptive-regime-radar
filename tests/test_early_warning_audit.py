"""Stage 6.1 tests: event matching conventions, onset-day rule and the Stage 7 handoff guard (synthetic data only)."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regime_radar import early_warning_audit as au  # noqa: E402
from regime_radar import early_warning_validation as ev  # noqa: E402
from regime_radar import stage7_handoff as h7  # noqa: E402


def events(onsets, peaks=None, ends=None):
    peaks = peaks or [o - 5 for o in onsets]
    ends = ends or [o + 10 for o in onsets]
    return pd.DataFrame({"onset_pos": onsets, "peak_pos": peaks, "end_pos": ends, "onset_date": onsets,
                         "peak_date": peaks, "trough_loss_pct": [-6.0] * len(onsets)})


def alerts(n, runs):
    a = np.zeros(n, int)
    for s, f in runs:
        a[s:f] = 1
    return a


class TestOnsetDayRule(unittest.TestCase):
    def test_alert_first_on_onset_day_is_late_in_both_conventions(self):
        a = alerts(200, [(100, 105)])
        for w in au.WINDOWS:
            _, tp = au.permissive_metrics(a, events([100]), w)
            _, ts, _ = au.strict_metrics(a, events([100]), w)
            for t in (tp, ts):
                self.assertEqual(t["status"].iloc[0], "detected_late")
                self.assertEqual(t["delay_days"].iloc[0], 0)
                self.assertTrue(t["first_alert_on_onset_day"].iloc[0])

    def test_alert_on_day_before_onset_is_a_warning_with_lead_one(self):
        a = alerts(200, [(99, 105)])
        _, ts, _ = au.strict_metrics(a, events([100]), 5)
        self.assertEqual((ts["status"].iloc[0], ts["lead_days"].iloc[0]), ("warned", 1))


class TestConventions(unittest.TestCase):
    def test_permissive_reproduces_stage6_at_63(self):
        rng = np.random.default_rng(0)
        a = (rng.uniform(size=600) > 0.85).astype(int)
        e = events([100, 250, 400], [90, 240, 380], [130, 300, 420])
        m6, t6 = ev.event_metrics(pd.Series(a), e)
        mp, tp = au.permissive_metrics(a, e, 63)
        for k in ("warned", "detected_late", "missed", "alert_episodes", "false_alert_episodes"):
            self.assertEqual(m6[k], mp[k], k)
        np.testing.assert_array_equal(t6["lead_days"].to_numpy(), tp["lead_days"].to_numpy())

    def test_one_long_episode_warns_once_under_strict(self):
        a = alerts(300, [(80, 160)])              # one episode covering two onsets
        e = events([100, 150], [95, 145], [110, 170])
        mp, _ = au.permissive_metrics(a, e, 63)
        ms, ts, es = au.strict_metrics(a, e, 63)
        self.assertEqual(mp["warned"], 2)
        self.assertEqual(ms["warned"], 1)
        self.assertEqual(ts["status"].tolist(), ["warned", "missed"])
        self.assertTrue(ts["stale_alert_active_at_onset"].iloc[1])
        self.assertEqual(es["matched_event"].dropna().tolist(), [0])

    def test_episode_started_before_window_is_not_a_strict_warning(self):
        a = alerts(300, [(50, 120)])              # starts 50 sessions before onset 100
        ms, ts, _ = au.strict_metrics(a, events([100]), 21)
        mp, tp = au.permissive_metrics(a, events([100]), 21)
        self.assertEqual(ts["status"].iloc[0], "missed")
        self.assertTrue(ts["stale_alert_active_at_onset"].iloc[0])
        self.assertEqual(tp["status"].iloc[0], "warned")
        self.assertEqual(tp["lead_days"].iloc[0], 21)    # clipped to the window

    def test_most_recent_episode_is_matched_and_others_are_redundant(self):
        a = alerts(300, [(60, 62), (90, 95)])
        ms, ts, es = au.strict_metrics(a, events([100]), 63)
        self.assertEqual(ts["lead_days"].iloc[0], 10)
        self.assertEqual(ts["gap_last_alert_to_onset"].iloc[0], 6)
        self.assertEqual(es["class"].tolist(), ["redundant_pre_event", "matched_early"])
        self.assertEqual(ms["false_alert_episodes"], 0)

    def test_episode_classes_and_balance(self):
        a = alerts(400, [(90, 92), (130, 135), (300, 305)])
        e = events([100], [95], [200])
        ms, ts, es = au.strict_metrics(a, e, 21)
        self.assertEqual(es["class"].tolist(), ["matched_early", "in_event_period", "false"])
        # 50 and 4 sessions early (premature for W=3, even after the peak); one starting late in the drawdown
        a2 = alerts(400, [(50, 52), (96, 97), (150, 152)])
        _, _, es2 = au.strict_metrics(a2, events([100], [80], [200]), 3)
        self.assertEqual(es2["class"].tolist(), ["premature", "premature", "in_event_period"])
        self.assertEqual(ms["false_alert_episodes"], 1)
        self.assertEqual(ms["alert_episodes"], len(es))
        self.assertAlmostEqual(ms["episode_precision"], 1 / 3)

    def test_strict_never_exceeds_permissive_and_warned_monotone_in_window(self):
        rng = np.random.default_rng(1)
        for seed in range(20):
            a = (np.random.default_rng(seed).uniform(size=800) > 0.9).astype(int)
            on = sorted(rng.choice(np.arange(70, 780), size=5, replace=False))
            e = events(list(on))
            prev = {"permissive": -1, "strict": -1}
            for w in au.WINDOWS:
                mp, _ = au.permissive_metrics(a, e, w)
                ms, _, _ = au.strict_metrics(a, e, w)
                self.assertLessEqual(ms["warned"], mp["warned"])
                self.assertGreaterEqual(mp["warned"], prev["permissive"])
                prev["permissive"] = mp["warned"]

    def test_exposure_matched_threshold_uses_train_only(self):
        s = pd.Series(np.r_[np.arange(100.0), np.full(50, 1000.0)])
        train = np.r_[np.ones(100, bool), np.zeros(50, bool)]
        thr = au.exposure_matched_threshold(s, train, 0.2)
        self.assertAlmostEqual((s[train] >= thr).mean(), 0.2, delta=0.011)

    def test_interval_statement_never_claims_equivalence(self):
        txt = au.interval_statement(-0.05, 0.03, "diff")
        self.assertIn("NOT evidence that the two scores are equivalent", txt)
        self.assertNotIn("equivalent.", au.interval_statement(0.01, 0.03, "diff"))


class TestHandoffGuard(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        idx = pd.bdate_range("2015-12-28", periods=12)
        daily = pd.DataFrame({"p_s1": 0.5, "p_s2": 0.3, "p_s3": 0.2, "split": ["train"] * 4 + ["validation"] * 8},
                             index=idx)
        daily["p_non_normal"] = daily["p_s2"] + daily["p_s3"]
        score = pd.Series(np.linspace(0, 1, 12), index=idx)
        levels = pd.Series(["GREEN", "AMBER", "ORANGE", "RED", "RECOVERY", "GREEN"] * 2, index=idx)
        frames = h7.build_role_frames(daily, score, levels, 0.4)
        hashes = h7.write_handoff(frames, self.tmp / h7.HANDOFF_DIRNAME)
        (self.tmp / h7.HANDOFF_DIRNAME / h7.MANIFEST_NAME).write_text(
            json.dumps({"handoff_status": "READY", "files": {k: {"sha256": v} for k, v in hashes.items()}}))
        self.levels = levels

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_roles_are_separate_and_prefixed(self):
        b = h7.load_stage7_inputs("baseline_regime", "validation", self.tmp)
        c = h7.load_stage7_inputs("challenger_ews", "validation", self.tmp)
        self.assertFalse(set(b.columns) & set(h7.ROLE_COLUMNS["challenger_ews"]))
        self.assertFalse(set(c.columns) & set(h7.ROLE_COLUMNS["baseline_regime"]))
        self.assertEqual(h7.assert_single_role(b), "baseline_regime")
        self.assertEqual(h7.assert_single_role(c), "challenger_ews")
        self.assertNotIn("p_non_normal", c.columns)

    def test_cross_role_request_and_mixing_refused(self):
        with self.assertRaises(h7.HandoffError):
            h7.load_stage7_inputs("baseline_regime", "train", self.tmp, columns=["challenger_alert_level"])
        with self.assertRaises(h7.HandoffError):
            h7.load_stage7_inputs("challenger_ews", "train", self.tmp, columns=["baseline_p_non_normal"])
        with self.assertRaises(h7.HandoffError):
            h7.load_stage7_inputs("combined", "train", self.tmp)
        with self.assertRaises(h7.HandoffError):
            h7.load_stage7_inputs("baseline_regime", "test", self.tmp)
        b = h7.load_stage7_inputs("baseline_regime", "train", self.tmp)
        c = h7.load_stage7_inputs("challenger_ews", "train", self.tmp)
        with self.assertRaises(h7.HandoffError):
            h7.assert_single_role(pd.concat([b, c], axis=1))

    def test_tampered_file_is_rejected(self):
        path = self.tmp / h7.HANDOFF_DIRNAME / "stage7_challenger_ews_validation.csv"
        path.write_text(path.read_text().replace("AMBER", "GREEN"))
        with self.assertRaises(h7.HandoffError):
            h7.load_stage7_inputs("challenger_ews", "validation", self.tmp)

    def test_alert_flags_and_timing(self):
        c = pd.concat([h7.load_stage7_inputs("challenger_ews", s, self.tmp) for s in ("train", "validation")])
        exp = self.levels.isin(["AMBER", "ORANGE", "RED"]).astype(int).to_numpy()
        np.testing.assert_array_equal(c["challenger_alert_ge_amber"].to_numpy(), exp)
        self.assertEqual(int(c.loc[c["challenger_alert_level"] == "RECOVERY", "challenger_alert_ge_amber"].sum()), 0)
        nxt = pd.to_datetime(c["execution_date"])
        nxt2 = pd.to_datetime(c["first_holding_return_date"])
        self.assertTrue((nxt.iloc[:-1].to_numpy() == c.index[1:].to_numpy()).all())
        self.assertTrue((nxt2.iloc[:-2].to_numpy() == c.index[2:].to_numpy()).all())
        self.assertTrue(pd.isna(nxt.iloc[-1]))           # Test calendar is never read
        self.assertEqual(c["signal_available_after"].iloc[0], "2015-12-28T16:00:00-05:00")
        self.assertEqual(c["execution_time"].iloc[0], "2015-12-29T16:00:00-05:00")

    def test_daylight_saving_offsets(self):
        t = h7.timing_frame(pd.DatetimeIndex(["2016-07-01", "2016-07-05", "2016-07-06"]))
        self.assertEqual(t["signal_available_after"].iloc[0], "2016-07-01T16:00:00-04:00")
        self.assertEqual(t["execution_date"].iloc[0], "2016-07-05")
        self.assertEqual(t["first_holding_return_date"].iloc[0], "2016-07-06")

    def test_non_ready_handoff_is_refused(self):
        mpath = self.tmp / h7.HANDOFF_DIRNAME / h7.MANIFEST_NAME
        m = json.loads(mpath.read_text())
        for status in ("PENDING", "BLOCKED"):
            m["handoff_status"] = status
            mpath.write_text(json.dumps(m))
            with self.assertRaises(h7.HandoffError):
                h7.load_stage7_inputs("baseline_regime", "train", self.tmp)
            h7.load_stage7_inputs("baseline_regime", "train", self.tmp, require_ready=False)


class TestSaturation(unittest.TestCase):
    """Regression test for the Stage 6.1 real-data failure: P(non-normal) saturated at 1.0."""

    def saturated(self, seed=0, n=2058, n_one=419, n_near=60):
        rng = np.random.default_rng(seed)
        p_s1 = np.r_[np.zeros(n_one), 10 ** rng.uniform(-16, -10, n_near), rng.uniform(1e-6, 1, n - n_one - n_near)]
        return pd.Series(1.0 - p_s1, index=pd.bdate_range("2008-01-01", periods=n))

    def test_handoff_round_trip_is_bit_exact_under_saturation(self):
        p = self.saturated()
        thr = float(np.nanquantile(p, 0.8))
        self.assertEqual(thr, 1.0)
        daily = pd.DataFrame({"p_non_normal": p, "split": "train"})
        daily.iloc[-100:, daily.columns.get_loc("split")] = "validation"
        frames = h7.build_role_frames(daily, p * 0.5, pd.Series("GREEN", index=p.index), thr)
        tmp = Path(tempfile.mkdtemp())
        try:
            hashes = h7.write_handoff(frames, tmp / h7.HANDOFF_DIRNAME)
            (tmp / h7.HANDOFF_DIRNAME / h7.MANIFEST_NAME).write_text(
                json.dumps({"handoff_status": "READY", "files": {k: {"sha256": v} for k, v in hashes.items()}}))
            b = pd.concat([h7.load_stage7_inputs("baseline_regime", s, tmp) for s in ("train", "validation")])
            np.testing.assert_array_equal(b["baseline_p_non_normal"].to_numpy(), p.to_numpy())   # bit-exact
            thr_back = float(np.nanquantile(b["baseline_p_non_normal"].iloc[:-100], 0.8))
            self.assertEqual(thr_back, float(np.nanquantile(p.iloc[:-100], 0.8)))
            np.testing.assert_array_equal(b["baseline_alert_ge_q80"].to_numpy(),
                                          (b["baseline_p_non_normal"] >= thr).astype(int).to_numpy())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_ten_digit_serialization_would_break_the_rule(self):
        """Documents the root cause: '%.10g' maps 1 - 1e-11 to '1'."""
        p = self.saturated()
        near = p[(p > 1 - 5e-11) & (p < 1)]          # exactly the values that 10 digits print as "1"
        self.assertTrue(len(near) > 0)
        self.assertTrue(all(float("%.10g" % v) == 1.0 for v in near))
        self.assertTrue(all(float("%.17g" % v) == v for v in near))

    def test_exposure_audit_flags_unattainable_budget_without_tie_breaking(self):
        p = self.saturated()
        train = np.r_[np.ones(len(p) - 100, bool), np.zeros(100, bool)]
        tr = p[train]
        top = (tr == 1.0).mean()
        ex13 = au.exposure_audit(p, train, ~train, 0.13)
        self.assertFalse(ex13["exposure_matched"])
        self.assertTrue(ex13["exposure_status"].startswith("NOT EXPOSURE-MATCHED"))
        self.assertEqual(ex13["threshold"], 1.0)
        self.assertAlmostEqual(ex13["achieved_train_share"], top)
        self.assertEqual(ex13["attainable_share_just_below"], 0.0)
        self.assertEqual(ex13["n_train_tied_at_threshold"], int((tr == 1.0).sum()))
        ex30 = au.exposure_audit(p, train, ~train, 0.30)
        self.assertTrue(ex30["exposure_matched"])
        self.assertEqual(repr(ex30["threshold"]), ex30["threshold_repr"])


class TestExecutableLeadAndVerification(unittest.TestCase):
    def test_executable_lead_is_signal_lead_minus_one(self):
        r = pd.Series(np.r_[np.zeros(95), np.full(10, -0.01), np.zeros(95)])
        a = alerts(200, [(99, 103), ])
        _, t, _ = au.strict_metrics(a, events([100], [95], [150]), 21)
        t = au.protection_columns(t, r)
        self.assertEqual(t["signal_lead_sessions"].iloc[0], 1)
        self.assertEqual(t["executable_lead_sessions"].iloc[0], 0)
        self.assertFalse(t["executable_warning"].iloc[0])
        a = alerts(200, [(97, 103)])
        _, t, _ = au.strict_metrics(a, events([100], [95], [150]), 21)
        t = au.protection_columns(t, r)
        self.assertEqual((t["signal_lead_sessions"].iloc[0], t["executable_lead_sessions"].iloc[0]), (3, 2))
        self.assertTrue(t["executable_warning"].iloc[0])
        # peak = close of 94; execution at the close of 98 after returns 95..98 (4 x -1%); returns 99..100 remain
        self.assertAlmostEqual(t["drawdown_at_signal_pct"].iloc[0], 100 * np.expm1(-0.03))
        self.assertAlmostEqual(t["drawdown_at_execution_pct"].iloc[0], 100 * np.expm1(-0.04))
        self.assertAlmostEqual(t["spy_return_execution_to_onset_pct"].iloc[0], 100 * np.expm1(-0.02))

    def test_verify_strict_accepts_correct_tables_and_rejects_tampering(self):
        for seed in range(30):
            rng = np.random.default_rng(seed)
            a = (rng.uniform(size=600) > 0.92).astype(int)
            on = sorted(rng.choice(np.arange(70, 570), size=4, replace=False))
            e = events(list(on))
            for w in au.WINDOWS:
                _, t, es = au.strict_metrics(a, e, w)
                self.assertEqual(au.verify_strict(a, e, w, t, es), [], (seed, w))
        a = alerts(300, [(60, 62), (90, 95)])
        e = events([100])
        _, t, es = au.strict_metrics(a, e, 63)
        bad = t.copy()
        bad.loc[0, "matched_episode"] = 0           # older episode instead of the most recent start
        bad.loc[0, "lead_days"] = 40
        self.assertTrue(au.verify_strict(a, e, 63, bad, es))


if __name__ == "__main__":
    unittest.main()
