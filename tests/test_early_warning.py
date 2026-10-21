"""Stage 6 tests: causal signals, Train-only scaling, alert state machine, outcomes and event evaluation.

Synthetic data only; these tests are not real-data validation.
"""

from __future__ import annotations

import copy
import inspect
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regime_radar import alert_state_machine as asm  # noqa: E402
from regime_radar import early_warning as ew  # noqa: E402
from regime_radar import early_warning_validation as ev  # noqa: E402
from regime_radar import innovation_signals as ins  # noqa: E402
from regime_radar.ar_hmm import ARParams  # noqa: E402
from regime_radar.dataset import FEATURE_COLUMNS  # noqa: E402
from regime_radar.hmm_models import HMMParams  # noqa: E402

D = len(FEATURE_COLUMNS)


def frozen_models():
    hmm = HMMParams(np.array([0.6, 0.3, 0.1]),
                    np.array([[0.97, 0.025, 0.005], [0.03, 0.94, 0.03], [0.01, 0.04, 0.95]]),
                    np.array([[0.5, 0, 0, -0.5, -0.5], [0, 0, 0, 0, 0], [-1.5, 0.5, 0.5, 1.5, 1.5]]),
                    np.array([[0.3] * D, [0.6] * D, [1.5] * D]), "diag")
    ar1 = ARParams(np.array([1.0]), np.array([[1.0]]), np.zeros((1, D)), np.full((1, D), 0.95),
                   np.full((1, D), 0.05))
    arhmm = ARParams(np.full(3, 1 / 3), np.array([[0.97, 0.02, 0.01], [0.02, 0.96, 0.02], [0.01, 0.03, 0.96]]),
                     np.zeros((3, D)), np.full((3, D), 0.95), np.array([[0.02] * D, [0.05] * D, [0.2] * D]))
    return hmm, ar1, arhmm


def synthetic_factors(n_tr=300, n_va=160, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros((n_tr + n_va, D))
    for t in range(1, len(x)):
        x[t] = 0.95 * x[t - 1] + rng.normal(scale=0.25, size=D)
    x[n_tr + 60:n_tr + 80, 3] += 2.0        # a volatility episode in "Validation"
    idx = pd.bdate_range("2010-01-01", periods=len(x))
    return x[:n_tr], x[n_tr:], idx[:n_tr], idx[n_tr:]


def build(x_tr, x_va, tr_idx, va_idx, models=None):
    hmm, ar1, arhmm = models or frozen_models()
    lv_tr = pd.DataFrame(x_tr, index=tr_idx, columns=FEATURE_COLUMNS)
    lv_va = pd.DataFrame(x_va, index=va_idx, columns=FEATURE_COLUMNS)
    return ew.build_signals(hmm, ar1, arhmm, x_tr, x_va, tr_idx, va_idx, lv_tr, lv_va)


def ews_pipeline(daily):
    train = (daily["split"] == "train").to_numpy()
    scaler = ew.fit_scaler(daily, train)
    comp = ew.components(daily, scaler)
    w, sub = ew.ews_specs()["full"]
    score, contrib = ew.ews_score(comp, w, sub)
    th = asm.Thresholds.from_train(score[train])
    st = asm.run_state_machine(score, th, daily["p_stress"], comp["confirmation"], comp["market_innovation_1d"])
    return comp, score, contrib, th, st


class TestSignals(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x_tr, cls.x_va, cls.tr_idx, cls.va_idx = synthetic_factors()
        cls.daily, cls.meta = build(cls.x_tr, cls.x_va, cls.tr_idx, cls.va_idx)

    def test_entropy_k3(self):
        p = np.array([[1, 0, 0], [1 / 3, 1 / 3, 1 / 3], [0.5, 0.5, 0.0]])
        np.testing.assert_allclose(ins.normalized_entropy(p), [0.0, 1.0, math.log(2) / math.log(3)], atol=1e-12)

    def test_probabilities_are_filtered_and_sum_to_one(self):
        np.testing.assert_allclose(self.daily[["p_s1", "p_s2", "p_s3"]].sum(axis=1), 1.0, atol=1e-12)
        np.testing.assert_allclose(self.daily["p_non_normal"], 1 - self.daily["p_s1"], atol=1e-15)

    def test_velocity_uses_five_session_lag(self):
        p = self.daily["p_non_normal"]
        np.testing.assert_allclose(self.daily["non_normal_probability_velocity"].iloc[5:],
                                   (p - p.shift(5)).iloc[5:], atol=1e-15)
        self.assertTrue(self.daily["non_normal_probability_velocity"].iloc[:5].isna().all())
        self.assertTrue((self.daily["non_normal_probability_velocity_positive"].dropna() >= 0).all())

    def test_innovations_use_previous_day_only(self):
        hmm, ar1, _ = frozen_models()
        z = self.daily[[f"z_{c}" for c in FEATURE_COLUMNS]].to_numpy()
        self.assertTrue(np.isnan(z[0]).all())
        sd = np.sqrt(ar1.var[0])
        first_va = (self.x_va[0] - ar1.intercept[0] - ar1.phi[0] * self.x_tr[-1]) / sd
        np.testing.assert_allclose(z[len(self.x_tr)], first_va, atol=1e-12)

    def test_channel_normalization_is_train_only(self):
        tr = self.daily["split"] == "train"
        self.assertAlmostEqual(float(self.daily.loc[tr, "market_innovation"].std()), 1.0, places=10)
        self.assertAlmostEqual(float(self.daily.loc[tr, "stress_index_level"].mean()), 0.0, places=10)

    def test_returns_are_not_an_input(self):
        params = " ".join(inspect.signature(ew.build_signals).parameters)
        self.assertNotIn("return", params)

    def test_leakage_injection_future_factors_do_not_change_past(self):
        cut = 80
        shocked = self.x_va.copy()
        shocked[cut:] = shocked[cut:] * -4 + 3
        d2, _ = build(self.x_tr, shocked, self.tr_idx, self.va_idx)
        n = len(self.x_tr) + cut
        num = [c for c in self.daily.columns if pd.api.types.is_numeric_dtype(self.daily[c])]
        a, b = self.daily[num].iloc[:n].to_numpy(float), d2[num].iloc[:n].to_numpy(float)
        np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
        np.testing.assert_array_equal(a[~np.isnan(a)], b[~np.isnan(b)])
        _, s1, _, _, st1 = ews_pipeline(self.daily)
        _, s2, _, _, st2 = ews_pipeline(d2)
        np.testing.assert_array_equal(s1.iloc[:n].to_numpy(), s2.iloc[:n].to_numpy())
        self.assertTrue(st1["alert_level"].iloc[:n].equals(st2["alert_level"].iloc[:n]))
        self.assertFalse(np.allclose(np.nan_to_num(s1.iloc[n:].to_numpy()), np.nan_to_num(s2.iloc[n:].to_numpy())))

    def test_validation_values_do_not_move_train_scales_or_thresholds(self):
        shocked = self.x_va * 5
        d2, m2 = build(self.x_tr, shocked, self.tr_idx, self.va_idx)
        self.assertEqual(self.meta["normalization"].to_dict(), m2["normalization"].to_dict())
        self.assertEqual(self.meta["bocpd_priors"], m2["bocpd_priors"])
        _, _, _, th1, _ = ews_pipeline(self.daily)
        _, _, _, th2, _ = ews_pipeline(d2)
        self.assertEqual(th1.to_dict(), th2.to_dict())

    def test_frozen_parameters_unchanged(self):
        models = frozen_models()
        before = copy.deepcopy(models)
        build(self.x_tr, self.x_va, self.tr_idx, self.va_idx, models)
        for a, b in zip(models, before):
            for k, v in vars(a).items():
                np.testing.assert_array_equal(v, vars(b)[k])

    def test_ews_in_unit_interval_and_weights_sum_to_one(self):
        comp, score, contrib, _, _ = ews_pipeline(self.daily)
        s = score.dropna()
        self.assertTrue(((s >= 0) & (s <= 1)).all())
        np.testing.assert_allclose(contrib.loc[s.index].sum(axis=1), s, atol=1e-12)
        ones = comp.copy()
        ones.loc[:, :] = 1.0
        for name, (w, sub) in ew.ews_specs().items():      # effective weights renormalize to one in every variant
            self.assertTrue(np.allclose(ew.ews_score(ones, w, sub)[0], 1.0), name)
        self.assertEqual(ew.ews_specs()["no_bocpd"][0]["bocpd"], 0.0)

    def test_bocpd_component_gated_by_adverse_innovation(self):
        comp, *_ = ews_pipeline(self.daily)
        calm = self.daily["market_innovation_5d"] <= 0
        self.assertTrue((comp.loc[calm, "bocpd"].dropna() == 0).all())

    def test_train_ecdf_strictly_below(self):
        e = ew.TrainECDF([1, 2, 2, 3])
        np.testing.assert_allclose(e([0, 1, 2, 3, 4]), [0, 0, 0.25, 0.75, 1.0])


class TestStateMachine(unittest.TestCase):
    TH = asm.Thresholds(amber_in=0.6, amber_out=0.5, orange_in=0.7, orange_out=0.65, red_in=0.8, red_out=0.75)

    def run_(self, values, **kw):
        return asm.run_state_machine(pd.Series(values, dtype=float), self.TH, **kw)["alert_level"].tolist()

    def test_single_spike_does_not_escalate(self):
        self.assertEqual(self.run_([0.1, 0.1, 0.9, 0.1, 0.1]), ["GREEN"] * 5)

    def test_three_of_five_escalates_and_levels_can_be_skipped(self):
        lv = self.run_([0.1, 0.72, 0.72, 0.72, 0.72])
        self.assertEqual(lv, ["GREEN", "GREEN", "GREEN", "ORANGE", "ORANGE"])

    def test_hysteresis_holds_level_inside_band(self):
        lv = self.run_([0.62] * 3 + [0.55] * 10)
        self.assertTrue(all(v == "AMBER" for v in lv[2:]))

    def test_deescalation_to_recovery_then_green_after_quiet_period(self):
        lv = self.run_([0.72] * 3 + [0.3] * 20)
        self.assertEqual(lv[2], "ORANGE")
        self.assertIn("RECOVERY", lv)
        first_rec = lv.index("RECOVERY")
        self.assertEqual(lv[first_rec + 1], "RECOVERY")
        self.assertEqual(lv[-1], "GREEN")
        first_green = len(lv) - lv[::-1].index("RECOVERY")      # first session after the last RECOVERY
        self.assertEqual(lv[first_green], "GREEN")
        self.assertEqual(first_green - 3 + 1, asm.RECOVERY_CLEAR_DAYS)  # quiet sessions 3..first_green inclusive

    def test_emergency_rule_is_immediate(self):
        n = 4
        lv = asm.run_state_machine(pd.Series([0.1] * n), self.TH, p_stress=pd.Series([0, 0, 0.95, 0.95]),
                                   confirmation=pd.Series([0, 0, 0.97, 0.97]),
                                   innovation_pct=pd.Series([0.5] * n))["alert_level"].tolist()
        self.assertEqual(lv[2], "RED")

    def test_no_hysteresis_variant_is_band_only(self):
        lv = self.run_([0.1, 0.9, 0.1], hysteresis=False)
        self.assertEqual(lv, ["GREEN", "RED", "GREEN"])

    def test_alert_flag_excludes_recovery(self):
        f = asm.alert_flag(pd.Series(["GREEN", "AMBER", "RECOVERY", "ORANGE", "RED"]), "AMBER")
        self.assertEqual(f.tolist(), [0, 1, 0, 1, 1])

    def test_state_machine_is_causal(self):
        rng = np.random.default_rng(3)
        s = pd.Series(rng.uniform(size=300))
        s2 = s.copy()
        s2.iloc[200:] = 0.99
        a = asm.run_state_machine(s, self.TH)["alert_level"]
        b = asm.run_state_machine(s2, self.TH)["alert_level"]
        self.assertTrue(a.iloc[:200].equals(b.iloc[:200]))


class TestOutcomesAndEvaluation(unittest.TestCase):
    def test_forward_outcomes_and_purged_tail(self):
        r = pd.Series([0.01, -0.02, -0.03, 0.04, 0.0, -0.01], index=pd.bdate_range("2020-01-01", periods=6))
        o = ev.forward_outcomes(r, horizons=(2,))
        self.assertAlmostEqual(o["fwd_ret_2"].iloc[0], -0.05)
        self.assertAlmostEqual(o["fwd_mdd_2"].iloc[0], -0.05)
        self.assertAlmostEqual(o["fwd_mdd_2"].iloc[2], 0.0)          # +0.04 then +0.04: no loss
        self.assertTrue(o["fwd_ret_2"].iloc[-2:].isna().all())

    def test_outcome_label_uses_only_future_returns(self):
        r = pd.Series(np.r_[np.full(30, -0.02), np.zeros(30)], index=pd.bdate_range("2020-01-01", periods=60))
        lab = ev.daily_labels(ev.forward_outcomes(r))
        r2 = r.copy()
        r2.iloc[:10] = 0.05                                          # past returns changed
        lab2 = ev.daily_labels(ev.forward_outcomes(r2))
        pd.testing.assert_series_equal(lab.iloc[10:], lab2.iloc[10:])

    def test_block_purging(self):
        idx = pd.bdate_range("2017-11-01", "2018-02-28")
        lab = pd.Series(1.0, index=idx)
        p = ev.purge_block_boundary(lab[idx <= "2017-12-31"], "2017-12-31")
        self.assertEqual(int(p.isna().sum()), ev.LABEL_HORIZON)
        self.assertTrue(p.iloc[-ev.LABEL_HORIZON:].isna().all())

    def test_drawdown_events_and_rearm(self):
        r = pd.Series(np.r_[np.full(10, 0.01), np.full(10, -0.01), np.full(20, 0.01), np.full(10, -0.01)],
                      index=pd.bdate_range("2020-01-01", periods=50))
        e = ev.drawdown_events(r)
        self.assertEqual(len(e), 2)
        self.assertEqual(int(e["onset_pos"].iloc[0]), 15)                # 6th down day: -0.06 <= log(0.95)
        self.assertTrue(bool(e["censored"].iloc[-1]))

    def test_event_metrics_lead_late_missed_and_false(self):
        n = 400
        events = pd.DataFrame({"onset_pos": [100, 200, 300], "peak_pos": [95, 195, 295], "end_pos": [110, 210, 310],
                               "onset_date": [0, 0, 0], "peak_date": [0, 0, 0], "trough_loss_pct": [-6, -6, -6]})
        a = np.zeros(n, int)
        a[60:63] = 1          # old episode inside the window, ended long before onset 100
        a[90:95] = 1          # most recent episode before onset 100 -> lead 10, gap 6
        a[205] = 1            # late for event 200 (delay 5)
        a[360:370] = 1        # false episode
        m, t = ev.event_metrics(pd.Series(a), events)
        self.assertEqual(t["status"].tolist(), ["warned", "detected_late", "missed"])
        self.assertEqual(t["lead_days"].iloc[0], 10)
        self.assertEqual(t["gap_last_alert_to_onset"].iloc[0], 6)
        self.assertEqual(t["delay_days"].iloc[1], 5)
        self.assertEqual(m["false_alert_episodes"], 1)
        self.assertEqual(m["alert_episodes"], 4)

    def test_equal_train_budget(self):
        s = pd.Series(np.random.default_rng(1).normal(size=1000))
        train = np.r_[np.ones(600, bool), np.zeros(400, bool)]
        a, thr = ev.budget_alert(s, train)
        self.assertAlmostEqual(a[train].mean(), ev.TRAIN_ALERT_BUDGET, delta=0.005)

    def test_selection_prefers_simpler_when_not_significantly_worse(self):
        comp = pd.DataFrame({"system": ["I_full_ews_score", "B_p_stress", "E_ar_innovation"],
                             "split": ["validation"] * 3, "pr_auc": [0.30, 0.28, 0.20]})
        boot = pd.DataFrame({"model": ["B_p_stress", "E_ar_innovation"], "ci95_low": [-0.08, -0.15],
                             "ci95_high": [0.03, -0.01], "mean_ap_diff": [-0.02, -0.1]})
        sel = ev.select_specification(boot, comp, "I_full_ews_score")
        self.assertEqual(sel["selected"], "B_p_stress")
        boot.loc[0, "ci95_high"] = -0.001
        self.assertEqual(ev.select_specification(boot, comp, "I_full_ews_score")["selected"], "I_full_ews_score")

    def test_portfolio_returns(self):
        r = pd.DataFrame({"SPY": [math.log(1.1)], "BIL": [0.0]})
        p = ev.portfolio_log_returns(r, {"SPY": 0.5, "BIL": 0.5})
        self.assertAlmostEqual(p.iloc[0], math.log(1.05))


if __name__ == "__main__":
    unittest.main()
