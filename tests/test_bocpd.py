"""Stage 6 tests: Bayesian online change-point detection (exactness, causality, synthetic behaviour).

Synthetic data only; these tests are not real-data validation.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regime_radar import bocpd as bo  # noqa: E402

PRIOR = bo.NIGPrior(0.0, 1.0, 2.0, 1.0)


class TestBOCPDExactness(unittest.TestCase):
    def test_matches_brute_force_enumeration(self):
        y = np.array([0.1, -0.3, 0.2, 2.5, 2.8, 2.2, 3.1, 2.6])
        res = bo.bocpd(y, PRIOR, 5.0, store_matrix=len(y) + 1)
        exact = bo.brute_force_run_length(y, PRIOR, 5.0)
        np.testing.assert_allclose(res.run_length_matrix[-1, :len(exact)], exact, atol=1e-12)

    def test_posterior_sums_to_one(self):
        y = np.random.default_rng(0).normal(size=300)
        res = bo.bocpd(y, PRIOR, 50.0, store_matrix=301)
        np.testing.assert_allclose(res.run_length_matrix.sum(axis=1), 1.0, atol=1e-10)
        for w, p in res.p_recent.items():
            self.assertTrue(np.all((p >= 0) & (p <= 1 + 1e-12)))

    def test_constant_hazard_makes_p_change_now_equal_hazard(self):
        y = np.random.default_rng(1).normal(size=200)
        res = bo.bocpd(y, PRIOR, 126.0)
        np.testing.assert_allclose(res.p_change_now, 1 / 126.0, rtol=1e-10)

    def test_prior_recent_probability(self):
        self.assertAlmostEqual(bo.prior_recent_probability(126, 1), 1 / 126)
        self.assertAlmostEqual(bo.prior_recent_probability(126, 10), 1 - (125 / 126) ** 10)

    def test_prior_predictive_variance_matches_train(self):
        y = np.random.default_rng(2).normal(3.0, 2.0, size=5000)
        p = bo.NIGPrior.from_train(y)
        df = 2 * p.alpha0
        scale2 = p.beta0 * (p.kappa0 + 1) / (p.alpha0 * p.kappa0)
        self.assertAlmostEqual(scale2 * df / (df - 2), y.var(), places=8)
        self.assertAlmostEqual(p.mu0, y.mean(), places=10)

    def test_truncation_exact_until_it_binds_and_stays_normalized(self):
        """Truncation is an approximation (the Stage 6 runs use none); before the limit it changes nothing."""
        y = np.random.default_rng(3).normal(size=400)
        full = bo.bocpd(y, PRIOR, 100.0)
        same = bo.bocpd(y, PRIOR, 100.0, max_run_length=1000)
        np.testing.assert_array_equal(full.p_recent[10], same.p_recent[10])
        trunc = bo.bocpd(y, PRIOR, 100.0, max_run_length=300, store_matrix=300)
        np.testing.assert_array_equal(trunc.p_recent[10][:299], full.p_recent[10][:299])
        np.testing.assert_allclose(trunc.run_length_matrix.sum(axis=1), 1.0, atol=1e-10)

    def test_nan_input_carries_posterior_forward(self):
        y = np.random.default_rng(4).normal(size=50)
        y[20] = np.nan
        res = bo.bocpd(y, PRIOR, 30.0)
        self.assertEqual(res.p_recent[10][20], res.p_recent[10][19])
        self.assertTrue(np.isnan(res.log_evidence[20]))


class TestBOCPDCausality(unittest.TestCase):
    def test_future_values_do_not_change_past_output(self):
        rng = np.random.default_rng(5)
        y = rng.normal(size=300)
        y2 = y.copy()
        y2[200:] = rng.normal(5, 3, size=100)          # leakage injection: a dramatic future change
        a, b = bo.bocpd(y, PRIOR, 126.0), bo.bocpd(y2, PRIOR, 126.0)
        for w in a.p_recent:
            np.testing.assert_array_equal(a.p_recent[w][:200], b.p_recent[w][:200])
        np.testing.assert_array_equal(a.expected_run_length[:200], b.expected_run_length[:200])
        self.assertGreater(b.p_recent[10][200:215].max(), a.p_recent[10][200:215].max())


class TestBOCPDScenarios(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = bo.synthetic_scenarios(seeds=range(3)).set_index("scenario")
        print("\nBOCPD synthetic scenarios (3 seeds):\n" + cls.s.round(3).to_string())

    def test_mean_shift_and_variance_increase_detected_quickly(self):
        for name in ("mean_shift", "variance_increase", "consecutive_shocks"):
            self.assertGreaterEqual(self.s.at[name, "detection_rate"], 2 / 3, name)
            self.assertLessEqual(self.s.at[name, "median_delay_sessions"], 20, name)

    def test_persistent_levels_create_spurious_changes_innovations_do_not(self):
        lv = self.s.at["ar1_no_change_on_levels", "false_alarm_episodes_per_252"]
        inn = self.s.at["ar1_no_change_on_innovations", "false_alarm_episodes_per_252"]
        self.assertGreater(lv, 3 * max(inn, 0.1))

    def test_gradual_drift_is_slow_to_detect(self):
        d = self.s.at["gradual_drift", "median_delay_sessions"]
        self.assertTrue(np.isnan(d) or d > self.s.at["mean_shift", "median_delay_sessions"])


if __name__ == "__main__":
    unittest.main()
