"""Stage 5.5 tests: AR-HMM estimation, causal filtering, predictive calibration and the adoption rules.

Synthetic data only; these tests are not real-data validation.
"""

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

from regime_radar import ar_hmm as ah  # noqa: E402
from regime_radar import ar_hmm_diagnostics as ad  # noqa: E402
from regime_radar.dataset import FEATURE_COLUMNS  # noqa: E402
from regime_radar.hmm_models import HMMParams  # noqa: E402

TRUE = ah.ARParams(np.full(3, 1 / 3),
                   np.array([[0.98, 0.02, 0.0], [0.02, 0.96, 0.02], [0.0, 0.03, 0.97]]),
                   np.zeros((3, 5)), np.full((3, 5), 0.9), np.array([[0.04] * 5, [0.09] * 5, [0.36] * 5]))
LEVELS = np.array([[0.5] * 5, [0.0] * 5, [-1.5] * 5])
TRUE.intercept = LEVELS * (1 - TRUE.phi)


def simulate(params: ah.ARParams, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = np.zeros(n, dtype=int)
    x = np.zeros((n, params.intercept.shape[1]))
    x[0] = params.long_run_mean[0]
    for t in range(1, n):
        z[t] = rng.choice(params.k, p=params.transmat[z[t - 1]])
        x[t] = params.intercept[z[t]] + params.phi[z[t]] * x[t - 1] + rng.normal(size=x.shape[1]) * np.sqrt(params.var[z[t]])
    return x, z


class ARHMMEstimationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x, cls.z = simulate(TRUE, 2000, 1)
        cls.fit = ah.fit_ar_hmm_best(cls.x, 3, 7, n_init=3)

    def test_recovers_regimes_and_persistence(self):
        from sklearn.metrics import adjusted_rand_score
        f = self.fit
        self.assertTrue(f.converged and f.monotone)
        self.assertTrue(np.all(np.diff(f.history) >= -1e-8 * f.n_obs))
        tr, _ = ah.filter_train_validation_ar(f.params, self.x, self.x[-5:])
        self.assertGreater(adjusted_rand_score(self.z[1:], tr.filtered.argmax(1)), 0.6)
        np.testing.assert_allclose(np.sort(np.diag(f.params.transmat)), np.sort(np.diag(TRUE.transmat)), atol=0.03)
        np.testing.assert_allclose(f.params.phi.mean(), 0.9, atol=0.05)
        np.testing.assert_allclose(np.sort(f.params.var.mean(1)), np.sort(TRUE.var.mean(1)), rtol=0.25)

    def test_k1_equals_ordinary_least_squares(self):
        f = ah.fit_ar_hmm(self.x, 1, 0)
        y, lag = ah.lagged_pairs(self.x)
        for j in range(5):
            b = np.polyfit(lag[:, j], y[:, j], 1)
            self.assertAlmostEqual(f.params.phi[0, j], b[0], places=8)
            self.assertAlmostEqual(f.params.intercept[0, j], b[1], places=8)

    def test_phi_bound_is_enforced_and_em_stays_monotone(self):
        walk = np.cumsum(np.random.default_rng(3).normal(size=(800, 5)), axis=0)     # unit roots
        f = ah.fit_ar_hmm(walk, 2, 1)
        self.assertLessEqual(np.abs(f.params.phi).max(), ah.PHI_BOUND + 1e-12)
        self.assertTrue(f.monotone)
        self.assertGreater(f.bound_share, 0)

    def test_constrained_regression_is_optimal(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(300, 1)).cumsum(axis=0)
        y = x[1:] * 1.0 + rng.normal(scale=0.1, size=(299, 1))
        c, phi, var, active = ah.weighted_ar_regression(y, x[:-1], np.ones(299), bound=0.9)
        self.assertTrue(active[0])
        sse = lambda cc, pp: float(((y[:, 0] - cc - pp * x[:-1, 0]) ** 2).sum())   # noqa: E731
        best = sse(c[0], phi[0])
        for pp in np.linspace(-0.9, 0.9, 37):
            cc = float((y[:, 0] - pp * x[:-1, 0]).mean())
            self.assertGreaterEqual(sse(cc, pp), best - 1e-9)

    def test_reproducible_and_train_only(self):
        a = ah.fit_ar_hmm_best(self.x[:1500], 3, 7, n_init=2)
        b = ah.fit_ar_hmm_best(self.x[:1500], 3, 7, n_init=2)
        np.testing.assert_array_equal(a.params.phi, b.params.phi)
        np.testing.assert_array_equal(a.params.transmat, b.params.transmat)
        # the estimator never sees Validation: fitting is a function of the Train array alone
        c = ah.fit_ar_hmm_best(self.x[:1500].copy(), 3, 7, n_init=2)
        np.testing.assert_array_equal(a.params.intercept, c.params.intercept)

    def test_empty_state_handled(self):
        flat = simulate(ah.ARParams(np.ones(1), np.ones((1, 1)), np.zeros((1, 5)), np.full((1, 5), 0.5),
                                    np.full((1, 5), 0.1)), 400, 5)[0]
        f = ah.fit_ar_hmm(flat, 3, 2, max_iter=300)
        self.assertTrue(np.isfinite(f.train_loglik))
        self.assertTrue(np.isfinite(f.params.var).all() and (f.params.var >= ah.REG_VAR).all())


class ARHMMCausalityTests(unittest.TestCase):
    def setUp(self):
        self.x, _ = simulate(TRUE, 600, 11)
        self.p = TRUE

    def test_brute_force_likelihood(self):
        x = self.x[:7]
        y, lag = ah.lagged_pairs(x)
        res = ah.filter_from_log_emission(ah.log_emission_ar(self.p, y, lag), self.p.transmat,
                                          initial_predicted=self.p.startprob)
        self.assertAlmostEqual(res.log_score.sum(), ah.brute_force_loglik_ar(self.p, x), places=10)

    def test_emission_uses_only_current_and_previous_observation(self):
        y, lag = ah.lagged_pairs(self.x)
        base = ah.log_emission_ar(self.p, y, lag)
        x2 = self.x.copy()
        x2[300] += 5.0                                     # affects emission rows of y = x[300] and x[301]
        y2, lag2 = ah.lagged_pairs(x2)
        changed = np.flatnonzero(np.abs(ah.log_emission_ar(self.p, y2, lag2) - base).max(axis=1) > 0)
        self.assertEqual(list(changed), [299, 300])        # rows indexed from x[1]

    def test_probabilities_sum_to_one_and_future_invariance(self):
        tr, va = ah.filter_train_validation_ar(self.p, self.x[:400], self.x[400:])
        np.testing.assert_allclose(va.filtered.sum(1), 1, atol=1e-12)
        np.testing.assert_allclose(va.predicted.sum(1), 1, atol=1e-12)
        shocked = self.x[400:].copy()
        shocked[100:] = shocked[100:] * -8 + 30
        _, va2 = ah.filter_train_validation_ar(self.p, self.x[:400], shocked)
        self.assertEqual(np.abs(va2.filtered[:100] - va.filtered[:100]).max(), 0.0)
        self.assertEqual(np.abs(va2.predicted[100] - va.predicted[100]).max(), 0.0)   # forecast before x_t
        self.assertGreater(np.abs(va2.log_score[100] - va.log_score[100]), 0)

    def test_validation_continuation_equals_joint_filter(self):
        tr, va = ah.filter_train_validation_ar(self.p, self.x[:400], self.x[400:])
        y, lag = ah.lagged_pairs(self.x)
        joint = ah.filter_from_log_emission(ah.log_emission_ar(self.p, y, lag), self.p.transmat,
                                            initial_predicted=self.p.startprob)
        np.testing.assert_allclose(va.filtered, joint.filtered[399:], atol=1e-12)
        np.testing.assert_allclose(va.log_score, joint.log_score[399:], atol=1e-10)

    def test_validation_changes_do_not_change_train_estimates(self):
        a = ah.fit_ar_hmm(self.x[:400], 2, 3)
        tr1, _ = ah.filter_train_validation_ar(a.params, self.x[:400], self.x[400:])
        tr2, _ = ah.filter_train_validation_ar(a.params, self.x[:400], self.x[400:] * 3)
        np.testing.assert_array_equal(tr1.filtered, tr2.filtered)


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.x, _ = simulate(TRUE, 3000, 21)
        tr, va = ah.filter_train_validation_ar(TRUE, self.x[:2000], self.x[2000:])
        y, lag = ah.lagged_pairs(self.x[2000:], previous=self.x[1999])
        self.p = ad.Predictive(va.predicted, ah.conditional_means(TRUE, lag), TRUE.var, y, va.log_score, va.filtered)

    def test_correct_model_is_calibrated(self):
        u = ad.pit(self.p)
        for lvl in ad.COVERAGE_LEVELS:
            cover = ((u > 0.5 - lvl / 2) & (u < 0.5 + lvl / 2)).mean()
            self.assertAlmostEqual(cover, lvl, delta=0.03)
        q = ad.quantile_residuals(self.p)
        self.assertLess(abs(np.mean([ad.acf(q[:, j], 1) for j in range(5)])), 0.06)

    def test_mixture_quantile_inverts_cdf(self):
        q90 = ad.mixture_quantile(self.p, 0.9)
        np.testing.assert_allclose(ad.mixture_cdf(self.p, q90), 0.9, atol=1e-8)

    def test_crps_matches_gaussian_formula_and_monte_carlo(self):
        from scipy.stats import norm
        one = ad.Predictive(np.ones((1, 1)), np.zeros((1, 1, 1)), np.ones((1, 1)), np.array([[0.7]]), np.zeros(1),
                            np.ones((1, 1)))
        z = 0.7
        exact = z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi)
        self.assertAlmostEqual(ad.mixture_crps(one)[0, 0], exact, places=10)
        rng = np.random.default_rng(0)
        w, m, s = np.array([0.3, 0.7]), np.array([-1.0, 2.0]), np.array([0.5, 1.5])
        mix = ad.Predictive(w[None], m[None, :, None], (s ** 2)[:, None], np.array([[0.4]]), np.zeros(1), w[None])
        comp = rng.choice(2, size=400000, p=w)
        a = rng.normal(m[comp], s[comp])
        b = rng.normal(m[rng.choice(2, size=400000, p=w)], s[rng.choice(2, size=400000, p=w)])
        mc = np.abs(a - 0.4).mean() - 0.5 * np.abs(a - b).mean()
        self.assertAlmostEqual(ad.mixture_crps(mix)[0, 0], mc, delta=0.01)

    def test_ljung_box(self):
        rng = np.random.default_rng(1)
        self.assertGreater(ad.ljung_box(rng.normal(size=1000))[1], 0.01)
        ar = np.zeros(1000)
        for t in range(1, 1000):
            ar[t] = 0.8 * ar[t - 1] + rng.normal()
        self.assertLess(ad.ljung_box(ar)[1], 1e-10)

    def test_independent_emission_hmm_misses_dependence_that_ar_captures(self):
        """Fitting an i.i.d.-emission view to AR data leaves residual autocorrelation; the AR model removes it."""
        y = self.p.x
        iid = ad.Predictive(np.ones((len(y), 1)), np.broadcast_to(y.mean(0)[None, None], (len(y), 1, 5)),
                            y.var(0)[None], y, np.zeros(len(y)), np.ones((len(y), 1)))
        a = np.mean([ad.acf(ad.quantile_residuals(iid)[:, j], 1) for j in range(5)])
        b = np.mean([ad.acf(ad.quantile_residuals(self.p)[:, j], 1) for j in range(5)])
        self.assertGreater(a, 0.8)
        self.assertLess(abs(b), 0.06)


class AdoptionAndProtocolTests(unittest.TestCase):
    def _tables(self, gain_ar1_t=3.0, coverage=0.9, ari=0.95, stress_vol=1.5):
        comp = pd.DataFrame([{"model_id": "B", "validation_mean_diff_vs_frozen_hmm": 5.0,
                              "validation_nw_t_vs_frozen_hmm": 20.0, "validation_mean_diff_vs_ar1": 0.2,
                              "validation_nw_t_vs_ar1": gain_ar1_t}])
        calib = pd.DataFrame([{"model_id": m, "split": "validation", "factor": "all factors", "coverage_90": c}
                              for m, c in (("A", 0.81), ("B", coverage))])
        resid = pd.DataFrame([{"model_id": m, "split": "validation", "qres_acf_lag1": v}
                              for m, v in (("A", 0.96), ("B", 0.05))])
        init = pd.DataFrame([{"model_id": "B", "converged_share": 1.0, "all_monotone": True,
                              "train_mean_pairwise_ari": ari}])
        regimes = pd.DataFrame([{"model_id": "B", "train_min_state_share": 0.2, "validation_switches_per_year": 8.0},
                                {"model_id": "A", "train_min_state_share": 0.2, "validation_switches_per_year": 6.8}])
        profiles = pd.DataFrame([{"model_id": "B", "split": "train", "is_stress_state": True, "volatility": stress_vol,
                                  "liquidity_credit": 1.2, "equity_trend": -1.0}])
        return comp, calib, resid, init, regimes, profiles

    def test_replace_only_when_all_rules_pass(self):
        d = ad.adoption_decision(*self._tables(), "B", "A", "C")
        self.assertEqual(d["verdict"], "replace")
        d = ad.adoption_decision(*self._tables(gain_ar1_t=1.0), "B", "A", "C")
        self.assertEqual(d["verdict"], "retain_ar_explains_gain")
        d = ad.adoption_decision(*self._tables(stress_vol=0.3), "B", "A", "C")
        self.assertEqual(d["verdict"], "challenger_density")
        d = ad.adoption_decision(*self._tables(coverage=0.7), "B", "A", "C")
        self.assertEqual(d["verdict"], "retain_mixed")
        t = list(self._tables())
        t[4].loc[t[4]["model_id"] == "B", "validation_switches_per_year"] = 80.0      # flickering states
        d = ad.adoption_decision(*t, "B", "A", "C")
        self.assertFalse(d["checks"]["5_stability_and_persistence"])
        self.assertEqual(d["verdict"], "challenger_density")

    def test_frozen_hmm_wrapper_scores_same_dates_without_refit(self):
        params = HMMParams(np.array([0.5, 0.5]), np.array([[0.95, 0.05], [0.05, 0.95]]),
                           np.array([[0.0] * 5, [1.0] * 5]), np.ones((2, 5)), "diag")
        x = np.random.default_rng(2).normal(size=(300, 5))
        m = ad.frozen_hmm_result(params, x[:200], x[200:], [1], "frozen")
        self.assertEqual(len(m.train.log_score), 199)       # Train day 1 filtered, not scored
        self.assertEqual(len(m.validation.log_score), 100)
        self.assertIs(m.params, params)

    def test_directory_hash_detects_changes(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            (tmp / "a.json").write_text(json.dumps({"x": 1}))
            before = ad.directory_hashes(tmp)
            self.assertEqual(before, ad.directory_hashes(tmp))
            (tmp / "a.json").write_text(json.dumps({"x": 2}))
            self.assertNotEqual(before, ad.directory_hashes(tmp))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_script_reads_no_test_split_and_versions_match(self):
        from regime_radar import ar_hmm_report as ar
        script = (ROOT / "scripts" / "07b_compare_ar_hmm.py").read_text()
        self.assertNotIn("test_features", script)
        self.assertNotIn("test_returns", script)
        versions = {m.STAGE55_CODE_VERSION for m in (ah, ad, ar)}
        self.assertEqual(len(versions), 1)
        v = versions.pop()
        self.assertIn(f'STAGE55_CODE_VERSION = "{v}"', script)
        audit = (ROOT / "scripts" / "07c_audit_ar_hmm_em.py").read_text()
        self.assertIn(f'STAGE55_CODE_VERSION = "{v}"', audit)
        self.assertNotIn("test_features", audit)
        self.assertEqual(list(FEATURE_COLUMNS), ["equity_trend", "interest_rate", "inflation", "volatility",
                                                 "liquidity_credit"])


def realistic(seed: int, n: int = 2058) -> np.ndarray:
    """Persistent states, phi 0.95-0.99, small innovations (like the real Stage 5.5 estimates), standardized."""
    rng = np.random.default_rng(seed)
    a = np.array([[0.92, 0.07, 0.01], [0.10, 0.87, 0.03], [0.30, 0.35, 0.35]])
    phi = np.array([[0.99, 0.98, 0.99, 0.98, 0.99], [0.97, 0.96, 0.99, 0.96, 0.98], [0.99, 0.97, 0.99, 0.95, 0.99]])
    sd = np.array([[0.10, 0.09, 0.06, 0.10, 0.08], [0.10, 0.20, 0.09, 0.15, 0.10], [0.17, 0.18, 0.15, 0.33, 0.18]])
    mu = np.array([[0.3, 0, 0.2, -0.6, -0.6], [-0.3, 0.2, -0.2, 0.1, 0.0], [-1.2, 0, -1, 1.6, 1.2]])
    z, x = np.zeros(n, int), np.zeros((n, 5))
    for t in range(1, n):
        z[t] = rng.choice(3, p=a[z[t - 1]])
        x[t] = mu[z[t]] + phi[z[t]] * (x[t - 1] - mu[z[t]]) + rng.normal(size=5) * sd[z[t]]
    return (x - x.mean(0)) / x.std(0)


class EMAuditTests(unittest.TestCase):
    """Regression tests for the Stage 5.5 audit (v1 variance update s2 + reg was not a Q maximizer)."""

    @classmethod
    def setUpClass(cls):
        from regime_radar.hmm_models import _forward_backward
        cls.x = realistic(2)
        cls.y, cls.lag = ah.lagged_pairs(cls.x)
        p = ah.initial_params_ar(cls.y, cls.lag, 3, 7)
        for _ in range(5):
            g, xi, _ = _forward_backward(p, ah.log_emission_ar(p, cls.y, cls.lag))
            p = ah._m_step(p, cls.y, cls.lag, g, xi)[0]
        cls.p = p
        cls.g, cls.xi, _ = _forward_backward(p, ah.log_emission_ar(p, cls.y, cls.lag))

    def test_variance_update_is_floor_not_additive(self):
        rng = np.random.default_rng(0)
        lagv = rng.normal(size=(200, 1))
        yv = 0.5 * lagv + rng.normal(scale=0.2, size=(200, 1))
        _, _, v, _ = ah.weighted_ar_regression(yv, lagv, np.ones(200))
        _, _, v_old, _ = ah.weighted_ar_regression(yv, lagv, np.ones(200), variance_rule="additive_legacy")
        self.assertAlmostEqual(v_old[0] - v[0], ah.REG_VAR, places=12)
        tiny = ah.weighted_ar_regression(lagv[1:] * 0 + 1.0, lagv[:-1], np.ones(199))[2]
        self.assertEqual(tiny[0], ah.REG_VAR)              # zero residual variance -> floor binds

    def test_exact_mstep_maximizes_q_and_legacy_does_not(self):
        new = ah._m_step(self.p, self.y, self.lag, self.g, self.xi)[0]
        old = ah._m_step(self.p, self.y, self.lag, self.g, self.xi, "additive_legacy")[0]
        q_new = ah.expected_complete_loglik(new, self.y, self.lag, self.g, self.xi)
        q_old = ah.expected_complete_loglik(old, self.y, self.lag, self.g, self.xi)
        self.assertGreater(q_new - q_old, 1e-3)
        o = ad.mstep_optimality(self.p, self.x)
        self.assertTrue(o["optimal"], o)

    def test_exact_em_is_monotone_for_every_start(self):
        from regime_radar.hmm_models import start_seeds
        for s in start_seeds(20240533, 5):
            f = ah.fit_ar_hmm(self.x, 3, s)
            self.assertEqual(f.n_decreases, 0)
            self.assertLessEqual(f.max_decrease, ah.MONOTONE_ABS_TOL)
        best = ah.fit_ar_hmm_best(self.x, 3, 20240533, n_init=5)
        self.assertEqual(len(best.start_histories), 5)
        self.assertTrue(best.all_starts_monotone)

    def test_legacy_rule_shows_genuine_decrease_and_the_check_fails(self):
        f = ah.fit_ar_hmm(self.x, 3, 202405332, variance_rule="additive_legacy")   # start found in the audit
        self.assertGreater(f.n_decreases, 0)
        self.assertGreater(f.max_decrease, 1e3 * ah.MONOTONE_ABS_TOL)
        fake = ad.ModelResult("legacy", "", 3, 0, None, None, f.params, fits=[f])
        f.start_histories = [f.history]
        status, detail = ad.monotonicity_check({"legacy": fake}, ["legacy"])
        self.assertEqual(status, "FAIL", detail)
        g = ah.fit_ar_hmm(self.x, 3, 202405332)
        g.start_histories = [g.history]
        ok = ad.ModelResult("exact", "", 3, 0, None, None, g.params, fits=[g])
        self.assertEqual(ad.monotonicity_check({"exact": ok}, ["exact"])[0], "PASS")

    def test_objective_equals_filter_score(self):
        f = ah.fit_ar_hmm(self.x, 3, 11)
        res = ah.filter_from_log_emission(ah.log_emission_ar(f.params, self.y, self.lag), f.params.transmat,
                                          initial_predicted=f.params.startprob)
        self.assertAlmostEqual(res.log_score.sum(), f.train_loglik, places=8)


if __name__ == "__main__":
    unittest.main()
