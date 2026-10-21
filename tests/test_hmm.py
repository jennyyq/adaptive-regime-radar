"""Stage 5 tests: Gaussian HMM estimation, sticky prior, causal filtering, alignment, selection and data access.

All tests use synthetic data; they are not real-data validation.
"""

from __future__ import annotations

import itertools
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regime_radar import hmm_diagnostics as hd  # noqa: E402
from regime_radar import hmm_filtering as hf  # noqa: E402
from regime_radar import hmm_models as hm  # noqa: E402
from regime_radar import regime_exploration as rx  # noqa: E402
from regime_radar.dataset import FEATURE_COLUMNS  # noqa: E402

MEANS = np.array([[0.6, -0.2, 0.0, -0.6, -0.5],     # calm
                  [-1.5, -1.0, -1.0, 2.5, 2.0]])    # stress


def two_state_sequence(n: int, seed: int, stay: float = 0.98) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = np.zeros(n, dtype=int)
    for t in range(1, n):
        z[t] = z[t - 1] if rng.random() < stay else 1 - z[t - 1]
    return MEANS[z] + rng.normal(scale=0.7, size=(n, 5)), z


def small_params(cov: str = "diag") -> hm.HMMParams:
    covars = np.array([[1.0, 0.5, 2.0], [0.7, 1.5, 0.9]])
    if cov == "full":
        covars = np.stack([np.diag(c) + 0.1 for c in covars])
    return hm.HMMParams(np.array([0.6, 0.4]), np.array([[0.9, 0.1], [0.2, 0.8]]),
                        np.array([[0.0, 0.5, -0.3], [1.0, -1.0, 0.8]]), covars, cov)


class LikelihoodAndFilterTests(unittest.TestCase):
    def test_forward_loglik_matches_brute_force(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(6, 3))
        for cov in ("diag", "full"):
            p = small_params(cov)
            res = hf.forward_filter(p, x)
            self.assertAlmostEqual(res.log_score.sum(), hf.brute_force_loglik(p, x), places=10)
            gamma, _, ll = hm._forward_backward(p, hm.log_emission(p, x))
            self.assertAlmostEqual(ll, res.log_score.sum(), places=10)

    def test_probabilities_sum_to_one_and_prediction_identity(self):
        x, _ = two_state_sequence(300, 2)
        p = hm.fit_hmm(x, hm.HMMSpec("standard", 2, "diag"), 3).params
        res = hf.forward_filter(p, x)
        np.testing.assert_allclose(res.filtered.sum(1), 1, atol=1e-12)
        np.testing.assert_allclose(res.predicted.sum(1), 1, atol=1e-12)
        np.testing.assert_allclose(res.predicted[1:], res.filtered[:-1] @ p.transmat, atol=1e-12)

    def test_future_changes_do_not_move_earlier_filtered_probabilities(self):
        x, _ = two_state_sequence(400, 4)
        p = hm.fit_hmm(x, hm.HMMSpec("standard", 2, "diag"), 5).params
        base = hf.forward_filter(p, x).filtered
        shocked = x.copy()
        shocked[250:] = shocked[250:] * -8 + 20            # dramatic change of the future
        new = hf.forward_filter(p, shocked).filtered
        self.assertEqual(np.abs(new[:250] - base[:250]).max(), 0.0)
        self.assertGreater(np.abs(new[250:] - base[250:]).max(), 0.1)

    def test_smoothed_probabilities_fail_the_causality_test(self):
        """Same past, two different futures: filtered probabilities of the past are identical, smoothed ones are not."""
        x, _ = two_state_sequence(400, 4)
        p = hm.fit_hmm(x, hm.HMMSpec("standard", 2, "diag"), 5).params
        past = np.repeat(((MEANS[0] + MEANS[1]) / 2)[None, :], 50, axis=0)     # ambiguous recent history
        calm, stress = np.repeat(MEANS[[0]], 30, axis=0), np.repeat(MEANS[[1]], 30, axis=0)
        a, b = np.vstack([x[:200], past, calm]), np.vstack([x[:200], past, stress])
        cut = 250
        self.assertEqual(np.abs(hf.forward_filter(p, a).filtered[:cut] - hf.forward_filter(p, b).filtered[:cut]).max(), 0)
        smooth_gap = np.abs(hf.smoothed_probabilities(p, a)[:cut] - hf.smoothed_probabilities(p, b)[:cut]).max()
        self.assertGreater(smooth_gap, 0.05)          # a smoother revises earlier dates: it fails causality

    def test_validation_continues_from_last_train_filter(self):
        x, _ = two_state_sequence(500, 6)
        p = hm.fit_hmm(x[:350], hm.HMMSpec("standard", 2, "diag"), 7).params
        tr, va = hf.filter_train_validation(p, x[:350], x[350:])
        joint = hf.forward_filter(p, x)
        np.testing.assert_allclose(va.filtered, joint.filtered[350:], atol=1e-12)
        np.testing.assert_allclose(va.log_score, joint.log_score[350:], atol=1e-10)
        np.testing.assert_allclose(va.predicted[0], tr.last_filtered @ p.transmat, atol=1e-12)

    def test_order_of_observations_matters(self):
        x, _ = two_state_sequence(300, 8)
        p = hm.fit_hmm(x, hm.HMMSpec("standard", 2, "diag"), 9).params
        fwd = hf.forward_filter(p, x).filtered
        rev = hf.forward_filter(p, x[::-1]).filtered[::-1]
        self.assertGreater(np.abs(fwd - rev).max(), 1e-6)

    def test_viterbi_matches_brute_force(self):
        p = small_params()
        x = np.random.default_rng(3).normal(size=(5, 3))
        log_b, log_a = hm.log_emission(p, x), np.log(p.transmat)
        best = max(itertools.product(range(2), repeat=5), key=lambda z: np.log(p.startprob[z[0]]) + log_b[0, z[0]] + sum(
            log_a[z[t - 1], z[t]] + log_b[t, z[t]] for t in range(1, 5)))
        np.testing.assert_array_equal(hf.viterbi(p, x), best)

    @unittest.skipUnless(__import__("importlib").util.find_spec("hmmlearn"), "hmmlearn not installed (optional)")
    def test_cross_check_with_hmmlearn(self):
        from hmmlearn.hmm import GaussianHMM

        p = small_params()
        x = np.random.default_rng(4).normal(size=(200, 3))
        ref = GaussianHMM(n_components=2, covariance_type="diag")
        ref.startprob_, ref.transmat_, ref.means_, ref.covars_ = p.startprob, p.transmat, p.means, p.covars
        self.assertAlmostEqual(ref.score(x), hf.forward_filter(p, x).log_score.sum(), places=6)
        np.testing.assert_allclose(ref.predict_proba(x), hf.smoothed_probabilities(p, x), atol=1e-8)


class EstimationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x, cls.z = two_state_sequence(1500, 11)

    def test_em_recovers_states_and_is_monotone(self):
        fit = hm.fit_hmm(self.x, hm.HMMSpec("standard", 2, "diag"), 1)
        self.assertTrue(fit.converged and fit.monotone)
        self.assertTrue(np.all(np.diff(fit.history) >= -1e-8 * len(self.x)))
        labels = hf.forward_filter(fit.params, self.x).filtered.argmax(1)
        from sklearn.metrics import adjusted_rand_score
        self.assertGreater(adjusted_rand_score(self.z, labels), 0.9)
        np.testing.assert_allclose(fit.params.transmat.sum(1), 1, atol=1e-12)

    def test_best_of_n_init_keeps_highest_objective(self):
        spec = hm.HMMSpec("standard", 3, "diag")
        best = hm.fit_hmm_best(self.x, spec, 7, n_init=4)
        singles = [hm.fit_hmm(self.x, spec, s) for s in hm.start_seeds(7, 4)]
        self.assertEqual(len(best.start_objectives), 4)
        self.assertAlmostEqual(best.objective, max(f.objective for f in singles if f.converged), places=8)
        self.assertEqual(best.seed, 7)
        again = hm.fit_hmm_best(self.x, spec, 7, n_init=4)
        np.testing.assert_allclose(again.params.means, best.params.means, atol=1e-12)
        self.assertEqual(len(set(hm.start_seeds(1, 5)) & set(hm.start_seeds(2, 5))), 0)

    def test_reproducible_for_a_seed(self):
        a = hm.fit_hmm(self.x, hm.HMMSpec("standard", 2, "full"), 5)
        b = hm.fit_hmm(self.x, hm.HMMSpec("standard", 2, "full"), 5)
        np.testing.assert_allclose(a.params.means, b.params.means, atol=1e-12)
        np.testing.assert_allclose(a.params.transmat, b.params.transmat, atol=1e-12)

    def test_stable_across_seeds_after_alignment(self):
        spec = hm.HMMSpec("standard", 2, "diag")
        fits = [hm.fit_hmm(self.x, spec, s) for s in (1, 2, 3, 4)]
        ref = fits[0]
        for f in fits[1:]:
            q = f.params.permuted(hm.align_by_means(ref.params, f.params))
            self.assertLess(np.abs(q.means - ref.params.means).max(), 1e-3)

    def test_train_only_estimation(self):
        """fit_hmm receives Train only; Validation can change arbitrarily without moving the parameters."""
        x_tr, x_va = self.x[:1000], self.x[1000:]
        spec = hm.HMMSpec("standard", 2, "diag")
        mean, scale = np.zeros(5), np.ones(5)
        a = hd.run_candidate(spec, x_tr, x_va, (1, 2), mean, scale, x_tr)
        b = hd.run_candidate(spec, x_tr, x_va * 10 + 3, (1, 2), mean, scale, x_tr)
        np.testing.assert_array_equal(a.reference.params.means, b.reference.params.means)
        np.testing.assert_array_equal(a.reference.params.transmat, b.reference.params.transmat)
        np.testing.assert_array_equal(a.train.filtered, b.train.filtered)

    def test_empty_states_handled(self):
        x = self.x[self.z == 0][:400]                       # one regime only, fitted with K=4
        fit = hm.fit_hmm(x, hm.HMMSpec("standard", 4, "full"), 3, max_iter=200)
        self.assertTrue(np.isfinite(fit.train_loglik))
        self.assertTrue(np.isfinite(fit.params.means).all() and np.isfinite(fit.params.covars).all())
        res = hf.forward_filter(fit.params, x)
        np.testing.assert_allclose(res.filtered.sum(1), 1, atol=1e-10)

    def test_mstep_keeps_parameters_of_empty_state(self):
        p = hm.HMMParams(np.array([0.5, 0.5]), np.full((2, 2), 0.5), np.array([[0.0] * 5, [9.0] * 5]),
                         np.ones((2, 5)), "diag")
        gamma = np.c_[np.ones(50), np.zeros(50)]
        new, empty = hm._m_step(p, self.x[:50], gamma, np.array([[49.0, 0.0], [0.0, 0.0]]), 0.0, 1e-4)
        self.assertEqual(empty, [1])
        np.testing.assert_array_equal(new.means[1], p.means[1])
        np.testing.assert_allclose(new.transmat.sum(1), 1)


class StickyPriorTests(unittest.TestCase):
    def test_map_transition_update(self):
        counts = np.array([[90.0, 10.0], [5.0, 45.0]])
        a = hm.map_transmat(counts, 20.0)
        np.testing.assert_allclose(a, [[110 / 120, 10 / 120], [5 / 70, 65 / 70]])
        np.testing.assert_allclose(hm.map_transmat(counts, 0.0), counts / counts.sum(1, keepdims=True))

    def test_kappa_definition_and_ids(self):
        spec = hm.HMMSpec("sticky", 2, "diag", 0.3)
        self.assertAlmostEqual(spec.kappa(2058), 0.3 * 2057 / 2)
        self.assertEqual(hm.HMMSpec("standard", 3, "diag").kappa(2058), 0.0)
        self.assertEqual(spec.model_id, "hmm_sticky_diag_k2_w0.30")
        ids = [s.model_id for s in hm.candidate_specs()]
        self.assertEqual(len(ids), len(set(ids)))

    def test_prior_enters_estimation_and_raises_persistence(self):
        x, _ = two_state_sequence(800, 21, stay=0.95)
        init = hm.initial_params(x, 2, "diag", 1)
        std = hm.fit_hmm(x, hm.HMMSpec("standard", 2, "diag"), 1, init=init)
        sticky = hm.fit_hmm(x, hm.HMMSpec("sticky", 2, "diag", 1.0), 1, init=init)
        self.assertTrue(sticky.monotone)
        self.assertGreater(np.diag(sticky.params.transmat).min(), np.diag(std.params.transmat).min())
        # the means move too: the prior acts through the E-step, not by relabelling
        self.assertGreater(np.abs(sticky.params.means - std.params.means).max(), 0)
        # MAP objective = loglik + kappa * sum log A_ii
        self.assertAlmostEqual(sticky.objective, sticky.train_loglik
                               + sticky.kappa * np.log(np.diag(sticky.params.transmat)).sum(), places=6)
        # the MAP estimate has a lower likelihood but a higher posterior than the ML estimate
        self.assertLessEqual(sticky.train_loglik, std.train_loglik + 1e-6)
        ml_post = std.train_loglik + hm.log_prior(std.params.transmat, sticky.kappa)
        self.assertGreaterEqual(sticky.objective, ml_post - 1e-6)

    def test_prior_is_not_post_hoc(self):
        """A fitted sticky model differs from the standard model with its diagonal overwritten afterwards."""
        x, _ = two_state_sequence(800, 22, stay=0.9)
        init = hm.initial_params(x, 2, "diag", 2)
        std = hm.fit_hmm(x, hm.HMMSpec("standard", 2, "diag"), 2, init=init)
        sticky = hm.fit_hmm(x, hm.HMMSpec("sticky", 2, "diag", 1.0), 2, init=init)
        forced = std.params.copy()
        forced.transmat = sticky.params.transmat.copy()
        self.assertGreater(np.abs(forced.means - sticky.params.means).max(), 1e-6)


class DiagnosticsTests(unittest.TestCase):
    def test_expected_durations(self):
        d = hm.expected_durations(np.array([[0.9, 0.1], [0.0, 1.0]]))
        self.assertAlmostEqual(d[0], 10.0)
        self.assertTrue(np.isinf(d[1]))

    def test_alignment_recovers_permutation(self):
        p = hm.HMMParams(np.full(3, 1 / 3), np.full((3, 3), 1 / 3), np.array([[0.0] * 5, [1.0] * 5, [-2.0] * 5]),
                         np.ones((3, 5)), "diag")
        shuffled = p.permuted([2, 0, 1])
        back = shuffled.permuted(hm.align_by_means(p, shuffled))
        np.testing.assert_array_equal(back.means, p.means)

    def test_canonical_order_by_stress_score(self):
        means = np.zeros((2, 5))
        means[0, FEATURE_COLUMNS.index("volatility")] = 2.0
        p = hm.HMMParams(np.full(2, 0.5), np.full((2, 2), 0.5), means, np.ones((2, 5)), "diag")
        order = hm.canonical_order(p, FEATURE_COLUMNS, np.zeros(5), np.ones(5))
        self.assertEqual(list(order), [1, 0])                # the high-volatility state becomes S2

    def test_permuted_transition_matrix(self):
        p = small_params()
        q = p.permuted([1, 0])
        self.assertAlmostEqual(q.transmat[0, 0], p.transmat[1, 1])
        self.assertAlmostEqual(q.transmat[0, 1], p.transmat[1, 0])

    def test_episode_rules(self):
        ind = np.zeros(120, dtype=bool)
        ind[10:30] = True
        ind[32:40] = True          # 2-day gap closed -> one episode 10..40
        ind[50:55] = True          # 5 days: too short
        ind[70:85] = True          # separate episode (gap >= 21 after closing? 40 -> 70 = 30)
        eps = hd.episodes_from_indicator(ind)
        self.assertEqual(eps, [(10, 40), (70, 85)])

    def test_onset_lags(self):
        p = np.zeros(100)
        p[25:] = 0.9
        out = hd.onset_lags(p, [20], 0.5, 63)
        self.assertEqual(out[0]["lag_days"], 5)
        out = hd.onset_lags(p, [40], 0.5, 63)
        self.assertEqual(out[0]["lag_days"], -15)            # stress run began 15 days before the onset
        self.assertTrue(out[0]["already_in_stress"])
        q = np.full(100, 0.9)
        self.assertEqual(hd.onset_lags(q, [80], 0.5, 63)[0]["lag_days"], -63)   # capped

    def test_paired_delay(self):
        std = np.zeros(200)
        std[50:90] = 0.9
        sticky = np.zeros(200)
        sticky[53:96] = 0.9
        r = hd.paired_delay(std, sticky)
        self.assertEqual(r["n_standard_episodes"], 1)
        self.assertEqual(r["median_entry_lag"], 3)
        self.assertEqual(r["median_exit_lag"], 6)

    def test_selection_prefers_parsimony_and_respects_switch_rule(self):
        idx = pd.bdate_range("2016-01-01", periods=300)
        rng = np.random.default_rng(0)
        base = rng.normal(size=300)
        scores = pd.DataFrame({"a2": base, "b3": base + 0.01 + rng.normal(scale=0.5, size=300),
                               "s2": base + 0.005 + rng.normal(scale=0.5, size=300)}, index=idx)
        comp = pd.DataFrame({"model_id": ["a2", "b3", "s2"], "family": ["standard", "standard", "sticky"],
                             "k": [2, 3, 2], "covariance_type": "diag", "prior_weight": [0, 0, 0.3],
                             "validation_avg_loglik": scores.mean().to_numpy(),
                             "validation_switches_per_year": [10.0, 8.0, 9.0]})
        elig = pd.DataFrame({"model_id": ["a2", "b3", "s2"], "eligible": True})
        det = pd.DataFrame({"model_id": ["a2", "b3", "s2"], "reference": "factor_stress_score", "split": "pooled",
                            "median_lag_p50": [1.0, 1.0, 1.0], "undetected_share_p50": [0.0, 0.0, 0.0]})
        sel = hd.select_model(comp, elig, scores, det)
        self.assertEqual(sel["selected_k"], 2)
        self.assertEqual(sel["selected"], "a2")        # sticky cuts switching by only 10% (< 20%)
        comp.loc[comp["model_id"] == "s2", "validation_switches_per_year"] = 5.0
        sel = hd.select_model(comp, elig, scores, det)
        self.assertEqual(sel["selected"], "s2")
        det.loc[det["model_id"] == "s2", "median_lag_p50"] = 9.0       # delay budget exceeded
        sel = hd.select_model(comp, elig, scores, det)
        self.assertEqual(sel["selected"], "a2")


class DataAccessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_loader_refuses_and_never_reads_test(self):
        idx = {"train": pd.bdate_range("2010-01-01", periods=30), "validation": pd.bdate_range("2011-01-03", periods=10),
               "test": pd.bdate_range("2012-01-02", periods=10)}
        for split, i in idx.items():
            pd.DataFrame(np.zeros((len(i), 5)), index=pd.Index(i, name="date"), columns=FEATURE_COLUMNS) \
                .to_csv(self.tmp / f"{split}_features.csv")
            pd.DataFrame(np.zeros((len(i), 2)), index=pd.Index(i, name="date"), columns=["SPY", "IEF"]) \
                .to_csv(self.tmp / f"{split}_returns.csv")
        (self.tmp / "split_manifest.json").write_text(json.dumps({"split_files": {}}))
        with self.assertRaises(ValueError):
            rx.load_stage4_data(self.tmp, ("train", "validation", "test"))
        read = []
        original = pd.read_csv
        with mock.patch("pandas.read_csv", side_effect=lambda p, *a, **k: (read.append(str(p)), original(p, *a, **k))[1]):
            data = rx.load_stage4_data(self.tmp)
        self.assertFalse(any("test_" in Path(p).name for p in read))
        self.assertFalse(any(f.startswith("test_") for f in data.files_read))

    def test_composition_reader_stops_at_train_end(self):
        from regime_radar.factors import COMPONENTS
        cols = [f"{c.factor}__{c.name}" for c in COMPONENTS if c.in_composite]
        path = self.tmp / "factor_components.csv"
        lines = ["date," + ",".join(cols)]
        dates = pd.bdate_range("2010-01-01", periods=10)
        for i, d in enumerate(dates):
            vals = ["1.0"] * len(cols)
            if i < 4:
                vals[0] = ""                       # first component appears on the 5th day
            lines.append(f"{d.date()}," + ",".join(vals))
        lines.append("2011-01-03," + ",".join(["1.0"] * len(cols)))
        lines.append("THIS ROW WOULD FAIL TO PARSE")  # after the train end: must never be parsed
        path.write_text("\n".join(lines))
        start, note = hd.composition_stable_from(path, dates[-1], dates[0])
        self.assertEqual(start, dates[4])


class MetadataTests(unittest.TestCase):
    def test_model_metadata_fields(self):
        x, _ = two_state_sequence(400, 31)
        idx = pd.bdate_range("2008-01-01", periods=300)
        c = hd.run_candidate(hm.HMMSpec("sticky", 2, "diag", 0.3), x[:300], x[300:], (1, 2), np.zeros(5), np.ones(5),
                             x[:300])
        meta = hd.model_metadata(c, {"protocol_version": "v"}, {"applied": True}, idx, {"train_features.csv": "abc"},
                                 hm.STAGE5_CODE_VERSION)
        for key in ("model_family", "n_states", "covariance_type", "persistence_prior", "random_seed",
                    "train_first_date", "train_last_date", "factor_columns", "preprocessing", "model_version",
                    "input_sha256", "parameters"):
            self.assertIn(key, meta)
        self.assertAlmostEqual(meta["persistence_prior"]["kappa"], 0.3 * 299 / 2)
        restored = hm.HMMParams.from_dict(json.loads(json.dumps(meta["parameters"])))
        np.testing.assert_allclose(restored.transmat, c.reference.params.transmat)

    def test_versions_match_script(self):
        from regime_radar import hmm_calibration as hcal
        from regime_radar import hmm_report as hr
        from regime_radar import hmm_robustness as hrob
        versions = {m.STAGE5_CODE_VERSION for m in (hm, hf, hd, hr, hcal, hrob)}
        self.assertEqual(len(versions), 1)
        script = (ROOT / "scripts" / "07_fit_hmm.py").read_text()
        self.assertIn(f'STAGE5_CODE_VERSION = "{versions.pop()}"', script)


def simulate_hmm(params: hm.HMMParams, n: int, seed: int, ar: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Draw from a Gaussian HMM; ar > 0 adds within-state autocorrelation (violates the HMM assumption)."""
    rng = np.random.default_rng(seed)
    z = np.zeros(n, dtype=int)
    z[0] = rng.choice(params.k, p=params.startprob)
    for t in range(1, n):
        z[t] = rng.choice(params.k, p=params.transmat[z[t - 1]])
    sd = np.sqrt(params.covars)
    e = np.zeros((n, params.means.shape[1]))
    for t in range(n):
        shock = rng.normal(size=params.means.shape[1]) * np.sqrt(1 - ar ** 2)
        e[t] = shock if t == 0 else ar * e[t - 1] + shock
    return params.means[z] + e * sd[z], z


TRUE = hm.HMMParams(np.array([0.5, 0.5]), np.array([[0.98, 0.02], [0.03, 0.97]]),
                    np.array([[0.5, 0.0, 0.0, -0.5, -0.5], [-1.0, 0.0, 0.0, 1.5, 1.5]]),
                    np.full((2, 5), 0.6), "diag")


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        from regime_radar import hmm_calibration as hcal
        self.hcal = hcal

    def test_correct_model_is_calibrated(self):
        x, _ = simulate_hmm(TRUE, 3000, 1)
        res = hf.forward_filter(TRUE, x)
        s = self.hcal.pit_summary(self.hcal.predictive_pit(TRUE, res, x), FEATURE_COLUMNS)
        allf = s[s["factor"] == "all factors"].iloc[0]
        self.assertAlmostEqual(allf["coverage_90"], 0.90, delta=0.02)
        self.assertLess(abs(allf["lag1_autocorrelation"]), 0.1)
        pred, filt = res.predicted[:, 1], res.filtered[:, 1]
        r = self.hcal.reliability_summary(pred, filt, res.filtered, TRUE.transmat, res.filtered.argmax(1))
        self.assertLess(abs(r["nw_t"]), 3)
        self.assertAlmostEqual(r["calibration_slope"], 1.0, delta=0.1)
        ac = self.hcal.residual_autocorrelation(TRUE, x, res.filtered.argmax(1), FEATURE_COLUMNS)
        self.assertLess(abs(ac["mean"]), 0.1)

    def test_within_state_dependence_is_detected(self):
        x, _ = simulate_hmm(TRUE, 3000, 2, ar=0.95)
        res = hf.forward_filter(TRUE, x)
        ac = self.hcal.residual_autocorrelation(TRUE, x, res.filtered.argmax(1), FEATURE_COLUMNS)
        self.assertGreater(ac["mean"], 0.8)
        s = self.hcal.pit_summary(self.hcal.predictive_pit(TRUE, res, x), FEATURE_COLUMNS)
        self.assertGreater(s[s["factor"] == "all factors"]["lag1_autocorrelation"].iloc[0], 0.5)

    def test_inflation_test_rejects_correct_and_fixes_overconfident_model(self):
        x, _ = simulate_hmm(TRUE, 3000, 3)
        x_tr, x_va = x[:1992], x[1992:]
        idx = pd.bdate_range("2016-01-01", periods=len(x_va))
        _, ok = self.hcal.inflation_curve(TRUE, [1], x_tr, x_va, idx)
        self.assertFalse(ok["adopted"])
        narrow = TRUE.copy()
        narrow.covars = TRUE.covars / 4                 # deliberately overconfident emissions
        curve, fix = self.hcal.inflation_curve(narrow, [1], x_tr, x_va, idx)
        self.assertGreaterEqual(fix["chosen_inflation_on_val_a"], 3.0)
        self.assertTrue(fix["adopted"])
        np.testing.assert_allclose(hm.inflate_covariance(TRUE, 2.0).covars, TRUE.covars * 2)
        self.assertEqual(len(curve), len(self.hcal.CALIBRATION_RULES["inflation_grid"]))

    def test_inflation_choice_uses_validation_a_only(self):
        """Changing 2018-2019 data cannot change the chosen inflation (it is picked on 2016-2017)."""
        x, _ = simulate_hmm(TRUE, 3000, 4)
        x_tr, x_va = x[:1992], x[1992:]
        idx = pd.bdate_range("2016-01-01", periods=len(x_va))
        later = np.asarray(idx >= pd.Timestamp("2018-01-01"))
        shocked = x_va.copy()
        shocked[later] = shocked[later] * 3
        a = self.hcal.inflation_curve(TRUE, [1], x_tr, x_va, idx)[1]["chosen_inflation_on_val_a"]
        b = self.hcal.inflation_curve(TRUE, [1], x_tr, shocked, idx)[1]["chosen_inflation_on_val_a"]
        self.assertEqual(a, b)


class RobustnessTests(unittest.TestCase):
    def setUp(self):
        from regime_radar import hmm_robustness as hrob
        self.hrob = hrob

    def test_phase_consistency(self):
        idx = pd.bdate_range("2016-01-01", "2019-12-31")
        base = pd.Series(np.zeros(len(idx)), index=idx)
        big = base.copy()
        big[idx.year == 2016] = 3.0                      # wins one year only
        big[idx.year != 2016] = -0.1
        r = self.hrob.phase_consistency(pd.DataFrame({"k3": big, "k2": base}), "k3", "k2")
        self.assertEqual(r["positive_years"], 1)
        self.assertFalse(r["consistent"])

    def test_selection_falls_back_when_gain_comes_from_one_year(self):
        idx = pd.bdate_range("2016-01-01", "2019-12-31")
        rng = np.random.default_rng(0)
        k2 = pd.Series(rng.normal(scale=0.1, size=len(idx)), index=idx)
        k3 = k2 + np.where(idx.year == 2016, 3.0, -0.05)
        scores = pd.DataFrame({"a2": k2, "b3": k3})
        comp = pd.DataFrame({"model_id": ["a2", "b3"], "family": "standard", "k": [2, 3], "covariance_type": "diag",
                             "prior_weight": 0.0, "validation_avg_loglik": scores.mean().to_numpy(),
                             "validation_switches_per_year": [2.0, 3.0]})
        elig = pd.DataFrame({"model_id": ["a2", "b3"], "eligible": True})
        det = pd.DataFrame({"model_id": ["a2", "b3"], "reference": "factor_stress_score", "split": "pooled",
                            "median_lag_p50": 0.0, "undetected_share_p50": 0.0})
        sel = hd.select_model(comp, elig, scores, det)
        self.assertTrue(sel["phase_fallback"])
        self.assertEqual(sel["selected"], "a2")
        scores["b3"] = k2 + 0.5                          # consistent gain in every year -> K=3 kept
        comp["validation_avg_loglik"] = scores.mean().to_numpy()
        sel = hd.select_model(comp, elig, scores, det)
        self.assertFalse(sel["phase_fallback"])
        self.assertEqual(sel["selected"], "b3")

    def test_window_definitions(self):
        idx = pd.bdate_range("2007-10-30", "2015-12-31")
        w = self.hrob.window_definitions(idx, pd.Timestamp("2011-11-29"))
        self.assertEqual(w["start_plus_2y"][0], pd.Timestamp("2009-10-30"))
        self.assertEqual(w["end_minus_2y"][1], pd.Timestamp("2013-12-31"))
        self.assertIn("post_composition", w)
        self.assertNotIn("post_composition", self.hrob.window_definitions(idx, None))
        for start, end in w.values():
            self.assertGreaterEqual(start, idx.min())
            self.assertLessEqual(end, idx.max())

    def test_gmm_rebuilt_without_unpickling_other_versions(self):
        import warnings
        x, _ = two_state_sequence(600, 5)
        feats = pd.DataFrame(x, columns=FEATURE_COLUMNS, index=pd.bdate_range("2008-01-01", periods=600))
        tmp = Path(tempfile.mkdtemp())
        try:
            (tmp / "gmm_diag_k2.json").write_text(json.dumps({"software": {"scikit-learn": "0.0.1"}}))
            (tmp / "gmm_diag_k2.pkl").write_bytes(b"not a pickle")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                out, info, note = self.hrob.rebuild_gmm_reference(x[:400], x[400:], feats.iloc[:400], tmp)
            self.assertFalse([w for w in caught if "version" in str(w.message).lower()])
            self.assertIn("not unpickled", note)
            self.assertEqual(len(out["validation"]["score"]), 200)
            again, _, _ = self.hrob.rebuild_gmm_reference(x[:400], x[400:], feats.iloc[:400], tmp)
            np.testing.assert_allclose(again["validation"]["score"], out["validation"]["score"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
