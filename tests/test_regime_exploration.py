"""Stage 4 tests (synthetic data only): train-only fitting, no test-split access, valid outputs, stability tools.

A pass here shows the workflow behaves as designed; it says nothing about results on the real dataset.

Run:  python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

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

from regime_radar import regime_decision as rdec  # noqa: E402
from regime_radar import regime_diagnostics as rd  # noqa: E402
from regime_radar import regime_exploration as rx  # noqa: E402
from regime_radar.dataset import FEATURE_COLUMNS, file_sha256  # noqa: E402
from regime_radar.factors import COMPONENTS  # noqa: E402

ETFS = ["SPY", "EFA", "EEM", "IEF", "TIP", "HYG", "GLD", "DBC", "BIL"]
FACTORS_PY = ROOT / "src" / "regime_radar" / "factors.py"

# Frozen stage-2 factor definitions: (name, factor, sign, window, in_composite)
FROZEN_COMPONENTS = [
    ("spy_mom_63d", "equity_trend", 1, 252, True), ("spy_mom_12_1", "equity_trend", 1, 252, True),
    ("sector_breadth_200d", "equity_trend", 1, 252, True), ("equal_vs_cap_weight_63d", "equity_trend", 1, 252, True),
    ("small_vs_large_63d", "equity_trend", 1, 252, True), ("ust2y_change_63d", "interest_rate", 1, 252, True),
    ("ust10y_change_63d", "interest_rate", 1, 252, True), ("curve_10y2y", "interest_rate", -1, 252, True),
    ("ust10y_realized_vol_21d", "interest_rate", 1, 252, True), ("cpi_yoy", "inflation", 1, 756, True),
    ("core_cpi_yoy", "inflation", 1, 756, True), ("breakeven_5y", "inflation", 1, 252, True),
    ("tip_vs_ief_63d", "inflation", 1, 252, True), ("commodity_mom_126d", "inflation", 1, 252, True),
    ("vix_log", "volatility", 1, 252, True), ("spy_realized_vol_21d", "volatility", 1, 252, True),
    ("spy_downside_vol_63d", "volatility", 1, 252, True), ("vix_term_structure", "volatility", 1, 252, True),
    ("risk_asset_correlation_63d", "volatility", 1, 252, True), ("baa_10y_spread", "liquidity_credit", 1, 252, True),
    ("hyg_vs_lqd_63d", "liquidity_credit", -1, 252, True), ("nfci", "liquidity_credit", 1, 252, True),
    ("hyg_illiquidity_21d", "liquidity_credit", 1, 252, True), ("hy_oas", "liquidity_credit", 1, 252, False),
]

REGIME_MEANS = np.array([
    [0.6, -0.2, 0.0, -0.6, -0.5],    # calm
    [-1.5, -1.0, -1.0, 2.5, 2.0],    # stress
    [0.0, 1.5, 1.5, 0.3, 0.2],       # rates / inflation
])


def synthetic_regimes(n: int, seed: int, stay: float = 0.97) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    states = np.zeros(n, dtype=int)
    for t in range(1, n):
        states[t] = states[t - 1] if rng.random() < stay else rng.integers(0, 3)
    x = REGIME_MEANS[states] + rng.normal(0, 0.45, (n, 5)) * np.array([1.0, 0.8, 0.8, 1.5, 1.2])
    return x, states


def write_splits(directory: Path, n_train: int = 900, n_val: int = 400, n_test: int = 300) -> Path:
    dates = pd.bdate_range("2010-01-04", periods=n_train + n_val + n_test, name="date")
    x, _ = synthetic_regimes(len(dates), seed=11)
    rng = np.random.default_rng(12)
    returns = rng.normal(0, 0.01, (len(dates), 9))
    parts = {"train": slice(0, n_train), "validation": slice(n_train, n_train + n_val),
             "test": slice(n_train + n_val, None)}
    files = {}
    for name, part in parts.items():
        f = pd.DataFrame(x[part], index=dates[part], columns=FEATURE_COLUMNS)
        r = pd.DataFrame(returns[part], index=dates[part], columns=ETFS)
        f.to_csv(directory / f"{name}_features.csv", date_format="%Y-%m-%d")
        r.to_csv(directory / f"{name}_returns.csv", date_format="%Y-%m-%d")
    for path in directory.glob("*.csv"):
        files[path.name] = file_sha256(path)
    manifest = {
        "protocol_version": "ARR-stage3-protocol-v1.2", "feature_columns": FEATURE_COLUMNS,
        "splits": [{"name": n, "rows": len(dates[p])} for n, p in parts.items()],
        "split_files": files,
        "feature_provenance": {"factor_construction": {"module_sha256": file_sha256(FACTORS_PY)}},
    }
    (directory / "split_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return directory


class SplitDirMixin:
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        write_splits(cls.tmp)
        cls.data = rx.load_stage4_data(cls.tmp)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)


# --------------------------------------------------------------------------------------
# Data access and protocol
# --------------------------------------------------------------------------------------
class DataAccessTests(SplitDirMixin, unittest.TestCase):
    def test_test_split_is_never_read(self):
        opened = []
        real = pd.read_csv
        with mock.patch("pandas.read_csv", side_effect=lambda p, *a, **k: opened.append(str(p)) or real(p, *a, **k)):
            data = rx.load_stage4_data(self.tmp)
            rx.run_exploration(data, ks=(2,), seeds=rx.SEEDS[:1])
        self.assertTrue(opened)
        self.assertFalse([p for p in opened if "test_" in Path(p).name], opened)
        self.assertNotIn("test_features.csv", data.files_read)

    def test_requesting_test_split_raises(self):
        with self.assertRaises(ValueError):
            rx.load_stage4_data(self.tmp, splits=("train", "validation", "test"))
        with self.assertRaises(ValueError):
            rx.load_stage4_data(self.tmp, splits=("train", "test"))

    def test_hashes_and_input_checks(self):
        self.assertTrue(all(self.data.hash_status.values()))
        checks = rx.input_checks(self.data)
        self.assertTrue(all(c.status == rx.PASS for c in checks), checks)

    def test_frozen_factor_definitions(self):
        self.assertEqual(rx.frozen_factor_check(self.data.manifest, FACTORS_PY).status, rx.PASS)
        tampered = json.loads(json.dumps(self.data.manifest))
        tampered["feature_provenance"]["factor_construction"]["module_sha256"] = "0" * 64
        self.assertEqual(rx.frozen_factor_check(tampered, FACTORS_PY).status, rx.FAIL)
        current = [(c.name, c.factor, c.sign, c.window, c.in_composite) for c in COMPONENTS]
        self.assertEqual(current, FROZEN_COMPONENTS)

    def test_candidate_k_validation(self):
        self.assertEqual(rx.validate_candidate_ks((4, 2, 3), 900), (2, 3, 4))
        for bad in ((1, 2), (2, 2), (2.5,), (), (True,), (2, 40)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                rx.validate_candidate_ks(bad, 900)


# --------------------------------------------------------------------------------------
# Train-only fitting
# --------------------------------------------------------------------------------------
class FittingTests(SplitDirMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.result = rx.run_exploration(cls.data, ks=(2, 3), seeds=rx.SEEDS[:3])

    def test_scaler_fitted_on_train_only(self):
        tr = self.data.train_features[FEATURE_COLUMNS]
        both = pd.concat([tr, self.data.validation_features[FEATURE_COLUMNS]])
        self.assertTrue(self.result.scaler.apply)
        np.testing.assert_allclose(self.result.scaler.mean_, tr.mean(), atol=1e-12)
        np.testing.assert_allclose(self.result.scaler.scale_, tr.std(ddof=0), atol=1e-12)
        self.assertFalse(np.allclose(self.result.scaler.mean_, both.mean(), atol=1e-6))
        self.assertEqual(self.result.scaler.fitted_on["rows"], len(tr))

    def test_validation_never_enters_fitted_parameters(self):
        altered = rx.Stage4Data(self.data.train_features, self.data.validation_features * 7 + 3,
                                self.data.train_returns, self.data.validation_returns, self.data.manifest)
        other = rx.run_exploration(altered, ks=(2, 3), seeds=rx.SEEDS[:3])
        for model_id, fit in self.result.reference.items():
            # identical up to floating-point rounding of multi-threaded sums
            np.testing.assert_allclose(rx._parameters(fit.model), rx._parameters(other.reference[model_id].model),
                                       rtol=0, atol=rx.PARAMETER_TOLERANCE)
            np.testing.assert_array_equal(fit.train_labels, other.reference[model_id].train_labels)

    def test_labels_and_probabilities_valid(self):
        for fit in self.result.reference.values():
            for labels in (fit.train_labels, fit.validation_labels):
                self.assertTrue(((labels >= 0) & (labels < fit.spec.k)).all())
            if fit.spec.family == "gmm":
                for proba in (fit.train_proba, fit.validation_proba):
                    self.assertTrue(np.isfinite(proba).all())
                    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)
                    np.testing.assert_array_equal(proba.argmax(axis=1), fit.validation_labels
                                                  if proba is fit.validation_proba else fit.train_labels)

    def test_seed_reproducibility(self):
        x_tr = self.result.scaler.transform(self.data.train_features[FEATURE_COLUMNS])
        for spec in (rx.CandidateSpec("kmeans", 3), rx.CandidateSpec("gmm", 3, "full")):
            a, *_ = rx.fit_model(spec, x_tr, rx.SEEDS[0])
            b, *_ = rx.fit_model(spec, x_tr, rx.SEEDS[0])
            np.testing.assert_allclose(rx._parameters(a), rx._parameters(b), rtol=0, atol=rx.PARAMETER_TOLERANCE)
            np.testing.assert_array_equal(a.predict(x_tr), b.predict(x_tr))

    def test_rounding_noise_passes_but_real_dependence_fails(self):
        """Regression test for the real-data run: 1e-16 differences must PASS, data-driven changes must FAIL."""
        sel = self.result.selection
        ids = [sel["gmm_preferred"], sel["kmeans_preferred"]]
        real_fit = rx.fit_candidate

        def jitter(spec, x_train, x_validation, train_features, seed):
            fit = real_fit(spec, x_train, x_validation, train_features, seed)
            if hasattr(fit.model, "cluster_centers_"):
                fit.model.cluster_centers_ = fit.model.cluster_centers_ + 2e-16
            else:
                fit.model.means_ = fit.model.means_ + 2e-16
            return fit

        with mock.patch.object(rx, "fit_candidate", jitter):
            status = {c.name: c.status for c in rx.leakage_checks(self.data, self.result, ids)}
        self.assertEqual(status["Same seed reproduces the same fit"], rx.PASS)
        self.assertEqual(status["Fitted parameters independent of validation data"], rx.PASS)

        def leaky(spec, x_train, x_validation, train_features, seed):    # uses validation in the fit
            fit = real_fit(spec, np.vstack([x_train, x_validation]), x_validation,
                           pd.concat([train_features, train_features.iloc[:len(x_validation)]]), seed)
            fit.train_labels = fit.train_labels[:len(x_train)]
            return fit

        with mock.patch.object(rx, "fit_candidate", leaky):
            status = {c.name: c.status for c in rx.leakage_checks(self.data, self.result, ids)}
        self.assertEqual(status["Fitted parameters independent of validation data"], rx.FAIL)

    def test_recovers_synthetic_regimes(self):
        _, states = synthetic_regimes(900 + 400 + 300, seed=11)
        fit = self.result.reference["gmm_full_k3"]
        from sklearn.metrics import adjusted_rand_score
        self.assertGreater(adjusted_rand_score(states[:900], fit.train_labels), 0.7)
        self.assertGreater(adjusted_rand_score(states[900:1300], fit.validation_labels), 0.6)

    def test_canonical_order_by_stress(self):
        fit = self.result.reference["kmeans_k3"]
        prof = rd.factor_profile(self.data.train_features[FEATURE_COLUMNS], fit.train_labels, 3)
        score = prof["volatility"] + prof["liquidity_credit"] - prof["equity_trend"]
        self.assertTrue(score.is_monotonic_increasing)

    def test_selection_and_outputs_identify_model_and_period(self):
        sel = self.result.selection
        self.assertIn(sel["gmm_preferred"], self.result.reference)
        self.assertIn(sel["kmeans_preferred"], self.result.reference)
        fit = self.result.reference[sel["gmm_preferred"]]
        meta = rx.model_metadata(fit, self.data, self.result.scaler)
        self.assertEqual(meta["model_type"], "GaussianMixture")
        self.assertEqual(meta["n_states"], fit.spec.k)
        self.assertEqual(meta["random_seed"], rx.SEEDS[0])
        self.assertEqual(meta["training_last_date"], str(self.data.train_features.index.max().date()))
        self.assertEqual(meta["factor_columns"], FEATURE_COLUMNS)
        checks = rx.leakage_checks(self.data, self.result, [sel["gmm_preferred"], sel["kmeans_preferred"]])
        self.assertTrue(all(c.status == rx.PASS for c in checks), checks)
        manifest = rx.stage4_manifest(self.data, self.result, checks, [])
        self.assertIn("candidate_k_range_for_hmm", manifest["preferred"])
        self.assertTrue(set(sel["candidate_k_range"]) <= {2, 3})
        self.assertFalse(manifest["test_data_loaded"])
        self.assertEqual(manifest["validation_period"]["rows"], len(self.data.validation_features))
        tr, va, ptr, pva = rx.assignments(self.result, self.data)
        self.assertTrue(va.index.max() <= self.data.validation_features.index.max())
        self.assertEqual(set(tr.columns), set(self.result.reference))
        np.testing.assert_allclose(pva.filter(like="gmm_full_k3__").sum(axis=1), 1.0, atol=1e-9)

    def test_elbow_curve_train_only_and_monotone(self):
        e = self.result.elbow
        self.assertEqual(e["k"].tolist(), list(rx.ELBOW_K))
        self.assertTrue((e["train_inertia"].diff().dropna() <= 1e-9).all())       # never increases with K
        x_tr = self.result.scaler.transform(self.data.train_features[FEATURE_COLUMNS])
        self.assertAlmostEqual(e.loc[e["k"] == 1, "train_inertia"].iloc[0], ((x_tr - x_tr.mean(0)) ** 2).sum(), 6)
        self.assertEqual(int(e["kneedle_elbow"].sum()), 1)
        noisy_val = self.result.scaler.transform(self.data.validation_features[FEATURE_COLUMNS]) * 3
        e2, k2, _ = rx.elbow_curve(x_tr, noisy_val, ks=rx.ELBOW_K)
        np.testing.assert_allclose(e2["train_inertia"], e["train_inertia"], rtol=1e-9)   # validation never fitted
        self.assertEqual(k2, self.result.elbow_k)

    def test_elbow_primary_range_and_sensitivity(self):
        e, sens = self.result.elbow, self.result.elbow_sensitivity
        self.assertEqual(rx.ELBOW_PRIMARY_MAX_K, max(rx.CANDIDATE_K))
        self.assertEqual(sens["max_k"].tolist(), list(range(4, max(rx.ELBOW_K) + 1)))
        self.assertEqual(int(sens["primary_range"].sum()), 1)
        primary = e[e["k"] <= rx.ELBOW_PRIMARY_MAX_K]
        self.assertEqual(self.result.elbow_k, rx.kneedle_elbow(primary["k"], primary["train_inertia"]))
        self.assertEqual(int(e.loc[e["kneedle_elbow"], "k"].iloc[0]), self.result.elbow_k)
        full_col = f"kneedle_elbow_range_1_{max(rx.ELBOW_K)}"
        self.assertEqual(int(e.loc[e[full_col], "k"].iloc[0]), rx.kneedle_elbow(e["k"], e["train_inertia"]))
        self.assertTrue((sens["margin_vs_runner_up"] >= 0).all())

    def test_k3_alternative_assessment(self):
        profiles = rx.profiles_table(self.result, self.data)
        alt = rx.alternative_state_assessment(profiles, self.result.stability)
        self.assertEqual(set(alt["model_id"]), {"kmeans_k3", "gmm_full_k3", "gmm_diag_k3"})
        row = alt.set_index("model_id").loc["kmeans_k3"]
        # the synthetic third regime is a rates / inflation state, distinct from the stress state
        self.assertGreaterEqual(max(row["train_interest_rate"], row["train_inflation"]), 0.5)
        self.assertEqual(row["accepted"], all(row[c] for c in ("c1_rates_or_inflation_state", "c2_validation_share",
                                                               "c3_profile_stable", "c4_seed_stability")))

    def test_collapsed_validation_state_is_flagged_not_excluded(self):
        table = rx.eligibility(self.result)
        self.assertIn("validation_collapsed_states", table.columns)
        self.assertNotIn("collapsed", " ".join(table["ineligible_reasons"]))


# --------------------------------------------------------------------------------------
# Recorded stage-4 decision (K=2)
# --------------------------------------------------------------------------------------
class VersionTests(unittest.TestCase):
    def test_stage4_modules_share_one_version(self):
        from regime_radar import regime_report as rr
        script = (ROOT / "scripts" / "06_explore_regimes.py").read_text()
        versions = {m.STAGE4_CODE_VERSION for m in (rx, rd, rr, rdec)}
        self.assertEqual(len(versions), 1)
        self.assertIn(f'STAGE4_CODE_VERSION = "{versions.pop()}"', script)


class DecisionTests(SplitDirMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.result = rx.run_exploration(cls.data, ks=(2, 3), seeds=rx.SEEDS[:2])
        cls.profiles = rx.profiles_table(cls.result, cls.data)

    def test_decision_record_is_complete(self):
        d = rdec.STAGE4_DECISION
        self.assertEqual(d["primary_k"], 2)
        self.assertEqual((d["reference_model"], d["baseline_model"]), ("gmm_diag_k2", "kmeans_k2"))
        self.assertEqual(d["alternatives_for_hmm"], [3, 4])
        self.assertEqual(set(d["state_names"]), {"S1", "S2"})
        self.assertIn("test split was not read", d["data_used"])

    def test_decision_checks_flag_disagreement(self):
        checks = rdec.decision_checks(self.result, self.profiles)
        self.assertEqual(len(checks), 4)
        self.assertTrue(all(c.status in (rdec.PASS, rdec.WARN) for c in checks))
        wrong = {**rdec.STAGE4_DECISION, "reference_model": "gmm_full_k2", "alternatives_for_hmm": [5]}
        status = {c.name: c.status for c in rdec.decision_checks(self.result, self.profiles, wrong)}
        self.assertEqual(status["Decision: reference model equals the rule-based GMM preference"], rdec.WARN)
        self.assertEqual(status["Decision: primary and alternative K inside the validation-supported range"], rdec.WARN)

    def test_selected_states_table(self):
        states = rdec.selected_states(self.result, self.data)
        self.assertEqual(len(states), len(self.data.train_features) + len(self.data.validation_features))
        self.assertLessEqual(states.index.max(), self.data.validation_features.index.max())   # no test dates
        self.assertEqual(set(states["split"]), {"train", "validation"})
        np.testing.assert_allclose(states[["p_S1", "p_S2"]].sum(axis=1), 1.0, atol=1e-9)
        self.assertEqual(set(states["state_name"]), set(rdec.STAGE4_DECISION["state_names"].values()))
        self.assertTrue(((states["normalized_entropy"] >= -1e-12) & (states["normalized_entropy"] <= 1 + 1e-12)).all())

    def test_summary_and_evidence(self):
        summary = rdec.summary_table(self.result, self.data, self.profiles)
        self.assertEqual(len(summary), 4)
        self.assertAlmostEqual(summary.groupby("split")["share"].sum().loc["train"], 1.0)
        assets = rx.asset_table(self.result, self.data, ["gmm_diag_k2"])
        lines = rdec.evidence_statements(summary, assets, "gmm_diag_k2")
        self.assertTrue(any("mechanical" in x for x in lines))

    def test_evidence_detects_sign_reversal(self):
        summary = pd.DataFrame([
            {"split": "train", "state": s, "share": sh, "equity_trend": e, "interest_rate": 0.1, "inflation": i,
             "volatility": vo, "liquidity_credit": c}
            for s, sh, e, i, vo, c in (("S1", 0.65, 0.2, 0.0, -0.5, -0.6), ("S2", 0.35, -1.4, -1.0, 1.5, 1.2))])
        rows = [{"split": sp, "horizon": h, "state": st, "etf": etf, "n_days": 300, "annualized_mean_pct": val}
                for sp, h, st, etf, val in (("train", "next_session", "S2", "SPY", -11.0),
                                            ("validation", "next_session", "S2", "SPY", 17.0),
                                            ("train", "same_day", "S1", "IEF", 0.6),
                                            ("train", "same_day", "S2", "IEF", 14.5))]
        assets = pd.DataFrame(rows).assign(model_id="gmm_diag_k2")
        text = " ".join(rdec.evidence_statements(summary, assets, "gmm_diag_k2"))
        self.assertIn("a risk-off profile", text)
        self.assertIn("flight to quality", text)
        self.assertIn("sign reverses", text)

    def test_transition_matrix_and_episodes(self):
        labels = np.array([0, 0, 0, 1, 1, 0, 0, 1, 1, 1])
        tm = rdec.transition_matrix(labels, 2)
        np.testing.assert_allclose(tm.sum(axis=1), 1.0)
        self.assertAlmostEqual(tm.loc["S1", "S2"], 2 / 5)
        self.assertAlmostEqual(tm.loc["S2", "S2"], 3 / 4)
        idx = pd.bdate_range("2015-01-01", periods=30)
        states = pd.DataFrame({"split": "train", "state": ["S1"] * 5 + ["S2"] * 12 + ["S1"] * 3 + ["S2"] * 4
                               + ["S1"] * 6, "p_S2": 0.9, "models_agree": True}, index=idx)
        eps = rdec.episodes(states, min_days=10)
        self.assertEqual(len(eps), 1)
        self.assertEqual((eps["start"].iloc[0], eps["trading_days"].iloc[0]), (idx[5].date().isoformat(), 12))


# --------------------------------------------------------------------------------------
# Diagnostics helpers
# --------------------------------------------------------------------------------------
class DiagnosticsTests(unittest.TestCase):
    def test_population_and_durations(self):
        labels = np.array([0, 0, 1, 1, 1, 0, 2, 2, 0, 0])
        np.testing.assert_allclose(rd.population(labels, 4), [0.5, 0.3, 0.2, 0.0])
        runs = rd.run_lengths(labels)
        self.assertEqual(runs["length"].tolist(), [2, 3, 1, 2, 2])
        self.assertEqual(runs["state"].tolist(), [0, 1, 0, 2, 0])
        stats = rd.switching_stats(labels)
        self.assertEqual(stats["n_switches"], 4)
        self.assertEqual(stats["median_duration_days"], 2.0)
        self.assertAlmostEqual(stats["switches_per_year"], 4 / 10 * 252)
        self.assertAlmostEqual(stats["share_single_day_runs"], 0.2)
        counts = rd.transition_counts(labels, 3)
        self.assertEqual(counts.sum(), 9)
        self.assertEqual(counts[0, 0], 2)

    def test_label_permutation_does_not_corrupt_agreement(self):
        rng = np.random.default_rng(1)
        labels = rng.integers(0, 4, 500)
        permuted = np.array([2, 0, 3, 1])[labels]
        agree = rd.pairwise_agreement([labels, permuted], 4)
        self.assertAlmostEqual(agree["mean_ari"], 1.0)
        self.assertAlmostEqual(agree["mean_aligned_agreement"], 1.0)
        mapping = rd.align_labels(labels, permuted, 4)
        np.testing.assert_array_equal(mapping[permuted], labels)
        noisy = permuted.copy()
        noisy[:50] = (noisy[:50] + 1) % 4
        self.assertLess(rd.pairwise_agreement([labels, noisy], 4)["mean_aligned_agreement"], 1.0)

    def test_entropy_bounds(self):
        ent = rd.normalized_entropy(np.array([[1.0, 0.0, 0.0], [1 / 3, 1 / 3, 1 / 3]]))
        np.testing.assert_allclose(ent, [0.0, 1.0], atol=1e-12)

    def test_newey_west_matches_iid_case(self):
        x = np.random.default_rng(3).normal(size=20000)
        self.assertAlmostEqual(rd.newey_west_se(x, 0), x.std() / np.sqrt(len(x)), places=6)
        persistent = np.repeat(np.random.default_rng(4).normal(size=2000), 10)
        self.assertGreater(rd.newey_west_se(persistent, 21), rd.newey_west_se(persistent, 0))

    def test_hypothesis_labels(self):
        crisis = pd.Series({"equity_trend": -1.5, "interest_rate": -1, "inflation": -1, "volatility": 2.5,
                            "liquidity_credit": 2.0})
        self.assertEqual(rd.hypothesis_label(crisis)[0], "Liquidity Crisis (hypothesis)")
        calm = pd.Series({"equity_trend": 0.6, "interest_rate": 0, "inflation": 0, "volatility": -0.5,
                          "liquidity_credit": -0.4})
        self.assertEqual(rd.hypothesis_label(calm)[0], "Growth / Risk-on (hypothesis)")
        mixed = pd.Series({"equity_trend": -0.5, "interest_rate": 1.2, "inflation": 1.0, "volatility": 0.5,
                           "liquidity_credit": 0.3})
        label, rule = rd.hypothesis_label(mixed)
        self.assertEqual(label, "Neutral")
        self.assertIn("mixed", rule)

    def test_elbow_statement_reports_sensitivity(self):
        from regime_radar import regime_report as rr
        elbow = pd.DataFrame({"k": [1, 2, 3, 4], "pct_reduction_vs_previous_k": [np.nan, 40.0, 18.0, 15.0]})
        sens = pd.DataFrame({"k_range": ["1-4", "1-10"], "max_k": [4, 10], "kneedle_elbow": [2, 3],
                             "runner_up": [3, 4], "margin_vs_runner_up": [0.05, 0.017],
                             "primary_range": [True, False]})
        text = rr.elbow_statement(elbow, sens)
        self.assertIn("K=1 to K=2 (40%)", text)
        self.assertIn("range 1-4 is K=2", text)
        self.assertIn("1-10 it is K=3", text)
        same = sens.assign(kneedle_elbow=[2, 2])
        self.assertIn("does not change", rr.elbow_statement(elbow, same))

    def test_kneedle_elbow(self):
        ks = np.arange(1, 11)
        self.assertEqual(rx.kneedle_elbow(ks, 100 / ks), 3)                  # textbook 1/K curve: Kneedle gives 3
        self.assertEqual(rx.kneedle_elbow(ks, [100, 40, 36, 33, 31, 29, 27, 25, 23, 21]), 2)   # one big drop
        self.assertEqual(rx.kneedle_elbow(ks, [100, 95, 90, 40, 37, 35, 33, 31, 30, 29]), 4)   # late drop

    def test_asset_statistics_horizons(self):
        idx = pd.bdate_range("2015-01-01", periods=6)
        returns = pd.DataFrame({"SPY": [0.01, 0.02, -0.01, 0.03, 0.0, 0.01]}, index=idx)
        labels = np.array([0, 0, 1, 1, 0, 1])
        stats = rd.asset_statistics(returns, labels, 2).set_index(["horizon", "state"])
        self.assertAlmostEqual(stats.loc[("same_day", "S1"), "mean_daily_bp"], 1e4 * np.mean([0.01, 0.02, 0.0]))
        # next_session: state_t paired with return_t+1 inside the split; the last day has no next return
        self.assertAlmostEqual(stats.loc[("next_session", "S1"), "mean_daily_bp"], 1e4 * np.mean([0.02, -0.01, 0.01]))
        self.assertEqual(stats.loc[("next_session", "S2"), "n_days"], 2)


if __name__ == "__main__":
    unittest.main()
