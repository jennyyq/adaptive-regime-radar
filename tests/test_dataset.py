"""Stage 3 tests: chronological splits, alignment, no fill, PIT audit and leakage detection.

All tests use SYNTHETIC data. A pass here shows that the checks behave as designed; it is not a
validation of the real dataset (that is what scripts/05_prepare_datasets.py does on the local files).

Run:  python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from regime_radar import dataset_validation as dv  # noqa: E402
from regime_radar.dataset import (  # noqa: E402
    SplitSpec, assign_partitions, build_manifest, build_splits, render_protocol, split_specs,
)
from regime_radar.factors import FACTORS, FactorInputs, build_factors  # noqa: E402
from regime_radar.pit import align_to_calendar  # noqa: E402
from test_factors import synthetic_inputs  # noqa: E402

ETFS = ["SPY", "EFA", "EEM", "IEF", "TIP", "HYG", "GLD", "DBC", "BIL"]
SPECS = (SplitSpec("train", "2015-01-01", "2016-06-30"),
         SplitSpec("validation", "2016-07-01", "2017-06-30"),
         SplitSpec("test", "2017-07-01", None))


def synthetic_panels():
    rng = np.random.default_rng(3)
    cal = pd.bdate_range("2014-06-02", "2018-06-29", name="date")
    features = pd.DataFrame(rng.normal(size=(len(cal), 5)), index=cal, columns=list(FACTORS))
    features.iloc[:60] = np.nan                                       # warm-up
    adj = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (len(cal), 9)), axis=0)),
                       index=cal, columns=ETFS)
    adj.loc[:"2014-12-31", "BIL"] = np.nan                            # BIL lists last -> binding fund
    prices = adj.loc["2015-01-01":]                                   # investment sample starts here
    returns = np.log(prices).diff().iloc[1:]
    return cal, features, prices, returns, adj


# --------------------------------------------------------------------------------------
# Splits and alignment
# --------------------------------------------------------------------------------------
class SplitSpecTests(unittest.TestCase):
    def test_default_specs_are_frozen(self):
        specs = split_specs()
        self.assertEqual([(s.name, s.start, s.end) for s in specs],
                         [("train", "2007-05-30", "2015-12-31"), ("validation", "2016-01-01", "2019-12-31"),
                          ("test", "2020-01-01", None)])

    def test_gap_or_overlap_rejected(self):
        with self.assertRaises(ValueError):
            split_specs({"splits": [{"name": "a", "start": "2010-01-01", "end": "2010-12-31"},
                                    {"name": "b", "start": "2011-01-05", "end": None}]})
        with self.assertRaises(ValueError):
            split_specs({"splits": [{"name": "a", "start": "2010-01-01", "end": "2010-12-31"},
                                    {"name": "b", "start": "2010-12-01", "end": None}]})

    def test_assign_partitions_inclusive_boundaries(self):
        dates = pd.DatetimeIndex(["2016-06-30", "2016-07-01", "2017-06-30", "2017-07-03"])
        labels = assign_partitions(dates, SPECS)
        self.assertEqual(list(labels), ["train", "validation", "validation", "test"])


class BuildSplitTests(unittest.TestCase):
    def setUp(self):
        self.cal, self.features, self.prices, self.returns, self.adj = synthetic_panels()

    def test_exclusive_exhaustive_chronological_aligned(self):
        result = build_splits(self.features, self.returns, SPECS, ETFS)
        frames = list(result.features.values())
        all_dates = pd.DatetimeIndex(np.concatenate([f.index.values for f in frames]))
        self.assertFalse(all_dates.duplicated().any())
        self.assertTrue(all_dates.equals(result.usable_dates))
        self.assertTrue(all_dates.is_monotonic_increasing)            # chronological, never shuffled
        for name in result.features:
            self.assertTrue(result.features[name].index.equals(result.returns[name].index))
            self.assertEqual(list(result.features[name].columns), list(FACTORS))   # no target columns
            self.assertEqual(list(result.returns[name].columns), ETFS)
        checks = dv.check_partitions(result) + dv.check_schema(result) + dv.check_alignment(result, self.cal)
        self.assertTrue(all(c.status == dv.PASS for c in checks), checks)

    def test_first_eligible_return_date(self):
        result = build_splits(self.features, self.returns, SPECS, ETFS)
        first_return = self.cal[self.cal > self.prices.index[0]][0]
        ret, usable = dv.check_first_eligible_return(self.adj, self.prices, self.returns, result, self.cal, ETFS,
                                                     documented=str(first_return.date()),
                                                     accepted_first_row=str(first_return.date()))
        self.assertEqual((ret.status, usable.status), (dv.PASS, dv.PASS), (ret, usable))
        self.assertIn("binding ETF: BIL", ret.detail)
        self.assertEqual(result.excluded["date"].iloc[0], self.prices.index[0])   # first price date excluded
        other, _ = dv.check_first_eligible_return(self.adj, self.prices, self.returns, result, self.cal, ETFS,
                                                  documented="2007-05-31")
        self.assertEqual(other.status, dv.WARN)                       # data differ from documented date
        late = build_splits(self.features, self.returns.iloc[5:], SPECS, ETFS)
        bad, _ = dv.check_first_eligible_return(self.adj, self.prices, self.returns.iloc[5:], late, self.cal, ETFS)
        self.assertEqual(bad.status, dv.FAIL)                         # return convention broken

    def test_factor_warm_up_delays_first_usable_row(self):
        """Real-data case: a factor missing after the first return. PASS if it matches the accepted start,
        WARN if the data disagree with the accepted start; never FAIL."""
        features = self.features.copy()
        features.loc[:"2015-03-31", "liquidity_credit"] = np.nan
        result = build_splits(features, self.returns, SPECS, ETFS)
        first_return = self.cal[self.cal > self.prices.index[0]][0]
        effective = features.loc["2015-04-01":].index[0]
        self.assertEqual(result.usable_dates.min(), effective)
        ret, usable = dv.check_first_eligible_return(self.adj, self.prices, self.returns, result, self.cal, ETFS,
                                                     documented=str(first_return.date()),
                                                     accepted_first_row=str(effective.date()))
        self.assertEqual((ret.status, usable.status), (dv.PASS, dv.PASS), usable)
        self.assertIn("liquidity_credit is missing", usable.detail)
        _, changed = dv.check_first_eligible_return(self.adj, self.prices, self.returns, result, self.cal, ETFS,
                                                    documented=str(first_return.date()),
                                                    accepted_first_row=str(first_return.date()))
        self.assertEqual(changed.status, dv.WARN)                     # e.g. after a re-download
        self.assertIn("DIFFERS from the accepted", changed.detail)
        excluded = {c.name: c.status for c in dv.check_alignment(result, self.cal)}
        self.assertEqual(excluded["Excluded dates inside split range are explained (leading block only)"], dv.PASS)
        self.assertEqual(SPECS[0].start, "2015-01-01")                # configured boundary unchanged

    def test_accepted_start_recorded_in_protocol_and_manifest(self):
        from regime_radar.dataset import ACCEPTED_FIRST_TRAIN_ROW, PROTOCOL_VERSION
        self.assertEqual(ACCEPTED_FIRST_TRAIN_ROW["date"], "2007-10-30")
        self.assertEqual(ACCEPTED_FIRST_TRAIN_ROW["binding_factor"], "liquidity_credit")
        self.assertTrue(PROTOCOL_VERSION.endswith("v1.2"))
        self.assertEqual(split_specs()[0].start, "2007-05-30")        # configured boundary unchanged

    def test_missing_values_are_excluded_not_filled(self):
        returns, features = self.returns.copy(), self.features.copy()
        gap_r, gap_f = returns.index[100], features.loc["2016-02-01":].index[0]
        returns.loc[gap_r, "GLD"] = np.nan
        features.loc[gap_f, "inflation"] = np.nan
        result = build_splits(features, returns, SPECS, ETFS)
        self.assertNotIn(gap_r, result.usable_dates)
        self.assertNotIn(gap_f, result.usable_dates)
        reasons = dict(zip(result.excluded["date"], result.excluded["reason"]))
        self.assertEqual(reasons[gap_r], "missing ETF return")
        self.assertEqual(reasons[gap_f], "missing factor value: inflation")
        for frame in list(result.features.values()) + list(result.returns.values()):
            self.assertTrue(np.isfinite(frame.to_numpy()).all())
        alignment = {c.name: c.status for c in dv.check_alignment(result, self.cal)}
        self.assertEqual(alignment["No master-calendar session missing inside the usable span"], dv.WARN)
        self.assertEqual(alignment["Excluded dates inside split range are explained (leading block only)"], dv.WARN)

    def test_returns_are_same_day_and_unfilled(self):
        name_ret = "Returns = same-day log price differences log(P_t/P_t-1) (no shift / fill)"
        name_stale = "No stale-price runs (>= 5 consecutive zero returns)"
        ok = {c.name: c.status for c in dv.check_returns_reproducible(self.prices, self.returns, ETFS)}
        self.assertEqual(ok[name_ret], dv.PASS)
        forward = self.returns.shift(-1)                                # a forward-return label in disguise
        bad = {c.name: c.status for c in dv.check_returns_reproducible(self.prices, forward, ETFS)}
        self.assertEqual(bad[name_ret], dv.FAIL)
        filled = self.prices.copy()
        filled.iloc[200:210, 0] = filled.iloc[199, 0]                   # simulated forward-fill of SPY
        stale = {c.name: c.status for c in
                 dv.check_returns_reproducible(filled, np.log(filled).diff().iloc[1:], ETFS)}
        self.assertEqual(stale[name_stale], dv.WARN)


class IntegrityDetectionTests(unittest.TestCase):
    def test_duplicate_or_unsorted_index_fails(self):
        _, features, _, _, _ = synthetic_panels()
        self.assertEqual(dv.check_index("dup", pd.concat([features, features.iloc[[10]]])).status, dv.FAIL)
        self.assertEqual(dv.check_index("unsorted", features.iloc[::-1]).status, dv.FAIL)
        self.assertEqual(dv.check_index("ok", features).status, dv.PASS)

    def test_factor_without_components_flagged(self):
        inputs = synthetic_inputs()
        _, comps, facs = build_factors(inputs)
        self.assertEqual(dv.check_no_backfill(facs, comps).status, dv.PASS)
        self.assertEqual(dv.check_no_backfill(facs.bfill(), comps).status, dv.FAIL)

    def test_redundancy_flags(self):
        rng = np.random.default_rng(5)
        train = pd.DataFrame(rng.normal(size=(500, 5)), columns=list(FACTORS))
        self.assertTrue(all(c.status == dv.PASS for c in dv.train_redundancy(train)[0]))
        dup = train.copy()
        dup["volatility"] = dup["liquidity_credit"]
        flags = {c.name: c.status for c in dv.train_redundancy(dup)[0]}
        self.assertEqual(flags["Train: no duplicate factor columns"], dv.FAIL)
        self.assertEqual(flags["Train: no severe multicollinearity (VIF < 10, |rho| < 0.9)"], dv.FAIL)


# --------------------------------------------------------------------------------------
# Point-in-time audit
# --------------------------------------------------------------------------------------
def pit_fixture():
    cal = pd.bdate_range("2016-01-04", periods=80)
    months = pd.date_range("2015-10-01", periods=6, freq="MS")
    cpi_avail = months + pd.offsets.MonthBegin(1) + pd.Timedelta(days=12)
    log = pd.concat([
        pd.DataFrame({"series_id": "DGS10", "date": cal - pd.offsets.BDay(1), "value": 2.0,
                      "available_date": cal, "availability_method": "rule:business_days"}),
        pd.DataFrame({"series_id": "CPIAUCSL", "date": months, "value": np.arange(6.0),
                      "available_date": cpi_avail,
                      "availability_method": ["fallback:pre_vintage_history"] * 2 + ["vintage:first_release"] * 4}),
    ], ignore_index=True)
    obs = pd.DataFrame({
        "DGS10": cal - pd.offsets.BDay(1),
        "CPIAUCSL": [months[cpi_avail <= d].max() for d in cal],
    }, index=cal)
    labels = pd.Series(["train"] * 40 + ["validation"] * 40, index=cal)
    specs = {"DGS10": {"frequency": "daily", "availability": {"rule": "business_days"}},
             "CPIAUCSL": {"frequency": "monthly", "availability": {"rule": "vintage"}}}
    return cal, log, obs, labels, specs


class PitAuditTests(unittest.TestCase):
    def test_category_mapping(self):
        self.assertEqual(dv.pit_category("vintage:first_release"), "A")
        self.assertEqual(dv.pit_category("fallback:missing_vintage"), "B")
        self.assertEqual(dv.pit_category("fallback:pre_vintage_history"), "C")
        self.assertEqual(dv.pit_category("fallback:no_vintage_data"), "C")
        self.assertEqual(dv.pit_category("rule:business_days"), "D")

    def test_categories_shares_and_verified_start(self):
        _, log, obs, labels, specs = pit_fixture()
        checks, table, verified = dv.macro_pit_audit(obs, log, labels, specs, {"CPIAUCSL": "fred_api_first_release"})
        status = {c.name: c.status for c in checks}
        self.assertEqual(status["No macro observation dated after the trade date"], dv.PASS)
        self.assertEqual(status["Macro values used only after their availability date"], dv.PASS)
        self.assertEqual(status["CPI / Core CPI / NFCI: every value used is a verified first release (category A)"], dv.WARN)
        self.assertEqual(status["Daily FRED series timing (category D) is a conservative assumption"], dv.WARN)
        self.assertEqual(status["Historical revisions of daily FRED series checked against ALFRED vintages"],
                         dv.NOT_RUN)
        cpi = table[(table["series"] == "CPIAUCSL") & (table["split"] == "train")]
        self.assertEqual(set(cpi["category"]), {"A", "C"})
        row = verified.set_index("series").loc["CPIAUCSL"]
        first_a_use = obs.index[obs["CPIAUCSL"] >= pd.Timestamp("2015-12-01")][0]
        self.assertEqual(row["category_A_from"], first_a_use.date().isoformat())
        self.assertEqual(row["validation_pct_A"], 100.0)
        self.assertGreater(row["train_pct_BC"], 0)

    def test_no_value_period_is_not_category_c(self):
        """Real-data NFCI case: back-filled history dated at its vintage -> 'no value' earlier, not C -> PASS."""
        cal, log, obs, labels, specs = pit_fixture()
        log.loc[log["series_id"] == "CPIAUCSL", "availability_method"] = "vintage:first_release"
        obs.loc[obs.index[:20], "CPIAUCSL"] = pd.NaT                     # no value published yet
        checks, _, verified = dv.macro_pit_audit(obs, log, labels, specs)
        status = {c.name: c for c in checks}
        check = status["CPI / Core CPI / NFCI: every value used is a verified first release (category A)"]
        self.assertEqual(check.status, dv.PASS)
        self.assertIn("no real-time value on part of the sample", check.detail)
        row = verified.set_index("series").loc["CPIAUCSL"]
        self.assertEqual(row["train_pct_no_value"], 50.0)
        self.assertEqual(row["category_A_from"], cal[20].date().isoformat())

    def test_all_first_release_passes(self):
        _, log, obs, labels, specs = pit_fixture()
        log.loc[log["series_id"] == "CPIAUCSL", "availability_method"] = "vintage:first_release"
        status = {c.name: c.status for c in dv.macro_pit_audit(obs, log, labels, specs)[0]}
        self.assertEqual(status["CPI / Core CPI / NFCI: every value used is a verified first release (category A)"], dv.PASS)

    def test_unpublished_or_future_observation_fails(self):
        cal, log, obs, labels, specs = pit_fixture()
        early = log.copy()                                            # CPI used before its release date
        early.loc[early["series_id"] == "CPIAUCSL", "available_date"] += pd.Timedelta(days=60)
        self.assertEqual(dv.check_macro_published(obs, early, cal).status, dv.FAIL)
        bad = obs.copy()
        bad.iloc[5, 0] = cal[6]                                        # value from the future
        status = {c.name: c.status for c in dv.macro_pit_audit(bad, log, labels, specs)[0]}
        self.assertEqual(status["No macro observation dated after the trade date"], dv.FAIL)
        same_day = obs.copy()
        same_day.iloc[3, 0] = cal[3]                                   # daily value used on its own date
        status = {c.name: c.status for c in dv.macro_pit_audit(same_day, log, labels, specs)[0]}
        self.assertEqual(status["Daily FRED values lag their observation date by >= 1 session"], dv.FAIL)

    def test_macro_alignment_prefix_check(self):
        cal, log, _, _, _ = pit_fixture()
        stored = pd.DataFrame({s: align_to_calendar(log[log["series_id"] == s], cal)["value"]
                               for s in ("DGS10", "CPIAUCSL")}, index=cal)
        checks, _ = dv.macro_prefix_check(log, cal, stored, [cal[30], cal[60]])
        self.assertTrue(all(c.status == dv.PASS for c in checks), checks)
        leaky = stored.copy()
        leaky["CPIAUCSL"] = leaky["CPIAUCSL"].shift(-15)             # macro value used before release
        checks, _ = dv.macro_prefix_check(log, cal, leaky, [cal[30]])
        self.assertEqual(checks[0].status, dv.FAIL)


# --------------------------------------------------------------------------------------
# Leakage detection: the causal pipeline passes, deliberately leaky pipelines are caught
# --------------------------------------------------------------------------------------
def _leaky(transform_factors=None, transform_inputs=None):
    def build(inputs: FactorInputs):
        if transform_inputs is not None:
            inputs = transform_inputs(inputs)
        raw, comps, facs = build_factors(inputs)
        return raw, comps, (transform_factors(facs) if transform_factors else facs)
    return build


def _future_macro(inputs: FactorInputs) -> FactorInputs:
    return FactorInputs(inputs.adj_close, inputs.close, inputs.volume, inputs.macro.shift(-5),
                        inputs.release_log, inputs.sector_tickers)


LEAKY_BUILDERS = {
    "full-sample z-score": _leaky(lambda f: (f - f.mean()) / f.std()),
    "centred rolling window": _leaky(lambda f: f.rolling(11, center=True, min_periods=1).mean()),
    "one-day look-ahead shift": _leaky(lambda f: f.shift(-1)),
    "macro values five days early": _leaky(transform_inputs=_future_macro),
}


class LeakageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = synthetic_inputs()
        _, cls.comps, cls.factors = build_factors(cls.inputs)
        cls.cutoffs = [cls.inputs.calendar[500], cls.inputs.calendar[750]]

    def test_causal_pipeline_passes(self):
        checks, table = dv.prefix_invariance_check(self.inputs, self.factors.round(6), self.comps.round(6),
                                                   self.cutoffs, self.cutoffs)
        self.assertTrue(all(c.status == dv.PASS for c in checks), checks)
        self.assertEqual(table.loc[table["test"] != "serialization", "max_abs_diff"].max(), 0.0)
        self.assertLessEqual(table.loc[table["test"] == "serialization", "max_abs_diff"].max(), 5e-7)

    def test_leaky_pipelines_are_detected(self):
        for label, leaky_build in LEAKY_BUILDERS.items():
            with self.subTest(leak=label):
                _, comps, facs = leaky_build(self.inputs)
                with mock.patch.object(dv, "build_factors", leaky_build):
                    checks, _ = dv.prefix_invariance_check(self.inputs, facs, comps, self.cutoffs, self.cutoffs)
                status = {c.name: c.status for c in checks}
                self.assertEqual(status["Stored factor files reproduce from current code and inputs "
                                        "(serialization only)"], dv.PASS)
                leaks = [status["Prefix invariance: removing later data leaves history unchanged"],
                         status["Prefix invariance: altering later data leaves history unchanged"]]
                self.assertIn(dv.FAIL, leaks)

    def test_stale_stored_file_is_detected(self):
        stored = self.factors.copy()
        stored.iloc[300, 0] += 0.01                                     # file not produced by current code
        checks, _ = dv.prefix_invariance_check(self.inputs, stored, self.comps, self.cutoffs[:1], [])
        self.assertEqual(checks[0].status, dv.FAIL)

    def test_changing_or_removing_future_keeps_history(self):
        cutoff = self.cutoffs[0]
        _, comps_cut, facs_cut = build_factors(self.inputs.truncate(cutoff))
        pd.testing.assert_frame_equal(facs_cut, self.factors.loc[:cutoff])
        _, comps_alt, facs_alt = build_factors(dv.perturb_future(self.inputs, cutoff))
        pd.testing.assert_frame_equal(facs_alt.loc[:cutoff], self.factors.loc[:cutoff])
        pd.testing.assert_frame_equal(comps_alt.loc[:cutoff], self.comps.loc[:cutoff])
        self.assertFalse(facs_alt.loc[cutoff:].iloc[1:].equals(self.factors.loc[cutoff:].iloc[1:]))

    def test_initial_features_prefix(self):
        adj = self.inputs.adj_close.copy()
        for t in ("QQQ",):
            adj[t] = adj["SPY"]
        from regime_radar.features import build_initial_features
        stored = build_initial_features(adj)
        self.assertEqual(dv.initial_features_prefix_check(adj, stored, self.cutoffs).status, dv.PASS)
        self.assertEqual(dv.initial_features_prefix_check(adj, None, self.cutoffs).status, dv.NOT_RUN)


# --------------------------------------------------------------------------------------
# Composition, status semantics and documentation
# --------------------------------------------------------------------------------------
class CompositionTests(unittest.TestCase):
    def test_late_component_and_inclusion_effect(self):
        inputs = synthetic_inputs()
        _, comps, facs = build_factors(inputs)
        comps = comps.copy()
        late_col = "volatility__vix_term_structure"
        comps.loc[: inputs.calendar[450], late_col] = np.nan
        dates = inputs.calendar[400:]
        changes = dv.composition_changes(comps, dates, facs, pd.Series("train", index=dates), dates)
        row = changes[(changes["factor"] == "volatility") & changes["added"].str.contains("vix_term_structure")]
        self.assertEqual(len(row), 1)
        day = inputs.calendar[451]
        self.assertEqual(row["date"].iloc[0], day.date().isoformat())
        cols = [c for c in comps if c.startswith("volatility__")]
        expected = comps.loc[day, cols].mean() - comps.loc[day, [c for c in cols if c != late_col]].mean()
        self.assertAlmostEqual(row["inclusion_effect"].iloc[0], round(expected, 3), places=3)
        self.assertIn("volatility", set(dv.composition_sensitivity(comps, facs, dates)["factor"]))


class DocumentationTests(unittest.TestCase):
    def test_status_vocabulary(self):
        self.assertEqual(set(dv.STATUSES), {"PASS", "WARN", "FAIL", "NOT RUN"})
        counts = dv.summarize([dv.Check("a", dv.NOT_RUN, ""), dv.Check("b", dv.PASS, "")])
        self.assertEqual((counts["NOT RUN"], counts["PASS"], counts["FAIL"]), (1, 1, 0))

    def test_protocol_states_exposure_and_walk_forward_rules(self):
        cal, features, prices, returns, _ = synthetic_panels()
        result = build_splits(features, returns, SPECS, ETFS)
        with tempfile.TemporaryDirectory() as tmp:
            manifest = build_manifest(result, Path(tmp), Path(tmp), [], {}, {"PASS": 1}, {})
        text = render_protocol(result, manifest, ETFS, ["- `NFCI`: verified first-release (category A) from 2011."])
        for phrase in ("Descriptive historical inspection", "Causal feature computation", "Model fitting",
                       "Hyperparameter / specification selection", "Final out-of-sample evaluation",
                       "not a never-inspected", "Sequential walk-forward", "Static holdout",
                       "because of test-period performance", "designated out-of-sample period",
                       "never credit decision_t with return_t"):
            self.assertIn(phrase, text)
        self.assertNotIn("never been inspected", text)
        self.assertEqual(manifest["test_period_exposure"]["model_fitting"], "not done")
        self.assertEqual(manifest["effective_first_train_row"]["date"], "2007-10-30")
        self.assertIn("Accepted effective train start (protocol v1.2)", text)

    def test_pit_limitations_are_documented(self):
        verified = pd.DataFrame([{"series": "NFCI", "used_in": "liquidity_credit: nfci", "download_source": "x",
                                  "category_A_from": "2011-05-25", "split_of_A_start": "train",
                                  "train_pct_A": 50.0, "train_pct_BC": 0.0, "train_pct_no_value": 50.0,
                                  "validation_pct_A": 100.0, "validation_pct_BC": 0.0, "validation_pct_no_value": 0.0,
                                  "test_pct_A": 100.0, "test_pct_BC": 0.0, "test_pct_no_value": 0.0}])
        lines = dv.pit_protocol_summary(verified)
        self.assertIn("category A) from 2011-05-25", lines[0])
        self.assertIn("train: A 50.0%, no value 50.0%", lines[0])


if __name__ == "__main__":
    unittest.main()
