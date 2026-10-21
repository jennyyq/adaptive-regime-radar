"""Offline tests for the factor layer: no look-ahead, safe standardization, sign conventions.

Run:  python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from regime_radar.factors import (  # noqa: E402
    COMPONENTS, FactorInputs, build_factors, composite_factors, pit_yoy, rolling_robust_z, validate,
)

SECTORS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]


class RobustZTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.x = pd.Series(rng.standard_t(4, 1500).cumsum(), index=pd.bdate_range("2010-01-01", periods=1500))

    def test_changing_future_does_not_change_history(self):
        z_before = rolling_robust_z(self.x, 252, 126)
        k = 900
        altered = self.x.copy()
        altered.iloc[k + 1:] = altered.iloc[k + 1:] * 50 + 1e6          # wild future
        z_after = rolling_robust_z(altered, 252, 126)
        pd.testing.assert_series_equal(z_before.iloc[: k + 1], z_after.iloc[: k + 1])
        self.assertFalse(z_before.iloc[k + 1:].equals(z_after.iloc[k + 1:]))

    def test_current_value_not_in_its_own_window(self):
        z = rolling_robust_z(self.x, 252, 126, clip=None)
        t = 1000
        window = self.x.iloc[t - 252:t]
        center = window.median()
        mad = (window - center).abs().median()
        expected = (self.x.iloc[t] - center) / (1.4826 * mad)
        self.assertAlmostEqual(z.iloc[t], expected, places=10)
        # changing x_t changes only z_t among dates <= t
        bumped = self.x.copy()
        bumped.iloc[t] += 100
        z2 = rolling_robust_z(bumped, 252, 126, clip=None)
        pd.testing.assert_series_equal(z.iloc[:t], z2.iloc[:t])

    def test_min_periods(self):
        z = rolling_robust_z(self.x, 252, 126)
        self.assertTrue(z.iloc[:126].isna().all())
        self.assertTrue(z.iloc[127:400].notna().all())

    def test_zero_mad_and_constant_series_are_safe(self):
        idx = pd.bdate_range("2020-01-01", periods=400)
        constant = pd.Series(1.0, index=idx)
        self.assertTrue(rolling_robust_z(constant).isna().all())
        discrete = pd.Series(np.where(np.arange(400) % 50 == 0, 1.0, 0.0), index=idx)  # MAD = 0, std > 0
        z = rolling_robust_z(discrete, clip=None)
        self.assertFalse(np.isinf(z).any())
        self.assertTrue(z.iloc[200:].notna().all())

    def test_missing_values_and_clip(self):
        x = self.x.copy()
        x.iloc[500:510] = np.nan
        x.iloc[700] = 1e9
        z = rolling_robust_z(x, 252, 126, clip=5.0)
        self.assertTrue(z.iloc[500:510].isna().all())
        self.assertEqual(z.iloc[700], 5.0)
        self.assertLessEqual(z.abs().max(), 5.0)


class CompositeTests(unittest.TestCase):
    def test_min_component_share(self):
        idx = pd.bdate_range("2020-01-01", periods=3)
        cols = [f"{c.factor}__{c.name}" for c in COMPONENTS]
        comps = pd.DataFrame(np.nan, index=idx, columns=cols)
        eq = [f"{c.factor}__{c.name}" for c in COMPONENTS if c.factor == "equity_trend"]
        comps.loc[idx[0], eq[:2]] = 1.0          # 2 of 5 -> not enough
        comps.loc[idx[1], eq[:3]] = [1.0, 2.0, 3.0]  # 3 of 5 -> mean
        out = composite_factors(comps)
        self.assertTrue(np.isnan(out.loc[idx[0], "equity_trend"]))
        self.assertAlmostEqual(out.loc[idx[1], "equity_trend"], 2.0)

    def test_monitor_only_component_excluded(self):
        idx = pd.bdate_range("2020-01-01", periods=1)
        comps = pd.DataFrame(0.0, index=idx, columns=[f"{c.factor}__{c.name}" for c in COMPONENTS])
        comps["liquidity_credit__hy_oas"] = 100.0
        self.assertEqual(composite_factors(comps).loc[idx[0], "liquidity_credit"], 0.0)


class PitYoyTests(unittest.TestCase):
    def test_yoy_known_only_after_release(self):
        months = pd.date_range("2019-01-01", "2020-03-01", freq="MS")
        log = pd.DataFrame({
            "series_id": "CPIAUCSL", "date": months, "value": np.linspace(100, 114, len(months)),
            "available_date": months + pd.offsets.MonthBegin(1) + pd.Timedelta(days=11),
        })
        cal = pd.bdate_range("2019-06-01", "2020-04-30")
        yoy = pit_yoy(log, "CPIAUCSL", cal)
        jan_2020_release = pd.Timestamp("2020-02-12")
        self.assertTrue(yoy.loc[:"2020-02-11"].isna().all())               # no YoY before Jan-20 published
        self.assertAlmostEqual(yoy.loc[jan_2020_release], 100 * (log.value.iloc[12] / log.value.iloc[0] - 1))
        self.assertTrue(pit_yoy(log, "MISSING", cal).isna().all())


def synthetic_inputs(n: int = 900, crash_start: int = 700, crash_len: int = 40) -> FactorInputs:
    """Calm market followed by a crash: equities fall, vol and credit spreads jump."""
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2015-01-01", periods=n, name="date")
    crash = np.zeros(n, dtype=bool)
    crash[crash_start:crash_start + crash_len] = True
    common = rng.normal(0, 0.006, n)
    common[crash] = rng.normal(-0.012, 0.03, crash.sum())

    def walk(beta, idio=0.004, drift=0.0003, extra=None):
        r = drift + beta * common + rng.normal(0, idio, n)
        if extra is not None:
            r = r + extra
        return 100 * np.exp(np.cumsum(r))

    hy_hit = np.where(crash, -0.006, 0.0)
    adj = pd.DataFrame({
        "SPY": walk(1.0), "RSP": walk(1.05, extra=np.where(crash, -0.002, 0)),
        "IWM": walk(1.3, extra=np.where(crash, -0.003, 0)), "EFA": walk(0.9), "EEM": walk(1.1),
        "TIP": walk(0.05, 0.002), "IEF": walk(-0.1, 0.002), "DBC": walk(0.6),
        "HYG": walk(0.35, 0.002, extra=hy_hit), "LQD": walk(0.05, 0.002), "BIL": walk(0, 0.0001),
        **{s: walk(1.0) for s in SECTORS},
    }, index=idx)
    vix = 15 + rng.normal(0, 1, n) + np.where(crash, 30, 0)
    adj["^VIX"] = vix
    adj["^VIX3M"] = 17 + rng.normal(0, 0.8, n) + np.where(crash, 18, 0)
    close = adj.copy()
    volume = pd.DataFrame(1e6, index=idx, columns=adj.columns)
    volume.loc[crash, "HYG"] = 6e5
    macro = pd.DataFrame({
        "DGS2": 2 + np.cumsum(rng.normal(0, 0.02, n)), "DGS10": 3 + np.cumsum(rng.normal(0, 0.02, n)),
        "T10Y2Y": 1 + rng.normal(0, 0.05, n), "T5YIE": 2 + np.cumsum(rng.normal(0, 0.01, n)),
        "BAA10Y": 2 + rng.normal(0, 0.03, n) + np.where(crash, 2.0, 0),
        "NFCI": -0.5 + rng.normal(0, 0.03, n) + np.where(crash, 1.5, 0),
    }, index=idx)
    months = pd.date_range("2013-01-01", idx[-1], freq="MS")
    log = pd.concat([pd.DataFrame({
        "series_id": sid, "date": months, "value": 100 * np.exp(np.cumsum(rng.normal(0.002, 0.002, len(months)))),
        "available_date": months + pd.offsets.MonthBegin(1) + pd.Timedelta(days=12),
    }) for sid in ("CPIAUCSL", "CPILFESL")], ignore_index=True)
    return FactorInputs(adj, close, volume, macro, log, SECTORS)


class FactorPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = synthetic_inputs()
        cls.raw, cls.comps, cls.factors = build_factors(cls.inputs)

    def test_signs_in_a_crash(self):
        calm = self.factors.iloc[600:690].mean()
        crash = self.factors.iloc[720:740].mean()
        self.assertLess(crash["equity_trend"], calm["equity_trend"] - 1)
        self.assertGreater(crash["volatility"], calm["volatility"] + 1)
        self.assertGreater(crash["liquidity_credit"], calm["liquidity_credit"] + 1)

    def test_truncated_rebuild_identical(self):
        cutoff = self.inputs.calendar[650]
        _, comps_cut, fac_cut = build_factors(self.inputs.truncate(cutoff))
        pd.testing.assert_frame_equal(self.factors.loc[:cutoff], fac_cut.loc[:cutoff])
        pd.testing.assert_frame_equal(self.comps.loc[:cutoff], comps_cut.loc[:cutoff])

    def test_validation_has_no_failures(self):
        checks = validate(self.inputs, self.raw, self.comps, self.factors, None, [], self.inputs.calendar[400])
        self.assertFalse([c for c in checks if c.status == "FAIL"], checks)
        self.assertFalse(np.isinf(self.factors.to_numpy()).any())


if __name__ == "__main__":
    unittest.main()
