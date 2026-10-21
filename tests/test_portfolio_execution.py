"""Stage 7 tests: weekly calendar, execution timing, split boundaries, drift and turnover (synthetic calendars)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regime_radar import portfolio_execution as pe  # noqa: E402


def calendar(dates):
    cal = pd.DatetimeIndex(dates)
    split = pd.Series(np.where(cal < pd.Timestamp("2016-01-01"), "train", "validation"), index=cal)
    return cal, split


class TestSchedule(unittest.TestCase):
    def setUp(self):
        days = pd.bdate_range("2015-11-02", "2016-01-29")
        holidays = pd.DatetimeIndex(["2015-11-26", "2015-12-25", "2016-01-01", "2016-01-18"])
        self.cal, self.split = calendar(days.difference(holidays))
        self.s = pe.weekly_schedule(self.cal, self.split)

    def test_one_decision_on_the_last_session_of_each_week(self):
        wk = self.s["signal_date"] - pd.to_timedelta(self.s["signal_date"].dt.weekday, unit="D")
        self.assertFalse(wk.duplicated().any())
        last = pd.Series(self.cal, index=self.cal).groupby(self.cal - pd.to_timedelta(self.cal.weekday, unit="D")).max()
        self.assertTrue((self.s["signal_date"].to_numpy() == last.loc[wk].to_numpy()).all())

    def test_holiday_weeks(self):
        self.assertIn(pd.Timestamp("2015-12-24"), set(self.s["signal_date"]))   # Christmas Friday closed
        r = self.s[self.s["signal_date"] == pd.Timestamp("2015-12-24")].iloc[0]
        self.assertEqual(r["execution_date"], pd.Timestamp("2015-12-28"))
        self.assertEqual(r["first_holding_return_date"], pd.Timestamp("2015-12-29"))
        r = self.s[self.s["signal_date"] == pd.Timestamp("2016-01-15")].iloc[0]  # MLK Monday closed
        self.assertEqual(r["execution_date"], pd.Timestamp("2016-01-19"))

    def test_execution_after_signal_and_once_per_week(self):
        self.assertTrue((self.s["execution_date"] > self.s["signal_date"]).all())
        self.assertTrue((self.s["first_holding_return_date"].dropna() > self.s["execution_date"][self.s["first_holding_return_date"].notna()]).all())
        ex = self.s["execution_date"] - pd.to_timedelta(self.s["execution_date"].dt.weekday, unit="D")
        self.assertFalse(ex.duplicated().any())
        pos = {d: i for i, d in enumerate(self.cal)}
        for _, r in self.s.iterrows():
            self.assertEqual(pos[r["execution_date"]], pos[r["signal_date"]] + 1)

    def test_split_boundary(self):
        r = self.s[self.s["signal_date"] == pd.Timestamp("2015-12-31")].iloc[0]
        self.assertEqual(r["split"], "train")
        self.assertTrue(r["split_crossing"])
        self.assertEqual(r["execution_date"], pd.Timestamp("2016-01-04"))

    def test_incomplete_final_week_is_skipped(self):
        cal, split = calendar(pd.bdate_range("2019-12-02", "2019-12-31"))
        s = pe.weekly_schedule(cal, split)
        self.assertNotIn(pd.Timestamp("2019-12-31"), set(s["signal_date"]))       # Tuesday: week end unknown
        self.assertEqual(s["signal_date"].max(), pd.Timestamp("2019-12-27"))
        self.assertEqual(s["execution_date"].max(), pd.Timestamp("2019-12-30"))
        self.assertTrue(s["execution_date"].max() <= cal.max())                    # no Test date generated
        cal2, split2 = calendar(pd.bdate_range("2019-12-02", "2019-12-27"))      # ends on a Friday
        s2 = pe.weekly_schedule(cal2, split2)
        self.assertNotIn(pd.Timestamp("2019-12-27"), set(s2["signal_date"]))      # no known execution session

    def test_non_monotone_calendar_rejected(self):
        with self.assertRaises(ValueError):
            pe.weekly_schedule(pd.DatetimeIndex(["2016-01-05", "2016-01-04"]), pd.Series(["v", "v"]))


class TestDriftAndTurnover(unittest.TestCase):
    def test_drift(self):
        w = np.array([0.5, 0.5])
        r = np.array([[0.10, 0.0], [0.0, 0.0]])
        np.testing.assert_allclose(pe.drift(w, r), [0.55 / 1.05, 0.5 / 1.05])
        np.testing.assert_array_equal(pe.drift(w, np.empty((0, 2))), w)

    def test_turnover_definition(self):
        self.assertAlmostEqual(pe.one_way_turnover(np.array([0.3, 0.7]), np.array([0.5, 0.5])), 0.2)


if __name__ == "__main__":
    unittest.main()
