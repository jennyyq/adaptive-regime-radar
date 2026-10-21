"""Offline tests (no network): parsing, session handling and point-in-time alignment.

Run:  python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from regime_radar.config import load_config, yahoo_tickers  # noqa: E402
from regime_radar.panels import investment_panel  # noqa: E402
from regime_radar.pit import align_to_calendar, availability  # noqa: E402
from regime_radar.sources.fred import parse_api_json, parse_graph_csv  # noqa: E402
from regime_radar.sources.yahoo import drop_incomplete_session, normalize_yahoo  # noqa: E402

DAILY = {"availability": {"rule": "business_days", "lag": 1}}
MONTHLY_VINTAGE = {"availability": {"rule": "vintage", "fallback": {"rule": "next_month_day", "day": 20}}}


def yf_like(dates, ticker="SPY", tz=None) -> pd.DataFrame:
    idx = pd.DatetimeIndex(dates, tz=tz, name="Date")
    n = len(idx)
    data = {
        ("Adj Close", ticker): np.linspace(90, 99, n),
        ("Close", ticker): np.linspace(100, 109, n),
        ("High", ticker): np.linspace(101, 110, n),
        ("Low", ticker): np.linspace(99, 108, n),
        ("Open", ticker): np.linspace(100, 109, n),
        ("Volume", ticker): np.full(n, 1e6),
    }
    frame = pd.DataFrame(data, index=idx)
    frame.columns = pd.MultiIndex.from_tuples(frame.columns, names=["Price", "Ticker"])
    return frame


class YahooTests(unittest.TestCase):
    def test_multiindex_and_timezone_are_flattened(self):
        raw = yf_like(["2024-01-02 00:00", "2024-01-03 00:00"], tz="America/New_York")
        out = normalize_yahoo(raw, now=pd.Timestamp("2024-02-01 12:00", tz="America/New_York"))
        self.assertEqual(list(out.columns), ["open", "high", "low", "close", "adj_close", "volume"])
        self.assertIsNone(out.index.tz)
        self.assertEqual(out.index[0], pd.Timestamp("2024-01-02"))
        self.assertAlmostEqual(out["adj_close"].iloc[0], 90.0)

    def test_index_without_adj_close_uses_close(self):
        raw = yf_like(["2024-01-02"]).drop(columns="Adj Close", level=0)
        out = normalize_yahoo(raw, now=pd.Timestamp("2024-02-01 12:00", tz="America/New_York"))
        self.assertEqual(out["adj_close"].iloc[0], out["close"].iloc[0])

    def test_open_session_bar_is_dropped(self):
        frame = pd.DataFrame({"adj_close": [1.0, 2.0]}, index=pd.to_datetime(["2024-03-04", "2024-03-05"]))
        during = drop_incomplete_session(frame, pd.Timestamp("2024-03-05 11:00", tz="America/New_York"))
        after = drop_incomplete_session(frame, pd.Timestamp("2024-03-05 17:00", tz="America/New_York"))
        # 2024-03-05 11:00 NY == 2024-03-06 00:00 Singapore: the bar is still forming
        self.assertEqual(len(during), 1)
        self.assertEqual(len(after), 2)


class FredParsingTests(unittest.TestCase):
    def test_graph_csv_missing_values(self):
        out = parse_graph_csv("observation_date,DGS10\n2024-01-02,3.95\n2024-01-03,.\n")
        self.assertEqual(len(out), 2)
        self.assertTrue(np.isnan(out["value"].iloc[1]))

    def test_api_json_first_release(self):
        payload = ('{"observations":[{"realtime_start":"2024-02-13","realtime_end":"2024-03-12",'
                   '"date":"2024-01-01","value":"308.417"}]}')
        out = parse_api_json(payload, vintage=True)
        self.assertEqual(out["realtime_start"].iloc[0], pd.Timestamp("2024-02-13"))
        self.assertAlmostEqual(out["value"].iloc[0], 308.417)


class PointInTimeTests(unittest.TestCase):
    calendar = pd.bdate_range("2024-01-02", "2024-03-29")

    def test_daily_rule_has_no_lookahead(self):
        obs = pd.DataFrame({"date": pd.bdate_range("2024-01-02", "2024-03-28", freq="C",
                                                   holidays=["2024-01-15", "2024-02-19"])})
        obs["value"] = np.arange(len(obs), dtype=float)
        aligned = align_to_calendar(availability(obs, DAILY), self.calendar)
        known = aligned.dropna()
        self.assertTrue((known["obs_date"] < known.index).all(), "same-day H.15 value leaked")
        # Friday 2024-01-12 -> Monday 2024-01-15 is MLK day -> usable Tuesday 2024-01-16
        self.assertEqual(aligned.loc["2024-01-16", "obs_date"], pd.Timestamp("2024-01-12"))

    def test_vintage_dates_and_pre_vintage_fallback(self):
        obs = pd.DataFrame({
            "date": pd.to_datetime(["2023-06-01", "2024-01-01", "2024-02-01"]),
            "value": [1.0, 2.0, 3.0],
            # first vintage 2024-02-13 back-fills June 2023 -> treated as pre-vintage history
            "realtime_start": pd.to_datetime(["2024-02-13", "2024-02-13", "2024-03-12"]),
        })
        out = availability(obs, MONTHLY_VINTAGE)
        self.assertEqual(out.loc[0, "availability_method"], "fallback:pre_vintage_history")
        self.assertEqual(out.loc[0, "available_date"], pd.Timestamp("2023-07-20"))
        self.assertEqual(out.loc[1, "available_date"], pd.Timestamp("2024-02-13"))
        aligned = align_to_calendar(out, self.calendar)
        self.assertEqual(aligned.loc["2024-02-12", "value"], 1.0)
        self.assertEqual(aligned.loc["2024-02-13", "value"], 2.0)
        self.assertEqual(aligned.loc["2024-03-12", "value"], 3.0)

    def test_late_release_never_flips_back(self):
        obs = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            "value": [10.0, 20.0],
            "available_date": pd.to_datetime(["2024-03-01", "2024-02-20"]),
        })
        aligned = align_to_calendar(obs, self.calendar)
        self.assertEqual(aligned.loc["2024-03-01", "value"], 20.0)


class PanelTests(unittest.TestCase):
    def test_common_sample_and_no_fill(self):
        dates = pd.bdate_range("2024-01-01", periods=6)
        adj = pd.DataFrame({"A": [1, 2, 3, 4, 5, 6.0], "B": [np.nan, np.nan, 1, np.nan, 1.1, 1.2]}, index=dates)
        prices, rets = investment_panel(adj, ["A", "B"])
        self.assertEqual(prices.index[0], dates[2])
        self.assertEqual(int(rets["B"].isna().sum()), 2)  # gap is not forward-filled

    def test_project_config_is_valid(self):
        config = load_config()
        self.assertEqual(yahoo_tickers(config)[:9], config["investment_etfs"])


if __name__ == "__main__":
    unittest.main()
