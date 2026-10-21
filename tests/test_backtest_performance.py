"""Stage 8 tests: performance metrics, statistics, attribution identity and the prespecified conclusion rules."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage8_fixtures import PROTOCOL  # noqa: E402

from regime_radar import backtest_attribution as bat  # noqa: E402
from regime_radar import backtest_performance as bp  # noqa: E402
from regime_radar import backtest_report as brep  # noqa: E402
from regime_radar import backtest_statistics as bst  # noqa: E402


class TestMetrics(unittest.TestCase):
    def test_known_path(self):
        idx = pd.bdate_range("2021-01-04", periods=4)
        r = pd.Series([0.10, -0.20, 0.05, 0.10], index=idx)
        rf = pd.Series(0.0, index=idx)
        m = bp.metrics(r, rf)
        w = np.cumprod(1 + r.to_numpy())
        self.assertAlmostEqual(m["cumulative_return"], w[-1] - 1, places=15)
        self.assertAlmostEqual(m["max_drawdown"], 0.88 / 1.1 - 1, places=15)
        self.assertAlmostEqual(m["cagr"], w[-1] ** (252 / 4) - 1, places=10)
        self.assertAlmostEqual(m["sharpe"], r.mean() * 252 / (r.std(ddof=1) * np.sqrt(252)), places=12)
        dd = np.sqrt(np.mean(np.minimum(r.to_numpy(), 0) ** 2)) * np.sqrt(252)
        self.assertAlmostEqual(m["sortino"], r.mean() * 252 / dd, places=12)
        self.assertAlmostEqual(m["worst_day"], -0.20)

    def test_max_drawdown_includes_initial_wealth(self):
        self.assertAlmostEqual(bp.max_drawdown([-0.1, 0.05]), -0.1, places=15)

    def test_sharpe_uses_excess_over_bil_once(self):
        idx = pd.bdate_range("2021-01-04", periods=300)
        rng = np.random.default_rng(0)
        r = pd.Series(rng.normal(0.0004, 0.01, 300), index=idx)
        rf = pd.Series(0.0001, index=idx)
        m = bp.metrics(r, rf)
        ex = r - rf
        self.assertAlmostEqual(m["sharpe"], ex.mean() * 252 / (ex.std(ddof=1) * np.sqrt(252)), places=12)
        self.assertAlmostEqual(m["arithmetic_annual_return"], r.mean() * 252, places=12)   # rf not added to returns

    def test_period_returns_compound(self):
        idx = pd.bdate_range("2021-01-25", periods=10)
        r = pd.Series(0.01, index=idx)
        mon = bp.period_returns(r, "M")
        self.assertAlmostEqual(mon.iloc[0], 1.01 ** 5 - 1, places=12)

    def test_drawdown_episodes(self):
        idx = pd.bdate_range("2021-01-04", periods=6)
        r = pd.Series([0.1, -0.1, -0.1, 0.3, -0.05, 0.0], index=idx)
        e = bp.drawdown_episodes(r)
        self.assertAlmostEqual(e["depth"].min(), 0.81 - 1, places=12)
        self.assertTrue(e.iloc[0]["recovered"])


class TestStatistics(unittest.TestCase):
    def test_newey_west_lag0_is_iid_variance(self):
        x = np.random.default_rng(1).normal(size=500)
        self.assertAlmostEqual(bst.nw_variance_of_mean(x, 0), x.var() / 500, places=15)
        self.assertEqual(bst.nw_lag(1707), 7)

    def test_ols_hac_matches_ols_point_estimates(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=400)
        y = 0.001 + 0.7 * x + rng.normal(scale=0.1, size=400)
        r = bst.ols_hac(y, x)
        b = np.polyfit(x, y, 1)
        self.assertAlmostEqual(r["beta"], b[0], places=10)
        self.assertAlmostEqual(r["alpha_annualized"] / 252, b[1], places=10)

    def test_stationary_bootstrap_deterministic_and_block_length(self):
        a = bst.stationary_indices(1000, 21, 50, 7)
        b = bst.stationary_indices(1000, 21, 50, 7)
        np.testing.assert_array_equal(a, b)
        starts = (np.diff(a, axis=1) != 1) & (np.diff(a, axis=1) != -999)
        self.assertAlmostEqual(1 / starts.mean(), 21, delta=2.5)

    def test_bootstrap_identical_series_gives_zero(self):
        idx = pd.bdate_range("2021-01-04", periods=200)
        r = pd.Series(np.random.default_rng(3).normal(0.0003, 0.01, 200), index=idx)
        df = pd.DataFrame({"A": r, "B": r})
        out = bst.bootstrap_differences(df, pd.Series(0.0, index=idx), [("A", "B")], 21, 200, 1)
        self.assertTrue((out["estimate"].abs() < 1e-15).all())
        self.assertTrue((out["ci_low"].abs() < 1e-12).all() and (out["ci_high"].abs() < 1e-12).all())

    def test_holm(self):
        np.testing.assert_allclose(bst.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


class TestAttribution(unittest.TestCase):
    def test_exact_daily_reconciliation(self):
        rng = np.random.default_rng(4)
        idx = pd.bdate_range("2021-01-04", periods=60)
        assets = ["SPY", "IEF", "BIL"]
        simple = pd.DataFrame(rng.normal(0, 0.01, (60, 3)), index=idx, columns=assets)
        wa = pd.DataFrame(rng.dirichlet([2, 2, 2], 60), index=idx, columns=assets)
        wb = pd.DataFrame(rng.dirichlet([2, 2, 2], 60), index=idx, columns=assets)
        gross = pd.DataFrame({"A": (wa * simple).sum(axis=1), "B": (wb * simple).sum(axis=1)})
        cost = pd.DataFrame({"A": 0.0005, "B": 0.0}, index=idx)
        net = (1 + gross) * (1 - cost) - 1
        s, d = bat.attribution("A", "B", {"A": wa, "B": wb}, simple, gross, net,
                               {"equity": ["SPY"], "duration": ["IEF"], "bil": ["BIL"]})
        self.assertLess(s["reconciliation_error"].iloc[0], 1e-15)
        g = s.set_index("component")["sum_daily"]
        self.assertAlmostEqual(g["average-exposure part (all groups)"] + g["timing part (all groups)"],
                               g["gross active return (ledger)"], places=14)
        self.assertAlmostEqual(g["gross active return (ledger)"] + g["transaction-cost effect"], g["net active return"],
                               places=14)


class TestConclusionRules(unittest.TestCase):
    def frames(self, s10, s20, s21, lo21):
        rows = []
        for (a, b), (e, lo, hi) in {("P1_regime_aware", "P0_strategic"): s10, ("P2_ews_enhanced", "P0_strategic"): s20,
                                    ("P2_ews_enhanced", "P1_regime_aware"): (s21, lo21, s21 + 0.3),
                                    ("P1_regime_aware", "B2_constant_defensiveness"): (0, -0.1, 0.1),
                                    ("P2_ews_enhanced", "B2_constant_defensiveness"): (0, -0.1, 0.1)}.items():
            for stat in ("sharpe_difference", "max_drawdown_difference"):
                rows.append({"portfolio_a": a, "portfolio_b": b, "statistic": stat, "estimate": e, "ci_low": lo,
                             "ci_high": hi, "method": "stationary bootstrap, mean block 21 (primary)"})
        stats = pd.DataFrame(rows)
        perf = pd.DataFrame([{"portfolio": p, "basis": "net_10bps", "max_drawdown": -0.2, "annualized_one_way_turnover": 1,
                              "total_cost_fraction_10bps": 0.01} for p in ("P0_strategic", "P1_regime_aware", "P2_ews_enhanced")])
        cs = pd.DataFrame([{"portfolio": f"{a} - {b}", "basis": f"net_{x}bps", "sharpe": e}
                           for (a, b), e in ((("P1_regime_aware", "P0_strategic"), s10[0]),
                                             (("P2_ews_enhanced", "P0_strategic"), s20[0]),
                                             (("P2_ews_enhanced", "P1_regime_aware"), s21)) for x in (5, 10, 25)])
        att = pd.DataFrame(columns=["portfolio_a", "portfolio_b", "component", "group", "annualized_mean"])
        return brep.conclusions(PROTOCOL, perf, stats, att, cs, [], pd.DataFrame(), pd.DataFrame(), 10)

    def test_outcomes(self):
        self.assertTrue(self.frames((0.2, 0.05, 0.4), (0.4, 0.1, 0.7), 0.2, 0.05)["outcome"].startswith("Outcome 1"))
        self.assertTrue(self.frames((-0.1, -0.3, 0.1), (-0.05, -0.3, 0.2), 0.05, -0.2)["outcome"].startswith("Outcome 3"))
        self.assertTrue(self.frames((0.3, 0.1, 0.5), (0.3, -0.1, 0.6), 0.0, -0.2)["outcome"].startswith("Outcome 2"))
        self.assertTrue(self.frames((0.1, -0.1, 0.3), (0.1, -0.1, 0.3), 0.0, -0.2)["outcome"].startswith("Outcome 4"))


if __name__ == "__main__":
    unittest.main()
