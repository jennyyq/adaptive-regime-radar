"""Stage 8 tests: execution-day rule, cost units and the causal cost-stress flag."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage8_fixtures import market  # noqa: E402

from regime_radar import backtest_execution as bx  # noqa: E402


def order(kind, target, cap=0.2, stress=False):
    return bx.Order("P", pd.Timestamp("2020-01-03"), pd.Timestamp("2020-01-06"), pd.Timestamp("2020-01-07"), kind,
                    np.asarray(target, float), cap, stress, "trade_to_target")


class TestExecutionRule(unittest.TestCase):
    def test_hold_executes_nothing(self):
        pre = np.array([0.52, 0.48])
        e = bx.execute(order("hold", [0.5, 0.5]), pre)
        np.testing.assert_array_equal(e.executed, pre)
        self.assertEqual(e.turnover, 0.0)
        self.assertEqual(e.gross_notional, 0.0)

    def test_trade_within_cap_reaches_decision_target_exactly(self):
        e = bx.execute(order("trade", [0.6, 0.4], cap=0.2), np.array([0.5, 0.5]))
        np.testing.assert_allclose(e.executed, [0.6, 0.4], atol=1e-15)
        self.assertAlmostEqual(e.turnover, 0.1, places=15)
        self.assertAlmostEqual(e.gross_notional, 0.2, places=15)
        self.assertFalse(e.projected)

    def test_cap_preserving_projection(self):
        pre = np.array([0.5, 0.3, 0.2])
        tgt = np.array([0.1, 0.5, 0.4])
        e = bx.execute(order("trade", tgt, cap=0.15), pre)
        self.assertTrue(e.projected)
        self.assertAlmostEqual(e.turnover, 0.15, places=14)
        np.testing.assert_allclose(e.executed, pre + 0.15 / 0.4 * (tgt - pre), atol=1e-15)
        self.assertTrue(e.exceptions)

    def test_execution_uses_execution_day_pre_trade_weights(self):
        # the same decision target executed from different drifted holdings gives different trades
        a = bx.execute(order("trade", [0.5, 0.5]), np.array([0.55, 0.45]))
        b = bx.execute(order("trade", [0.5, 0.5]), np.array([0.53, 0.47]))
        self.assertAlmostEqual(a.turnover, 0.05, places=15)
        self.assertAlmostEqual(b.turnover, 0.03, places=15)


class TestCosts(unittest.TestCase):
    def test_basis_point_units_on_gross_notional(self):
        self.assertAlmostEqual(bx.cost_fraction(1.0, 10, False, 2.0), 0.001, places=16)
        self.assertAlmostEqual(bx.cost_fraction(0.4, 10, False, 2.0), 0.0004, places=16)   # 0.2 one-way = 0.4 notional
        self.assertAlmostEqual(bx.cost_fraction(0.4, 25, True, 2.0), 0.002, places=16)
        self.assertAlmostEqual(bx.cost_fraction(0.4, 5, False, 2.0), 0.0002, places=16)

    def test_not_applied_to_one_way_turnover(self):
        e = bx.execute(order("trade", [0.6, 0.4]), np.array([0.5, 0.5]))
        self.assertAlmostEqual(bx.cost_fraction(e.gross_notional, 10, False, 2.0), 0.0002, places=16)

    def test_stress_flag_is_causal_and_role_free(self):
        simple, *_ = market(n=200)
        s = bx.stress_indicator(simple, 0.15)
        pert = simple.copy()
        cut = simple.index[120]
        pert.loc[pert.index > cut, "SPY"] *= 10
        s2 = bx.stress_indicator(pert, 0.15)
        pd.testing.assert_series_equal(s.loc[:cut], s2.loc[:cut])
        self.assertFalse(s.loc[:simple.index[19]].any())               # no flag before 21 observations
        import inspect
        src = inspect.getsource(bx.stress_indicator)
        for banned in ("baseline_", "challenger_", "alert", "p_non_normal"):
            self.assertNotIn(banned, src)


if __name__ == "__main__":
    unittest.main()
