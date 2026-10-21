"""Stage 7 tests: signal -> defensiveness -> risk budget mappings and economic risk responses (synthetic data)."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regime_radar import risk_budgeting as rb  # noqa: E402

CFG = json.loads((ROOT / "configs" / "portfolio_policy.json").read_text(encoding="utf-8"))
COEF = CFG["risk_budget"]["coefficients"]
LEVELS = CFG["risk_budget"]["P2_ews_levels"]


class TestBudgetMapping(unittest.TestCase):
    def test_endpoints_match_documented_policy(self):
        b0, b1 = rb.budget_from_d(0.0, COEF), rb.budget_from_d(1.0, COEF)
        self.assertEqual((b0.vol_multiplier, b0.equity_max, b0.credit_max, b0.liquidity_min, b0.turnover_cap),
                         (1.0, 0.55, 0.10, 0.02, 0.20))
        self.assertAlmostEqual(b1.vol_multiplier, 0.65)
        self.assertAlmostEqual(b1.equity_max, 0.30)
        self.assertAlmostEqual(b1.credit_max, 0.05)
        self.assertAlmostEqual(b1.liquidity_min, 0.20)
        self.assertAlmostEqual(b1.turnover_cap, 0.30)

    def test_budget_monotone_in_defensiveness(self):
        bs = [rb.budget_from_d(d, COEF) for d in np.linspace(0, 1, 11)]
        for a, b in zip(bs, bs[1:]):
            self.assertLessEqual(b.vol_multiplier, a.vol_multiplier)
            self.assertLessEqual(b.equity_max, a.equity_max)
            self.assertLessEqual(b.credit_max, a.credit_max)          # credit stress never raises credit room
            self.assertGreaterEqual(b.liquidity_min, a.liquidity_min)

    def test_red_never_larger_budget_than_green(self):
        g = rb.budget_from_d(rb.p2_defensiveness("GREEN", LEVELS), COEF)
        r = rb.budget_from_d(rb.p2_defensiveness("RED", LEVELS), COEF)
        self.assertLess(r.vol_multiplier, g.vol_multiplier)
        self.assertLess(r.equity_max, g.equity_max)
        order = [rb.p2_defensiveness(x, LEVELS) for x in ("GREEN", "AMBER", "ORANGE", "RED")]
        self.assertEqual(order, sorted(order))
        self.assertEqual(order[-1], 1.0)                              # RED = documented maximum defensive target

    def test_recovery_keeps_amber_posture(self):
        self.assertEqual(rb.p2_defensiveness("RECOVERY", LEVELS), rb.p2_defensiveness("AMBER", LEVELS))
        self.assertGreater(rb.p2_defensiveness("RECOVERY", LEVELS), rb.p2_defensiveness("GREEN", LEVELS))

    def test_invalid_inputs_raise(self):
        with self.assertRaises(ValueError):
            rb.p2_defensiveness("PURPLE", LEVELS)
        with self.assertRaises(ValueError):
            rb.budget_from_d(1.2, COEF)
        with self.assertRaises(ValueError):
            rb.p1_defensiveness(np.nan, 0.3)

    def test_p1_lipschitz_and_extremes(self):
        a = 0.25
        self.assertEqual(rb.p1_defensiveness(0.0, a), 0.0)
        self.assertAlmostEqual(rb.p1_defensiveness(1.0, a), a)
        self.assertEqual(rb.p1_defensiveness(1.0, 5.0), 1.0)          # capped at the maximum defensive target
        for p in (0.9999, 0.999999999, 1.0 - 1e-15):                 # no jump near the saturation point
            self.assertLessEqual(abs(rb.p1_defensiveness(p, a) - rb.p1_defensiveness(1.0, a)), a * (1 - p) + 1e-15)

    def test_parity_scale_equalizes_mean_defensiveness(self):
        rng = np.random.default_rng(0)
        p = pd.Series(np.r_[np.ones(400), np.zeros(800), rng.uniform(size=300)])
        lv = pd.Series(rng.choice(["GREEN", "AMBER", "ORANGE", "RED", "RECOVERY"], size=1500, p=[.7, .1, .08, .05, .07]))
        a = rb.parity_scale(p, lv, LEVELS)
        self.assertAlmostEqual(np.minimum(1, a * p).mean(), lv.map(LEVELS).mean(), places=9)

    def test_continuous_score_mapping(self):
        self.assertEqual(rb.p2_defensiveness_continuous(0.3, 0.5, 0.8), 0.0)
        self.assertAlmostEqual(rb.p2_defensiveness_continuous(0.65, 0.5, 0.8), 0.5)
        self.assertEqual(rb.p2_defensiveness_continuous(0.95, 0.5, 0.8), 1.0)
        self.assertEqual(rb.p2_defensiveness_continuous(np.nan, 0.5, 0.8), 0.0)


if __name__ == "__main__":
    unittest.main()
