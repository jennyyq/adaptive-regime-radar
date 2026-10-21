"""Stage 7 tests: optimizer feasibility, fallback, portfolio mathematics and covariance estimation (synthetic data)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regime_radar import expected_returns as er  # noqa: E402
from regime_radar import portfolio_constraints as pc  # noqa: E402
from regime_radar import portfolio_execution as pe  # noqa: E402
from regime_radar import portfolio_optimizer as po  # noqa: E402
from regime_radar import portfolio_risk as pr  # noqa: E402
from regime_radar import risk_budgeting as rb  # noqa: E402

POLICY = pc.load_policy(ROOT / "configs" / "portfolio_policy.json", ROOT / "configs" / "project.json")
COEF = POLICY.raw["risk_budget"]["coefficients"]


def synthetic_returns(n=600, seed=0):
    rng = np.random.default_rng(seed)
    vols = np.array([.18, .20, .25, .06, .05, .09, .16, .20, .004]) / np.sqrt(252)
    load = np.array([[.8, .8, .8, -.3, -.1, .6, .1, .4, 0.0],      # equity / risk factor
                     [0.0, 0.0, 0.0, .8, .8, .1, .3, 0.0, 0.0]])    # rates factor (a valid PSD factor model)
    idio = np.sqrt(np.maximum(1 - (load ** 2).sum(axis=0), 0.05))
    z = rng.normal(size=(n, 2)) @ load + rng.normal(size=(n, 9)) * idio
    x = z / z.std(axis=0) * vols
    return pd.DataFrame(x, columns=POLICY.assets, index=pd.bdate_range("2010-01-01", periods=n))


def inputs(seed=0):
    r = synthetic_returns(seed=seed)
    cov = pr.estimate_cov(r, "ledoit_wolf", 504).cov_annual
    mu, delta, sb = er.equilibrium(cov, POLICY.benchmark, 0.3, 0.01)
    return r, cov, mu, delta, sb


def solve(d, w_pre=None, gamma_mult=1.0, lam_to=0.008, lam_te=2.0, budget=None, seed=0):
    r, cov, mu, delta, sb = inputs(seed)
    b = budget or rb.budget_from_d(d, COEF)
    return po.solve_mvo(POLICY, mu, cov, delta * gamma_mult, lam_to, lam_te,
                        POLICY.benchmark if w_pre is None else w_pre, b, sb), cov, sb, b


class TestOptimizer(unittest.TestCase):
    def assert_hard(self, res, b, w_pre):
        w = res.weights
        lo, hi = pc.bounds(POLICY)
        self.assertAlmostEqual(w.sum(), 1.0, places=9)
        self.assertTrue((w >= lo - 1e-9).all() and (w <= hi + 1e-9).all())
        self.assertTrue((w >= -1e-12).all())
        self.assertLessEqual(0.5 * np.abs(w - w_pre).sum(), b.turnover_cap + 1e-7)
        self.assertEqual(pc.hard_violations(res.checks), [])

    def test_normal_budget_reproduces_benchmark(self):
        res, cov, sb, b = solve(0.0)
        self.assertEqual(res.status, "optimal")
        np.testing.assert_allclose(res.weights, POLICY.benchmark, atol=1e-6)

    def test_defensive_budgets_feasible_and_respected(self):
        for d in (1 / 3, 2 / 3, 1.0):
            res, cov, sb, b = solve(d)
            self.assert_hard(res, b, POLICY.benchmark)
            if res.status == "optimal":
                self.assertLessEqual(pr.portfolio_vol(res.weights, cov), b.vol_multiplier * sb * (1 + 1e-6))
                self.assertLessEqual(POLICY.group_vector("equity") @ res.weights, b.equity_max + 1e-7)
                self.assertGreaterEqual(res.weights[-1], b.liquidity_min - 1e-7)

    def test_turnover_cap_and_unmet_budget_reported(self):
        tight = rb.RiskBudget(1.0, 0.4, 0.30, 0.05, 0.20, 0.02)       # infeasible within a 2% turnover cap
        res, cov, sb, b = solve(1.0, budget=tight)
        self.assertEqual(res.status, "transition_fallback")
        self.assertEqual(res.stage, 2)
        self.assertTrue(res.unmet_budget)
        self.assert_hard(res, tight, POLICY.benchmark)
        self.assertLess(pr.portfolio_vol(res.weights, cov), sb)            # moved toward the target

    def test_deterministic_and_start_independent(self):
        a, *_ = solve(2 / 3)
        b, *_ = solve(2 / 3)
        np.testing.assert_array_equal(a.weights, b.weights)
        self.assertTrue(np.isnan(a.start_disagreement) or a.start_disagreement < 1e-5)

    def test_infeasible_policy_detected(self):
        import copy
        bad = copy.deepcopy(POLICY)
        bad.group_min["equity"] = 0.60
        self.assertFalse(pc.linear_feasible(bad)[0])

    def test_hold_fallback_satisfies_hard_constraints(self):
        import copy
        bad = copy.deepcopy(POLICY)
        bad.group_min["equity"], bad.group_max["equity"] = 0.6, 0.55      # nothing is feasible
        r, cov, mu, delta, sb = inputs()
        res = po.solve_mvo(bad, mu, cov, delta, 0.008, 2.0, POLICY.benchmark, rb.budget_from_d(0, COEF), sb)
        self.assertEqual(res.stage, 3)
        self.assertEqual(res.status, "fallback_violates_hard_constraints")   # reported, never silent

    def test_higher_risk_aversion_does_not_raise_risk(self):
        loose = rb.RiskBudget(0.0, 10.0, 0.55, 0.10, 0.02, 1.0)          # risk budget and turnover not binding
        vols = []
        for g in (0.5, 1.0, 2.0, 4.0):
            res, cov, sb, b = solve(0.0, gamma_mult=g, budget=loose, lam_te=0.0, lam_to=0.0)
            vols.append(pr.portfolio_vol(res.weights, cov))
        self.assertTrue(all(vols[i + 1] <= vols[i] + 1e-7 for i in range(len(vols) - 1)), vols)

    def test_higher_defensiveness_does_not_raise_credit(self):
        credit = [POLICY.group_vector("credit") @ solve(d)[0].weights for d in (0, 1 / 3, 2 / 3, 1)]
        self.assertTrue(all(credit[i + 1] <= credit[i] + 1e-7 for i in range(3)), credit)

    def test_cvar_lp_respects_cap(self):
        r, cov, mu, delta, sb = inputs()
        sc = r.to_numpy()[-504:]
        b = rb.budget_from_d(0.5, COEF)
        _, cvar_b = pr.historical_var_cvar(sc, POLICY.benchmark)
        w, st = po.solve_cvar_lp(POLICY, mu, sc, POLICY.benchmark, b, b.vol_multiplier * cvar_b, 0.008, 0.02)
        self.assertEqual(st, "optimal")
        self.assertLessEqual(pr.historical_var_cvar(sc, w)[1], b.vol_multiplier * cvar_b + 1e-8)
        self.assertEqual(pc.hard_violations(pc.check_weights(POLICY, w, b, POLICY.benchmark, b.turnover_cap)), [])

    def test_frontier_increasing(self):
        r, cov, mu, delta, sb = inputs()
        f = pd.DataFrame(po.efficient_frontier(POLICY, mu, cov, rb.budget_from_d(0, COEF), points=8))
        self.assertTrue((np.diff(f["expected_return"]) >= -1e-9).all())
        self.assertTrue((np.diff(f["vol"]) >= -1e-9).all())

    def test_benchmark_feasible(self):
        self.assertTrue(pc.linear_feasible(POLICY)[0])
        self.assertEqual(pc.hard_violations(pc.check_weights(POLICY, POLICY.benchmark)), [])
        self.assertTrue(pc.linear_feasible(POLICY, rb.budget_from_d(1.0, COEF))[0])


class TestTargetThenTrade(unittest.TestCase):
    def test_posture_is_path_independent(self):
        r, cov, mu, delta, sb = inputs()
        b = rb.budget_from_d(2 / 3, COEF)
        w1, s1, _, _ = po.solve_target(POLICY, mu, cov, delta, 2.0, b, sb)
        w_far = np.array([0.20, 0.05, 0.02, 0.30, 0.15, 0.03, 0.10, 0.05, 0.10])
        res, post = po.solve_policy(POLICY, mu, cov, delta, 2.0, w_far, b, sb, 0.01)
        np.testing.assert_allclose(post, w1, atol=1e-9)
        w0, _, _, _ = po.solve_target(POLICY, mu, cov, delta, 2.0, rb.budget_from_d(0, COEF), sb)
        np.testing.assert_allclose(w0, POLICY.benchmark, atol=1e-6)       # normal budget = benchmark

    def test_hold_inside_band_and_trade_outside(self):
        r, cov, mu, delta, sb = inputs()
        b = rb.budget_from_d(0, COEF)
        near = POLICY.benchmark + np.r_[0.004, -0.004, np.zeros(7)]
        res, post = po.solve_policy(POLICY, mu, cov, delta, 2.0, near, b, sb, 0.01)
        self.assertEqual(res.status, "hold_within_no_trade_band")
        np.testing.assert_array_equal(res.weights, near)
        far = POLICY.benchmark + np.r_[0.03, -0.03, np.zeros(7)]
        res, post = po.solve_policy(POLICY, mu, cov, delta, 2.0, far, b, sb, 0.01)
        self.assertEqual(res.status, "trade_to_target")

    def test_turnover_limited_transition_is_on_the_line_and_feasible(self):
        r, cov, mu, delta, sb = inputs()
        b = rb.RiskBudget(1.0, 0.65, 0.30, 0.05, 0.20, 0.05)              # 5% cap: cannot reach RED in one week
        res, post = po.solve_policy(POLICY, mu, cov, delta, 2.0, POLICY.benchmark, b, sb, 0.01)
        self.assertEqual(res.status, "turnover_limited_transition")
        k = 0.05 / (0.5 * np.abs(post - POLICY.benchmark).sum())
        np.testing.assert_allclose(res.weights, POLICY.benchmark + k * (post - POLICY.benchmark), atol=1e-12)
        self.assertAlmostEqual(0.5 * np.abs(res.weights - POLICY.benchmark).sum(), 0.05)
        self.assertEqual(pc.hard_violations(res.checks), [])
        self.assertTrue(res.unmet_budget)

    def test_p0_rule_without_budget(self):
        r, cov, mu, delta, sb = inputs()
        drifted = POLICY.benchmark + np.r_[0.02, -0.02, np.zeros(7)]
        res, post = po.solve_policy(POLICY, mu, cov, delta, 2.0, drifted, None, sb, 0.01)
        np.testing.assert_array_equal(post, POLICY.benchmark)
        np.testing.assert_allclose(res.weights, POLICY.benchmark)


class TestPortfolioMath(unittest.TestCase):
    def test_simple_vs_log_returns(self):
        lr = pd.DataFrame({"A": [np.log(1.1), np.log(0.8)], "B": [0.0, np.log(1.05)]})
        sr = pr.to_simple(lr)
        np.testing.assert_allclose(sr["A"], [0.1, -0.2])
        w = np.array([0.5, 0.5])
        self.assertAlmostEqual(pe.portfolio_simple_return(w, sr.iloc[0].to_numpy()), 0.05)
        self.assertNotAlmostEqual(np.expm1(w @ lr.iloc[0].to_numpy()), 0.05)   # weighted log returns are not exact

    def test_covariance_annualization_and_volatility(self):
        r = synthetic_returns()
        est = pr.estimate_cov(r, "sample", 504)
        np.testing.assert_allclose(est.cov_annual, 252 * np.cov(r.to_numpy()[-504:], rowvar=False))
        w = np.ones(9) / 9
        self.assertAlmostEqual(pr.portfolio_vol(w, est.cov_annual), np.sqrt(w @ est.cov_annual @ w))

    def test_ledoit_wolf_matches_sklearn(self):
        from sklearn.covariance import LedoitWolf
        x = synthetic_returns().to_numpy()
        cov, s = pr.ledoit_wolf(x)
        np.testing.assert_allclose(cov, LedoitWolf().fit(x).covariance_, atol=1e-15)
        self.assertTrue(0 <= s <= 1)

    def test_ewma_effective_sample_size(self):
        x = synthetic_returns(n=2000).to_numpy()
        _, ess = pr.ewma_cov(x, 63)
        lam = 0.5 ** (1 / 63)
        self.assertAlmostEqual(ess, (1 + lam) / (1 - lam), delta=0.5)

    def test_psd_stabilization(self):
        bad = np.array([[1.0, 1.2], [1.2, 1.0]])                    # indefinite
        fixed, n = pr.psd_stabilize(bad, 1e-8)
        self.assertEqual(n, 1)
        self.assertGreater(np.linalg.eigvalsh(fixed).min(), 0)

    def test_risk_contributions_sum_to_volatility(self):
        cov = np.array([[0.04, 0.006], [0.006, 0.01]])
        w = np.array([0.6, 0.4])
        mcr, ccr = pr.risk_contributions(w, cov)
        s = np.sqrt(w @ cov @ w)
        self.assertAlmostEqual(ccr.sum(), s)
        np.testing.assert_allclose(mcr, cov @ w / s)

    def test_var_cvar_small_example(self):
        sc = np.array([[-0.10], [-0.05], [0.0], [0.01], [0.02], [0.03], [0.04], [0.05], [0.06], [0.07]])
        var, cvar = pr.historical_var_cvar(sc, np.array([1.0]), alpha=0.8)
        self.assertAlmostEqual(var, 0.0)                             # ceil(0.8 x 10) = 8th smallest loss of 10
        self.assertAlmostEqual(cvar, 0.075)                          # mean of the worst 20%: (0.10 + 0.05) / 2

    def test_concentration_tracking_cost(self):
        w = np.array([0.5, 0.5, 0.0])
        self.assertAlmostEqual(pr.herfindahl(w), 0.5)
        self.assertAlmostEqual(pr.effective_n(w), 2.0)
        cov = np.diag([0.04, 0.01, 0.0001])
        self.assertAlmostEqual(pr.tracking_error(w, np.array([0.5, 0.5, 0.0]), cov), 0.0)
        self.assertAlmostEqual(pr.tracking_error(np.array([0.6, 0.4, 0]), w, cov), np.sqrt(0.01 * 0.04 + 0.01 * 0.01))
        self.assertAlmostEqual(pe.transaction_cost(0.10, 10), 0.0002)           # 10 bps x 20% traded notional
        self.assertAlmostEqual(pe.transaction_cost(0.10, 10, True, 2.0), 0.0004)
        self.assertAlmostEqual(pe.one_way_turnover(np.array([0.6, 0.4]), np.array([0.5, 0.5])), 0.1)

    def test_expected_return_equilibrium(self):
        cov = np.diag([0.04, 0.01])
        wb = np.array([0.5, 0.5])
        mu, delta, sb = er.equilibrium(cov, wb, 0.3, 0.02)
        self.assertAlmostEqual(sb, np.sqrt(0.25 * 0.05))
        np.testing.assert_allclose(mu, 0.02 + 0.3 / sb * cov @ wb)
        self.assertAlmostEqual((mu - 0.02) @ wb / sb, 0.3)           # benchmark Sharpe recovered


if __name__ == "__main__":
    unittest.main()
