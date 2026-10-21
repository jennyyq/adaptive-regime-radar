"""Stage 8 tests: self-financing ledger, analytic examples and reconciliation (with deliberate corruption)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage8_fixtures import market  # noqa: E402

from regime_radar import backtest_accounting as ba  # noqa: E402
from regime_radar import backtest_execution as bx  # noqa: E402
from regime_radar import stage8_preflight as sp  # noqa: E402


def two_asset_run(n=30, seed=1, cap=0.05, stress_every=3):
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range("2021-01-04", periods=n)
    simple = pd.DataFrame(rng.normal(0.0005, 0.01, (n, 2)), index=cal, columns=["A", "B"])
    lg = ba.Ledger("X", ["A", "B"], cal[0], np.array([0.5, 0.5]), bps=(5, 10, 25), primary_bps=10)
    pending, exe_dates = None, set()
    for i in range(1, n):
        o = pending if pending is not None and pending.execution_date == cal[i] else None
        lg.session(cal[i], simple.iloc[i].to_numpy(), o)
        pending = None if o is not None else pending
        if i % 5 == 0 and i + 1 < n:
            kind = "hold" if i % 10 == 0 else "trade"
            tgt = np.array([0.8, 0.2]) if (i // 5) % 2 else np.array([0.3, 0.7])
            pending = bx.Order("X", cal[i], cal[i + 1], cal[i + 2] if i + 2 < n else None, kind, tgt, cap,
                               bool(i % stress_every == 0), "test")
            exe_dates.add(cal[i + 1])
    return lg, simple, exe_dates


class TestAnalyticExamples(unittest.TestCase):
    def test_all_examples_pass(self):
        rows = pd.DataFrame(sp.analytic_examples())
        self.assertGreaterEqual(len(rows), 10)
        self.assertTrue((rows["status"] == "PASS").all(), rows[rows["status"] != "PASS"].to_string())


class TestLedger(unittest.TestCase):
    def test_reconciliation_passes(self):
        lg, simple, ex = two_asset_run()
        rows = pd.DataFrame(ba.reconcile(lg, simple, ex))
        self.assertTrue((rows["status"] == "PASS").all(), rows[rows["status"] != "PASS"].to_string())
        self.assertTrue(lg.frame()["traded"].any())

    def test_turnover_cap_respected_and_costs_reduce_wealth(self):
        lg, simple, ex = two_asset_run(cap=0.05)
        f = lg.frame()
        self.assertLessEqual(f["turnover"].max(), 0.05 + 1e-12)
        self.assertGreater(f["wealth_gross"].iloc[-1], f["wealth_net_5bps"].iloc[-1])
        self.assertGreater(f["wealth_net_5bps"].iloc[-1], f["wealth_net_10bps"].iloc[-1])
        self.assertGreater(f["wealth_net_10bps"].iloc[-1], f["wealth_net_25bps"].iloc[-1])

    def test_zero_cost_equals_gross(self):
        cal = pd.bdate_range("2021-01-04", periods=4)
        lg = ba.Ledger("X", ["A", "B"], cal[0], np.array([0.5, 0.5]), bps=(0, 10), primary_bps=10)
        lg.session(cal[1], np.array([0.01, -0.02]), None)
        lg.session(cal[2], np.array([0.0, 0.01]), bx.Order("X", cal[1], cal[2], cal[3], "trade", np.array([0.9, 0.1]),
                                                          1.0, False, "t"))
        lg.session(cal[3], np.array([0.02, 0.0]), None)
        f = lg.frame()
        np.testing.assert_array_equal(f["wealth_gross"].to_numpy(), f["wealth_net_0bps"].to_numpy())

    def test_order_on_wrong_date_refused(self):
        cal = pd.bdate_range("2021-01-04", periods=3)
        lg = ba.Ledger("X", ["A", "B"], cal[0], np.array([0.5, 0.5]))
        with self.assertRaises(AssertionError):
            lg.session(cal[1], np.zeros(2), bx.Order("X", cal[0], cal[2], None, "trade", np.array([1.0, 0.0]), 1.0,
                                                     False, "t"))

    def test_invalid_initial_weights(self):
        with self.assertRaises(ValueError):
            ba.Ledger("X", ["A", "B"], pd.Timestamp("2021-01-04"), np.array([0.7, 0.7]))
        with self.assertRaises(ValueError):
            ba.Ledger("X", ["A", "B"], pd.Timestamp("2021-01-04"), np.array([1.2, -0.2]))


class TestReconciliationDetectsCorruption(unittest.TestCase):
    """Leakage / accounting-injection cases: every identity must catch a deliberate error."""

    def failed(self, lg, simple, ex):
        rows = pd.DataFrame(ba.reconcile(lg, simple, ex))
        return set(rows.loc[rows["status"] == "FAIL", "check"])

    def test_double_counted_cost(self):
        lg, simple, ex = two_asset_run()
        i = next(k for k, r in enumerate(lg.records) if r["traded"])
        lg.records[i]["cost_fraction_10bps"] *= 2
        self.assertTrue(any("Transaction cost" in c for c in self.failed(lg, simple, ex)))

    def test_unrecorded_rebalance(self):
        lg, simple, ex = two_asset_run()
        i = next(k for k, r in enumerate(lg.records) if not r["traded"] and k > 0)
        lg.w_post[i] = np.array([0.9, 0.1])
        self.assertTrue(self.failed(lg, simple, ex))

    def test_new_holdings_earning_execution_day_return(self):
        lg, simple, ex = two_asset_run()
        i = next(k for k, r in enumerate(lg.records) if r["traded"])
        lg.records[i]["gross_return"] = float(lg.w_post[i] @ simple.loc[lg.dates[i]].to_numpy())
        self.assertTrue(any("Gross return" in c for c in self.failed(lg, simple, ex)))

    def test_phantom_execution_date(self):
        lg, simple, ex = two_asset_run()
        traded = {d for d, r in zip(lg.dates, lg.records) if r["traded"]}
        self.assertTrue(any("phantom" in c for c in self.failed(lg, simple, ex - {sorted(traded)[0]})))

    def test_omitted_session(self):
        lg, simple, ex = two_asset_run()
        for lst in (lg.dates, lg.records, lg.w_post, lg.w_pre):
            del lst[5]
        self.assertTrue(any("omitted" in c for c in self.failed(lg, simple, ex)))

    def test_wealth_not_compounded(self):
        lg, simple, ex = two_asset_run()
        lg.records[-1]["wealth_net_10bps"] *= 1.01
        self.assertTrue(any("Net wealth" in c for c in self.failed(lg, simple, ex)))


class TestRealisticRun(unittest.TestCase):
    def test_nine_asset_market_reconciles(self):
        from stage8_fixtures import CFG, COSTS, POLICY, no_stress
        from regime_radar import backtest_engine as be
        from regime_radar import portfolio_engine as eng
        simple, split, sched, d = market(n=330)
        inp = eng.prepare_inputs(simple, sched, POLICY.benchmark, CFG)
        t0 = min(inp)
        run = be.run_sequential(be.PortfolioSpec("P2_ews_enhanced", "optimizer", "challenger_ews", d,
                                                 start_weights=POLICY.benchmark), simple, sched, inp, POLICY, CFG, COSTS,
                                no_stress(simple), t0)
        rows = pd.DataFrame(ba.reconcile(run.ledger, simple, set(sched["execution_date"])))
        self.assertTrue((rows["status"] == "PASS").all(), rows[rows["status"] != "PASS"].to_string())


if __name__ == "__main__":
    unittest.main()
