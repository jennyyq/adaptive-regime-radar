"""Stage 8 tests: sequential engine reuses the frozen Stage 7 decision exactly; portfolio specs behave as frozen."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage8_fixtures import CFG, COSTS, POLICY, market, no_stress  # noqa: E402

from regime_radar import backtest_engine as be  # noqa: E402
from regime_radar import portfolio_engine as eng  # noqa: E402
from regime_radar.portfolio_execution import drift  # noqa: E402

W = [f"w_{a}" for a in POLICY.assets]


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.simple, cls.split, cls.sched, cls.d = market(n=360)
        cls.inp = eng.prepare_inputs(cls.simple, cls.sched, POLICY.benchmark, CFG)
        cls.t0 = min(cls.inp)

    def run_spec(self, spec, start=None):
        return be.run_sequential(spec, self.simple, self.sched, self.inp, POLICY, CFG, COSTS, no_stress(self.simple),
                                 start or self.t0)


class TestStage7Reuse(Base):
    def test_decision_with_stage7_proxy_is_bit_identical_to_run_portfolio(self):
        for pid, role, d in (("P2_ews_enhanced", "challenger_ews", self.d), ("P0_strategic", None, None)):
            ref, _, _, _ = eng.run_portfolio(pid, role, self.simple, self.sched, self.inp, POLICY, CFG, COSTS,
                                             no_stress(self.simple), d, contributions=False)
            prev, hold = None, None
            for _, r in ref.iterrows():
                t = r["signal_date"]
                if prev is None:
                    w_pre = POLICY.benchmark.copy()
                else:
                    held = self.simple.loc[hold:t] if pd.notna(hold) and hold <= t else self.simple.iloc[0:0]
                    w_pre = drift(prev, held.to_numpy())
                row, res, _, _ = be.decide_stage7_row(POLICY, CFG, COSTS, self.inp[t], w_pre, prev,
                                                      None if d is None else d.loc[t], False)
                np.testing.assert_array_equal(res.weights, r[W].to_numpy(float))
                self.assertEqual(row["solver_status"], r["solver_status"])
                prev, hold = res.weights, r["first_holding_return_date"]


class TestSequentialEngine(Base):
    def test_execution_next_session_and_holdings_at_decision_are_actual(self):
        run = self.run_spec(be.PortfolioSpec("P2_ews_enhanced", "optimizer", "challenger_ews", self.d,
                                             start_weights=POLICY.benchmark))
        cal = self.simple.index
        pos = {d: i for i, d in enumerate(cal)}
        ex = run.executions
        self.assertTrue(len(ex) > 5)
        self.assertTrue((ex["execution_date"].map(pos) == ex["signal_date"].map(pos) + 1).all())
        post = run.ledger.weights("post")
        dec = run.decisions.set_index("signal_date")
        for t in dec.index:
            np.testing.assert_array_equal(dec.loc[t, [f"pretrade_{a}" for a in POLICY.assets]].to_numpy(float),
                                          post.loc[t].to_numpy())
        # new holdings start earning at t+2: weights held over the execution date are the signal-date holdings
        f = run.ledger.frame().set_index("date")
        for r in ex[ex["gross_notional"] > 0].itertuples():
            self.assertAlmostEqual(f.loc[r.execution_date, "gross_return"],
                                   float(post.loc[r.signal_date] @ self.simple.loc[r.execution_date]), places=15)
            nxt = cal[pos[r.execution_date] + 1] if pos[r.execution_date] + 1 < len(cal) else None
            if nxt is not None:
                self.assertAlmostEqual(f.loc[nxt, "gross_return"],
                                       float(post.loc[r.execution_date] @ self.simple.loc[nxt]), places=15)

    def test_p0_posture_is_benchmark_and_no_signal(self):
        run = self.run_spec(be.PortfolioSpec("P0_strategic", "optimizer", None, None, start_weights=POLICY.benchmark))
        post = run.decisions[[f"posture_{a}" for a in POLICY.assets]].to_numpy()
        np.testing.assert_allclose(post, np.tile(POLICY.benchmark, (len(post), 1)))
        self.assertTrue(run.decisions["budget_d_input"].isna().all())
        self.assertTrue((run.decisions["signal_role"] == "none").all())

    def test_constant_defensiveness_budget(self):
        run = self.run_spec(be.PortfolioSpec("B2", "optimizer", None, 0.4, start_weights=POLICY.benchmark))
        self.assertTrue((run.decisions["budget_d_input"] == 0.4).all())
        self.assertTrue((run.decisions["budget_d"] == 0.4).all())

    def test_buy_and_hold_never_trades(self):
        run = self.run_spec(be.PortfolioSpec("B3", "buy_and_hold", start_weights=POLICY.benchmark))
        self.assertTrue(run.decisions.empty)
        self.assertEqual(run.ledger.frame()["gross_notional"].sum(), 0.0)

    def test_fixed_6040_rebalances_only_outside_band(self):
        w = np.zeros(len(POLICY.assets))
        w[POLICY.assets.index("SPY")], w[POLICY.assets.index("IEF")] = 0.6, 0.4
        run = self.run_spec(be.PortfolioSpec("B1", "fixed", None, None, fixed_target=w, fixed_cap=1.0, start_weights=w))
        dec = run.decisions
        to = 0.5 * np.abs(dec[[f"pretrade_{a}" for a in POLICY.assets]].to_numpy() - w).sum(axis=1)
        self.assertTrue(((to > CFG.no_trade_band) == (dec["order_kind"] == "trade")).all())
        self.assertEqual(set(np.flatnonzero(run.ledger.weights().to_numpy().sum(axis=0) > 0)),
                         {POLICY.assets.index("SPY"), POLICY.assets.index("IEF")})

    def test_spec_requires_inputs_and_records_warnings_list(self):
        run = self.run_spec(be.PortfolioSpec("P1_regime_aware", "optimizer", "baseline_regime", self.d,
                                             start_weights=POLICY.benchmark))
        self.assertIsInstance(run.warnings, list)
        self.assertEqual(run.decisions["signal_date"].min(), self.t0)
        self.assertTrue((run.decisions["hard_violations"].fillna("") == "").all())

    def test_restart_and_continuation_share_calendar(self):
        a = self.run_spec(be.PortfolioSpec("P0_strategic", "optimizer", None, None, start_weights=POLICY.benchmark))
        later = self.sched["signal_date"].iloc[-10]
        b = self.run_spec(be.PortfolioSpec("P0_strategic", "optimizer", None, None, start_weights=POLICY.benchmark),
                          start=later)
        self.assertTrue(set(b.decisions["signal_date"]) <= set(a.decisions["signal_date"]))


if __name__ == "__main__":
    unittest.main()
