"""Stage 7 tests: causality, role separation and absence of realized outcomes in portfolio construction."""

from __future__ import annotations

import inspect
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regime_radar import portfolio_constraints as pc  # noqa: E402
from regime_radar import portfolio_engine as eng  # noqa: E402
from regime_radar import portfolio_execution as pe  # noqa: E402

POLICY = pc.load_policy(ROOT / "configs" / "portfolio_policy.json", ROOT / "configs" / "project.json")
CFGJ = json.loads((ROOT / "configs" / "portfolio_policy.json").read_text(encoding="utf-8"))
CFG = eng.EngineConfig(budget_coef=CFGJ["risk_budget"]["coefficients"], min_obs=252, lookback=504)


def setup(n=420, seed=3):
    rng = np.random.default_rng(seed)
    vols = np.array([.18, .20, .25, .06, .05, .09, .16, .20, .004]) / np.sqrt(252)
    x = rng.normal(size=(n, 9)) * vols + 0.0002
    cal = pd.bdate_range("2014-06-02", periods=n)
    simple = pd.DataFrame(x, index=cal, columns=POLICY.assets)
    split = pd.Series(np.where(cal < pd.Timestamp("2015-06-01"), "train", "validation"), index=cal)
    sched = pe.weekly_schedule(cal, split)
    d = pd.Series(np.clip(np.sin(np.arange(n) / 15.0), 0, 1), index=cal)
    return simple, sched, d


def run(simple, sched, d, role="challenger_ews", pid="P2_ews_enhanced"):
    inp = eng.prepare_inputs(simple, sched, POLICY.benchmark, CFG)
    tg, _, _, _ = eng.run_portfolio(pid, role, simple, sched, inp, POLICY, CFG, CFGJ["transaction_costs"],
                                    pd.Series(False, index=simple.index), d, contributions=False)
    return tg


class TestCausality(unittest.TestCase):
    def test_future_return_perturbation(self):
        simple, sched, d = setup()
        base = run(simple, sched, d)
        cut = base["signal_date"].iloc[len(base) // 2]
        pert = simple.copy()
        pert.loc[pert.index > cut] = pert.loc[pert.index > cut] * -5 + 0.03
        new = run(pert, sched, d)
        w = [f"w_{a}" for a in POLICY.assets]
        m = base["signal_date"] <= cut
        np.testing.assert_array_equal(base.loc[m, w].to_numpy(), new.loc[m, w].to_numpy())
        for c in ("vol_ex_ante", "cvar95_daily_ex_ante", "expected_return_ex_ante", "turnover_ex_ante_proxy"):
            np.testing.assert_array_equal(base.loc[m, c].to_numpy(), new.loc[m, c].to_numpy())
        self.assertFalse(np.allclose(base.loc[~m, w].to_numpy(), new.loc[~m, w].to_numpy()))

    def test_future_signals_do_not_change_earlier_targets(self):
        simple, sched, d = setup()
        base = run(simple, sched, d)
        cut = base["signal_date"].iloc[len(base) // 2]
        d2 = d.copy()
        d2[d2.index > cut] = 1.0
        new = run(simple, sched, d2)
        m = base["signal_date"] <= cut
        w = [f"w_{a}" for a in POLICY.assets]
        np.testing.assert_array_equal(base.loc[m, w].to_numpy(), new.loc[m, w].to_numpy())

    def test_risk_inputs_use_only_past_returns(self):
        simple, sched, _ = setup()
        t = sched["signal_date"].iloc[-10]
        a = eng.risk_inputs(simple.loc[:t], POLICY.benchmark, CFG)
        b = eng.prepare_inputs(simple, sched[sched["signal_date"] == t], POLICY.benchmark, CFG)[t]
        np.testing.assert_array_equal(a.cov, b.cov)
        np.testing.assert_array_equal(a.mu, b.mu)
        np.testing.assert_array_equal(a.scenarios, b.scenarios)
        self.assertEqual(a.scenarios.shape[0], min(504, len(simple.loc[:t])))

    def test_execution_convention_in_targets(self):
        simple, sched, d = setup()
        tg = run(simple, sched, d)
        pos = {x: i for i, x in enumerate(simple.index)}
        for _, r in tg.iterrows():
            self.assertEqual(pos[r["execution_date"]], pos[r["signal_date"]] + 1)
            if pd.notna(r["first_holding_return_date"]):
                self.assertEqual(pos[r["first_holding_return_date"]], pos[r["signal_date"]] + 2)

    def test_role_separation_enforced(self):
        simple, sched, d = setup()
        inp = eng.prepare_inputs(simple, sched, POLICY.benchmark, CFG)
        with self.assertRaises(ValueError):
            eng.run_portfolio("P0_strategic", None, simple, sched, inp, POLICY, CFG, CFGJ["transaction_costs"],
                              pd.Series(False, index=simple.index), d)
        with self.assertRaises(ValueError):
            eng.run_portfolio("P1_regime_aware", "baseline_regime", simple, sched, inp, POLICY, CFG,
                              CFGJ["transaction_costs"], pd.Series(False, index=simple.index), None)
        tg = run(simple, sched, d, role="baseline_regime", pid="P1_regime_aware")
        self.assertTrue((tg["signal_role"] == "baseline_regime").all())

    def test_no_forward_outcomes_or_realized_performance_in_engine(self):
        from regime_radar import (expected_returns, portfolio_diagnostics, portfolio_execution, portfolio_optimizer,
                                  portfolio_risk, risk_budgeting)
        for m in (eng, expected_returns, portfolio_diagnostics, portfolio_execution, portfolio_optimizer, portfolio_risk,
                  risk_budgeting):
            src = inspect.getsource(m)
            for banned in ("forward_outcomes", "fwd_ret", "fwd_mdd", "daily_labels", "test_features", "test_returns"):
                self.assertNotIn(banned, src, f"{m.__name__} references {banned}")

    def test_p0_has_no_signal_and_is_benchmark(self):
        simple, sched, _ = setup()
        inp = eng.prepare_inputs(simple, sched, POLICY.benchmark, CFG)
        tg, _, _, _ = eng.run_portfolio("P0_strategic", None, simple, sched, inp, POLICY, CFG, CFGJ["transaction_costs"],
                                        pd.Series(False, index=simple.index), None)
        np.testing.assert_allclose(tg[[f"posture_{a}" for a in POLICY.assets]].to_numpy(),
                                   np.tile(POLICY.benchmark, (len(tg), 1)))
        dist = 0.5 * np.abs(tg[[f"w_{a}" for a in POLICY.assets]].to_numpy() - POLICY.benchmark).sum(axis=1)
        self.assertTrue((dist <= CFG.no_trade_band + 1e-12).all())        # drift tolerated only inside the band
        self.assertTrue((tg["signal_role"] == "none").all())

    def test_no_ratchet_after_a_defensive_spell(self):
        simple, sched, _ = setup()
        d = pd.Series(0.0, index=simple.index)
        dates = sched["signal_date"][sched["signal_date"].isin(eng.prepare_inputs(simple, sched, POLICY.benchmark, CFG))]
        d.loc[dates.iloc[2]:dates.iloc[8]] = 1.0                     # six defensive weeks, then normal again
        tg = run(simple, sched, d)
        late = tg.iloc[14:]                                          # several weeks after the signal normalized
        dist = 0.5 * np.abs(late[[f"w_{a}" for a in POLICY.assets]].to_numpy() - POLICY.benchmark).sum(axis=1)
        self.assertTrue((dist <= CFG.no_trade_band + 1e-9).all(), dist.max())

    def test_defensive_signal_produces_defensive_targets(self):
        simple, sched, _ = setup()
        low = run(simple, sched, pd.Series(0.0, index=simple.index))
        high = run(simple, sched, pd.Series(1.0, index=simple.index))
        tail = slice(5, None)                                        # after the first turnover-limited transitions
        self.assertTrue((high["vol_ex_ante"].iloc[tail].to_numpy() <= low["vol_ex_ante"].iloc[tail].to_numpy() + 1e-9).all())
        self.assertTrue((high["exposure_equity"].iloc[tail].to_numpy() <= low["exposure_equity"].iloc[tail].to_numpy() + 1e-9).all())
        first = high.iloc[0]
        if first["solver_stage"] == 2:                               # unmet target must be reported
            self.assertNotEqual(first["unmet_budget"], "")


if __name__ == "__main__":
    unittest.main()
