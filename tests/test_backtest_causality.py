"""Stage 8 tests: prefix invariance, future-perturbation invariance, role separation and leakage injection."""

from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage8_fixtures import CFG, COSTS, POLICY, market, no_stress  # noqa: E402

from regime_radar import backtest_engine as be  # noqa: E402
from regime_radar import backtest_signals as bs  # noqa: E402
from regime_radar import portfolio_engine as eng  # noqa: E402
from regime_radar import stage7_handoff as h7  # noqa: E402

W = [f"w_{a}" for a in POLICY.assets]


def run(simple, sched, d, inputs=None, start=None, end=None):
    inp = inputs if inputs is not None else eng.prepare_inputs(simple, sched, POLICY.benchmark, CFG)
    spec = be.PortfolioSpec("P2_ews_enhanced", "optimizer", "challenger_ews", d, start_weights=POLICY.benchmark)
    return be.run_sequential(spec, simple, sched, inp, POLICY, CFG, COSTS, no_stress(simple), start or min(inp),
                             end_date=end)


def leaky_inputs(simple, sched):
    """DELIBERATE LEAK: risk inputs at t computed from returns through t+1."""
    out = {}
    cal = simple.index
    for t in sched["signal_date"]:
        i = cal.get_loc(t)
        hist = simple.iloc[:i + 2]
        if i + 1 >= CFG.min_obs:
            out[t] = eng.risk_inputs(hist, POLICY.benchmark, CFG)
    return out


def prefix_violation(run_a, run_b, cut) -> float:
    a = run_a.decisions[run_a.decisions["signal_date"] <= cut][W].to_numpy()
    b = run_b.decisions[run_b.decisions["signal_date"] <= cut][W].to_numpy()
    if a.shape != b.shape:
        return np.inf
    wa, wb = run_a.ledger.weights().loc[:cut].to_numpy(), run_b.ledger.weights().loc[:cut].to_numpy()
    return float(max(np.abs(a - b).max(initial=0), np.abs(wa - wb).max(initial=0)))


class TestPrefixInvariance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.simple, cls.split, cls.sched, cls.d = market(n=380)
        cls.full = run(cls.simple, cls.sched, cls.d)
        dates = cls.full.decisions["signal_date"]
        cls.cut = dates.iloc[len(dates) // 2]

    def test_truncated_run_equals_full_prefix(self):
        part = run(self.simple, self.sched, self.d, end=self.cut + pd.Timedelta(days=5))
        self.assertEqual(prefix_violation(self.full, part, self.cut), 0.0)

    def test_future_returns_do_not_change_past(self):
        pert = self.simple.copy()
        pert.loc[pert.index > self.cut] = pert.loc[pert.index > self.cut] * -4 + 0.02
        other = run(pert, self.sched, self.d)
        self.assertEqual(prefix_violation(self.full, other, self.cut), 0.0)
        later = self.full.decisions["signal_date"] > self.cut
        self.assertFalse(np.allclose(self.full.decisions.loc[later, W].to_numpy(),
                                     other.decisions.loc[later.to_numpy(), W].to_numpy()))

    def test_future_signals_do_not_change_past(self):
        d2 = self.d.copy()
        d2.loc[d2.index > self.cut] = 1.0 - d2.loc[d2.index > self.cut]
        other = run(self.simple, self.sched, d2)
        self.assertEqual(prefix_violation(self.full, other, self.cut), 0.0)

    def test_leak_injection_is_detected(self):
        leak = leaky_inputs(self.simple, self.sched)
        pert = self.simple.copy()
        pert.loc[pert.index > self.cut] = pert.loc[pert.index > self.cut] * -4 + 0.02
        a = run(self.simple, self.sched, self.d, inputs=leak)
        b = run(pert, self.sched, self.d, inputs=leaky_inputs(pert, self.sched))
        self.assertGreater(prefix_violation(a, b, self.cut), 0.0, "the prefix test must detect a t+1 leak")

    def test_covariance_inputs_use_returns_through_t_only(self):
        t = self.full.decisions["signal_date"].iloc[3]
        ri = eng.risk_inputs(self.simple.loc[:t], POLICY.benchmark, CFG)
        np.testing.assert_array_equal(ri.cov, eng.prepare_inputs(self.simple, self.sched[self.sched["signal_date"] == t],
                                                                 POLICY.benchmark, CFG)[t].cov)


class TestRoleSeparation(unittest.TestCase):
    def frames(self):
        idx = pd.bdate_range("2020-01-02", periods=5)
        t = h7.timing_frame(idx)
        base = t.copy()
        base.insert(0, "split", "test")
        base["baseline_p_non_normal"] = [0.0, 0.5, 1.0, 1.0, 0.2]
        base["baseline_alert_ge_q80"] = [0, 0, 1, 1, 0]
        ch = t.copy()
        ch.insert(0, "split", "test")
        ch["challenger_ews_score"] = [0.1, 0.5, 0.9, 0.95, 0.3]
        ch["challenger_alert_level"] = ["GREEN", "AMBER", "ORANGE", "RED", "RECOVERY"]
        ch["challenger_alert_ge_amber"] = [0, 1, 1, 1, 0]
        ch["challenger_alert_ge_orange"] = [0, 0, 1, 1, 0]
        return base, ch

    def test_defensiveness_from_own_role_only(self):
        base, ch = self.frames()
        d = bs.defensiveness_series({"baseline_regime": base, "challenger_ews": ch}, 0.3,
                                    {"GREEN": 0.0, "AMBER": 1 / 3, "ORANGE": 2 / 3, "RED": 1.0, "RECOVERY": 1 / 3})
        np.testing.assert_allclose(d["P1_regime_aware"].to_numpy(), [0, 0.15, 0.3, 0.3, 0.06])
        np.testing.assert_allclose(d["P2_ews_enhanced"].to_numpy(), [0, 1 / 3, 2 / 3, 1, 1 / 3])

    def test_mixed_frame_refused(self):
        base, ch = self.frames()
        mixed = base.copy()
        mixed["challenger_alert_level"] = ch["challenger_alert_level"]
        with self.assertRaises(h7.HandoffError):
            bs.defensiveness_series({"baseline_regime": mixed, "challenger_ews": ch}, 0.3, {"GREEN": 0.0})

    def test_engine_has_no_outcome_or_test_file_reference(self):
        from regime_radar import backtest_accounting, backtest_execution
        for m in (be, backtest_accounting, backtest_execution):
            src = inspect.getsource(m)
            for banned in ("forward_outcomes", "fwd_ret", "fwd_mdd", "test_returns", "test_features", "challenger_",
                           "baseline_"):
                self.assertNotIn(banned, src, f"{m.__name__} references {banned}")

    def test_signal_generator_takes_no_returns(self):
        params = inspect.signature(bs.sequential_signals).parameters
        self.assertFalse(any("return" in p for p in params))


if __name__ == "__main__":
    unittest.main()
