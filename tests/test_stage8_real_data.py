"""Stage 8 checks on the REAL outputs of scripts/10_backtest_portfolios.py (skipped until they exist).

They re-derive the reported numbers from the written files: wealth compounding, cost units, timing, constraint
compliance, role separation and the frozen protocol settings. Run automatically by script 10 after the Test
evaluation and by `python3 -m unittest discover -s tests -v`.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "data" / "processed" / "backtesting"
NEEDED = ("stage8_pretest_lock.json", "portfolio_test_manifest.json", "portfolio_daily_wealth.csv",
          "portfolio_daily_net_returns.csv", "portfolio_decision_targets.csv", "portfolio_executed_weights.csv")
PRIMARY = ("P0_strategic", "P1_regime_aware", "P2_ews_enhanced")


@unittest.skipUnless(all((OUT / n).exists() for n in NEEDED), "Stage 8 outputs not found (run scripts/10 first)")
class TestStage8RealOutputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rd = lambda n, **k: pd.read_csv(OUT / n, **k)  # noqa: E731
        cls.lock = json.loads((OUT / "stage8_pretest_lock.json").read_text())
        cls.manifest = json.loads((OUT / "portfolio_test_manifest.json").read_text())
        cls.log = json.loads((OUT / "test_access_log.json").read_text())
        cls.wealth = rd("portfolio_daily_wealth.csv", parse_dates=["date"])
        cls.net = rd("portfolio_daily_net_returns.csv", parse_dates=["date"])
        cls.gross = rd("portfolio_daily_gross_returns.csv", parse_dates=["date"])
        cls.dec = rd("portfolio_decision_targets.csv", parse_dates=["signal_date", "execution_date", "first_holding_return_date"])
        cls.exe = rd("portfolio_executed_weights.csv", parse_dates=["signal_date", "execution_date"])
        cls.hold = rd("portfolio_daily_holdings.csv", parse_dates=["date"])
        cls.costs = rd("portfolio_transaction_costs.csv", parse_dates=["execution_date"])
        cls.perf = rd("portfolio_performance_summary.csv")
        cls.stats = rd("portfolio_statistical_tests.csv")
        cls.protocol = json.loads((ROOT / "configs" / "stage8_protocol.json").read_text())
        from regime_radar import portfolio_constraints as pc
        cls.policy = pc.load_policy(ROOT / "configs" / "portfolio_policy.json", ROOT / "configs" / "project.json")
        cls.cal = pd.DatetimeIndex(sorted(cls.net["date"].unique()))

    def test_lock_preceded_test_access_and_is_unchanged(self):
        from regime_radar import backtest_integrity as bi
        self.assertTrue(self.lock["test_access_authorized"])
        self.assertEqual(bi.fingerprint(self.lock["frozen"]), self.lock["fingerprint"])
        self.assertTrue(self.log, "Test access log is empty")
        self.assertEqual(self.log[-1]["lock_fingerprint"], self.lock["fingerprint"])
        if self.log[-1]["mode"] != "amendment":
            self.assertEqual(self.log[0]["lock_fingerprint"], self.lock["fingerprint"])
        locked = self.lock["frozen"]["hashes"]["code"]
        now = {str(p.relative_to(ROOT)): bi.sha256(p) for p in bi.code_files(ROOT) if p.exists()}
        self.assertEqual(locked, now, "a project module changed after the pre-Test lock")
        self.assertEqual(self.lock["frozen"]["protocol"], self.protocol)

    def test_test_window_complete(self):
        tp = self.manifest["test_period"]
        self.assertEqual(str(self.cal[0].date()), tp["first"])
        self.assertEqual(str(self.cal[-1].date()), tp["last"])
        self.assertEqual(len(self.cal), tp["sessions"])
        for p, g in self.net.groupby("portfolio"):
            self.assertTrue(pd.DatetimeIndex(g["date"]).equals(self.cal), p)

    def test_wealth_compounds_net_and_gross_returns(self):
        for b in ("5bps", "10bps", "25bps"):
            for p, g in self.net.groupby("portfolio"):
                w = self.wealth[self.wealth["portfolio"] == p]
                np.testing.assert_allclose(np.cumprod(1 + g[f"net_return_{b}"].to_numpy()), w[f"wealth_net_{b}"].to_numpy(),
                                           rtol=1e-8)
        for p, g in self.gross.groupby("portfolio"):
            w = self.wealth[self.wealth["portfolio"] == p]
            np.testing.assert_allclose(np.cumprod(1 + g["gross_return"].to_numpy()), w["wealth_gross"].to_numpy(), rtol=1e-8)

    def test_costs_only_on_executions_and_in_bps_of_gross_notional(self):
        m = self.protocol["transaction_costs"]["stress_multiplier"]
        c = self.costs
        for b in (5, 10, 25):
            exp = b / 1e4 * np.where(c["stress_on"], m, 1.0) * c["gross_notional"]
            np.testing.assert_allclose(c[f"cost_fraction_{b}bps"], exp, rtol=1e-9, atol=1e-15)
        np.testing.assert_allclose(c["gross_notional"], 2 * c["one_way_turnover"], rtol=1e-9)
        mg = self.net.merge(self.gross, on=["date", "portfolio"])
        traded = set(zip(c["portfolio"], c["execution_date"]))
        no = ~pd.Series([(p, d) in traded for p, d in zip(mg["portfolio"], mg["date"])])
        np.testing.assert_allclose(mg.loc[no.to_numpy(), "net_return_10bps"], mg.loc[no.to_numpy(), "gross_return"],
                                   rtol=0, atol=1e-12)
        self.assertTrue((mg["net_return_10bps"] <= mg["gross_return"] + 1e-12).all())

    def test_execution_timing_t_plus_1_and_t_plus_2(self):
        pos = {d: i for i, d in enumerate(self.cal)}
        ex = self.exe[self.exe["signal_date"].isin(pos)]
        self.assertTrue((ex["execution_date"].map(pos) == ex["signal_date"].map(pos) + 1).all())
        d = self.dec.dropna(subset=["first_holding_return_date"])
        d = d[d["first_holding_return_date"].isin(pos) & d["execution_date"].isin(pos)]
        self.assertTrue((d["first_holding_return_date"].map(pos) == d["execution_date"].map(pos) + 1).all())

    def test_holdings_valid_and_executions_within_hard_limits(self):
        assets = self.policy.assets
        W = self.hold[assets].to_numpy()
        self.assertLess(np.abs(W.sum(axis=1) - 1).max(), 1e-8)
        self.assertGreaterEqual(W.min(), -1e-12)
        from regime_radar import portfolio_constraints as pc
        for p in PRIMARY + ("B2_constant_defensiveness",):
            for _, r in self.exe[self.exe["portfolio"] == p].iterrows():
                if r["order_kind"] == "hold":
                    continue
                w = r[[f"executed_{a}" for a in assets]].to_numpy(float)
                rows = pc.check_weights(self.policy, w, tol=1e-7)
                if not r["projected_to_cap"]:
                    self.assertEqual(pc.hard_violations(rows), [], f"{p} {r['execution_date']}")
                self.assertLessEqual(r["turnover"], r["turnover_cap"] + 1e-9)

    def test_role_separation_in_signals_and_decisions(self):
        base = pd.read_csv(OUT / "signals_baseline_regime_test.csv", index_col="signal_date", parse_dates=True,
                           float_precision="round_trip")
        ch = pd.read_csv(OUT / "signals_challenger_ews_test.csv", index_col="signal_date", parse_dates=True)
        self.assertFalse(any(c.startswith("challenger_") for c in base.columns))
        self.assertFalse(any(c.startswith("baseline_") for c in ch.columns))
        a = self.lock["frozen"]["frozen_constants"]["P1_parity_scale_a"]
        lm = self.lock["frozen"]["frozen_policies"]["P2_ews_enhanced"]["d"]
        p1 = self.dec[self.dec["portfolio"] == "P1_regime_aware"].set_index("signal_date")
        np.testing.assert_allclose(p1["budget_d_input"], np.minimum(1, a * base.loc[p1.index, "baseline_p_non_normal"]),
                                   rtol=1e-9)
        p2 = self.dec[self.dec["portfolio"] == "P2_ews_enhanced"].set_index("signal_date")
        np.testing.assert_allclose(p2["budget_d_input"], ch.loc[p2.index, "challenger_alert_level"].map(lm), rtol=1e-9)
        self.assertTrue((self.dec.loc[self.dec["portfolio"] == "P1_regime_aware", "signal_role"] == "baseline_regime").all())
        self.assertTrue((self.dec.loc[self.dec["portfolio"] == "P2_ews_enhanced", "signal_role"] == "challenger_ews").all())
        self.assertTrue(self.dec.loc[self.dec["portfolio"] == "P0_strategic", "budget_d_input"].isna().all())

    def test_reported_metrics_reproducible(self):
        for p in PRIMARY:
            r = self.net[self.net["portfolio"] == p]["net_return_10bps"].to_numpy()
            row = self.perf[(self.perf["portfolio"] == p) & (self.perf["basis"] == "net_10bps")].iloc[0]
            w = np.r_[1.0, np.cumprod(1 + r)]
            self.assertAlmostEqual(row["cumulative_return"], w[-1] - 1, delta=1e-8)
            self.assertAlmostEqual(row["cagr"], w[-1] ** (252 / len(r)) - 1, delta=1e-8)
            self.assertAlmostEqual(row["max_drawdown"], (w / np.maximum.accumulate(w) - 1).min(), delta=1e-8)
            self.assertAlmostEqual(row["annual_volatility"], r.std(ddof=1) * np.sqrt(252), delta=1e-8)

    def test_statistics_follow_the_frozen_protocol(self):
        bs = self.protocol["statistics"]["bootstrap"]
        boot = self.stats[self.stats["method"].str.contains("stationary bootstrap")]
        self.assertTrue(set(boot["method"].str.extract(r"mean block (\d+)")[0].astype(int)) ==
                        {bs["mean_block_length_primary"], bs["mean_block_length_secondary"]})
        pairs = set(zip(self.stats["portfolio_a"], self.stats["portfolio_b"]))
        for a, b in (("P1_regime_aware", "P0_strategic"), ("P2_ews_enhanced", "P0_strategic"),
                     ("P2_ews_enhanced", "P1_regime_aware")):
            self.assertIn((a, b), pairs)
        nw = self.stats[self.stats["statistic"] == "mean_active_return"]
        from regime_radar.backtest_statistics import nw_lag
        self.assertTrue((nw["nw_lag"] == nw_lag(len(self.cal))).all())


if __name__ == "__main__":
    unittest.main()
