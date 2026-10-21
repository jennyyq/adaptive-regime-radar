"""Stage 7 integration tests on the REAL Train/Validation outputs (skipped when the outputs do not exist).

scripts/09_optimize_portfolio.py runs them after writing its outputs and treats any skip as a failure.
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

from regime_radar import portfolio_constraints as pc  # noqa: E402
from regime_radar import portfolio_engine as eng  # noqa: E402
from regime_radar import portfolio_execution as pe  # noqa: E402
from regime_radar import portfolio_risk as pr  # noqa: E402
from regime_radar import risk_budgeting as rb  # noqa: E402
from regime_radar import stage7_handoff as h7  # noqa: E402
from regime_radar.ar_hmm_diagnostics import file_sha256  # noqa: E402

SPLITS = ROOT / "data" / "splits"
EW = ROOT / "data" / "processed" / "early_warning"
OUT = ROOT / "data" / "processed" / "portfolio_optimization"
REQUIRED = [OUT / "portfolio_targets_train.csv", OUT / "portfolio_targets_validation.csv",
            OUT / "portfolio_policy_manifest.json", EW / h7.HANDOFF_DIRNAME / h7.MANIFEST_NAME]
DATES = ["signal_date", "execution_date", "first_holding_return_date"]


@unittest.skipUnless(all(p.exists() for p in REQUIRED), "real Stage 7 outputs not present")
class TestStage7RealData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from regime_radar import regime_exploration as rx
        cls.data = rx.load_stage4_data(SPLITS)
        cls.policy = pc.load_policy(ROOT / "configs" / "portfolio_policy.json", ROOT / "configs" / "project.json")
        cls.t = pd.concat([pd.read_csv(OUT / f"portfolio_targets_{s}.csv", parse_dates=DATES) for s in ("train", "validation")],
                          ignore_index=True)
        cls.man = json.loads((OUT / "portfolio_policy_manifest.json").read_text(encoding="utf-8"))
        cls.cal = cls.data.train_features.index.append(cls.data.validation_features.index)
        cls.simple = pr.to_simple(pd.concat([cls.data.train_returns, cls.data.validation_returns])[cls.policy.assets])
        cls.w = [f"w_{a}" for a in cls.policy.assets]

    def test_no_test_data(self):
        self.assertFalse(any(f.startswith("test_") for f in self.data.files_read))
        test = [s for s in self.data.manifest["splits"] if s["name"] == "test"]
        if test:
            start = pd.Timestamp(test[0]["start"])
            for c in DATES:
                self.assertTrue((self.t[c].dropna() < start).all(), c)

    def test_hard_constraints_on_every_published_target(self):
        lo, hi = pc.bounds(self.policy)
        W = self.t[self.w].to_numpy()
        np.testing.assert_allclose(W.sum(axis=1), 1.0, atol=1e-8)
        self.assertTrue((W >= lo - 1e-8).all() and (W <= hi + 1e-8).all())
        for g in self.policy.groups:
            e = W @ self.policy.group_vector(g)
            self.assertTrue((e <= self.policy.group_max[g] + 1e-8).all() and (e >= self.policy.group_min[g] - 1e-8).all(), g)
        dyn = self.t[self.t["portfolio"] != "P0_strategic"]
        self.assertTrue((dyn["turnover_ex_ante_proxy"] <= dyn["budget_turnover_cap"] + 1e-7).all())
        self.assertTrue(self.t["hard_violations"].fillna("").eq("").all())

    def test_budget_constraints_met_unless_reported(self):
        dyn = self.t[self.t["portfolio"] != "P0_strategic"]
        ok = dyn["unmet_budget"].fillna("").eq("")
        e = dyn.loc[ok]
        self.assertTrue((e["vol_ex_ante"] <= e["budget_vol_target"] * (1 + 1e-6)).all())
        self.assertTrue((e["exposure_equity"] <= e["budget_equity_max"] + 1e-7).all())
        self.assertTrue((e["w_BIL"] >= e["budget_liquidity_min"] - 1e-7).all())
        bad = dyn.loc[~ok]
        self.assertTrue(((bad["solver_stage"] == 2) | (bad["solver_status"] == "hold_within_no_trade_band")).all())
        held = bad[bad["solver_status"] == "hold_within_no_trade_band"]
        self.assertTrue((held["distance_to_posture"] <= json.loads((ROOT / "configs" / "portfolio_policy.json")
                                                                   .read_text())["optimizer"]["no_trade_band"] + 1e-12).all())

    def test_p0_is_benchmark_and_portfolios_share_dates(self):
        p0 = self.t[self.t["portfolio"] == "P0_strategic"]
        post = [f"posture_{a}" for a in self.policy.assets]
        np.testing.assert_allclose(p0[post].to_numpy(), np.tile(self.policy.benchmark, (len(p0), 1)), atol=1e-12)
        band = json.loads((ROOT / "configs" / "portfolio_policy.json").read_text())["optimizer"]["no_trade_band"]
        self.assertTrue((0.5 * np.abs(p0[self.w].to_numpy() - self.policy.benchmark).sum(axis=1) <= band + 1e-12).all())
        dates = {p: tuple(g["signal_date"]) for p, g in self.t.groupby("portfolio")}
        self.assertEqual(len(set(dates.values())), 1)

    def test_role_bindings_and_signal_values(self):
        b = pd.concat([h7.load_stage7_inputs("baseline_regime", s, EW) for s in ("train", "validation")])
        c = pd.concat([h7.load_stage7_inputs("challenger_ews", s, EW) for s in ("train", "validation")])
        p1 = self.t[self.t["portfolio"] == "P1_regime_aware"].set_index("signal_date")
        p2 = self.t[self.t["portfolio"] == "P2_ews_enhanced"].set_index("signal_date")
        self.assertTrue((p1["signal_role"] == "baseline_regime").all() and (p2["signal_role"] == "challenger_ews").all())
        np.testing.assert_allclose(p1["signal_value"].astype(float), b.loc[p1.index, "baseline_p_non_normal"], rtol=1e-9, atol=1e-12)
        self.assertTrue((p2["signal_value"] == c.loc[p2.index, "challenger_alert_level"]).all())
        a = self.man["provenance"]["train_calibrated"]["P1_parity_scale_a"]
        np.testing.assert_allclose(p1["budget_d"], np.minimum(1, a * b.loc[p1.index, "baseline_p_non_normal"]), atol=1e-9)
        lm = json.loads((ROOT / "configs" / "portfolio_policy.json").read_text())["risk_budget"]["P2_ews_levels"]
        np.testing.assert_allclose(p2["budget_d"], p2["signal_value"].map(lm), atol=1e-12)

    def test_execution_convention_and_weekly_limit(self):
        pos = {d: i for i, d in enumerate(self.cal)}
        for _, r in self.t.drop_duplicates("signal_date").iterrows():
            i = pos[r["signal_date"]]
            self.assertEqual(r["execution_date"], self.cal[i + 1])
            if pd.notna(r["first_holding_return_date"]):
                self.assertEqual(r["first_holding_return_date"], self.cal[i + 2])
        for _, g in self.t.groupby("portfolio"):
            wk = g["execution_date"] - pd.to_timedelta(g["execution_date"].dt.weekday, unit="D")
            self.assertFalse(wk.duplicated().any())
            sw = g["signal_date"] - pd.to_timedelta(g["signal_date"].dt.weekday, unit="D")
            last = pd.Series(self.cal, index=self.cal).groupby(self.cal - pd.to_timedelta(self.cal.weekday, unit="D")).max()
            self.assertTrue((g["signal_date"].to_numpy() == last.loc[sw].to_numpy()).all())

    def test_pretrade_proxy_is_drift_through_t(self):
        p2 = self.t[self.t["portfolio"] == "P2_ews_enhanced"].sort_values("signal_date").reset_index(drop=True)
        pre = [f"pretrade_{a}" for a in self.policy.assets]
        for k in range(1, min(len(p2), 40)):
            prev, cur = p2.iloc[k - 1], p2.iloc[k]
            held = self.simple.loc[prev["first_holding_return_date"]:cur["signal_date"]].to_numpy()
            np.testing.assert_allclose(cur[pre].to_numpy(float), pe.drift(prev[self.w].to_numpy(float), held), atol=1e-8)

    def test_future_return_perturbation_does_not_change_earlier_targets(self):
        cfgj = json.loads((ROOT / "configs" / "portfolio_policy.json").read_text())
        ec = self.man["engine_config"]
        cfg = eng.EngineConfig(**{k: v for k, v in ec.items() if k in eng.EngineConfig.__dataclass_fields__})
        b = pd.concat([h7.load_stage7_inputs("baseline_regime", s, EW) for s in ("train", "validation")])
        a = self.man["provenance"]["train_calibrated"]["P1_parity_scale_a"]
        d1 = b["baseline_p_non_normal"].map(lambda p: rb.p1_defensiveness(p, a))
        split = pd.Series(["train"] * len(self.data.train_features) + ["validation"] * len(self.data.validation_features),
                          index=self.cal)
        sched = pe.weekly_schedule(self.cal, split)
        sched = sched[sched["split"] == "validation"].head(12)
        cut = sched["signal_date"].iloc[-1]
        pert = self.simple.copy()
        pert.loc[pert.index > cut] = 0.05
        stress = pd.Series(False, index=self.cal)
        runs = []
        for s in (self.simple, pert):
            inp = eng.prepare_inputs(s, sched, self.policy.benchmark, cfg)
            tg, _, _, _ = eng.run_portfolio("P1_regime_aware", "baseline_regime", s, sched, inp, self.policy, cfg,
                                            cfgj["transaction_costs"], stress, d1, contributions=False)
            runs.append(tg[self.w].to_numpy())
        np.testing.assert_array_equal(runs[0], runs[1])

    def test_ex_ante_risk_recomputed(self):
        p1 = self.t[self.t["portfolio"] == "P1_regime_aware"].sort_values("signal_date").tail(5)
        for _, r in p1.iterrows():
            hist = self.simple.loc[:r["signal_date"]]
            est = pr.estimate_cov(hist, "ledoit_wolf", 504)
            w = r[self.w].to_numpy(float)
            self.assertAlmostEqual(pr.portfolio_vol(w, est.cov_annual), r["vol_ex_ante"], places=7)
            self.assertAlmostEqual(pr.historical_var_cvar(hist.to_numpy()[-504:], w)[1], r["cvar95_daily_ex_ante"], places=7)

    def test_defensive_postures_are_not_riskier(self):
        p2 = self.t[self.t["portfolio"] == "P2_ews_enhanced"]
        m = p2.groupby("signal_value")["budget_vol_multiplier"].mean()
        if "RED" in m and "GREEN" in m:
            self.assertLess(m["RED"], m["GREEN"])
        self.assertTrue((p2.groupby("signal_value")["budget_credit_max"].mean().get("RED", 0) <=
                         p2.groupby("signal_value")["budget_credit_max"].mean().get("GREEN", 1)))

    def test_frozen_inputs_unchanged(self):
        hm = EW / h7.HANDOFF_DIRNAME / h7.MANIFEST_NAME
        self.assertEqual(self.man["handoff_manifest_sha256"], file_sha256(hm))
        h = json.loads(hm.read_text())
        for k, v in h["input_hashes"]["frozen_models"].items():
            self.assertEqual(file_sha256(ROOT / v["file"]), v["sha256"], k)

    def test_stage6_and_6_1_artifacts_keep_their_hashes(self):
        h = json.loads((EW / h7.HANDOFF_DIRNAME / h7.MANIFEST_NAME).read_text())
        for k, v in h["input_hashes"]["stage6_outputs"].items():
            self.assertEqual(file_sha256(EW / k.split("/", 1)[1]), v, k)
        for name, meta in h["files"].items():
            self.assertEqual(file_sha256(EW / h7.HANDOFF_DIRNAME / name), meta["sha256"], name)
        for n, digest in h["input_hashes"]["split_files"].items():
            self.assertEqual(file_sha256(SPLITS / n), digest, n)

    def test_no_realized_performance_in_outputs(self):
        banned = ("sharpe", "cagr", "drawdown", "wealth", "realized_return", "cumulative")
        for f in OUT.glob("*.csv"):
            cols = pd.read_csv(f, nrows=0).columns.str.lower()
            self.assertFalse(any(b in c for c in cols for b in banned), f.name)


if __name__ == "__main__":
    unittest.main()
