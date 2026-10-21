"""POST-TEST EXPLORATORY: four pre-declared variants of the frozen P2 policy (no new model, no search).

The reserved Test period was already examined in Stage 8, so nothing produced here is an independent
out-of-sample result. The official result remains data/processed/backtesting/ (Stage 8, V1).

Variants of P2_ews_enhanced (only the listed parameter changes; everything else is the frozen V1 policy):
  A  no-trade band 1% -> 2%
  B  RECOVERY defensiveness 1/3 -> 0
  C  level map GREEN 0, AMBER 0.25, ORANGE 0.50, RED 0.75, RECOVERY 0   (band 1%)
  D  C + no-trade band 2%

Reused unchanged: the frozen Stage 6.1 handoff signals (Train/Validation) and the Stage 8 Test signal files, the
Stage 8 sequential engine and daily ledger (backtest_engine / backtest_accounting / backtest_execution), the Stage 7
optimizer, Ledoit-Wolf 504 risk inputs, equilibrium returns, constraints, weekly calendar, t+1 execution and the
cost model (10 bps of gross notional, x2 in cost stress; 5 / 25 bps sensitivities). Every strategy is run as in V1:
continuation from the first decision (2008-10-31), wealth normalized to 1.0 at the last Validation close.

Reads the Test split only through regime_radar.backtest_integrity.load_test_split (the V1 lock must still be
valid). Writes only data/processed/quick_sharpe_experiment/, reports/quick_sharpe_improvement.md and
reports/figures/quick_sharpe_comparison.png. No V1 file is written.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regime_radar.paths import PROCESSED, REPORTS  # noqa: E402

SPLITS_DIR = ROOT / "data" / "splits"
EW_DIR = PROCESSED / "early_warning"
S7_DIR = PROCESSED / "portfolio_optimization"
V1_DIR = PROCESSED / "backtesting"
OUT_DIR = PROCESSED / "quick_sharpe_experiment"
FIG_PATH = REPORTS / "figures" / "quick_sharpe_comparison.png"
REPORT_PATH = REPORTS / "quick_sharpe_improvement.md"
POLICY_PATH = ROOT / "configs" / "portfolio_policy.json"
PROJECT_PATH = ROOT / "configs" / "project.json"
PROTOCOL_PATH = ROOT / "configs" / "stage8_protocol.json"
PASS, WARN, FAIL = "PASS", "WARN", "FAIL"

MODERATE = {"GREEN": 0.0, "AMBER": 0.25, "ORANGE": 0.50, "RED": 0.75, "RECOVERY": 0.0}
EXPERIMENTS = {          # declared before running; no other value is tried
    "A_band_2pct": {"band": 0.02, "levels": None},
    "B_recovery_0": {"band": None, "levels": {"RECOVERY": 0.0}},
    "C_moderate_map": {"band": None, "levels": MODERATE},
    "D_moderate_map_band_2pct": {"band": 0.02, "levels": MODERATE},
}
ORIGINALS = ("P0_strategic", "P1_regime_aware", "P2_ews_enhanced", "B2_constant_defensiveness")
LABEL = {"P0_strategic": "P0 original", "P1_regime_aware": "P1 original", "P2_ews_enhanced": "P2 original",
         "B2_constant_defensiveness": "B2 constant d (V1)", "A_band_2pct": "Exp A: band 2%",
         "B_recovery_0": "Exp B: RECOVERY = 0", "C_moderate_map": "Exp C: moderate map",
         "D_moderate_map_band_2pct": "Exp D: moderate map + band 2%"}
P0_MDD_GUARDRAIL = None   # set from the reproduced P0 (V1: -20.08%)


def main() -> int:
    from regime_radar import backtest_accounting as ba
    from regime_radar import backtest_attribution as bat
    from regime_radar import backtest_engine as be
    from regime_radar import backtest_execution as bx
    from regime_radar import backtest_integrity as bi
    from regime_radar import backtest_performance as bp
    from regime_radar import backtest_signals as bs
    from regime_radar import backtest_statistics as bst
    from regime_radar import early_warning_validation as ev
    from regime_radar import portfolio_constraints as pc
    from regime_radar import portfolio_engine as eng
    from regime_radar import portfolio_execution as pe
    from regime_radar import portfolio_risk as pr
    from regime_radar import regime_exploration as rx
    from regime_radar import risk_budgeting as rb
    from regime_radar import stage7_handoff as h7
    from regime_radar.factors import Check

    started = time.perf_counter()
    checks: list = []
    lock_path = V1_DIR / bi.LOCK_NAME
    tman_path = V1_DIR / "portfolio_test_manifest.json"
    for p in (lock_path, tman_path):
        if not p.exists():
            raise SystemExit(f"{p.relative_to(ROOT)} not found: run Stage 8 (scripts/10) first.")
    v1_before = {p.name: bi.sha256(p) for p in sorted(V1_DIR.glob("*")) if p.is_file()}
    frozen_before = {f"{d}/{k}": v for d in ("hmm", "ar_hmm", "early_warning", "portfolio_optimization")
                     for k, v in __import__("regime_radar.ar_hmm_diagnostics", fromlist=["x"]).directory_hashes(PROCESSED / d).items()}
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    tman = json.loads(tman_path.read_text(encoding="utf-8"))
    fc = lock["frozen"]["frozen_constants"]
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    policy = pc.load_policy(POLICY_PATH, PROJECT_PATH)
    pcfg = policy.raw
    assets = policy.assets
    print("POST-TEST EXPLORATORY experiment (not an independent out-of-sample evaluation).")

    # ------------------------------------------------------------------ data (Test only through the V1 lock)
    data = rx.load_stage4_data(SPLITS_DIR)
    h61_path = EW_DIR / h7.HANDOFF_DIRNAME / h7.MANIFEST_NAME
    h61 = json.loads(h61_path.read_text(encoding="utf-8"))
    models = bs.load_frozen_models(ROOT, h61)
    artifacts = {"configs/project.json": PROJECT_PATH, "configs/portfolio_policy.json": POLICY_PATH,
                 "configs/stage8_protocol.json": PROTOCOL_PATH,
                 "portfolio_optimization/stage8_handoff_manifest.json": S7_DIR / "stage8_handoff_manifest.json",
                 "portfolio_optimization/portfolio_policy_manifest.json": S7_DIR / "portfolio_policy_manifest.json",
                 "early_warning/stage7_handoff/stage7_handoff_manifest.json": h61_path,
                 "splits/split_manifest.json": SPLITS_DIR / "split_manifest.json"}
    artifacts.update({f"splits/{n}": SPLITS_DIR / n for n in bi.MODEL_FILES})
    artifacts.update({f"models/{k}": ROOT / v for k, v in models.files.items()})
    artifacts.update({f"portfolio_optimization/portfolio_targets_{s}.csv": S7_DIR / f"portfolio_targets_{s}.csv"
                      for s in ("train", "validation")})
    for role in h7.ROLES:
        for s in ("train", "validation"):
            artifacts[f"early_warning/stage7_handoff/stage7_{role}_{s}.csv"] = EW_DIR / h7.HANDOFF_DIRNAME / f"stage7_{role}_{s}.csv"
    test = bi.load_test_split(SPLITS_DIR, lock_path, bi.snapshot(ROOT, artifacts))
    checks.append(Check("V1 pre-Test lock still valid (no frozen code, config or artifact changed)", PASS,
                        f"lock {lock['fingerprint'][:16]}...; Test read through the gated loader"))
    tr, va = data.train_features, data.validation_features
    last_val = va.index.max()
    cal = tr.index.append(va.index).append(test.features.index)
    split = pd.Series(["train"] * len(tr) + ["validation"] * len(va) + ["test"] * len(test.features), index=cal)
    log_ret = pd.concat([data.train_returns, data.validation_returns, test.returns])[assets]
    simple = pr.to_simple(log_ret)

    # ------------------------------------------------------------------ signals: frozen handoff + saved V1 Test signals
    role_frames = {}
    for role in h7.ROLES:
        f61 = pd.concat([h7.load_stage7_inputs(role, s, EW_DIR) for s in ("train", "validation")])
        name = f"signals_{role}_test.csv"
        ok = tman["outputs"].get(name) == bi.sha256(V1_DIR / name)
        checks.append(Check(f"Saved V1 Test signals ({role}) unchanged", PASS if ok else FAIL, name))
        ft = pd.read_csv(V1_DIR / name, index_col="signal_date", parse_dates=["signal_date"], float_precision="round_trip")
        f = pd.concat([f61, ft[f61.columns]])
        f.attrs["stage7_role"] = role
        h7.assert_single_role(f)
        role_frames[role] = f
    same_cal = all(role_frames[r].index.equals(cal) for r in h7.ROLES)
    checks.append(Check("Signal calendar = split calendar (Train + Validation + Test)", PASS if same_cal else FAIL,
                        f"{len(cal)} sessions"))
    if any(c.status == FAIL for c in checks):
        for c in checks:
            print(f"  [{c.status}] {c.name}: {c.detail}")
        return 1
    level_v1 = pcfg["risk_budget"]["P2_ews_levels"]
    a_scale, d_const = float(fc["P1_parity_scale_a"]), float(fc["B2_constant_d"])
    d_v1 = bs.defensiveness_series(role_frames, a_scale, level_v1)
    levels = role_frames["challenger_ews"]["challenger_alert_level"]

    def d2_for(over):
        m = dict(level_v1, **(over or {}))
        return levels.map(lambda lv: rb.p2_defensiveness(lv, m)).astype(float), m

    # ------------------------------------------------------------------ engine configuration (identical to V1)
    ocfg, tp = pcfg["optimizer"], pcfg["optimizer"]["turnover_penalty"]
    cfg = eng.EngineConfig(cov_method=pcfg["risk_estimation"]["primary"]["method"],
                           lookback=pcfg["risk_estimation"]["primary"]["lookback"],
                           min_obs=pcfg["risk_estimation"]["primary"]["min_observations"],
                           psd_floor=pcfg["risk_estimation"]["psd_floor_relative"],
                           sharpe=pcfg["expected_returns"]["equilibrium"]["benchmark_sharpe_assumption"],
                           er_shrinkage=pcfg["expected_returns"]["sensitivity"]["shrunk_historical"]["shrinkage_to_prior"],
                           lambda_turnover=2 * tp["cost_rate"] / tp["holding_horizon_years"],
                           lambda_tracking=ocfg["tracking_penalty"], cvar_alpha=pcfg["cvar"]["confidence"],
                           no_trade_band=ocfg["no_trade_band"], execution_mode="target_then_trade",
                           budget_coef=pcfg["risk_budget"]["coefficients"], solver=ocfg["solver"])
    same = {k: v for k, v in vars(cfg).items()} == lock["frozen"]["engine_config"]
    checks.append(Check("Engine configuration identical to the locked V1 configuration", PASS if same else FAIL,
                        f"band {cfg.no_trade_band}, lambda_tracking {cfg.lambda_tracking}"))
    cost_cfg = pcfg["transaction_costs"]
    stress = bx.stress_indicator(simple, float(fc["cost_stress_threshold_spy_21d"]))
    sched = pe.weekly_schedule(cal, split)
    t0 = time.perf_counter()
    inputs = eng.prepare_inputs(simple, sched, policy.benchmark, cfg)
    t_first = min(inputs)
    print(f"Risk inputs for {len(inputs)} decisions in {time.perf_counter() - t0:.0f}s; first decision {t_first.date()}.")
    bps_all = (5, 10, 25)
    wb = policy.benchmark

    specs = {"P0_strategic": (be.PortfolioSpec("P0_strategic", "optimizer", None, None, start_weights=wb), cfg),
             "P1_regime_aware": (be.PortfolioSpec("P1_regime_aware", "optimizer", "baseline_regime",
                                                  d_v1["P1_regime_aware"], start_weights=wb), cfg),
             "P2_ews_enhanced": (be.PortfolioSpec("P2_ews_enhanced", "optimizer", "challenger_ews",
                                                  d_v1["P2_ews_enhanced"], start_weights=wb), cfg),
             "B2_constant_defensiveness": (be.PortfolioSpec("B2_constant_defensiveness", "optimizer", None, d_const,
                                                            start_weights=wb), cfg)}
    maps = {}
    for name, e in EXPERIMENTS.items():
        d2, m = d2_for(e["levels"])
        maps[name] = m
        cv = cfg if e["band"] is None else cfg.variant(no_trade_band=e["band"])
        specs[name] = (be.PortfolioSpec(name, "optimizer", "challenger_ews", d2, start_weights=wb), cv)

    # ------------------------------------------------------------------ runs (continuation, as in V1)
    runs, solver_warn = {}, []
    for pid, (spec, cv) in specs.items():
        t0 = time.perf_counter()
        runs[pid] = be.run_sequential(spec, simple, sched, inputs, policy, cv, cost_cfg, stress, t_first, bps_all, 10)
        solver_warn += runs[pid].warnings
        print(f"  {pid}: {len(runs[pid].decisions)} decisions in {time.perf_counter() - t0:.0f}s")

    test_dates = test.features.index
    years = len(test_dates) / bp.ANN
    rf = simple.loc[test_dates, "BIL"]

    def series(pid, col):
        return runs[pid].ledger.frame().set_index("date").loc[test_dates, col]

    gross = pd.DataFrame({p: series(p, "gross_return") for p in runs})
    net = {b: pd.DataFrame({p: series(p, f"net_return_{b}bps") for p in runs}) for b in bps_all}
    post = {p: runs[p].ledger.weights("post") for p in runs}
    held = {p: post[p].shift(1).loc[test_dates] for p in runs}

    # ------------------------------------------------------------------ checks
    v1_net = pd.read_csv(V1_DIR / "portfolio_daily_net_returns.csv", parse_dates=["date"])
    rep = {}
    for p in ORIGINALS:
        v = v1_net[v1_net["portfolio"] == p].set_index("date")["net_return_10bps"].reindex(test_dates)
        rep[p] = float(np.max(np.abs(v.to_numpy() - net[10][p].to_numpy())))
    checks.append(Check("Reused engine reproduces the official V1 net returns of P0, P1, P2, B2 (10 significant digits)",
                        PASS if max(rep.values()) < 1e-9 else FAIL,
                        ", ".join(f"{k.split('_')[0]} {v:.1e}" for k, v in rep.items())))
    rec = pd.DataFrame([r for p in runs for r in ba.reconcile(runs[p].ledger, simple, set(sched["execution_date"]))])
    bad = rec[rec["status"] == FAIL]
    checks.append(Check("Daily ledger reconciliation for every strategy (returns, drift, notional, costs, wealth)",
                        PASS if bad.empty else FAIL, f"{len(rec)} identity checks; failed {len(bad)}"))
    def exe_viol(r, kind):
        x = r.executions
        if not len(x):
            return 0
        x = x[(x["execution_date"] > last_val) & (x["order_kind"] == kind)]
        if kind == "trade":
            x = x[~x["projected_to_cap"]]
        return int(x["executed_hard_violations"].fillna("").ne("").sum())
    hv = {p: int(runs[p].decisions["hard_violations"].fillna("").ne("").sum()) for p in runs}
    hx = {p: exe_viol(runs[p], "trade") for p in runs}
    drift = {p: exe_viol(runs[p], "hold") for p in runs}
    proj = {p: int(runs[p].executions.loc[runs[p].executions["execution_date"] > last_val, "projected_to_cap"].sum())
            for p in runs}
    checks.append(Check("Hard portfolio constraints: every decision-time target and every executed trade compliant "
                        "(long only, no leverage, caps, groups, active limit, turnover cap)",
                        PASS if not sum(hv.values()) and not sum(hx.values()) else FAIL,
                        f"target violations {sum(hv.values())} (all history); Test trade executions violating "
                        f"{sum(hx.values())}; reported separately - passive drift during no-trade holds: "
                        + (", ".join(f"{k.split('_')[0]} {v}" for k, v in drift.items() if v) or "none") + "; cap projections: "
                        + (", ".join(f"{k.split('_')[0]} {v}" for k, v in proj.items() if v) or "none")))
    pos = {d: i for i, d in enumerate(cal)}
    lag_ok = all((r.executions["execution_date"].map(pos) == r.executions["signal_date"].map(pos) + 1).all()
                 for r in runs.values() if len(r.executions))
    checks.append(Check("Execution lag: every order executed at the close of the next session (t+1)",
                        PASS if lag_ok else FAIL, "all strategies"))
    mono = all(m["GREEN"] <= m["AMBER"] <= m["ORANGE"] <= m["RED"] and m["RECOVERY"] <= m["AMBER"] for m in maps.values())
    dchk = all(np.array_equal(runs[n].decisions["budget_d_input"].to_numpy(),
                              runs[n].decisions["signal_date"].map(levels).map(maps[n]).to_numpy(float)) for n in EXPERIMENTS)
    checks.append(Check("Modified level maps applied exactly; alert states unchanged; defensiveness monotone in severity",
                        PASS if mono and dchk else FAIL,
                        "; ".join(f"{n[0]}: " + ", ".join(f"{k} {v:.2f}" for k, v in maps[n].items()) for n in EXPERIMENTS)))
    held_in_band = {n: float((0.5 * np.abs(runs[n].decisions[[f"pretrade_{a}" for a in assets]].to_numpy() -
                                           runs[n].decisions[[f"posture_{a}" for a in assets]].to_numpy()).sum(axis=1)
                              [runs[n].decisions["order_kind"].eq("hold").to_numpy()]).max(initial=0.0))
                    for n in EXPERIMENTS}
    band_ok = all(held_in_band[n] <= specs[n][1].no_trade_band + 1e-12 for n in EXPERIMENTS)
    checks.append(Check("No-trade band applied as declared (every hold is within the variant's band)", PASS if band_ok else FAIL,
                        ", ".join(f"{n[0]} max held distance {v:.4f} (band {specs[n][1].no_trade_band})"
                                  for n, v in held_in_band.items())))
    # causality: perturb returns after a mid-Test decision; decisions / holdings up to that date must not change
    tdec = [t for t in sched["signal_date"] if t > last_val and t in inputs]
    cut = tdec[len(tdec) // 2]
    pert = simple.copy()
    pert.loc[pert.index > cut] = pert.loc[pert.index > cut] * -3 + 0.01
    sched_t = sched[sched["signal_date"] > last_val]
    inp_p = dict(inputs)
    inp_p.update(eng.prepare_inputs(pert, sched_t, wb, cfg))          # every Test input recomputed from perturbed data
    stress_p = bx.stress_indicator(pert, float(fc["cost_stress_threshold_spy_21d"]))
    caus = True
    for n in ("C_moderate_map", "D_moderate_map_band_2pct"):
        spec, cv = specs[n]
        o = be.run_sequential(spec, simple, sched, inputs, policy, cv, cost_cfg, stress, last_val, bps_all, 10)
        q = be.run_sequential(spec, pert, sched, inp_p, policy, cv, cost_cfg, stress_p, last_val, bps_all, 10)
        w = [f"w_{a}" for a in assets]
        caus &= np.array_equal(o.decisions[o.decisions["signal_date"] <= cut][w].to_numpy(),
                               q.decisions[q.decisions["signal_date"] <= cut][w].to_numpy())
        caus &= np.array_equal(o.ledger.weights().loc[:cut].to_numpy(), q.ledger.weights().loc[:cut].to_numpy())
    checks.append(Check("Causality: changing returns after a Test decision date leaves earlier decisions and holdings "
                        "unchanged (experiments C and D)", PASS if caus else FAIL, f"cut {cut.date()}"))

    # ------------------------------------------------------------------ metrics
    eq = policy.groups["equity"]
    rows = []
    for p in list(ORIGINALS) + list(EXPERIMENTS):
        mg = bp.metrics(gross[p], rf)
        m = {b: bp.metrics(net[b][p], rf) for b in bps_all}
        lg = runs[p].ledger.frame().set_index("date").loc[test_dates]
        yr = bp.period_returns(net[10][p], "Y")
        r2425 = net[10][p].loc["2024-01-01":"2025-12-31"]
        dec = runs[p].decisions[runs[p].decisions["signal_date"] > last_val]
        rows.append({"strategy": p, "label": LABEL[p], "exploratory": p in EXPERIMENTS,
                     "gross_sharpe": mg["sharpe"], "net_sharpe": m[10]["sharpe"], "net_cagr": m[10]["cagr"],
                     "annual_volatility": m[10]["annual_volatility"], "max_drawdown": m[10]["max_drawdown"],
                     "annual_one_way_turnover": float(lg["turnover"].sum() / years),
                     "cost_drag_cagr": mg["cagr"] - m[10]["cagr"],
                     "cost_paid_per_year": float(lg["cost_fraction_10bps"].sum() / years),
                     "return_2020": float(yr.get(pd.Period("2020", "Y"), np.nan)),
                     "return_2022": float(yr.get(pd.Period("2022", "Y"), np.nan)),
                     "return_2024_2025": float((1 + r2425).prod() - 1),
                     "mean_equity": float(held[p][eq].sum(axis=1).mean()), "mean_bil": float(held[p]["BIL"].mean()),
                     "mean_d_test": float(dec["budget_d_input"].mean()) if dec["budget_d_input"].notna().any() else 0.0,
                     "net_sharpe_5bps": m[5]["sharpe"], "net_sharpe_25bps": m[25]["sharpe"],
                     "net_cagr_25bps": m[25]["cagr"], "sortino": m[10]["sortino"], "calmar": m[10]["calmar"]})
    tab = pd.DataFrame(rows).set_index("strategy")
    p0_mdd = tab.loc["P0_strategic", "max_drawdown"]
    tab["mdd_worse_than_P0"] = tab["max_drawdown"] < p0_mdd - 1e-12

    # recovery participation in the objective SPY drawdown events (frozen Stage 6 rule)
    events = ev.drawdown_events(test.returns["SPY"])
    lvl_c = test.returns["SPY"].cumsum()
    events["trough_date"] = [lvl_c.loc[r.onset_date:r.end_date].idxmin() for r in events.itertuples()]
    evt = bat.event_analysis(events, net[10], post, {p: runs[p].executions for p in runs}, policy.groups, {}, cal,
                             strategic_equity=float(policy.group_vector("equity") @ wb))
    rp = evt.groupby("portfolio")["recovery_participation_vs_P0"].mean()
    prot = evt.groupby("portfolio")["protection_vs_P0"].mean()
    tab["mean_recovery_participation_vs_P0"] = rp.reindex(tab.index)
    tab["mean_peak_to_trough_protection_vs_P0"] = prot.reindex(tab.index)

    # comparisons vs P2, P1, B2: bootstrap Sharpe differences, attribution (exposure vs timing), beta vs P0
    bsc = protocol["statistics"]["bootstrap"]
    pairs = [(e, r) for e in EXPERIMENTS for r in ("P2_ews_enhanced", "P1_regime_aware", "B2_constant_defensiveness")]
    boot = bst.bootstrap_differences(net[10], rf, pairs, bsc["mean_block_length_primary"], bsc["replications"], bsc["seed"])
    boot = boot[boot["statistic"].isin(["sharpe_difference", "max_drawdown_difference"])]
    cmp_rows = []
    for e, r in pairs:
        s_, _ = bat.attribution(e, r, held, simple, gross, net[10], protocol["attribution"]["groups"])
        a = s_.set_index(["component", "group"])["annualized_mean"]
        avg, tim = a[("average-exposure part (all groups)", "all")], a[("timing part (all groups)", "all")]
        b = boot[(boot["portfolio_a"] == e) & (boot["portfolio_b"] == r)].set_index("statistic")
        cmp_rows.append({"candidate": e, "reference": r, "net_sharpe_difference": b.loc["sharpe_difference", "estimate"],
                         "sharpe_ci_low": b.loc["sharpe_difference", "ci_low"], "sharpe_ci_high": b.loc["sharpe_difference", "ci_high"],
                         "max_drawdown_difference": b.loc["max_drawdown_difference", "estimate"],
                         "net_sharpe_difference_25bps": tab.loc[e, "net_sharpe_25bps"] - tab.loc[r, "net_sharpe_25bps"],
                         "mean_equity_difference": tab.loc[e, "mean_equity"] - tab.loc[r, "mean_equity"],
                         "gross_active_return": a[("gross active return (ledger)", "all")],
                         "average_exposure_part": avg, "timing_part": tim,
                         "cost_effect": a[("transaction-cost effect", "all")],
                         "driver": ("average-exposure component dominates" if abs(avg) > abs(tim) else "timing component dominates")
                         + ("" if b.loc["sharpe_difference", "ci_low"] > 0 or b.loc["sharpe_difference", "ci_high"] < 0
                            else "; Sharpe interval contains 0")})
    comp = pd.DataFrame(cmp_rows)
    for p in tab.index:
        if p != "P0_strategic":
            rr = bst.ols_hac((net[10][p] - rf).to_numpy(), (net[10]["P0_strategic"] - rf).to_numpy())
            tab.loc[p, "beta_vs_P0"] = rr["beta"]
            tab.loc[p, "alpha_vs_P0"] = rr["alpha_annualized"]

    # ------------------------------------------------------------------ selection summary (rules declared in the brief)
    exp = tab.loc[list(EXPERIMENTS)].sort_values("net_sharpe", ascending=False)
    best = exp.index[0]
    ok = exp[~exp["mdd_worse_than_P0"]]
    score = ok.assign(rank_sharpe=ok["net_sharpe"].rank(ascending=False), rank_mdd=ok["max_drawdown"].rank(ascending=False),
                      rank_turn=ok["annual_one_way_turnover"].rank())
    score["rank_sum"] = score["rank_sharpe"] + score["rank_mdd"] + score["rank_turn"]
    defensible = score.sort_values(["rank_sum", "net_sharpe"], ascending=[True, False]).index[0] if len(score) else None
    beats = [e for e in EXPERIMENTS if tab.loc[e, "net_sharpe"] > tab.loc["P2_ews_enhanced", "net_sharpe"]
             and tab.loc[e, "net_sharpe"] > tab.loc["P1_regime_aware", "net_sharpe"]]
    survive25 = [e for e in beats if tab.loc[e, "net_sharpe_25bps"] > tab.loc["P2_ews_enhanced", "net_sharpe_25bps"]
                 and tab.loc[e, "net_sharpe_25bps"] > tab.loc["P1_regime_aware", "net_sharpe_25bps"]]
    summary = {"ranking_by_net_sharpe_10bps": list(exp.index), "highest_net_sharpe": best,
               "most_defensible_risk_return": defensible,
               "defensible_rule": "among candidates whose max drawdown is not worse than P0, lowest sum of ranks of net "
                                  "Sharpe, max drawdown and turnover (ties: higher net Sharpe)",
               "improves_on_P2_and_P1": beats, "improvement_survives_25bps": survive25,
               "flagged_mdd_worse_than_P0": [e for e in EXPERIMENTS if tab.loc[e, "mdd_worse_than_P0"]],
               "thresholds_exceeded": {t: [e for e in EXPERIMENTS if tab.loc[e, "net_sharpe"] > t] for t in (0.70, 0.75, 0.80)}}

    # ------------------------------------------------------------------ outputs (exploratory folder only)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tab.reset_index().to_csv(OUT_DIR / "quick_sharpe_comparison.csv", index=False, float_format="%.10g")
    comp.to_csv(OUT_DIR / "candidate_vs_references.csv", index=False, float_format="%.10g")
    evt.to_csv(OUT_DIR / "event_recovery_participation.csv", index=False, float_format="%.10g")
    long = []
    for p in runs:
        f = pd.DataFrame({"date": test_dates, "strategy": p, "gross_return": gross[p].to_numpy()})
        for b in bps_all:
            f[f"net_return_{b}bps"] = net[b][p].to_numpy()
        f["equity_weight_held"] = held[p][eq].sum(axis=1).to_numpy()
        f["bil_weight_held"] = held[p]["BIL"].to_numpy()
        long.append(f)
    pd.concat(long).to_csv(OUT_DIR / "daily_returns_and_exposure.csv", index=False, float_format="%.10g",
                           date_format="%Y-%m-%d")
    pd.concat([runs[e].decisions[runs[e].decisions["signal_date"] > last_val].assign(experiment=e) for e in EXPERIMENTS]
              ).to_csv(OUT_DIR / "experiment_decisions_test.csv", index=False, float_format="%.10g", date_format="%Y-%m-%d")
    rec.to_csv(OUT_DIR / "ledger_reconciliation.csv", index=False)
    v1_after = {p.name: bi.sha256(p) for p in sorted(V1_DIR.glob("*")) if p.is_file()}
    frozen_after = {f"{d}/{k}": v for d in ("hmm", "ar_hmm", "early_warning", "portfolio_optimization")
                    for k, v in __import__("regime_radar.ar_hmm_diagnostics", fromlist=["x"]).directory_hashes(PROCESSED / d).items()}
    still = not bi.verify_lock(lock, bi.snapshot(ROOT, artifacts))
    checks.append(Check("Original Stage 8 outputs, pre-Test lock and frozen Stage 5-7 artifacts unchanged",
                        PASS if v1_before == v1_after and frozen_before == frozen_after and still else FAIL,
                        f"{len(v1_before)} Stage 8 files, {len(frozen_before)} frozen files compared; lock valid {still}"))
    nwarn = len([w for w in solver_warn if w["signal_date"] > last_val])
    checks.append(Check("Solver warnings (documented, solutions verified by the hard-constraint check)",
                        PASS if not nwarn else WARN, f"{nwarn} records in Test decisions"))
    pd.DataFrame([vars(c) for c in checks]).to_csv(OUT_DIR / "checks.csv", index=False)
    (OUT_DIR / "experiment_manifest.json").write_text(json.dumps(bi.jsonable({
        "status": "POST-TEST EXPLORATORY - not an independent out-of-sample evaluation",
        "official_result": "data/processed/backtesting (Stage 8 V1, lock " + lock["fingerprint"] + ")",
        "experiments": {k: {"no_trade_band": v["band"] or cfg.no_trade_band, "level_map": maps[k]} for k, v in EXPERIMENTS.items()},
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "summary": summary,
        "checks": [vars(c) for c in checks]}), indent=2), encoding="utf-8")
    fig(tab, net[10], comp)
    REPORT_PATH.write_text(report(tab, comp, summary, checks, maps, lock, years, test_dates), encoding="utf-8")

    # ------------------------------------------------------------------ console
    show = ["net_sharpe", "gross_sharpe", "net_cagr", "annual_volatility", "max_drawdown", "annual_one_way_turnover",
            "cost_drag_cagr", "return_2020", "return_2022", "return_2024_2025", "mean_equity", "mean_bil",
            "net_sharpe_25bps", "mean_recovery_participation_vs_P0"]
    pd.set_option("display.width", 250)
    print("\n" + tab[show].round(4).to_string())
    print("\nCandidates vs references (net 10 bps):")
    print(comp[["candidate", "reference", "net_sharpe_difference", "sharpe_ci_low", "sharpe_ci_high",
                "mean_equity_difference", "average_exposure_part", "timing_part", "driver"]].round(4).to_string(index=False))
    print("\nSummary:", json.dumps(summary, indent=1))
    print("Checks:")
    for c in checks:
        print(f"  [{c.status}] {c.name}: {c.detail}")
    fails = [c for c in checks if c.status == FAIL]
    print(f"{'FAILED' if fails else 'Completed'} in {time.perf_counter() - started:.0f}s. POST-TEST EXPLORATORY ONLY; "
          f"official result unchanged. Report: {REPORT_PATH}")
    return 1 if fails else 0


def fig(tab, net10, comp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from regime_radar.regime_report import GRID, INK, INK_2, MUTED, SURFACE, _save, _style, _title
    col = {"P0_strategic": "#898781", "P1_regime_aware": "#2a78d6", "P2_ews_enhanced": "#eb6834",
           "B2_constant_defensiveness": "#6250d6", "A_band_2pct": "#1baf7a", "B_recovery_0": "#eda100",
           "C_moderate_map": "#e87ba4", "D_moderate_map_band_2pct": "#008300"}
    order = list(tab.index)
    fig_, axes = plt.subplots(2, 2, figsize=(15, 9), facecolor=SURFACE)
    ax = axes[0, 0]
    _style(ax)
    for p in order:
        w = (1 + net10[p]).cumprod()
        ax.plot(w.index, w, color=col[p], lw=2.0 if p in ("P2_ews_enhanced", "D_moderate_map_band_2pct") else 1.2,
                ls="-" if not tab.loc[p, "exploratory"] else (0, (4, 2)), label=f"{tab.loc[p, 'label']} ({w.iloc[-1]:.3f})")
    ax.set_title("Net wealth (10 bps), 1.0 at the last Validation close", fontsize=9, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK_2, loc="upper left")
    y = np.arange(len(order))[::-1]
    ax = axes[0, 1]
    _style(ax, grid_y=False)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    for yy, p in zip(y, order):
        n_, g_ = tab.loc[p, "net_sharpe"], tab.loc[p, "gross_sharpe"]
        ax.plot([n_, g_], [yy, yy], color=col[p], lw=1.5)
        ax.plot(g_, yy, "o", ms=7, mfc=SURFACE, mec=col[p], mew=1.5)
        ax.plot(n_, yy, "o", ms=9, color=col[p], mec=SURFACE, mew=1.5)
        ax.text(n_, yy + 0.28, f"{n_:.3f}", ha="center", fontsize=8, color=INK_2)
    ax.axvline(tab.loc["P2_ews_enhanced", "net_sharpe"], color=MUTED, lw=0.8, ls="--")
    ax.set_yticks(y, [tab.loc[p, "label"] for p in order], fontsize=8, color=INK_2)
    ax.set_title("net Sharpe at 10 bps (filled) and gross Sharpe (open); dashed = P2 original", fontsize=9, color=INK,
                 loc="left")
    ax = axes[1, 0]
    _style(ax, grid_y=False)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    v = 100 * tab["max_drawdown"].to_numpy()
    ax.barh(y, v, color=[col[p] for p in order], height=0.6)
    for yy, x in zip(y, v):
        ax.text(x * 0.98, yy, f"{x:.1f}", va="center", ha="left", fontsize=8, color=SURFACE)
    ax.axvline(100 * tab.loc["P0_strategic", "max_drawdown"], color=INK, lw=1, ls="--")
    ax.set_xlim(v.min() * 1.15, 0)
    ax.set_yticks(y, [tab.loc[p, "label"] for p in order], fontsize=8, color=INK_2)
    ax.set_title("maximum drawdown, % (dashed = P0 guardrail)", fontsize=9, color=INK, loc="left")
    ax = axes[1, 1]
    _style(ax)
    for p in order:
        ax.scatter(100 * tab.loc[p, "mean_equity"], tab.loc[p, "net_sharpe"], s=40 + 60 * tab.loc[p, "annual_one_way_turnover"],
                   color=col[p], edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.annotate(tab.loc[p, "label"].split(":")[0], (100 * tab.loc[p, "mean_equity"], tab.loc[p, "net_sharpe"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=7.5, color=INK_2)
    ax.set_xlabel("mean equity weight held, %", fontsize=8, color=INK_2)
    ax.set_ylabel("net Sharpe", fontsize=8, color=INK_2)
    ax.set_title("Sharpe vs equity exposure (marker size = turnover)", fontsize=9, color=INK, loc="left")
    _title(fig_, "POST-TEST EXPLORATORY: four pre-declared P2 policy variants vs the official Stage 8 portfolios",
           "Same engine, ledger, signals, constraints, t+1 execution and costs. The Test period was already examined: these "
           "are not independent out-of-sample results.")
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _save(fig_, FIG_PATH)


def report(tab, comp, summary, checks, maps, lock, years, test_dates) -> str:
    from regime_radar.quality import _md_table

    def pct(v):
        return "" if pd.isna(v) else f"{100 * v:.2f}%"

    def num(v):
        return "" if pd.isna(v) else f"{v:.3f}"
    L = ["# Quick Sharpe-improvement experiment (POST-TEST EXPLORATORY)\n",
         "**Status: exploratory, after the Test period was examined.** None of these results is an independent "
         "out-of-sample evaluation. The official result remains Stage 8 V1 (`data/processed/backtesting/`, lock "
         f"`{lock['fingerprint'][:16]}...`).\n",
         f"Test {test_dates[0].date()} to {test_dates[-1].date()} ({len(test_dates)} sessions, {years:.2f} years). "
         "Primary cost 10 bps of gross traded notional (x2 in cost stress).\n", "## 1. Declared experiments (no other values tried)\n"]
    L.append(_md_table(pd.DataFrame([{"experiment": LABEL[k], "no-trade band": f"{100 * (v['band'] or 0.01):.0f}%",
                                      **{lv: f"{maps[k][lv]:.3f}" for lv in ("GREEN", "AMBER", "ORANGE", "RED", "RECOVERY")}}
                                     for k, v in EXPERIMENTS.items()]).set_index("experiment")) + "\n")
    L.append("Original P2: band 1%, GREEN 0, AMBER 1/3, ORANGE 2/3, RED 1, RECOVERY 1/3. Alert states, signals, optimizer, "
             "constraints, calendar, execution lag and costs are unchanged.\n")
    L.append("## 2. Comparison table (net of 10 bps unless stated)\n")
    t = pd.DataFrame({"gross Sharpe": tab["gross_sharpe"].map(num), "net Sharpe": tab["net_sharpe"].map(num),
                      "net CAGR": tab["net_cagr"].map(pct), "volatility": tab["annual_volatility"].map(pct),
                      "max drawdown": tab["max_drawdown"].map(pct) + np.where(tab["mdd_worse_than_P0"], " (worse than P0)", ""),
                      "turnover / yr": tab["annual_one_way_turnover"].map(lambda v: f"{v:.2f}"),
                      "cost drag / yr": tab["cost_drag_cagr"].map(pct), "2020": tab["return_2020"].map(pct),
                      "2022": tab["return_2022"].map(pct), "2024-2025": tab["return_2024_2025"].map(pct),
                      "mean equity": tab["mean_equity"].map(pct), "mean BIL": tab["mean_bil"].map(pct)})
    t.index = [tab.loc[p, "label"] for p in tab.index]
    L.append(_md_table(t) + "\n")
    t2 = pd.DataFrame({"net Sharpe 25 bps": tab["net_sharpe_25bps"].map(num), "net CAGR 25 bps": tab["net_cagr_25bps"].map(pct),
                       "mean d (Test)": tab["mean_d_test"].map(num), "beta vs P0": tab["beta_vs_P0"].map(num),
                       "alpha vs P0": tab["alpha_vs_P0"].map(pct),
                       "recovery participation vs P0 (mean over events)": tab["mean_recovery_participation_vs_P0"].map(pct),
                       "peak-to-trough protection vs P0": tab["mean_peak_to_trough_protection_vs_P0"].map(pct)})
    t2.index = t.index
    L.append(_md_table(t2) + "\n")
    L.append("## 3. Candidates against P2, P1 and the constant-defensiveness benchmark\n")
    c = comp.copy()
    c["candidate"] = c["candidate"].map(LABEL)
    c["reference"] = c["reference"].map(LABEL)
    for k in ("net_sharpe_difference", "sharpe_ci_low", "sharpe_ci_high", "net_sharpe_difference_25bps"):
        c[k] = c[k].map(num)
    for k in ("max_drawdown_difference", "mean_equity_difference", "gross_active_return", "average_exposure_part",
              "timing_part", "cost_effect"):
        c[k] = c[k].map(pct)
    L.append(_md_table(c.set_index("candidate")) + "\n")
    L.append("Sharpe intervals: paired stationary bootstrap (mean block 21, 5000 draws, Stage 8 seed). Attribution: "
             "holdings-based daily identity; 'average-exposure part' = Test-average weight difference x returns "
             "(descriptive). A difference driven mainly by the average-exposure part comes from holding more (or less) "
             "risk on average, not from better timing.\n")
    L.append("## 4. Selection (rules stated in the brief; no second round)\n")
    L.append(f"- Ranking by net Sharpe (10 bps): " + " > ".join(f"{LABEL[e]} ({tab.loc[e, 'net_sharpe']:.3f})"
                                                                for e in summary["ranking_by_net_sharpe_10bps"]) + ".")
    L.append(f"- Highest net Sharpe: {LABEL[summary['highest_net_sharpe']]}.")
    L.append(f"- Most defensible risk-return candidate: {LABEL[summary['most_defensible_risk_return']] if summary['most_defensible_risk_return'] else 'none'} "
             f"({summary['defensible_rule']}).")
    L.append(f"- Improves on both P2 and P1 (net Sharpe, 10 bps): {', '.join(LABEL[e] for e in summary['improves_on_P2_and_P1']) or 'none'}.")
    L.append(f"- Of those, still above both at 25 bps: {', '.join(LABEL[e] for e in summary['improvement_survives_25bps']) or 'none'}.")
    L.append(f"- Maximum drawdown worse than P0 ({pct(tab.loc['P0_strategic', 'max_drawdown'])}): "
             f"{', '.join(LABEL[e] for e in summary['flagged_mdd_worse_than_P0']) or 'none'}.")
    L.append("- Net Sharpe above 0.70 / 0.75 / 0.80: " + "; ".join(f"{k}: {', '.join(LABEL[e] for e in v) or 'none'}"
                                                              for k, v in summary["thresholds_exceeded"].items()) + ".\n")
    L.append("## 5. Integrity checks\n")
    L.append(_md_table(pd.DataFrame([vars(x) for x in checks]).set_index("name")) + "\n")
    L.append("## 6. Interpretation limits\n")
    L.append("- Four variants were evaluated on the same, already-seen Test period; the best of four is selected with "
             "hindsight and its Sharpe is biased upward.\n- Differences of a few hundredths of Sharpe over ~7 years are "
             "inside the bootstrap intervals; they are not evidence of a better strategy.\n- Any candidate would need a "
             "new, untouched evaluation period (live paper trading) before it could replace the frozen P2 policy.\n")
    L.append("Figure: `reports/figures/quick_sharpe_comparison.png`\n")
    return "\n".join(L)


if __name__ == "__main__":
    warnings.simplefilter("default")
    sys.exit(main())
