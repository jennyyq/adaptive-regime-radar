"""Stage 8 Phase 8A (preflight, no Test data): Stage 7 handoff verification, analytically tractable ledger
examples and the pre-Test protocol document.

Nothing in this module reads a Test file. The pre-Test lock itself is written by scripts/10_backtest_portfolios.py
through backtest_integrity.build_lock once every preflight check passes.
"""

from __future__ import annotations

STAGE8_CODE_VERSION = "stage8-2026-10-21"  # must match scripts/10_backtest_portfolios.py

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest_accounting import Ledger
from .backtest_execution import Order, execute
from .backtest_integrity import sha256


# --------------------------------------------------------------------------------------
# Stage 7 handoff verification
# --------------------------------------------------------------------------------------
def verify_stage7_handoff(root: Path, s8: dict, policy_cfg: dict, project_cfg: dict, policy_path: Path,
                          project_path: Path, h61_path: Path) -> list[tuple[str, str, str]]:
    """(name, status, detail) rows. `s8` = data/processed/portfolio_optimization/stage8_handoff_manifest.json."""
    rows = []
    rows.append(("Stage 7 handoff status READY", "PASS" if s8.get("handoff_status") == "READY" else "FAIL",
                 str(s8.get("handoff_status_reason", ""))[:200]))
    h = s8.get("hashes", {})
    rows.append(("Frozen configuration hashes match (portfolio_policy.json, project.json)",
                 "PASS" if h.get("portfolio_policy.json") == sha256(policy_path) and h.get("project.json") == sha256(project_path)
                 else "FAIL", "configs/portfolio_policy.json, configs/project.json"))
    rows.append(("Stage 6.1 handoff manifest hash matches", "PASS" if h.get("stage7_handoff_manifest") == sha256(h61_path)
                 else "FAIL", str(h61_path.relative_to(root))))
    bad = [k for k, v in h.get("stage7_outputs", {}).items() if not (root / k).exists() or sha256(root / k) != v]
    rows.append(("Stage 7 output hashes match the handoff (targets, risk estimates, sensitivity, ...)",
                 "PASS" if not bad else "FAIL", f"{len(h.get('stage7_outputs', {}))} files; mismatched: "
                 f"{', '.join(bad[:4]) or 'none'}"))
    badc = [k for k, v in h.get("stage7_code", {}).items() if sha256(root / "src" / "regime_radar" / k) != v]
    rows.append(("Stage 7 code unchanged since the handoff", "PASS" if not badc else "FAIL",
                 f"{len(h.get('stage7_code', {}))} files; changed: {', '.join(badc) or 'none'}"))
    badm = [k for k, v in h.get("frozen_models", {}).items() if sha256(root / v["file"]) != v["sha256"]]
    rows.append(("Frozen model files (HMM K=3, AR(1), AR-HMM) unchanged - no refit", "PASS" if not badm else "FAIL",
                 ", ".join(f"{k}: {v['file']}" for k, v in h.get("frozen_models", {}).items())))
    bads = [k for k, v in h.get("split_files", {}).items() if sha256(root / "data" / "splits" / k) != v]
    rows.append(("Train / Validation split files unchanged", "PASS" if not bads else "FAIL",
                 f"{len(h.get('split_files', {}))} files; changed: {', '.join(bads) or 'none'}"))
    same_c = s8.get("constraints") == policy_cfg.get("constraints") and s8.get("risk_budget") == policy_cfg.get("risk_budget")
    bench = s8.get("benchmark_weights", {})
    same_b = all(abs(bench.get(a, -1) - project_cfg["strategic_weights"][a]) < 1e-15 for a in project_cfg["investment_etfs"])
    rows.append(("Portfolio constraints, risk-budget mapping and benchmark match the frozen Stage 7 policy",
                 "PASS" if same_c and same_b else "FAIL",
                 f"constraints {'=' if same_c else '!='}; benchmark {'=' if same_b else '!='} configs/project.json"))
    rb = s8.get("role_bindings", {})
    rows.append(("Signal roles separated: P1 -> baseline_regime, P2 -> challenger_ews",
                 "PASS" if rb == {"P1_regime_aware": "baseline_regime", "P2_ews_enhanced": "challenger_ews"} else "FAIL",
                 json.dumps(rb)))
    conv = (s8.get("execution_convention") or {}).get("name")
    rows.append(("Execution timing is the approved conservative convention", "PASS" if conv == "conservative_close_t_plus_1"
                 else "FAIL", str(conv)))
    rows.append(("Stage 7 read no Test data", "PASS" if s8.get("test_data_loaded") is False else "FAIL",
                 f"test_data_loaded = {s8.get('test_data_loaded')}"))
    tc = s8.get("train_calibrated", {})
    ok = all(isinstance(tc.get(k), (int, float)) and np.isfinite(tc.get(k)) for k in
             ("P1_parity_scale_a", "stress_vol_threshold_spy_21d_annualized"))
    rows.append(("Train-calibrated constants available (P1 scale a, cost-stress threshold)", "PASS" if ok else "FAIL",
                 f"a = {tc.get('P1_parity_scale_a')!r}; stress threshold = {tc.get('stress_vol_threshold_spy_21d_annualized')!r}"))
    return rows


# --------------------------------------------------------------------------------------
# Analytically tractable ledger examples (synthetic, no market data)
# --------------------------------------------------------------------------------------
def _order(kind, target, t, e, cap=1.0, stress=False):
    return Order("X", pd.Timestamp(t), pd.Timestamp(e), None, kind, np.asarray(target, float), cap, stress, "test")


def analytic_examples() -> list[dict]:
    """Each row: example, quantity, expected (closed form, dollar arithmetic), computed (ledger), abs error, status."""
    rows = []

    def add(ex, q, exp, got, tol=1e-12):
        err = float(np.max(np.abs(np.asarray(exp, float) - np.asarray(got, float))))
        rows.append({"example": ex, "quantity": q, "expected": json.dumps(np.round(np.atleast_1d(exp), 12).tolist()),
                     "computed": json.dumps(np.round(np.atleast_1d(got), 12).tolist()), "abs_error": err,
                     "status": "PASS" if err <= tol else "FAIL"})

    d = pd.bdate_range("2021-01-04", periods=4)
    R = np.array([[0.0, 0.0], [0.10, 0.0], [-0.10, 0.05], [0.02, 0.01]])

    # E1 two assets, signal on day 1, executed at the close of day 2 (t+1), first holding return day 3 (t+2)
    lg = Ledger("X", ["A", "B"], d[0], np.array([0.5, 0.5]), bps=(0, 10), primary_bps=10)
    lg.session(d[1], R[1], None)
    lg.session(d[2], R[2], _order("trade", [0.5, 0.5], d[1], d[2]))
    lg.session(d[3], R[3], None)
    dollars = np.array([0.5, 0.5]) * (1 + R[1]) * (1 + R[2])          # held through t+1: old holdings earn day 2
    v_pre = dollars.sum()
    notional = np.abs(np.array([0.5, 0.5]) - dollars / v_pre).sum()
    v_post = v_pre * (1 - 0.001 * notional)
    v3 = v_post * (0.5 * (1 + R[3, 0]) + 0.5 * (1 + R[3, 1]))         # new holdings earn day 3 only
    f = lg.frame()
    add("E1 two-asset delayed rebalance", "pre-trade wealth at the execution close", v_pre, f["wealth_pretrade_net_primary"].iloc[2])
    add("E1 two-asset delayed rebalance", "gross notional at execution", notional, f["gross_notional"].iloc[2])
    add("E1 two-asset delayed rebalance", "net wealth at day 3 (10 bps)", v3, f["wealth_net_10bps"].iloc[3])
    add("E1 two-asset delayed rebalance", "day-2 gross return earned by the OLD holdings",
        float(np.array([0.55, 0.5]) @ R[2] / 1.05), f["gross_return"].iloc[2])
    add("E1 two-asset delayed rebalance", "zero-cost scenario equals gross wealth", f["wealth_gross"].iloc[3],
        f["wealth_net_0bps"].iloc[3], 0.0)

    # E2 zero trades / E4 pure drift
    lg2 = Ledger("X", ["A", "B"], d[0], np.array([0.6, 0.4]), bps=(10,), primary_bps=10)
    for i in (1, 2, 3):
        lg2.session(d[i], R[i], None)
    px = np.cumprod(1 + R[1:], axis=0)
    add("E2 zero trades", "terminal wealth = sum w0 x price relative", float(np.array([0.6, 0.4]) @ px[-1]),
        lg2.frame()["wealth_net_10bps"].iloc[-1])
    wt = np.array([0.6, 0.4]) * px[-1] / (np.array([0.6, 0.4]) @ px[-1])
    add("E4 drift without rebalancing", "terminal weights = w0 x price relative / sum", wt, lg2.weights().iloc[-1].to_numpy())
    add("E2 zero trades", "no transaction cost", 0.0, lg2.frame()["cost_fraction_10bps"].sum(), 0.0)

    # E5 turnover cap: execution-day projection along the straight line
    o = _order("trade", [0.9, 0.1], d[0], d[1], cap=0.1)
    ex = execute(o, np.array([0.5, 0.5]))
    add("E5 turnover-capped execution", "executed weights", [0.6, 0.4], ex.executed)
    add("E5 turnover-capped execution", "one-way turnover = cap", 0.1, ex.turnover)

    # E6 alert at t executes at t+1 close: t+1 return with old weights, t+2 with new weights
    lg3 = Ledger("X", ["EQ", "BIL"], d[0], np.array([1.0, 0.0]), bps=(10,), primary_bps=10)
    lg3.session(d[1], np.array([-0.05, 0.0]), None)                                      # signal close d1
    lg3.session(d[2], np.array([-0.05, 0.0]), _order("trade", [0.0, 1.0], d[1], d[2]))   # execute close d2
    lg3.session(d[3], np.array([-0.05, 0.0]), None)
    g = lg3.frame()["gross_return"].to_numpy()
    add("E6 alert on t executed at t+1", "gross returns (t+1 still fully exposed, t+2 in BIL)", [-0.05, -0.05, 0.0], g[1:])

    # E7 repeated simulation identical
    def sim():
        lg_ = Ledger("X", ["A", "B"], d[0], np.array([0.5, 0.5]), bps=(5, 10, 25), primary_bps=10)
        lg_.session(d[1], R[1], None)
        lg_.session(d[2], R[2], _order("trade", [0.7, 0.3], d[1], d[2], cap=0.15, stress=True))
        lg_.session(d[3], R[3], None)
        return lg_.frame()[["wealth_net_5bps", "wealth_net_10bps", "wealth_net_25bps"]].to_numpy()
    a, b = sim(), sim()
    add("E7 round trip", "identical wealth under repeated simulation", a, b, 0.0)
    return rows


# --------------------------------------------------------------------------------------
# Pre-Test protocol document
# --------------------------------------------------------------------------------------
def render_protocol(lock: dict, checks: list, examples: pd.DataFrame, protocol: dict, ctx: dict) -> str:
    from .quality import _md_table
    fz = lock["frozen"]
    L = []
    L.append("# Stage 8 pre-Test protocol (Phase 8A)\n")
    L.append(f"**Lock fingerprint:** `{lock['fingerprint']}`  ")
    L.append(f"**Created (UTC):** {lock['created_utc']}  ")
    L.append(f"**Test access authorized:** {'YES' if lock['test_access_authorized'] else 'NO'} - "
             f"{lock['authorization_reason']}  ")
    L.append("**Test data read before this lock:** none (the Test split files were not opened; only their recorded "
             "hashes in the Stage 3 split manifest exist).\n")
    L.append("This document was generated before the reserved Test split was opened. Everything below is frozen in "
             "`data/processed/backtesting/stage8_pretest_lock.json` together with the hashes of every project module, "
             "configuration file and frozen artifact. Changing any of them invalidates the lock; the gated Test loader "
             "then refuses to read Test.\n")
    L.append("## 1. Research question\n")
    L.append(protocol["research_question"] + "\n")
    L.append("Terminology: " + protocol["terminology"] + "\n")
    L.append("## 2. Frozen strategies\n")
    rows = [{"id": k, "definition": v} for k, v in protocol["primary_portfolios"].items()]
    rows += [{"id": k, "definition": v["rule"]} for k, v in protocol["supplementary_benchmarks"].items()]
    L.append(_md_table(pd.DataFrame(rows).set_index("id")) + "\n")
    fc = fz["frozen_constants"]
    L.append("Frozen constants: " + ", ".join(f"{k} = {v!r}" for k, v in fc.items()) + "\n")
    L.append("## 3. Models and versions\n")
    L.append(_md_table(pd.DataFrame([{"artifact": k, "value": str(v)} for k, v in fz["model_versions"].items()])
                       .set_index("artifact")) + "\n")
    L.append("## 4. Data definitions and boundary\n")
    for k, v in fz["data_definitions"].items():
        L.append(f"- **{k}**: {v}")
    L.append("")
    L.append("## 5. Decision, execution and return accounting\n")
    for sec in ("decision_rule", "execution", "return_accounting"):
        for k, v in protocol[sec].items():
            L.append(f"- **{sec}.{k}**: {v}")
    L.append("")
    L.append("## 6. Transaction costs\n")
    for k, v in protocol["transaction_costs"].items():
        L.append(f"- **{k}**: {v}")
    L.append("")
    L.append("## 7. Performance metrics\n")
    for k, v in protocol["metrics"].items():
        L.append(f"- **{k}**: {v}")
    L.append("")
    L.append("## 8. Stress windows and objective events (fixed before Test)\n")
    sa = protocol["stress_analysis"]
    L.append(_md_table(pd.DataFrame([{"window": k, "start": v[0], "end": v[1] or "sample end"}
                                     for k, v in sa["fixed_windows"].items()]).set_index("window")) + "\n")
    for k in ("window_origin", "objective_events", "executable_protection", "de_risking_execution", "recovery_participation"):
        L.append(f"- **{k}**: {sa[k]}")
    L.append("")
    L.append("## 9. Statistical procedures\n")
    for k, v in protocol["statistics"].items():
        L.append(f"- **{k}**: {v}")
    L.append("")
    L.append("## 10. Decision rules for the final conclusion (set before Test)\n")
    for k, v in protocol["decision_rules"].items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                L.append(f"- **{kk}**: {vv}")
        else:
            L.append(f"- **{k}**: {v}")
    L.append("")
    L.append("## 11. Allowed sensitivity analyses\n")
    for k, v in protocol["sensitivity"].items():
        L.append(f"- **{k}**: {v}")
    L.append("")
    L.append("## 12. Model-selection restrictions\n")
    for r in protocol["restrictions"]:
        L.append(f"- {r}")
    L.append("")
    L.append("## 13. Implementation ambiguities resolved before Test\n")
    for r in ctx.get("ambiguities", []):
        L.append(f"- {r}")
    L.append("")
    L.append("## 14. Preflight checks\n")
    L.append(_md_table(pd.DataFrame(checks, columns=["check", "status", "detail"]).set_index("check")) + "\n")
    L.append("## 15. Analytic ledger examples\n")
    L.append(_md_table(examples.set_index("example")) + "\n")
    st = ctx.get("initial_state")
    if st is not None:
        L.append("## 16. Holdings carried into Test (pre-Test information only)\n")
        L.append(_md_table(st) + "\n")
        L.append(f"Initial-state hash (recorded in the lock): `{fz.get('initial_state_sha256')}`\n")
    L.append("## 17. Hash snapshot\n")
    L.append(f"{len(fz['hashes']['code'])} code files and {len(fz['hashes']['artifacts'])} configuration / artifact "
             "files are hashed in the lock.\n")
    return "\n".join(L)
