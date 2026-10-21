"""Stage 4 decision record and diagnostics of the selected K=2 regime model (Train / Validation only).

The decision below was taken by the researcher on 2026-10-21 after reviewing the stage-4 comparison
(Train fit, Validation comparison). The test split had not been read by stage 4. The record is checked
against every new run: if a re-download or code change makes the data disagree with it, the checks WARN.
"""

from __future__ import annotations

STAGE4_CODE_VERSION = "stage4-2026-10-21b"  # must match scripts/06_explore_regimes.py

import numpy as np
import pandas as pd

from . import regime_diagnostics as rd
from .dataset import FEATURE_COLUMNS
from .factors import Check

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"

STAGE4_DECISION = {
    "decision_date": "2026-10-21",
    "primary_k": 2,
    "reference_model": "gmm_diag_k2",          # probabilistic reference for the HMM stage
    "baseline_model": "kmeans_k2",             # hard-clustering baseline
    "alternatives_for_hmm": [3, 4],            # pre-registered alternatives inside the HMM search range
    "state_names": {"S1": "Normal / Low stress", "S2": "Stress / Risk-off"},
    "data_used": "Train for fitting, Validation for comparison; the test split was not read",
    "basis": [
        "Predefined rules select K=2 in both families (GMM: best validation log-likelihood and most parsimonious; "
        "K-Means: best validation silhouette).",
        "K=2 is the most stable solution (mean pairwise ARI 1.00 across 10 seeds for both families) and the GMM and "
        "K-Means K=2 partitions agree (ARI 0.82 Train, 0.67 Validation).",
        "K=3 and K=4 are not significantly worse on Validation (Newey-West t < 2) and stay in the HMM search range.",
        "K=3 as the primary model was rejected under criteria fixed before the K=3 profiles were inspected: the "
        "K-Means rates state is not stable between Train and Validation, and the GMM third state is a "
        "stress-intensity split, so the third dimension is model-dependent.",
        "Elbow diagnostic (supporting, not decisive): the primary Kneedle elbow is computed over the candidate "
        "range K=1-6, with every other range reported as sensitivity; the figures from this run are in the "
        "evidence list below and in sections 8-9.",
    ],
    "state_naming_rule": "S2 is the state with the higher train stress score (volatility + liquidity_credit - "
                         "equity_trend). Names are descriptive; the stage-4 rule label 'Liquidity Crisis' was not "
                         "adopted because the state covers about a third of train days, not only crisis episodes.",
    "known_limitation": "K=2 represents the risk-on / risk-off axis only. Interest-rate conditions are almost "
                        "identical in the two states and inflation pressure is lower in the stress state "
                        "(disinflationary stress), so a rates or inflation shock in which bonds and equities fall "
                        "together has no dedicated state. The HMM stage tests K=3 for this reason.",
}
MIN_EPISODE_DAYS = 10


# --------------------------------------------------------------------------------------
# Consistency of the decision with the current run
# --------------------------------------------------------------------------------------
def decision_checks(result, profiles: pd.DataFrame, decision: dict = STAGE4_DECISION) -> list[Check]:
    sel = result.selection
    checks = []
    ref, base = decision["reference_model"], decision["baseline_model"]
    checks.append(Check("Decision: reference model equals the rule-based GMM preference",
                        PASS if sel.get("gmm_preferred") == ref else WARN,
                        f"decision {ref}; rule output {sel.get('gmm_preferred')}"))
    checks.append(Check("Decision: baseline equals the rule-based K-Means preference",
                        PASS if sel.get("kmeans_preferred") == base else WARN,
                        f"decision {base}; rule output {sel.get('kmeans_preferred')}"))
    k_range = sel.get("candidate_k_range") or []
    wanted = [decision["primary_k"]] + list(decision["alternatives_for_hmm"])
    checks.append(Check("Decision: primary and alternative K inside the validation-supported range",
                        PASS if set(wanted) <= set(k_range) else WARN,
                        f"decision K {wanted}; not-significantly-worse range {k_range}"))
    ok_names, detail = True, []
    for model_id in (ref, base):
        for split in ("train", "validation"):
            rows = profiles[(profiles["model_id"] == model_id) & (profiles["split"] == split)].set_index("state")
            if rows.empty or not {"S1", "S2"} <= set(rows.index):
                ok_names = False
                detail.append(f"{model_id} {split}: states missing")
                continue
            score = rows["volatility"] + rows["liquidity_credit"] - rows["equity_trend"]
            if not score["S2"] > score["S1"]:
                ok_names = False
            detail.append(f"{model_id} {split}: stress score S1 {score['S1']:.2f} < S2 {score['S2']:.2f}")
    checks.append(Check("Decision: state names match the data (S2 = higher stress in Train and Validation)",
                        PASS if ok_names else WARN, "; ".join(detail)))
    return checks


# --------------------------------------------------------------------------------------
# Selected-model diagnostics
# --------------------------------------------------------------------------------------
def selected_states(result, data, decision: dict = STAGE4_DECISION) -> pd.DataFrame:
    """Tidy daily table for the reference model (Train + Validation), with the baseline for comparison."""
    ref, base = result.reference[decision["reference_model"]], result.reference[decision["baseline_model"]]
    names = decision["state_names"]
    frames = []
    for split, idx, labels, proba, base_labels in (
            ("train", data.train_features.index, ref.train_labels, ref.train_proba, base.train_labels),
            ("validation", data.validation_features.index, ref.validation_labels, ref.validation_proba,
             base.validation_labels)):
        states = [f"S{i + 1}" for i in labels]
        frame = pd.DataFrame({"split": split, "state": states, "state_name": [names.get(s, s) for s in states]},
                             index=idx)
        for j in range(ref.spec.k):
            frame[f"p_S{j + 1}"] = proba[:, j]
        frame["max_posterior"] = proba.max(axis=1)
        frame["normalized_entropy"] = rd.normalized_entropy(proba)
        frame["baseline_state"] = [f"S{i + 1}" for i in base_labels]
        frame["models_agree"] = frame["state"] == frame["baseline_state"]
        frames.append(frame)
    out = pd.concat(frames)
    out.index.name = "date"
    return out


def transition_matrix(labels: np.ndarray, k: int) -> pd.DataFrame:
    """Empirical one-day transition probabilities P(state_t+1 | state_t)."""
    counts = rd.transition_counts(labels, k).astype(float)
    rows = counts.sum(axis=1, keepdims=True)
    probs = np.divide(counts, rows, out=np.full_like(counts, np.nan), where=rows > 0)
    names = [f"S{i + 1}" for i in range(k)]
    return pd.DataFrame(probs, index=pd.Index(names, name="from"), columns=names)


def transition_table(result, decision: dict = STAGE4_DECISION) -> pd.DataFrame:
    rows = []
    for model_id in (decision["reference_model"], decision["baseline_model"]):
        fit = result.reference[model_id]
        for split, labels in (("train", fit.train_labels), ("validation", fit.validation_labels)):
            tm = transition_matrix(labels, fit.spec.k)
            for origin in tm.index:
                p_stay = tm.loc[origin, origin]
                rows.append({"model_id": model_id, "split": split, "from": origin,
                             **{f"to_{c}": tm.loc[origin, c] for c in tm.columns},
                             "implied_expected_duration_days": 1 / (1 - p_stay) if p_stay < 1 else np.inf})
    return pd.DataFrame(rows)


def episodes(states: pd.DataFrame, state: str = "S2", min_days: int = MIN_EPISODE_DAYS) -> pd.DataFrame:
    """Consecutive runs of `state` lasting at least `min_days` trading days, within each split."""
    rows = []
    for split, frame in states.groupby("split", sort=False):
        labels = (frame["state"] == state).to_numpy()
        runs = rd.run_lengths(labels.astype(int))
        for r in runs[(runs["state"] == 1) & (runs["length"] >= min_days)].itertuples():
            block = frame.iloc[r.start:r.start + r.length]
            rows.append({"split": split, "state": state, "start": block.index[0].date().isoformat(),
                         "end": block.index[-1].date().isoformat(), "trading_days": int(r.length),
                         "mean_posterior": round(float(block[f"p_{state}"].mean()), 3),
                         "baseline_agreement": round(float(block["models_agree"].mean()), 3)})
    return pd.DataFrame(rows, columns=["split", "state", "start", "end", "trading_days", "mean_posterior",
                                       "baseline_agreement"])


def summary_table(result, data, profiles: pd.DataFrame, decision: dict = STAGE4_DECISION) -> pd.DataFrame:
    """One row per state and split: share, factor means and persistence of the reference model."""
    ref = result.reference[decision["reference_model"]]
    rows = []
    for split, labels in (("train", ref.train_labels), ("validation", ref.validation_labels)):
        prof = profiles[(profiles["model_id"] == ref.spec.model_id) & (profiles["split"] == split)].set_index("state")
        runs = rd.run_lengths(labels)
        for s in range(ref.spec.k):
            name = f"S{s + 1}"
            lengths = runs.loc[runs["state"] == s, "length"]
            rows.append({"split": split, "state": name, "state_name": decision["state_names"].get(name, name),
                         "share": prof.loc[name, "share"], **{c: prof.loc[name, c] for c in FEATURE_COLUMNS},
                         "episodes": len(lengths), "median_run_days": float(lengths.median()) if len(lengths) else np.nan,
                         "mean_run_days": float(lengths.mean()) if len(lengths) else np.nan})
    return pd.DataFrame(rows)


def evidence_statements(summary: pd.DataFrame, assets: pd.DataFrame, model_id: str) -> list[str]:
    """Data-driven reading of the selected model; every statement is computed from this run."""
    out = []
    tr = summary[summary["split"] == "train"].set_index("state")
    if {"S1", "S2"} <= set(tr.index):
        risk_off = (tr.loc["S2", "equity_trend"] < tr.loc["S1", "equity_trend"]
                    and tr.loc["S2", "volatility"] > tr.loc["S1", "volatility"]
                    and tr.loc["S2", "liquidity_credit"] > tr.loc["S1", "liquidity_credit"])
        out.append(f"S2 has {'weaker' if tr.loc['S2', 'equity_trend'] < tr.loc['S1', 'equity_trend'] else 'stronger'} "
                   f"equity trend ({tr.loc['S2', 'equity_trend']:+.2f} vs {tr.loc['S1', 'equity_trend']:+.2f}), higher "
                   f"volatility ({tr.loc['S2', 'volatility']:+.2f} vs {tr.loc['S1', 'volatility']:+.2f}) and credit "
                   f"stress ({tr.loc['S2', 'liquidity_credit']:+.2f} vs {tr.loc['S1', 'liquidity_credit']:+.2f}) on "
                   f"Train: {'a risk-off profile' if risk_off else 'not a clean risk-off profile'}; it covers "
                   f"{tr.loc['S2', 'share']:.0%} of Train days, "
                   + ("so it is broader than crisis episodes alone." if tr.loc["S2", "share"] >= 0.15
                      else "so it is concentrated in a few severe episodes."))
        out.append(f"Interest-rate factor S1 {tr.loc['S1', 'interest_rate']:+.2f} vs S2 {tr.loc['S2', 'interest_rate']:+.2f}; "
                   f"inflation factor S1 {tr.loc['S1', 'inflation']:+.2f} vs S2 {tr.loc['S2', 'inflation']:+.2f}.")
    a = assets[assets["model_id"] == model_id]

    def value(split, horizon, state, etf):
        row = a[(a["split"] == split) & (a["horizon"] == horizon) & (a["state"] == state) & (a["etf"] == etf)]
        n_ok = not row.empty and row["n_days"].iloc[0] >= MIN_EPISODE_DAYS * 2
        return float(row["annualized_mean_pct"].iloc[0]) if n_ok else np.nan

    ief_s1, ief_s2 = value("train", "same_day", "S1", "IEF"), value("train", "same_day", "S2", "IEF")
    if np.isfinite(ief_s1) and np.isfinite(ief_s2):
        out.append(f"Treasuries (IEF) same-day on Train: S2 {ief_s2:+.1f}% vs S1 {ief_s1:+.1f}% annualized "
                   f"({'flight to quality' if ief_s2 > ief_s1 else 'no flight-to-quality pattern'}).")
    out.append("Same-day differences are partly mechanical, because the factors are built from the same prices.")
    spy_tr, spy_va = value("train", "next_session", "S2", "SPY"), value("validation", "next_session", "S2", "SPY")
    if np.isfinite(spy_tr) and np.isfinite(spy_va):
        same = np.sign(spy_tr) == np.sign(spy_va)
        out.append(f"SPY next-session return after S2 days: Train {spy_tr:+.1f}%, Validation {spy_va:+.1f}% annualized; "
                   + ("same sign in both periods, still in-sample for Train and not evidence of predictability."
                      if same else "the sign reverses between periods, so the state is descriptive and carries no "
                                   "stable information about the next day's equity return."))
    else:
        out.append("Next-session returns after S2 days cannot be compared across periods (too few S2 days).")
    return out
