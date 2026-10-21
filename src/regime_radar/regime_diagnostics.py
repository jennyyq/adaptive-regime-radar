"""Model-agnostic regime diagnostics (stage 4).

Populations, run lengths, switching, posterior entropy, label alignment, clustering agreement,
Newey-West inference, factor profiles, rule-based hypothesis labels and conditional asset statistics.
Every function works on arrays of integer state labels 0..K-1 and never looks beyond the data passed in.
"""

from __future__ import annotations

STAGE4_CODE_VERSION = "stage4-2026-10-21b"  # must match scripts/06_explore_regimes.py

import math

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

TRADING_DAYS = 252
STRESS_SCORE = {"volatility": 1.0, "liquidity_credit": 1.0, "equity_trend": -1.0}


def state_names(k: int) -> list[str]:
    return [f"S{i + 1}" for i in range(k)]


# --------------------------------------------------------------------------------------
# Populations, durations, switching
# --------------------------------------------------------------------------------------
def population(labels: np.ndarray, k: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    counts = np.bincount(labels, minlength=k)[:k]
    return counts / max(len(labels), 1)


def run_lengths(labels: np.ndarray) -> pd.DataFrame:
    """Consecutive runs of the same state. First and last runs are censored by the sample boundaries."""
    labels = np.asarray(labels)
    if len(labels) == 0:
        return pd.DataFrame({"state": pd.Series(dtype=int), "length": pd.Series(dtype=int),
                             "start": pd.Series(dtype=int)})
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(labels)]
    return pd.DataFrame({"state": labels[starts].astype(int), "length": (ends - starts).astype(int),
                         "start": starts.astype(int)})


def switching_stats(labels: np.ndarray) -> dict:
    labels = np.asarray(labels)
    runs = run_lengths(labels)
    n = len(labels)
    switches = max(len(runs) - 1, 0)
    return {
        "n_obs": n,
        "n_switches": switches,
        "switches_per_year": switches / n * TRADING_DAYS if n > 1 else np.nan,
        "mean_duration_days": float(runs["length"].mean()) if len(runs) else np.nan,
        "median_duration_days": float(runs["length"].median()) if len(runs) else np.nan,
        "share_single_day_runs": float((runs["length"] == 1).mean()) if len(runs) else np.nan,
    }


def transition_counts(labels: np.ndarray, k: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    counts = np.zeros((k, k), dtype=int)
    if len(labels) > 1:
        np.add.at(counts, (labels[:-1], labels[1:]), 1)
    return counts


def normalized_entropy(proba: np.ndarray) -> np.ndarray:
    """Posterior entropy divided by log(K): 0 = certain, 1 = uniform."""
    proba = np.asarray(proba, dtype=float)
    k = proba.shape[1]
    if k < 2:
        return np.zeros(len(proba))
    p = np.clip(proba, 1e-300, 1.0)
    return -(proba * np.log(p)).sum(axis=1) / math.log(k)


# --------------------------------------------------------------------------------------
# Agreement between label sets (permutation-aware)
# --------------------------------------------------------------------------------------
def align_labels(reference: np.ndarray, other: np.ndarray, k: int) -> np.ndarray:
    """Mapping m (length k) such that m[other_label] is the reference label with maximal overlap (Hungarian)."""
    table = np.zeros((k, k), dtype=int)
    np.add.at(table, (np.asarray(other, dtype=int), np.asarray(reference, dtype=int)), 1)
    rows, cols = linear_sum_assignment(-table)
    mapping = np.arange(k)
    mapping[rows] = cols
    return mapping


def aligned_agreement(reference: np.ndarray, other: np.ndarray, k: int) -> float:
    mapping = align_labels(reference, other, k)
    return float(np.mean(mapping[np.asarray(other, dtype=int)] == np.asarray(reference, dtype=int)))


def pairwise_agreement(label_sets: list[np.ndarray], k: int) -> dict:
    """ARI (permutation invariant by construction) and label-aligned agreement over all seed pairs."""
    ari, agree = [], []
    for i in range(len(label_sets)):
        for j in range(i + 1, len(label_sets)):
            ari.append(adjusted_rand_score(label_sets[i], label_sets[j]))
            agree.append(aligned_agreement(label_sets[i], label_sets[j], k))
    if not ari:
        return {"mean_ari": np.nan, "min_ari": np.nan, "mean_aligned_agreement": np.nan}
    return {"mean_ari": float(np.mean(ari)), "min_ari": float(np.min(ari)),
            "mean_aligned_agreement": float(np.mean(agree))}


# --------------------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------------------
def newey_west_se(x: np.ndarray, lags: int) -> float:
    """Newey-West (Bartlett kernel) standard error of the mean of an autocorrelated series."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return np.nan
    e = x - x.mean()
    lrv = e @ e / n
    for lag in range(1, min(lags, n - 1) + 1):
        weight = 1 - lag / (lags + 1)
        lrv += 2 * weight * (e[lag:] @ e[:-lag]) / n
    return float(math.sqrt(max(lrv, 0.0) / n))


# --------------------------------------------------------------------------------------
# Profiles, ordering, hypothesis labels
# --------------------------------------------------------------------------------------
def factor_profile(features: pd.DataFrame, labels: np.ndarray, k: int) -> pd.DataFrame:
    """Mean factor values per state (original factor units), with observation counts and shares."""
    labels = np.asarray(labels, dtype=int)
    rows = []
    for s in range(k):
        block = features[labels == s]
        rows.append({"state": f"S{s + 1}", "n_obs": len(block), "share": len(block) / max(len(features), 1),
                     **{c: (block[c].mean() if len(block) else np.nan) for c in features.columns}})
    return pd.DataFrame(rows).set_index("state")


def stress_order(centroids: pd.DataFrame) -> np.ndarray:
    """Permutation of raw cluster ids sorted by ascending stress score (volatility + liquidity_credit - equity).

    This is only an ordering convention so that state ids and colours are comparable across models;
    it is not an economic label. Empty clusters (NaN centroid) are placed last.
    """
    score = sum(w * centroids[c] for c, w in STRESS_SCORE.items() if c in centroids)
    score = score.fillna(np.inf)
    return np.argsort(score.to_numpy(), kind="stable")


HYPOTHESIS_RULES = (
    ("Liquidity Crisis",
     "volatility >= 1.0 and liquidity_credit >= 1.0 and equity_trend <= -0.5",
     lambda r: r["volatility"] >= 1.0 and r["liquidity_credit"] >= 1.0 and r["equity_trend"] <= -0.5),
    ("Inflation / Rates Shock",
     "max(inflation, interest_rate) >= 0.75 and volatility < 1.0",
     lambda r: max(r["inflation"], r["interest_rate"]) >= 0.75 and r["volatility"] < 1.0),
    ("Fragile / Transition",
     "equity_trend < 0 and max(volatility, liquidity_credit) >= 0.25 and not Liquidity Crisis",
     lambda r: r["equity_trend"] < 0 and max(r["volatility"], r["liquidity_credit"]) >= 0.25
     and not (r["volatility"] >= 1.0 and r["liquidity_credit"] >= 1.0 and r["equity_trend"] <= -0.5)),
    ("Growth / Risk-on",
     "equity_trend >= 0.25 and volatility <= 0 and liquidity_credit <= 0",
     lambda r: r["equity_trend"] >= 0.25 and r["volatility"] <= 0 and r["liquidity_credit"] <= 0),
)


def hypothesis_label(profile_row: pd.Series) -> tuple[str, str]:
    """Rule-based hypothesis from a TRAIN factor profile. Exactly one matching rule -> that label;
    none or several -> neutral (the state keeps its S# identifier). Thresholds were fixed before modelling."""
    if profile_row[["equity_trend", "volatility", "liquidity_credit", "inflation", "interest_rate"]].isna().any():
        return "Neutral", "empty state"
    matches = [name for name, _, rule in HYPOTHESIS_RULES if rule(profile_row)]
    if len(matches) == 1:
        return f"{matches[0]} (hypothesis)", matches[0]
    if not matches:
        return "Neutral", "no rule matched"
    return "Neutral", "mixed: " + " + ".join(matches)


# --------------------------------------------------------------------------------------
# Conditional asset statistics (descriptive)
# --------------------------------------------------------------------------------------
def asset_statistics(returns: pd.DataFrame, labels: np.ndarray, k: int) -> pd.DataFrame:
    """Per state and ETF, two descriptive horizons:

    same_day     : state_t with return_t (contemporaneous; both known at the close of t; NOT tradable)
    next_session : state_t with return_(t+1) inside the same split (closer to an implementable timing, but
                   still in-sample and conditional on a classifier fitted on the whole train period)
    """
    labels = pd.Series(np.asarray(labels, dtype=int), index=returns.index)
    rows = []
    for horizon, frame in (("same_day", returns), ("next_session", returns.shift(-1))):
        for s in range(k):
            block = frame[(labels == s).to_numpy()].dropna(how="all")
            for etf in returns.columns:
                r = block[etf].dropna()
                rows.append({
                    "horizon": horizon, "state": f"S{s + 1}", "etf": etf, "n_days": len(r),
                    "mean_daily_bp": 1e4 * r.mean() if len(r) else np.nan,
                    "annualized_mean_pct": 100 * TRADING_DAYS * r.mean() if len(r) else np.nan,
                    "annualized_vol_pct": 100 * math.sqrt(TRADING_DAYS) * r.std() if len(r) > 1 else np.nan,
                    "hit_rate": float((r > 0).mean()) if len(r) else np.nan,
                })
    return pd.DataFrame(rows)
