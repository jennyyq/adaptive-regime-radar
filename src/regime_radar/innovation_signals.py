"""Stage 6: causal HMM regime features and AR(1) predictive innovations from the FROZEN Stage 5 / 5.5 models.

Nothing here is estimated except Train-only normalization constants (documented in TrainNormalization).

HMM features (frozen Stage 5 HMM K=3, causal forward filter; never smoothed)
    p_s1, p_s2, p_s3, p_non_normal = p_s2 + p_s3, normalized entropy H / log 3,
    velocity_5d = p_t - p_(t-5) for p_s3 and p_non_normal (positive part = deterioration velocity).
    Train-period values use parameters estimated on all of Train: retrospective, not live.

AR(1) innovations (frozen Stage 5.5 single-regime AR(1), diagonal)
    e_t,d = x_t,d - (c_d + phi_d x_t-1,d) computed before x_t is seen; z_t,d = e_t,d / sigma_d.
    Adverse direction: equity_trend -1 (weakness), interest_rate +1 (tightening), inflation +1 (pressure),
    volatility +1, liquidity_credit +1. Directional z = sign_d * z_d (positive = adverse surprise).
    Channels (aggregation fixed in advance; correlated factors are combined ONCE, not counted separately):
        market_innovation = (-z_eq + z_vol + z_liq) / k_market     (common market-risk shock)
        macro_innovation  = (z_rate + z_infl) / k_macro            (rates / inflation shock)
        k = Train standard deviation of the unscaled sum, so each channel has unit variance on Train.
    5-session versions: rolling sums over the last 5 sessions, rescaled by their Train standard deviation.
    surprise_chi2 = sum_d z_t,d^2 (direction-free magnitude of the surprise).
AR-HMM predictive surprise (supplementary): -log p(x_t | x_1..x_t-1) of the frozen corrected AR-HMM K=3.
"""

from __future__ import annotations

STAGE6_CODE_VERSION = "stage6-2026-10-21"  # must match scripts/08_build_early_warning.py

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .ar_hmm import ARParams, filter_from_log_emission, lagged_pairs, log_emission_ar
from .dataset import FEATURE_COLUMNS
from .hmm_filtering import forward_filter
from .hmm_models import HMMParams

ADVERSE_SIGN = {"equity_trend": -1.0, "interest_rate": 1.0, "inflation": 1.0, "volatility": 1.0,
                "liquidity_credit": 1.0}
MARKET_CHANNEL = ("equity_trend", "volatility", "liquidity_credit")
MACRO_CHANNEL = ("interest_rate", "inflation")
VELOCITY_LAG = 5
ROLLING_DAYS = 5


# --------------------------------------------------------------------------------------
# HMM features
# --------------------------------------------------------------------------------------
def normalized_entropy(p: np.ndarray, floor: float = 1e-300) -> np.ndarray:
    p = np.asarray(p, float)
    k = p.shape[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.where(p > 0, p * np.log(np.clip(p, floor, 1.0)), 0.0).sum(axis=1)
    return np.clip(h / math.log(k), 0.0, 1.0)


def hmm_features(params: HMMParams, x_train: np.ndarray, x_validation: np.ndarray, train_index, val_index
                 ) -> pd.DataFrame:
    """Causal filtered probabilities over Train (from pi) continued into Validation, plus derived features.
    The state order is the frozen canonical order (S1 lowest stress ... S3 highest)."""
    tr = forward_filter(params, x_train)
    va = forward_filter(params, x_validation, previous_filtered=tr.last_filtered)
    p = np.vstack([tr.filtered, va.filtered])
    idx = pd.DatetimeIndex(train_index).append(pd.DatetimeIndex(val_index))
    f = pd.DataFrame({f"p_s{k + 1}": p[:, k] for k in range(p.shape[1])}, index=idx)
    f["p_stress"] = f[f"p_s{p.shape[1]}"]
    f["p_non_normal"] = 1.0 - f["p_s1"]
    f["normalized_entropy"] = normalized_entropy(p)
    f["hmm_state"] = p.argmax(axis=1) + 1
    for col, name in (("p_stress", "stress_probability_velocity"), ("p_non_normal", "non_normal_probability_velocity")):
        f[name] = f[col] - f[col].shift(VELOCITY_LAG)
        f[f"{name}_positive"] = f[name].clip(lower=0)
    f["hmm_log_predictive"] = np.r_[tr.log_score, va.log_score]
    f["split"] = ["train"] * len(train_index) + ["validation"] * len(val_index)
    return f


# --------------------------------------------------------------------------------------
# AR(1) innovations
# --------------------------------------------------------------------------------------
@dataclass
class TrainNormalization:
    """Constants estimated on Train only."""
    k_market: float = 1.0
    k_macro: float = 1.0
    k_market_5d: float = 1.0
    k_macro_5d: float = 1.0
    stress_index_mean: float = 0.0
    stress_index_sd: float = 1.0
    surprise_ecdf: np.ndarray = field(default_factory=lambda: np.array([0.0]))

    def to_dict(self) -> dict:
        return {"k_market": self.k_market, "k_macro": self.k_macro, "k_market_5d": self.k_market_5d,
                "k_macro_5d": self.k_macro_5d, "stress_index_mean": self.stress_index_mean,
                "stress_index_sd": self.stress_index_sd, "surprise_train_n": int(len(self.surprise_ecdf))}


def ar_innovations(params: ARParams, x_train: np.ndarray, x_validation: np.ndarray, train_index, val_index
                   ) -> pd.DataFrame:
    """Standardized one-step innovations z_t,d. Train row 1 has no lag (NaN); Validation row 1 uses the last
    Train row as its lag."""
    if params.k != 1:
        raise ValueError("expects the single-regime AR(1) model")
    y_tr, lag_tr = lagged_pairs(x_train)
    y_va, lag_va = lagged_pairs(x_validation, previous=x_train[-1])
    sd = np.sqrt(params.var[0])
    z_tr = (y_tr - (params.intercept[0] + params.phi[0] * lag_tr)) / sd
    z_va = (y_va - (params.intercept[0] + params.phi[0] * lag_va)) / sd
    z = np.vstack([np.full((1, len(sd)), np.nan), z_tr, z_va])
    idx = pd.DatetimeIndex(train_index).append(pd.DatetimeIndex(val_index))
    return pd.DataFrame(z, index=idx, columns=[f"z_{c}" for c in FEATURE_COLUMNS])


def fit_normalization(z_train: pd.DataFrame, levels_train: pd.DataFrame, surprise_train: pd.Series
                      ) -> TrainNormalization:
    raw = _raw_channels(z_train)
    n = TrainNormalization()
    n.k_market = float(raw["market"].std())
    n.k_macro = float(raw["macro"].std())
    n.k_market_5d = float((raw["market"] / n.k_market).rolling(ROLLING_DAYS).sum().std())
    n.k_macro_5d = float((raw["macro"] / n.k_macro).rolling(ROLLING_DAYS).sum().std())
    s = stress_index_raw(levels_train)
    n.stress_index_mean, n.stress_index_sd = float(s.mean()), float(s.std())
    n.surprise_ecdf = np.sort(surprise_train.dropna().to_numpy())
    return n


def _raw_channels(z: pd.DataFrame) -> pd.DataFrame:
    d = {c: ADVERSE_SIGN[c] * z[f"z_{c}"] for c in FEATURE_COLUMNS}
    return pd.DataFrame({"market": sum(d[c] for c in MARKET_CHANNEL), "macro": sum(d[c] for c in MACRO_CHANNEL)})


def stress_index_raw(levels: pd.DataFrame) -> pd.Series:
    """Factor-LEVEL stress index (standardized factor units): -equity_trend + volatility + liquidity_credit."""
    return sum(ADVERSE_SIGN[c] * levels[c] for c in MARKET_CHANNEL)


def innovation_frame(z: pd.DataFrame, levels: pd.DataFrame, surprise: pd.Series, norm: TrainNormalization
                     ) -> pd.DataFrame:
    raw = _raw_channels(z)
    out = pd.DataFrame(index=z.index)
    for c in FEATURE_COLUMNS:
        out[f"directional_z_{c}"] = ADVERSE_SIGN[c] * z[f"z_{c}"]
        out[f"abs_z_{c}"] = z[f"z_{c}"].abs()
    out["market_innovation"] = raw["market"] / norm.k_market
    out["macro_innovation"] = raw["macro"] / norm.k_macro
    out["market_innovation_5d"] = out["market_innovation"].rolling(ROLLING_DAYS).sum() / norm.k_market_5d
    out["macro_innovation_5d"] = out["macro_innovation"].rolling(ROLLING_DAYS).sum() / norm.k_macro_5d
    out["ar_innovation_stress"] = out["market_innovation_5d"].clip(lower=0)
    out["surprise_chi2"] = (z ** 2).sum(axis=1, min_count=len(FEATURE_COLUMNS))
    out["stress_index_level"] = (stress_index_raw(levels) - norm.stress_index_mean) / norm.stress_index_sd
    out["arhmm_surprise"] = surprise
    out["arhmm_surprise_probit"] = probit_ecdf(surprise, norm.surprise_ecdf)
    return out


def probit_ecdf(values: pd.Series, train_sorted: np.ndarray) -> pd.Series:
    """Normal score of the Train empirical CDF (mid-rank, clipped): maps a skewed input to ~N(0,1) on Train."""
    from scipy.stats import norm

    v = np.asarray(values, float)
    n = len(train_sorted)
    rank = (np.searchsorted(train_sorted, v, side="left") + np.searchsorted(train_sorted, v, side="right")) / 2
    u = np.clip((rank + 0.5) / (n + 1), 0.5 / (n + 1), 1 - 0.5 / (n + 1))
    out = norm.ppf(u)
    out[~np.isfinite(v)] = np.nan
    return pd.Series(out, index=values.index)


def arhmm_surprise(params: ARParams, x_train: np.ndarray, x_validation: np.ndarray, train_index, val_index
                   ) -> pd.Series:
    """-log one-step predictive density of the frozen AR-HMM (causal filter; Validation continues from Train)."""
    y_tr, lag_tr = lagged_pairs(x_train)
    tr = filter_from_log_emission(log_emission_ar(params, y_tr, lag_tr), params.transmat,
                                  initial_predicted=params.startprob)
    y_va, lag_va = lagged_pairs(x_validation, previous=x_train[-1])
    va = filter_from_log_emission(log_emission_ar(params, y_va, lag_va), params.transmat,
                                  previous_filtered=tr.last_filtered)
    idx = pd.DatetimeIndex(train_index).append(pd.DatetimeIndex(val_index))
    return pd.Series(np.r_[np.nan, -tr.log_score, -va.log_score], index=idx, name="arhmm_surprise")
