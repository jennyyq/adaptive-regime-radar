"""Initial observable daily features; the five-factor model is a later phase."""

from __future__ import annotations

import numpy as np
import pandas as pd


def lagged_robust_z(series: pd.Series, window: int = 252, min_periods: int = 126) -> pd.Series:
    """Scale today's signal with the *previous* window's median and MAD."""
    history = series.shift(1)
    median = history.rolling(window, min_periods=min_periods).median()
    deviation = (history - median).abs()
    mad = deviation.rolling(window, min_periods=min_periods).median()
    return ((series - median) / (1.4826 * mad.replace(0, np.nan))).clip(-8, 8)


def build_initial_features(close: pd.DataFrame) -> pd.DataFrame:
    required = ["SPY", "HYG", "LQD", "IWM", "QQQ", "^VIX"]
    missing = sorted(set(required) - set(close.columns))
    if missing:
        raise ValueError(f"Missing market tickers: {missing}")
    price = close[required].replace(0, np.nan)
    logp = np.log(price)
    logret = logp.diff()
    features = pd.DataFrame(index=close.index)
    features["equity_trend_21d"] = logp["SPY"].diff(21)
    features["equity_trend_63d"] = logp["SPY"].diff(63)
    features["volatility_21d"] = logret["SPY"].rolling(21, min_periods=21).std() * np.sqrt(252)
    features["vix_level"] = price["^VIX"]
    features["credit_hyg_lqd_21d"] = (logp["HYG"] - logp["LQD"]).diff(21)
    features["small_cap_iwm_qqq_21d"] = (logp["IWM"] - logp["QQQ"]).diff(21)
    for column in list(features):
        features[f"{column}_z"] = lagged_robust_z(features[column])
    features.index.name = "date"
    return features
