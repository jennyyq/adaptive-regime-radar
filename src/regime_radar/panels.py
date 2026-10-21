"""Turn raw per-series files into aligned date x series panels."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .pit import align_to_calendar, availability
from .sources.yahoo import FIELDS


def trading_calendar(raw: dict[str, pd.DataFrame], calendar_ticker: str, start: str | None = None) -> pd.DatetimeIndex:
    if calendar_ticker not in raw:
        raise ValueError(f"Calendar ticker {calendar_ticker} was not downloaded")
    dates = raw[calendar_ticker]["adj_close"].dropna().index
    if start:
        dates = dates[dates >= pd.Timestamp(start)]
    return pd.DatetimeIndex(dates, name="date")


def market_tables(raw: dict[str, pd.DataFrame], calendar: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    """One wide table per field on the trading calendar. Nothing is filled."""
    tables = {}
    for field in FIELDS:
        table = pd.DataFrame({ticker: frame[field] for ticker, frame in raw.items()})
        tables[field] = table.reindex(calendar)
    return tables


def investment_panel(adj_close: pd.DataFrame, etfs: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prices from the first date all ETFs trade, and daily log returns (missing stays NaN)."""
    missing = [t for t in etfs if t not in adj_close.columns or adj_close[t].dropna().empty]
    if missing:
        raise ValueError(f"No price history for investment ETFs: {missing}")
    common_start = max(adj_close[t].first_valid_index() for t in etfs)
    prices = adj_close.loc[common_start:, etfs]
    if (prices <= 0).any().any():
        raise ValueError("Non-positive adjusted prices in the investment panel")
    log_returns = np.log(prices).diff().iloc[1:]
    return prices, log_returns


def macro_panel(
    fred_raw: dict[str, pd.DataFrame],
    specs: dict[str, dict],
    calendar: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Point-in-time values, the observation date behind each value, and a release log."""
    values, obs_dates, release_log = {}, {}, []
    for series_id, raw in fred_raw.items():
        spec = specs[series_id]
        obs = availability(raw, spec)
        aligned = align_to_calendar(obs, calendar)
        values[series_id] = aligned["value"]
        obs_dates[series_id] = aligned["obs_date"]
        log = obs[["date", "value", "available_date", "availability_method"]].copy()
        log.insert(0, "series_id", series_id)
        release_log.append(log)
    values_df = pd.DataFrame(values, index=calendar)
    obs_df = pd.DataFrame(obs_dates, index=calendar)
    log_df = pd.concat(release_log, ignore_index=True) if release_log else pd.DataFrame()
    return values_df, obs_df, log_df
