"""Point-in-time availability for macro data.

Convention: a value is usable on trading day d if it was published on or before d.
Signals are formed after the close of d and executed no earlier than d+1, so a
value published during day d may inform the d+1 trade but never d's return.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import MonthBegin



def add_us_business_days(dates: pd.Series, lag: int) -> pd.Series:
    """Shift by `lag` US federal business days (weekend/holiday dates roll forward first)."""
    if dates.empty:
        return dates
    holidays = USFederalHolidayCalendar().holidays(dates.min(), dates.max() + pd.Timedelta(days=30))
    shifted = np.busday_offset(
        dates.values.astype("datetime64[D]"), lag, roll="forward",
        holidays=holidays.values.astype("datetime64[D]"),
    )
    return pd.Series(pd.to_datetime(shifted), index=dates.index)


def rule_dates(dates: pd.Series, rule: dict) -> pd.Series:
    dates = pd.to_datetime(dates)
    kind = rule["rule"]
    if kind == "business_days":
        return add_us_business_days(dates, rule.get("lag", 1))
    if kind == "calendar_days":
        return dates + pd.to_timedelta(rule["days"], unit="D")
    if kind == "next_month_day":
        return dates + MonthBegin(1) + pd.to_timedelta(rule["day"] - 1, unit="D")
    raise ValueError(f"Unsupported rule {kind!r}")


def availability(obs: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Add `available_date` and `availability_method` to obs[date, value, realtime_start]."""
    out = obs.copy()
    out["date"] = pd.to_datetime(out["date"])
    realtime = pd.to_datetime(out.get("realtime_start"), errors="coerce") if "realtime_start" in out else None
    rule = spec["availability"]

    if rule["rule"] != "vintage":
        out["available_date"] = rule_dates(out["date"], rule)
        out["availability_method"] = f"rule:{rule['rule']}"
        return out

    fallback = rule_dates(out["date"], rule["fallback"])
    if realtime is None or realtime.isna().all():
        out["available_date"] = fallback
        out["availability_method"] = "fallback:no_vintage_data"
        return out

    # ALFRED's first vintage back-fills all earlier history on one date. For those rows the
    # true first-release date is unknown; if the rule says the value would already have been
    # published before that vintage, use the rule date instead of the (much later) vintage date.
    first_vintage = realtime.min()
    pre_vintage = realtime.eq(first_vintage) & fallback.lt(first_vintage)
    use_vintage = realtime.notna() & ~pre_vintage
    out["available_date"] = realtime.where(use_vintage, fallback)
    out["availability_method"] = "vintage:first_release"
    out.loc[pre_vintage, "availability_method"] = "fallback:pre_vintage_history"
    out.loc[realtime.isna(), "availability_method"] = "fallback:missing_vintage"
    return out


def align_to_calendar(obs: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """For each trading day, the latest observation already published by that day.

    Returns columns value / obs_date indexed by calendar. If a delayed release is
    published after a later observation, the newer observation is kept (no flip-back).
    """
    valid = obs.dropna(subset=["value"]).sort_values(["available_date", "date"])
    valid = valid[valid["date"] >= valid["date"].cummax()]
    # pandas >= 3 may mix datetime resolutions (s/us/ns); merge_asof needs identical key dtypes
    left = pd.DataFrame({"trade_date": pd.DatetimeIndex(calendar).astype("datetime64[ns]")})
    right = valid.rename(columns={"date": "obs_date"})[["available_date", "obs_date", "value"]].copy()
    right["available_date"] = pd.to_datetime(right["available_date"]).astype("datetime64[ns]")
    right["obs_date"] = pd.to_datetime(right["obs_date"]).astype("datetime64[ns]")
    merged = pd.merge_asof(left, right, left_on="trade_date", right_on="available_date", direction="backward")
    return merged.set_index("trade_date")[["value", "obs_date"]].rename_axis("date")
