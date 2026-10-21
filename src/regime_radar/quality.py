"""Data-quality checks. They flag, never silently fix."""

from __future__ import annotations

import numpy as np
import pandas as pd

EXTREME_LOG_RETURN = 0.20  # |daily log return| above this is flagged for manual review
STALE_RUN_DAYS = 5         # identical adjusted closes for this many consecutive days


def _md_table(frame: pd.DataFrame) -> str:
    """Minimal Markdown table (avoids the optional `tabulate` dependency)."""
    header = [frame.index.name or ""] + [str(c) for c in frame.columns]
    rows = [[str(i)] + ["" if pd.isna(v) else str(v) for v in row] for i, row in zip(frame.index, frame.values)]
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def _longest_run(mask: pd.Series) -> int:
    if not mask.any():
        return 0
    groups = (~mask).cumsum()
    return int(mask.groupby(groups).sum().max())


def yahoo_quality(raw: dict[str, pd.DataFrame], calendar: pd.DatetimeIndex, required: set[str]) -> pd.DataFrame:
    rows = []
    for ticker, frame in raw.items():
        price = frame["adj_close"].dropna()
        live = calendar[(calendar >= price.index.min()) & (calendar <= price.index.max())] if len(price) else calendar[:0]
        on_cal = price.reindex(live)
        logret = np.log(on_cal.where(on_cal > 0)).diff()
        extreme = logret[logret.abs() > EXTREME_LOG_RETURN]
        unchanged = on_cal.diff().eq(0)
        is_index = ticker.startswith("^")
        bad_bar = (frame["high"] < frame["low"]) | (frame["close"] > frame["high"] * 1.0001) | (frame["close"] < frame["low"] * 0.9999)
        rows.append({
            "ticker": ticker,
            "required": ticker in required,
            "first_date": price.index.min().date().isoformat() if len(price) else "",
            "last_date": price.index.max().date().isoformat() if len(price) else "",
            "observations": len(price),
            "missing_in_live_range": int(on_cal.isna().sum()),
            "coverage_in_live_range_pct": round(100 * on_cal.notna().mean(), 2) if len(live) else 0.0,
            "off_calendar_rows": int((~price.index.isin(calendar)).sum()),
            "non_positive_prices": int((price <= 0).sum()),
            "extreme_returns": len(extreme),
            "extreme_return_dates": ";".join(d.date().isoformat() for d in extreme.index[:5]),
            "longest_unchanged_run": _longest_run(unchanged) + (1 if unchanged.any() else 0),
            "zero_volume_days": 0 if is_index else int((frame["volume"].fillna(0) == 0).sum()),
            "inconsistent_ohlc_bars": int(bad_bar.fillna(False).sum()),
        })
    return pd.DataFrame(rows).set_index("ticker")


def fred_quality(
    release_log: pd.DataFrame,
    obs_dates: pd.DataFrame,
    specs: dict[str, dict],
    calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    rows = []
    for series_id, spec in specs.items():
        log = release_log[release_log["series_id"] == series_id] if not release_log.empty else release_log
        if log.empty:
            rows.append({"series_id": series_id, "required": spec.get("required", False), "status": "missing"})
            continue
        valid = log.dropna(subset=["value"])
        lag = (valid["available_date"] - valid["date"]).dt.days
        age = (calendar.to_series() - obs_dates[series_id]).dt.days
        methods = valid["availability_method"].value_counts()
        rows.append({
            "series_id": series_id,
            "required": spec.get("required", False),
            "status": "ok",
            "factor": spec.get("factor", ""),
            "frequency": spec.get("frequency", ""),
            "first_obs": valid["date"].min().date().isoformat(),
            "last_obs": valid["date"].max().date().isoformat(),
            "observations": len(valid),
            "missing_values": int(log["value"].isna().sum()),
            "median_release_lag_days": float(lag.median()),
            "max_release_lag_days": int(lag.max()),
            "first_usable_trade_date": obs_dates[series_id].first_valid_index().date().isoformat()
            if obs_dates[series_id].notna().any() else "",
            "max_staleness_days": int(age.max()) if age.notna().any() else None,
            "availability_methods": "; ".join(f"{k}={v}" for k, v in methods.items()),
        })
    return pd.DataFrame(rows).set_index("series_id")


def markdown_report(
    yahoo_q: pd.DataFrame,
    fred_q: pd.DataFrame,
    prices: pd.DataFrame,
    log_returns: pd.DataFrame,
    manifest: dict,
) -> str:
    lines = ["# Data quality report", ""]
    lines.append(f"Generated from raw files last updated {manifest.get('updated_utc', 'unknown')} UTC.")
    lines += ["", "## Investment panel (9 ETFs)", ""]
    lines.append(f"- Common sample: **{prices.index.min().date()} → {prices.index.max().date()}** "
                 f"({len(prices)} price rows, {len(log_returns)} return rows)")
    binding = yahoo_q.loc[list(prices.columns), "first_date"].sort_values().iloc[-1:]
    lines.append(f"- Binding start ETF: {binding.index[0]} (first trade {binding.iloc[0]})")
    lines.append(f"- Missing daily returns inside the common sample (left as NaN): "
                 f"{int(log_returns.isna().sum().sum())}")

    issues = []
    for ticker, row in yahoo_q.iterrows():
        if row["missing_in_live_range"]:
            issues.append(f"{ticker}: {row['missing_in_live_range']} missing trading days after first listing")
        if row["extreme_returns"]:
            issues.append(f"{ticker}: {row['extreme_returns']} |log return| > {EXTREME_LOG_RETURN:.0%} "
                          f"({row['extreme_return_dates']}) — check against a second source")
        if row["longest_unchanged_run"] >= STALE_RUN_DAYS:
            issues.append(f"{ticker}: adjusted close unchanged for {row['longest_unchanged_run']} days")
        if row["inconsistent_ohlc_bars"]:
            issues.append(f"{ticker}: {row['inconsistent_ohlc_bars']} bars with close outside [low, high]")
    for series_id, row in fred_q.iterrows():
        if row.get("status") == "missing":
            issues.append(f"FRED {series_id}: not downloaded")
        elif row.get("frequency") == "daily" and (row.get("max_staleness_days") or 0) > 10:
            issues.append(f"FRED {series_id}: value up to {row['max_staleness_days']} days old on some trade dates")
        if "fallback" in str(row.get("availability_methods", "")):
            issues.append(f"FRED {series_id}: some release dates are rule-based approximations "
                          f"({row['availability_methods']})")

    lines += ["", "## Flags for review", ""]
    lines += [f"- {issue}" for issue in issues] or ["- None"]
    lines += ["", "## Yahoo coverage", "", _md_table(yahoo_q.drop(columns=["extreme_return_dates"]))]
    if not fred_q.empty:
        cols = [c for c in ["status", "frequency", "first_obs", "last_obs", "observations",
                            "median_release_lag_days", "first_usable_trade_date", "max_staleness_days"]
                if c in fred_q.columns]
        lines += ["", "## FRED coverage (point-in-time)", "", _md_table(fred_q[cols])]
    return "\n".join(lines) + "\n"
