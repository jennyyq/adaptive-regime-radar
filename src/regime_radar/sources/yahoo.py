"""Yahoo Finance daily OHLCV via yfinance, one ticker at a time with retries.

We request auto_adjust=False so both the raw Close (for dollar volume / illiquidity)
and the split-and-dividend Adj Close (for total-return log returns) are kept.
"""

from __future__ import annotations

import time

import pandas as pd

FIELD_MAP = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}
FIELDS = list(FIELD_MAP.values())
NEW_YORK = "America/New_York"


def normalize_yahoo(raw: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Flatten yfinance output into a date-indexed frame with FIELDS columns."""
    frame = raw.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        for level in range(frame.columns.nlevels):
            values = frame.columns.get_level_values(level)
            if set(FIELD_MAP) & set(values):
                frame.columns = values
                break
    frame = frame.loc[:, ~pd.Index(frame.columns).duplicated()]

    if "Close" not in frame.columns:
        raise ValueError(f"Yahoo response has no Close column: {list(frame.columns)}")
    if "Adj Close" not in frame.columns:
        # Index levels (e.g. ^VIX) have no distributions; Close is the total-return level.
        frame["Adj Close"] = frame["Close"]
    for name in FIELD_MAP:
        if name not in frame.columns:
            frame[name] = float("nan")

    frame = frame[list(FIELD_MAP)].rename(columns=FIELD_MAP)
    index = pd.to_datetime(frame.index)
    if index.tz is not None:
        index = index.tz_convert(NEW_YORK).tz_localize(None)
    frame.index = index.normalize()
    frame.index.name = "date"
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    frame = frame.dropna(how="all", subset=["open", "high", "low", "close", "adj_close"])
    return drop_incomplete_session(frame, now)


def drop_incomplete_session(frame: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Drop today's bar if the US session has not closed yet (Singapore runs often overlap it)."""
    if frame.empty:
        return frame
    now_ny = (now or pd.Timestamp.now(tz=NEW_YORK)).tz_convert(NEW_YORK)
    today_ny = now_ny.tz_localize(None).normalize()
    session_done = (now_ny.hour, now_ny.minute) >= (16, 30)
    if frame.index[-1] >= today_ny and not session_done:
        return frame[frame.index < today_ny]
    return frame


def _yf_call(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    import yfinance as yf

    kwargs = dict(
        tickers=ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=False,
        timeout=30,
    )
    try:
        return yf.download(multi_level_index=False, **kwargs)
    except TypeError:  # older yfinance without multi_level_index
        return yf.download(**kwargs)


def download_ticker(
    ticker: str,
    start: str,
    end: str | None = None,
    retries: int = 4,
    pause: float = 3.0,
) -> pd.DataFrame:
    """Download one ticker. `end` is inclusive here (yfinance itself treats it as exclusive)."""
    end_exclusive = None if end is None else (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            raw = _yf_call(ticker, start, end_exclusive)
            if raw is not None and not raw.empty:
                frame = normalize_yahoo(raw)
                if not frame.empty:
                    return frame
            last_error = RuntimeError("empty response (bad symbol, rate limit, or no network)")
        except Exception as exc:  # yfinance raises a variety of network / rate-limit errors
            last_error = exc
        if attempt < retries:
            time.sleep(pause * 2 ** (attempt - 1))
    raise RuntimeError(f"{ticker}: failed after {retries} attempts: {last_error}")
