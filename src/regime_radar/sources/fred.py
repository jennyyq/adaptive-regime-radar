"""FRED / ALFRED downloads using only the standard library.

* With FRED_API_KEY: the official API. Series whose availability rule is "vintage"
  are requested with output_type=4 (initial release only), so each observation
  carries its first-publication date in `realtime_start`.
* Without a key: the public fredgraph.csv endpoint (current vintage only). Vintage
  series then fall back to the conservative publication rule in the config.
"""

from __future__ import annotations

import io
import json
import os
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

API_URL = "https://api.stlouisfed.org/fred/series/observations"
GRAPH_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
USER_AGENT = "adaptive-regime-radar/0.2 (research)"


def _ssl_context() -> ssl.SSLContext:
    # python.org builds on macOS often lack root certificates; certifi ships with yfinance's deps.
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def http_get(url: str, params: dict, retries: int = 4, timeout: int = 30, pause: float = 2.0) -> bytes:
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(f"{url}?{query}", headers={"User-Agent": USER_AGENT})
    redacted = {k: ("***" if k == "api_key" else v) for k, v in params.items()}
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout, context=_ssl_context()) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code in (400, 401, 403, 404):  # not transient: bad id, bad key, forbidden
                break
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_error = exc
        if attempt < retries:
            time.sleep(pause * 2 ** (attempt - 1))
    raise RuntimeError(f"GET {url} {redacted} failed: {last_error}")


def parse_graph_csv(text: str) -> pd.DataFrame:
    """fredgraph.csv: first column is the date (header DATE or observation_date), second the value."""
    frame = pd.read_csv(io.StringIO(text), na_values=["."])
    if frame.shape[1] < 2:
        raise ValueError(f"Unexpected fredgraph response: {text[:200]!r}")
    out = pd.DataFrame({
        "date": pd.to_datetime(frame.iloc[:, 0]),
        "value": pd.to_numeric(frame.iloc[:, 1], errors="coerce"),
    })
    out["realtime_start"] = pd.NaT
    return out


def parse_api_json(payload: bytes | str, vintage: bool) -> pd.DataFrame:
    data = json.loads(payload)
    if "observations" not in data:
        raise ValueError(f"FRED API error: {data.get('error_message', data)}")
    frame = pd.DataFrame(data["observations"])
    if frame.empty:
        return pd.DataFrame(columns=["date", "value", "realtime_start"])
    out = pd.DataFrame({
        "date": pd.to_datetime(frame["date"]),
        "value": pd.to_numeric(frame["value"].replace(".", None), errors="coerce"),
    })
    out["realtime_start"] = pd.to_datetime(frame["realtime_start"]) if vintage else pd.NaT
    return out


def download_series(
    series_id: str,
    spec: dict,
    start: str,
    end: str | None = None,
    api_key: str | None = None,
) -> tuple[pd.DataFrame, str]:
    """Return (frame[date, value, realtime_start], source label)."""
    api_key = api_key or os.environ.get("FRED_API_KEY") or None
    wants_vintage = spec["availability"]["rule"] == "vintage"

    if api_key:
        params = {"series_id": series_id, "api_key": api_key, "file_type": "json", "observation_start": start}
        if end:
            params["observation_end"] = end
        if wants_vintage:
            params.update(realtime_start="1776-07-04", realtime_end="9999-12-31", output_type=4)
        frame = parse_api_json(http_get(API_URL, params), vintage=wants_vintage)
        source = "fred_api_first_release" if wants_vintage else "fred_api_current"
    else:
        params = {"id": series_id, "cosd": start}
        if end:
            params["coed"] = end
        frame = parse_graph_csv(http_get(GRAPH_URL, params).decode("utf-8"))
        source = "fredgraph_csv_current"

    frame = frame.sort_values(["date", "realtime_start"]).drop_duplicates("date", keep="first")
    frame = frame[frame["date"] >= pd.Timestamp(start)]
    if end:
        frame = frame[frame["date"] <= pd.Timestamp(end)]
    if frame["value"].notna().sum() == 0:
        raise RuntimeError(f"{series_id}: no numeric observations returned")
    return frame.reset_index(drop=True), source
