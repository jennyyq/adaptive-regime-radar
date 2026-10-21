"""Orchestrate raw downloads: one CSV per series plus a manifest for reproducibility."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from importlib import metadata

import pandas as pd

from .paths import MANIFEST, RAW_FRED, RAW_YAHOO, ensure_dirs, safe_name
from .sources import fred, yahoo


def yahoo_path(ticker: str):
    return RAW_YAHOO / f"{safe_name(ticker)}.csv"


def fred_path(series_id: str):
    return RAW_FRED / f"{series_id}.csv"


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed"


def _summary(frame: pd.DataFrame, value_col: str) -> dict:
    valid = frame.dropna(subset=[value_col])
    dates = valid.index if value_col != "value" else valid["date"]
    return {
        "rows": int(len(valid)),
        "first_date": dates.min().date().isoformat() if len(valid) else None,
        "last_date": dates.max().date().isoformat() if len(valid) else None,
    }


def read_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {"yahoo": {}, "fred": {}}


def write_manifest(manifest: dict) -> None:
    manifest["updated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest["versions"] = {p: _package_version(p) for p in ("pandas", "numpy", "yfinance")}
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def download_yahoo(
    tickers: list[str],
    start: str,
    end: str | None,
    skip_existing: bool = False,
    pause: float = 0.6,
) -> dict[str, dict]:
    ensure_dirs()
    results: dict[str, dict] = {}
    for i, ticker in enumerate(tickers, 1):
        path = yahoo_path(ticker)
        label = f"[{i:>2}/{len(tickers)}] Yahoo {ticker:<7}"
        if skip_existing and path.exists():
            print(f"{label} skipped (file exists)")
            results[ticker] = {"status": "skipped", "file": str(path.name)}
            continue
        try:
            frame = yahoo.download_ticker(ticker, start, end)
            frame.to_csv(path)
            info = {"status": "ok", "file": path.name, **_summary(frame, "adj_close")}
            print(f"{label} {info['rows']:>5} rows  {info['first_date']} -> {info['last_date']}")
        except Exception as exc:
            info = {"status": "failed", "error": str(exc)[:300]}
            print(f"{label} FAILED: {info['error']}")
        info["downloaded_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        results[ticker] = info
        time.sleep(pause)  # be polite; Yahoo rate-limits bursts
    return results


def download_fred(
    series: dict[str, dict],
    start: str,
    end: str | None,
    skip_existing: bool = False,
) -> dict[str, dict]:
    ensure_dirs()
    has_key = bool(os.environ.get("FRED_API_KEY"))
    if not has_key:
        print("FRED_API_KEY not set: using fredgraph.csv (current vintage). "
              "Vintage series will use the conservative fallback release rule.")
    results: dict[str, dict] = {}
    for i, (series_id, spec) in enumerate(series.items(), 1):
        path = fred_path(series_id)
        label = f"[{i:>2}/{len(series)}] FRED  {series_id:<13}"
        if skip_existing and path.exists():
            print(f"{label} skipped (file exists)")
            results[series_id] = {"status": "skipped", "file": path.name}
            continue
        try:
            frame, source = fred.download_series(series_id, spec, start, end)
            frame.to_csv(path, index=False, date_format="%Y-%m-%d")
            info = {"status": "ok", "file": path.name, "source": source, **_summary(frame, "value")}
            print(f"{label} {info['rows']:>5} obs   {info['first_date']} -> {info['last_date']}  ({source})")
        except Exception as exc:
            info = {"status": "failed", "error": str(exc)[:300]}
            print(f"{label} FAILED: {info['error']}")
        info["downloaded_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        results[series_id] = info
        time.sleep(0.6 if has_key else 1.0)
    return results


def load_yahoo_raw(tickers: list[str]) -> dict[str, pd.DataFrame]:
    frames = {}
    for ticker in tickers:
        path = yahoo_path(ticker)
        if path.exists():
            frames[ticker] = pd.read_csv(path, index_col="date", parse_dates=["date"])
    return frames


def load_fred_raw(series_ids: list[str]) -> dict[str, pd.DataFrame]:
    frames = {}
    for series_id in series_ids:
        path = fred_path(series_id)
        if path.exists():
            frames[series_id] = pd.read_csv(path, parse_dates=["date", "realtime_start"])
    return frames
