"""Load and validate configs/project.json."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .paths import CONFIG_FILE

AVAILABILITY_RULES = {"business_days", "calendar_days", "next_month_day", "vintage"}


def load_config(path: str | Path = CONFIG_FILE) -> dict:
    with open(path, encoding="utf-8") as handle:
        config = json.load(handle)

    etfs = config["investment_etfs"]
    weights = config["strategic_weights"]
    if len(etfs) != len(set(etfs)) or set(etfs) != set(weights):
        raise ValueError("investment_etfs and strategic_weights keys must match without duplicates")
    if not np.isclose(sum(weights.values()), 1.0):
        raise ValueError(f"Strategic weights sum to {sum(weights.values()):.4f}, not 1")
    if any(w < 0 for w in weights.values()):
        raise ValueError("Strategic weights must be non-negative (no leverage / shorting)")

    if config.get("calendar_ticker", "SPY") not in yahoo_tickers(config):
        raise ValueError("calendar_ticker must be one of the downloaded Yahoo tickers")
    unknown = set(config.get("required_auxiliary", [])) - set(yahoo_tickers(config))
    if unknown:
        raise ValueError(f"required_auxiliary not in auxiliary_tickers: {sorted(unknown)}")

    for series_id, spec in config.get("fred_series", {}).items():
        rule = spec["availability"]["rule"]
        if rule not in AVAILABILITY_RULES:
            raise ValueError(f"{series_id}: unknown availability rule {rule!r}")
        if rule == "vintage":
            fallback = spec["availability"].get("fallback", {}).get("rule")
            if fallback not in AVAILABILITY_RULES - {"vintage"}:
                raise ValueError(f"{series_id}: vintage rule needs a non-vintage fallback rule")
    return config


def yahoo_tickers(config: dict) -> list[str]:
    """Investment ETFs first, then auxiliary tickers, de-duplicated in order."""
    tickers = list(config["investment_etfs"])
    for group in config.get("auxiliary_tickers", {}).values():
        tickers.extend(group)
    return list(dict.fromkeys(tickers))


def required_yahoo(config: dict) -> set[str]:
    return set(config["investment_etfs"]) | set(config.get("required_auxiliary", []))


def required_fred(config: dict) -> set[str]:
    return {sid for sid, spec in config.get("fred_series", {}).items() if spec.get("required")}
