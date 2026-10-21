"""Shared synthetic fixtures for the Stage 8 unit tests (no market data, no Test split)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regime_radar import portfolio_constraints as pc  # noqa: E402
from regime_radar import portfolio_engine as eng  # noqa: E402
from regime_radar import portfolio_execution as pe  # noqa: E402

POLICY = pc.load_policy(ROOT / "configs" / "portfolio_policy.json", ROOT / "configs" / "project.json")
CFGJ = json.loads((ROOT / "configs" / "portfolio_policy.json").read_text(encoding="utf-8"))
CFG = eng.EngineConfig(budget_coef=CFGJ["risk_budget"]["coefficients"], min_obs=252, lookback=504,
                       solver=CFGJ["optimizer"]["solver"])
COSTS = CFGJ["transaction_costs"]
PROTOCOL = json.loads((ROOT / "configs" / "stage8_protocol.json").read_text(encoding="utf-8"))


def market(n=400, seed=3, start="2014-06-02", split_at="2015-06-01"):
    """Factor-model simple returns (PSD by construction), calendar, split labels and a weekly schedule."""
    rng = np.random.default_rng(seed)
    vols = np.array([.18, .20, .25, .06, .05, .09, .16, .20, .004]) / np.sqrt(252)
    common = rng.normal(size=(n, 1)) * 0.5
    x = (rng.normal(size=(n, 9)) + common * np.array([1, 1, 1, -.3, -.2, .6, .1, .5, 0])) * vols + 0.0002
    cal = pd.bdate_range(start, periods=n)
    simple = pd.DataFrame(x, index=cal, columns=POLICY.assets)
    split = pd.Series(np.where(cal < pd.Timestamp(split_at), "train", "validation"), index=cal)
    sched = pe.weekly_schedule(cal, split)
    d = pd.Series(np.clip(np.sin(np.arange(n) / 12.0), 0, 1), index=cal)
    return simple, split, sched, d


def no_stress(simple):
    return pd.Series(False, index=simple.index)
