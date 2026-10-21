"""Stage 8: self-financing daily portfolio ledger and its reconciliation checks.

Per session s (after the start session):
  1. holdings held at the close of s-1 (weights w_{s-1}) earn the close-to-close simple returns R_s:
         gross_s = sum_i w_{s-1,i} R_{s,i};  wealth_pre = wealth_{s-1} (1 + gross_s);  w_pre = w_{s-1} (1 + R_s) / (1 + gross_s)
  2. if an order executes at the close of s: executed weights from backtest_execution.execute(order, w_pre);
         cost_s = c * gross_notional_s;  wealth_post = wealth_pre (1 - cost_s);  w_s = executed weights
     otherwise w_s = w_pre (holdings drift, nothing is traded)
  3. net_s = wealth_post / wealth_{s-1} - 1 = (1 + gross_s)(1 - cost_s) - 1
The trading path (weights) does not depend on the cost rate (costs are deducted pro rata), so one pass gives the
gross path and the net paths for every cost scenario. All portfolios are fully invested; there is no cash
account, no borrowing and no leverage (BIL is the liquidity sleeve).
"""

from __future__ import annotations

STAGE8_CODE_VERSION = "stage8-2026-10-21"  # must match scripts/10_backtest_portfolios.py

import numpy as np
import pandas as pd

from .backtest_execution import Execution, Order, cost_fraction, execute


class Ledger:
    def __init__(self, portfolio: str, assets: list[str], start_date: pd.Timestamp, w0: np.ndarray,
                 bps: tuple = (5, 10, 25), primary_bps: float = 10, stress_multiplier: float = 2.0):
        w0 = np.asarray(w0, float)
        if abs(w0.sum() - 1) > 1e-12 or (w0 < 0).any():
            raise ValueError("initial weights must be long-only and sum to one")
        self.portfolio, self.assets = portfolio, list(assets)
        self.bps, self.primary = tuple(bps), primary_bps
        if primary_bps not in self.bps:
            raise ValueError("primary cost scenario must be one of the scenarios")
        self.mult = stress_multiplier
        self.w = w0.copy()
        self.wealth = {"gross": 1.0, **{b: 1.0 for b in self.bps}}
        self.dates, self.records, self.w_post, self.w_pre = [], [], [], []
        self.executions: list[Execution] = []
        self._log(start_date, None, w0.copy(), None, start=True)

    def _log(self, date, gross, w_pre, exe: Execution | None, start=False, wealth_begin=None, wealth_pre=None,
             costs=None):
        rec = {"date": date, "portfolio": self.portfolio, "start_session": start,
               "gross_return": np.nan if gross is None else gross,
               "wealth_begin_net_primary": np.nan if wealth_begin is None else wealth_begin,
               "wealth_pretrade_net_primary": np.nan if wealth_pre is None else wealth_pre,
               "traded": bool(exe is not None and exe.gross_notional > 0),
               "execution_kind": "" if exe is None else exe.kind,
               "executed_signal_date": pd.NaT if exe is None else exe.signal_date,
               "turnover": 0.0 if exe is None else exe.turnover,
               "gross_notional": 0.0 if exe is None else exe.gross_notional,
               "stress_on": bool(exe.stress_on) if exe is not None else False,
               "wealth_gross": self.wealth["gross"]}
        for b in self.bps:
            c = 0.0 if costs is None else costs[b]
            rec[f"cost_fraction_{b:g}bps"] = c
            rec[f"wealth_net_{b:g}bps"] = self.wealth[b]
            rec[f"net_return_{b:g}bps"] = np.nan if gross is None else (1 + gross) * (1 - c) - 1
        rec["cost_amount_primary"] = 0.0 if costs is None else costs[self.primary] * wealth_pre
        rec["wealth_posttrade_net_primary"] = self.wealth[self.primary]
        self.dates.append(date)
        self.records.append(rec)
        self.w_pre.append(np.asarray(w_pre, float).copy())
        self.w_post.append(self.w.copy())

    def session(self, date: pd.Timestamp, simple_row: np.ndarray, order: Order | None) -> Execution | None:
        """One session: earn the return with yesterday's holdings, then execute an order due at this close."""
        r = np.asarray(simple_row, float)
        gross = float(self.w @ r)
        if not np.isfinite(gross) or 1 + gross <= 0:
            raise FloatingPointError(f"{self.portfolio} {date}: invalid gross return {gross}")
        w_pre = self.w * (1 + r) / (1 + gross)
        wealth_begin = self.wealth[self.primary]
        for k in self.wealth:
            self.wealth[k] *= (1 + gross)
        wealth_pre = self.wealth[self.primary]
        exe, costs = None, None
        if order is not None:
            if order.execution_date != date:
                raise AssertionError(f"order for {order.execution_date} executed on {date}")
            exe = execute(order, w_pre)
            costs = {b: cost_fraction(exe.gross_notional, b, exe.stress_on, self.mult) for b in self.bps}
            for b in self.bps:
                self.wealth[b] *= (1 - costs[b])
            self.w = exe.executed.copy()
            self.executions.append(exe)
        else:
            self.w = w_pre
        self._log(date, gross, w_pre, exe, wealth_begin=wealth_begin, wealth_pre=wealth_pre, costs=costs)
        return exe

    # ------------------------------------------------------------------ outputs
    def frame(self) -> pd.DataFrame:
        f = pd.DataFrame(self.records)
        f["date"] = pd.to_datetime(f["date"])
        return f

    def weights(self, which: str = "post") -> pd.DataFrame:
        arr = np.vstack(self.w_post if which == "post" else self.w_pre)
        return pd.DataFrame(arr, index=pd.DatetimeIndex(self.dates, name="date"), columns=self.assets)


# --------------------------------------------------------------------------------------
# Reconciliation (independent recomputation from the stored ledger)
# --------------------------------------------------------------------------------------
def reconcile(ledger: Ledger, simple: pd.DataFrame, execution_dates: set, tol: float = 1e-10) -> list[dict]:
    """Accounting identities for one ledger. Returns rows {check, status, detail, max_error}."""
    f = ledger.frame()
    wpost = ledger.weights("post")
    wpre = ledger.weights("pre")
    dates = pd.DatetimeIndex(f["date"])
    rows = []

    def add(name, ok, detail, err=np.nan):
        rows.append({"portfolio": ledger.portfolio, "check": name, "status": "PASS" if ok else "FAIL", "detail": detail,
                     "max_error": err})

    cal = simple.index
    i0 = cal.get_loc(dates[0])
    expected = cal[i0:i0 + len(dates)]
    add("Every session after the start is in the ledger (no omitted holding returns, no phantom dates)",
        dates.equals(expected) and len(dates) == len(cal) - i0, f"{len(dates)} sessions from {dates[0].date()}")
    wealth_cols = ["wealth_gross"] + [f"wealth_net_{b:g}bps" for b in ledger.bps]
    pos = bool(np.isfinite(f[wealth_cols].to_numpy()).all() and (f[wealth_cols].to_numpy() > 0).all())
    add("Wealth finite and positive", pos, "gross and every net cost scenario")
    s = np.abs(wpost.sum(axis=1) - 1).max()
    neg = float(min(wpost.min().min(), wpre.min().min()))
    add("Weights sum to one, no short positions, no leverage", s < 1e-10 and neg >= -1e-12,
        f"max |sum - 1| {s:.1e}; min weight {neg:.2e}", float(s))
    R = simple.loc[dates[1:]].to_numpy(float)
    held = wpost.to_numpy()[:-1]
    g = np.einsum("ij,ij->i", held, R)
    e1 = float(np.max(np.abs(g - f["gross_return"].to_numpy()[1:]))) if len(g) else 0.0
    add("Gross return = sum of previous-close holdings x asset simple returns", e1 < 1e-13, f"max error {e1:.1e}", e1)
    drift = held * (1 + R) / (1 + g)[:, None]
    e2 = float(np.max(np.abs(drift - wpre.to_numpy()[1:]))) if len(g) else 0.0
    add("Holdings continuity: pre-trade weights = previous holdings drifted by the session return", e2 < 1e-12,
        f"max error {e2:.1e}", e2)
    traded = f["gross_notional"].to_numpy() > 0
    notional = np.abs(wpost.to_numpy() - wpre.to_numpy()).sum(axis=1)
    e3 = float(np.max(np.abs(notional - f["gross_notional"].to_numpy())))
    e3b = float(np.max(np.abs(f["gross_notional"] - 2 * f["turnover"])))
    no_trade_change = float(np.max(np.abs(wpost.to_numpy()[~traded] - wpre.to_numpy()[~traded]))) if (~traded).any() else 0.0
    add("Trades recorded: gross notional = sum |post - pre| = 2 x one-way turnover; no unrecorded rebalancing",
        e3 < 1e-12 and e3b < 1e-15 and no_trade_change < 1e-15,
        f"notional error {e3:.1e}; unrecorded weight change {no_trade_change:.1e}", max(e3, no_trade_change))
    tdates = set(dates[traded])
    phantom = sorted(d for d in tdates if d not in execution_dates)
    add("Trades only on scheduled execution sessions (no phantom executions)", not phantom,
        f"{len(tdates)} trading sessions; unscheduled: {', '.join(str(d.date()) for d in phantom[:3]) or 'none'}")
    worst_cost = 0.0
    for b in ledger.bps:
        mult = np.where(f["stress_on"], ledger.mult, 1.0)
        c = b / 1e4 * mult * f["gross_notional"].to_numpy()
        worst_cost = max(worst_cost, float(np.max(np.abs(c - f[f"cost_fraction_{b:g}bps"].to_numpy()))))
    add("Transaction cost = bps / 1e4 x stress multiplier x gross notional (10 bps = 0.001)", worst_cost < 1e-15,
        f"max error {worst_cost:.1e}", worst_cost)
    worst_w = 0.0
    gross_full = np.r_[0.0, f["gross_return"].to_numpy()[1:]]
    wg = np.cumprod(1 + gross_full)
    worst_w = max(worst_w, float(np.max(np.abs(wg - f["wealth_gross"].to_numpy()) / wg)))
    for b in ledger.bps:
        c = f[f"cost_fraction_{b:g}bps"].to_numpy()
        wn = np.cumprod((1 + gross_full) * (1 - c))
        worst_w = max(worst_w, float(np.max(np.abs(wn - f[f"wealth_net_{b:g}bps"].to_numpy()) / wn)))
        nr = f[f"net_return_{b:g}bps"].to_numpy()[1:]
        worst_w = max(worst_w, float(np.max(np.abs(nr - ((1 + gross_full[1:]) * (1 - c[1:]) - 1)))))
    add("Net wealth = gross wealth path x (1 - cost) at each execution, compounded daily (costs counted once)",
        worst_w < tol, f"max relative error {worst_w:.1e}", worst_w)
    body = f.iloc[1:]
    e4 = float(np.max(np.abs(body["wealth_posttrade_net_primary"] - (body["wealth_pretrade_net_primary"] -
                                                                      body["cost_amount_primary"]))
                      / body["wealth_pretrade_net_primary"])) if len(body) else 0.0
    e5 = float(np.max(np.abs(body["wealth_pretrade_net_primary"] - body["wealth_begin_net_primary"] *
                             (1 + body["gross_return"])) / body["wealth_pretrade_net_primary"])) if len(body) else 0.0
    add("Self-financing: pre-trade wealth = previous wealth x (1 + gross); post-trade wealth = pre-trade wealth - "
        "cost amount", e4 < 1e-13 and e5 < 1e-13, f"max relative errors {e4:.1e}, {e5:.1e}", max(e4, e5))
    return rows
