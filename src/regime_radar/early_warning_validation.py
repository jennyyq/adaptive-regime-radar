"""Stage 6: independent adverse outcomes, event-level and daily evaluation, baselines, ablations, selection.

Outcomes (evaluation only; never inputs to any signal). r_t = SPY daily log return from the split files.
  fwd_ret_h(t)  = r_t+1 + ... + r_t+h                     (terminal forward return, h = 5, 10, 21)
  fwd_mdd_h(t)  = min_{1<=j<=h} (r_t+1 + ... + r_t+j), capped at 0   (worst cumulative loss inside the horizon)
  Daily adverse label: y_t = 1[fwd_mdd_21(t) <= log(0.95)]  (a >= 5% SPY loss at some point in the next 21 sessions)
  Secondary: fixed strategic 9-ETF reference portfolio (weights from configs/project.json, daily rebalanced),
             y_t = 1[fwd_mdd_21 <= log(0.97)].
  A forward window must lie inside the same split; the last 21 sessions of each split are unlabeled (purged).
  For the 2016-17 / 2018-19 block split, labels of 2016-17 dates whose window reaches 2018 are also purged.

Drawdown events (event-level evaluation; defined from SPY prices alone):
  Running peak of cumulative log price within the split; an event ONSET is the first session at which the
  drawdown from the peak reaches -5%; the event ends when the drawdown recovers above -2.5% (re-arm). The
  onset is observable at its close; it is not an HMM switch nor a factor threshold.
  Note: the equity_trend factor contains SPY momentum, so signals and events are not fully independent.
Event metrics for a binary alert series a_t:
  warned       : alert on at least one session in [onset - 63, onset - 1]; lead = onset - start of the MOST RECENT
                 alert episode before onset (clipped to the window), so an old alert that has already ended does not
                 inflate the lead; gap = onset - last alert session (1 = alert still on the day before onset)
  detected late: no pre-onset alert but an alert in [onset, onset + 21]; delay = first alert - onset
  missed       : neither
  alert episode: maximal run of alert sessions; FALSE if no event onset in [start, start + 63] and it does not
                 overlap any event period [peak, end].
"""

from __future__ import annotations

STAGE6_CODE_VERSION = "stage6-2026-10-21"  # must match scripts/08_build_early_warning.py

import math

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

HORIZONS = (5, 10, 21)
LABEL_HORIZON = 21
SPY_LOSS = math.log(0.95)
PORTFOLIO_LOSS = math.log(0.97)
EVENT_DRAWDOWN = math.log(0.95)
EVENT_REARM = math.log(0.975)
LEAD_WINDOW = 63
LATE_WINDOW = 21
TRAIN_ALERT_BUDGET = 0.20          # every continuous baseline alerts on its top 20% of Train days
BOOTSTRAP_BLOCK = 63
BOOTSTRAP_REPS = 1000
VALIDATION_BLOCKS = {"2016-2017": ("2016-01-01", "2017-12-31"), "2018-2019": ("2018-01-01", "2019-12-31")}
COMPLEXITY = {"B_p_stress": 1, "C_p_non_normal": 1, "D_volatility_level": 1, "E_ar_innovation": 1, "F_bocpd": 1,
              "G_hmm_bocpd": 2, "H_hmm_innovation_bocpd": 3, "I_full_ews_score": 6}


# --------------------------------------------------------------------------------------
# Outcomes and events
# --------------------------------------------------------------------------------------
def portfolio_log_returns(returns: pd.DataFrame, weights: dict) -> pd.Series:
    w = pd.Series(weights, dtype=float)
    w = w / w.sum()
    simple = np.expm1(returns[w.index])
    return np.log1p(simple @ w)


def forward_outcomes(r: pd.Series, horizons=HORIZONS) -> pd.DataFrame:
    """Forward terminal return and worst cumulative loss within each horizon; NaN where the window leaves the
    series (the series must be a single split)."""
    v = r.to_numpy(float)
    n = len(v)
    out = {}
    cum = np.r_[0.0, np.cumsum(v)]
    for h in horizons:
        ret = np.full(n, np.nan)
        mdd = np.full(n, np.nan)
        for t in range(n - h):
            path = cum[t + 2:t + h + 2] - cum[t + 1]
            ret[t] = path[-1]
            mdd[t] = min(path.min(), 0.0)
        out[f"fwd_ret_{h}"] = ret
        out[f"fwd_mdd_{h}"] = mdd
    return pd.DataFrame(out, index=r.index)


def drawdown_events(r: pd.Series, onset_level: float = EVENT_DRAWDOWN, rearm_level: float = EVENT_REARM
                    ) -> pd.DataFrame:
    level = r.cumsum()
    peak_val, peak_idx, armed, current = -np.inf, None, True, None
    rows = []
    for i, (t, v) in enumerate(level.items()):
        if v > peak_val:
            peak_val, peak_idx = v, i
        dd = v - peak_val
        if armed and dd <= onset_level:
            current = {"peak_date": level.index[peak_idx], "onset_date": t, "onset_pos": i, "peak_pos": peak_idx,
                       "trough_dd": dd}
            armed = False
        elif current is not None:
            current["trough_dd"] = min(current["trough_dd"], dd)
            if dd > rearm_level:
                current["end_date"], current["end_pos"] = t, i
                rows.append(current)
                current, armed = None, True
    if current is not None:
        current["end_date"], current["end_pos"] = level.index[-1], len(level) - 1
        current["censored"] = True
        rows.append(current)
    ev = pd.DataFrame(rows)
    if not ev.empty:
        ev["censored"] = ev.get("censored", False)
        ev["censored"] = ev["censored"].fillna(False).astype(bool)
        ev["trough_loss_pct"] = 100 * np.expm1(ev["trough_dd"])
    return ev


def daily_labels(out: pd.DataFrame, column: str = f"fwd_mdd_{LABEL_HORIZON}", loss: float = SPY_LOSS) -> pd.Series:
    return (out[column] <= loss).astype(float).where(out[column].notna())


def purge_block_boundary(labels: pd.Series, block_end: str, horizon: int = LABEL_HORIZON) -> pd.Series:
    """Remove labels whose forward window crosses the end of an evaluation block."""
    lab = labels.copy()
    pos = np.flatnonzero(lab.index <= pd.Timestamp(block_end))
    if len(pos):
        lab.iloc[pos[-horizon:]] = np.nan
    return lab


# --------------------------------------------------------------------------------------
# Event-level metrics
# --------------------------------------------------------------------------------------
def alert_episodes(a: np.ndarray) -> list[tuple[int, int]]:
    a = np.asarray(a, int)
    eps, start = [], None
    for i, v in enumerate(a):
        if v and start is None:
            start = i
        if not v and start is not None:
            eps.append((start, i))
            start = None
    if start is not None:
        eps.append((start, len(a)))
    return eps


def event_metrics(a: pd.Series, events: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    a_ = a.fillna(0).to_numpy(int)
    n = len(a_)
    years = n / 252
    rows = []
    for e in events.itertuples():
        o = e.onset_pos
        pre = np.flatnonzero(a_[max(o - LEAD_WINDOW, 0):o]) + max(o - LEAD_WINDOW, 0)
        post = np.flatnonzero(a_[o:min(o + LATE_WINDOW + 1, n)]) + o
        gap = np.nan
        if len(pre):
            s0, lo = pre[-1], max(o - LEAD_WINDOW, 0)
            while s0 - 1 >= lo and a_[s0 - 1]:
                s0 -= 1
            status, lead, delay, gap = "warned", o - s0, np.nan, o - pre[-1]
        elif len(post):
            status, lead, delay = "detected_late", np.nan, post[0] - o
        else:
            status, lead, delay = "missed", np.nan, np.nan
        rows.append({"onset_date": e.onset_date, "peak_date": e.peak_date, "trough_loss_pct": e.trough_loss_pct,
                     "status": status, "lead_days": lead, "gap_last_alert_to_onset": gap, "delay_days": delay,
                     "alert_on_onset_day": bool(a_[o])})
    ev = pd.DataFrame(rows)
    eps = alert_episodes(a_)
    false_eps, true_eps = [], []
    for s, t in eps:
        hits_onset = any(s <= e.onset_pos <= s + LEAD_WINDOW for e in events.itertuples())
        overlaps = any(not (t <= e.peak_pos or s > e.end_pos) for e in events.itertuples())
        (true_eps if hits_onset or overlaps else false_eps).append((s, t))
    m = {"n_events": len(events), "warned": int((ev["status"] == "warned").sum()) if len(ev) else 0,
         "detected_late": int((ev["status"] == "detected_late").sum()) if len(ev) else 0,
         "missed": int((ev["status"] == "missed").sum()) if len(ev) else 0,
         "median_lead_days": float(ev["lead_days"].median()) if len(ev) and ev["lead_days"].notna().any() else np.nan,
         "median_gap_to_onset": float(ev["gap_last_alert_to_onset"].median())
         if len(ev) and ev["gap_last_alert_to_onset"].notna().any() else np.nan,
         "median_delay_days": float(ev["delay_days"].median()) if len(ev) and ev["delay_days"].notna().any() else np.nan,
         "alert_episodes": len(eps), "false_alert_episodes": len(false_eps),
         "false_alerts_per_year": len(false_eps) / years if years else np.nan,
         "mean_false_episode_days": float(np.mean([t - s for s, t in false_eps])) if false_eps else 0.0,
         "episode_precision": len(true_eps) / len(eps) if eps else np.nan,
         "time_share_alerted": float(a_.mean()),
         "alert_transitions_per_year": float(np.abs(np.diff(a_)).sum() / years) if years else np.nan,
         "mean_alert_episode_days": float(np.mean([t - s for s, t in eps])) if eps else 0.0}
    m["event_recall_warned"] = m["warned"] / m["n_events"] if m["n_events"] else np.nan
    m["event_recall_any"] = (m["warned"] + m["detected_late"]) / m["n_events"] if m["n_events"] else np.nan
    return m, ev


def daily_metrics(score: pd.Series, a: pd.Series, labels: pd.Series) -> dict:
    ok = labels.notna() & score.notna()
    y, s, b = labels[ok].to_numpy(int), score[ok].to_numpy(float), a[ok].fillna(0).to_numpy(int)
    out = {"n_days": int(ok.sum()), "base_rate": float(y.mean()) if len(y) else np.nan}
    out["pr_auc"] = float(average_precision_score(y, s)) if 0 < y.sum() < len(y) else np.nan
    tp = int(((b == 1) & (y == 1)).sum())
    out["daily_precision"] = tp / b.sum() if b.sum() else np.nan
    out["daily_recall"] = tp / y.sum() if y.sum() else np.nan
    out["daily_lift"] = out["daily_precision"] / out["base_rate"] if out["base_rate"] and b.sum() else np.nan
    return out


def block_bootstrap_ap(scores: pd.DataFrame, labels: pd.Series, reference: str, reps: int = BOOTSTRAP_REPS,
                       block: int = BOOTSTRAP_BLOCK, seed: int = 20240531) -> pd.DataFrame:
    """Moving-block bootstrap (blocks of 63 sessions) of the PR-AUC difference of every score vs `reference`."""
    ok = labels.notna() & scores.notna().all(axis=1)
    y = labels[ok].to_numpy(int)
    s = scores[ok]
    n = len(y)
    rng = np.random.default_rng(seed)
    diffs = {c: [] for c in s.columns if c != reference}
    nb = int(np.ceil(n / block))
    for _ in range(reps):
        starts = rng.integers(0, max(n - block, 1), size=nb)
        idx = np.concatenate([np.arange(st, st + block) for st in starts])[:n]
        yy = y[idx]
        if yy.sum() in (0, len(yy)):
            continue
        ref = average_precision_score(yy, s[reference].to_numpy()[idx])
        for c in diffs:
            diffs[c].append(average_precision_score(yy, s[c].to_numpy()[idx]) - ref)
    rows = []
    for c, d in diffs.items():
        d = np.asarray(d)
        point = average_precision_score(y, s[c].to_numpy()) - average_precision_score(y, s[reference].to_numpy())
        rows.append({"model": c, "reference": reference, "point_ap_diff": float(point), "mean_ap_diff": float(d.mean()),
                     "ci95_low": float(np.quantile(d, 0.025)), "ci95_high": float(np.quantile(d, 0.975)),
                     "share_positive": float((d > 0).mean()), "valid_reps": len(d)})
    return pd.DataFrame(rows)


def budget_alert(score: pd.Series, train_mask, budget: float = TRAIN_ALERT_BUDGET) -> tuple[pd.Series, float]:
    """Alert when the score is in its top `budget` share of Train days (threshold from Train only)."""
    thr = float(np.nanquantile(score[train_mask].to_numpy(float), 1 - budget))
    return (score >= thr).astype(int).where(score.notna(), 0), thr


def evaluate_system(name: str, score: pd.Series, alerts: pd.Series, split_data: dict) -> list[dict]:
    """Metrics for each split / block. split_data[split] = (mask, labels, events, portfolio_labels)."""
    rows = []
    for split, (mask, labels, events, plabels) in split_data.items():
        m, _ = event_metrics(alerts[mask], events)
        d = daily_metrics(score[mask], alerts[mask], labels)
        dp = daily_metrics(score[mask], alerts[mask], plabels)
        rows.append({"system": name, "split": split, **m, **d,
                     **{f"portfolio_{k}": v for k, v in dp.items() if k in ("pr_auc", "base_rate", "daily_lift")}})
    return rows


def select_specification(boot: pd.DataFrame, comparison: pd.DataFrame, best: str) -> dict:
    """Rule fixed in advance: on pooled Validation PR-AUC (SPY 21-session >= 5% drawdown label), take the
    least complex score whose PR-AUC is not significantly below the best (block-bootstrap 95% interval of the
    difference vs the best includes zero). Ties in complexity: higher PR-AUC."""
    v = comparison[(comparison["split"] == "validation") & comparison["system"].isin(COMPLEXITY)]
    ap = v.set_index("system")["pr_auc"]
    ok = [best]
    for r in boot.itertuples():
        if r.ci95_high >= 0:
            ok.append(r.model)
    cand = sorted(set(ok) & set(COMPLEXITY), key=lambda m: (COMPLEXITY[m], -ap.get(m, -np.inf)))
    chosen = cand[0] if cand else best
    return {"best_validation_pr_auc": best, "not_significantly_worse": sorted(set(ok)), "selected": chosen,
            "rule": "least complex score whose block-bootstrap 95% PR-AUC interval vs the best includes 0"}


# --------------------------------------------------------------------------------------
# Entropy and velocity diagnostics
# --------------------------------------------------------------------------------------
def conditional_event_rates(daily: pd.DataFrame, labels: pd.Series, col: str, cond_col: str = "p_non_normal"
                            ) -> pd.DataFrame:
    """Event rate by tercile of `col` within low / high regime probability (does `col` add information?)."""
    d = pd.DataFrame({"x": daily[col], "c": daily[cond_col], "y": labels}).dropna()
    rows = []
    for name, sub in (("p_non_normal < 0.5", d[d["c"] < 0.5]), ("p_non_normal >= 0.5", d[d["c"] >= 0.5])):
        if len(sub) < 30 or sub["x"].nunique() < 3:
            rows.append({"regime_bucket": name, "tercile": "all", "n": len(sub),
                         "event_rate": float(sub["y"].mean()) if len(sub) else np.nan})
            continue
        q = pd.qcut(sub["x"].rank(method="first"), 3, labels=["low", "mid", "high"])
        for t, g in sub.groupby(q, observed=True):
            rows.append({"regime_bucket": name, "tercile": str(t), "n": len(g), "event_rate": float(g["y"].mean()),
                         "mean_value": float(g["x"].mean())})
    return pd.DataFrame(rows)


def velocity_lead(daily: pd.DataFrame, threshold: float, window: int = 10) -> pd.DataFrame:
    """For each hard HMM switch into a non-normal state, did positive velocity exceed `threshold` in the
    `window` sessions BEFORE the switch day?"""
    s = daily["hmm_state"].to_numpy()
    v = daily["non_normal_probability_velocity_positive"].to_numpy()
    rows = []
    for t in range(1, len(s)):
        if s[t - 1] == 1 and s[t] > 1:
            pre = v[max(t - window, 0):t]
            hit = np.flatnonzero(pre >= threshold)
            rows.append({"switch_date": daily.index[t], "to_state": f"S{s[t]}",
                         "velocity_exceeded_before": bool(len(hit)),
                         "lead_days": int(len(pre) - hit[0]) if len(hit) else np.nan,
                         "velocity_on_switch_day": float(v[t]) if np.isfinite(v[t]) else np.nan})
    return pd.DataFrame(rows)
