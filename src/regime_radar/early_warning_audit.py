"""Stage 6.1: non-tuning audit of the Stage 6 event evaluation (Train/Validation only).

Nothing is fitted or re-optimized here. Frozen Stage 6 signals, thresholds and alert rules are re-evaluated under
several warning windows and two event-matching conventions.

Warning windows W in {5, 10, 21, 63} sessions; late window L = 21 sessions (as in Stage 6).

PERMISSIVE convention (Stage 6 convention, generalized to W; identical to Stage 6 at W = 63)
  * warned        : at least one alert session in [onset - W, onset - 1];
                    lead = onset - start of the most recent alert episode before onset, clipped to the window;
  * detected late : no alert in [onset - W, onset - 1] but an alert in [onset, onset + L]; delay = first alert - onset;
  * missed        : neither;
  * false episode : alert episode with no onset in [start, start + W] and no overlap with any event period
                    [peak, end]. One episode may be credited to several events; episodes that start long before the
                    window still count if they are active inside it.

STRICT convention (one-to-one chronological episode matching)
  Episodes are maximal runs of alert sessions; an episode STARTS on its first alert session. Events are processed in
  chronological order and each episode can be matched to at most one event:
  * warned        : an unmatched episode STARTS in [onset - W, onset - 1]; if several, the most recent start is
                    matched (conservative lead = onset - start); the others are labelled `redundant_pre_event`;
  * detected late : otherwise, an unmatched episode starts in [onset, onset + L]; the earliest is matched
                    (delay = start - onset);
  * missed        : otherwise (even if an older episode, started before the window or already matched to an
                    earlier event, is still active - it is reported as `stale_alert_active_at_onset`).
  Unmatched episodes are classified, in this order, as
    `redundant_pre_event` : starts in [onset - W, onset - 1] of some event but another episode was matched;
    `premature`           : starts in [onset - 63, onset - W - 1] (too early for window W; only possible if W < 63);
    `in_event_period`     : starts inside an event period [peak, end] (drawdown already under way);
    `false`               : none of the above.
  Strict false alerts = the `false` class only; strict episode precision = matched episodes / all episodes.
  Caveat: strict matching counts episode STARTS, so a flickering alert that switches off and on creates new starts
  and can be credited where one continuous alert would not; read it together with episodes per year.
  In both conventions an alert that is first triggered ON the onset date is never an early warning: early windows
  end at onset - 1.

Episodes are evaluated inside one split series; an episode active on the first session of a split is
left-censored (its true start is unknown) and flagged.
"""

from __future__ import annotations

STAGE61_CODE_VERSION = "stage6.1-2026-10-21b"  # must match scripts/08b_audit_early_warning.py

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

WINDOWS = (5, 10, 21, 63)
LATE_WINDOW = 21
PREMATURE_HORIZON = 63
CONVENTIONS = ("permissive", "strict")
TRADING_DAYS = 252


def episodes(a) -> list[tuple[int, int]]:
    """Maximal runs of 1s as (start, end_exclusive)."""
    a = np.asarray(a, int)
    out, start = [], None
    for i, v in enumerate(a):
        if v and start is None:
            start = i
        if not v and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(a)))
    return out


def exposure(a) -> dict:
    a = np.asarray(a, int)
    n = len(a)
    eps = episodes(a)
    years = n / TRADING_DAYS if n else np.nan
    return {"n_sessions": n, "time_share_alerted": float(a.mean()) if n else np.nan,
            "alert_episodes": len(eps), "episodes_per_year": len(eps) / years if years else np.nan,
            "mean_episode_days": float(np.mean([t - s for s, t in eps])) if eps else 0.0,
            "alert_switches_per_year": float(np.abs(np.diff(a)).sum() / years) if years and n > 1 else np.nan,
            "left_censored_episode": bool(len(a) and a[0] == 1)}


def _events_sorted(events: pd.DataFrame) -> pd.DataFrame:
    return events.sort_values("onset_pos").reset_index(drop=True) if len(events) else events


def permissive_metrics(a, events: pd.DataFrame, window: int, late: int = LATE_WINDOW) -> tuple[dict, pd.DataFrame]:
    a_ = np.asarray(a, int)
    n = len(a_)
    ev = _events_sorted(events)
    rows = []
    for e in ev.itertuples():
        o = int(e.onset_pos)
        lo = max(o - window, 0)
        pre = np.flatnonzero(a_[lo:o]) + lo
        post = np.flatnonzero(a_[o:min(o + late + 1, n)]) + o
        lead = gap = delay = np.nan
        if len(pre):
            s0 = pre[-1]
            while s0 - 1 >= lo and a_[s0 - 1]:
                s0 -= 1
            status, lead, gap = "warned", o - s0, o - pre[-1]
        elif len(post):
            status, delay = "detected_late", post[0] - o
        else:
            status = "missed"
        rows.append({"onset_date": e.onset_date, "onset_pos": o, "status": status, "lead_days": lead,
                     "gap_last_alert_to_onset": gap, "delay_days": delay, "alert_on_onset_day": bool(o < n and a_[o]),
                     "first_alert_on_onset_day": bool(o < n and a_[o] and (o == 0 or not a_[o - 1]))})
    t = pd.DataFrame(rows)
    eps = episodes(a_)
    false = 0
    for s, f in eps:
        hit = any(s <= e.onset_pos <= s + window for e in ev.itertuples())
        overlap = any(not (f <= e.peak_pos or s > e.end_pos) for e in ev.itertuples())
        false += not (hit or overlap)
    m = _summary(t, n, len(eps), false, len(eps) - false)
    return m, t


def strict_metrics(a, events: pd.DataFrame, window: int, late: int = LATE_WINDOW
                   ) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    a_ = np.asarray(a, int)
    n = len(a_)
    ev = _events_sorted(events)
    eps = episodes(a_)
    starts = np.array([s for s, _ in eps])
    matched: dict[int, int] = {}
    rows = []
    for k, e in enumerate(ev.itertuples()):
        o = int(e.onset_pos)
        free = [i for i in range(len(eps)) if i not in matched]
        early = [i for i in free if o - window <= starts[i] <= o - 1]
        lead = gap = delay = np.nan
        ep = None
        if early:
            ep = max(early, key=lambda i: starts[i])
            s, f = eps[ep]
            status, lead, gap = "warned", o - s, o - (min(f, o) - 1)
        else:
            lt = [i for i in free if o <= starts[i] <= o + late]
            if lt:
                ep = min(lt, key=lambda i: starts[i])
                status, delay = "detected_late", starts[ep] - o
            else:
                status = "missed"
        if ep is not None:
            matched[ep] = k
        # an older episode (started before the window, or already matched) still active on onset - 1 or onset
        stale = any((s <= o - 1 < f) or (s < o < f) for i, (s, f) in enumerate(eps) if i != ep)
        rows.append({"onset_date": e.onset_date, "onset_pos": o, "status": status, "lead_days": lead,
                     "gap_last_alert_to_onset": gap, "delay_days": delay, "matched_episode": ep,
                     "matched_episode_start_pos": starts[ep] if ep is not None else np.nan,
                     "stale_alert_active_at_onset": bool(stale and status != "warned"),
                     "alert_on_onset_day": bool(o < n and a_[o]),
                     "first_alert_on_onset_day": bool(o < n and a_[o] and (o == 0 or not a_[o - 1]))})
    t = pd.DataFrame(rows)
    ep_rows = []
    for i, (s, f) in enumerate(eps):
        if i in matched:
            cls = "matched_early" if t.loc[matched[i], "status"] == "warned" else "matched_late"
        elif any(e.onset_pos - window <= s <= e.onset_pos - 1 for e in ev.itertuples()):
            cls = "redundant_pre_event"
        elif any(e.onset_pos - PREMATURE_HORIZON <= s <= e.onset_pos - window - 1 for e in ev.itertuples()):
            cls = "premature"
        elif any(e.peak_pos <= s <= e.end_pos for e in ev.itertuples()):
            cls = "in_event_period"
        else:
            cls = "false"
        ep_rows.append({"episode": i, "start_pos": s, "end_pos_exclusive": f, "days": f - s, "class": cls,
                        "matched_event": matched.get(i, np.nan), "left_censored": s == 0 and n > 0 and a_[0] == 1})
    et = pd.DataFrame(ep_rows)
    n_false = int((et["class"] == "false").sum()) if len(et) else 0
    n_match = int(et["class"].str.startswith("matched").sum()) if len(et) else 0
    m = _summary(t, n, len(eps), n_false, n_match)
    for c in ("redundant_pre_event", "premature", "in_event_period"):
        m[f"{c}_episodes"] = int((et["class"] == c).sum()) if len(et) else 0
    m["stale_alert_active_at_missed_or_late_onset"] = int(t["stale_alert_active_at_onset"].sum()) if len(t) else 0
    return m, t, et


def _summary(t: pd.DataFrame, n: int, n_eps: int, n_false: int, n_true: int) -> dict:
    years = n / TRADING_DAYS if n else np.nan
    st = t["status"] if len(t) else pd.Series(dtype=str)
    m = {"n_events": len(t), "warned": int((st == "warned").sum()), "detected_late": int((st == "detected_late").sum()),
         "missed": int((st == "missed").sum()),
         "median_lead_days": float(t["lead_days"].median()) if len(t) and t["lead_days"].notna().any() else np.nan,
         "median_gap_to_onset": float(t["gap_last_alert_to_onset"].median())
         if len(t) and t["gap_last_alert_to_onset"].notna().any() else np.nan,
         "median_delay_days": float(t["delay_days"].median()) if len(t) and t["delay_days"].notna().any() else np.nan,
         "first_alert_on_onset_day": int(t["first_alert_on_onset_day"].sum()) if len(t) else 0,
         "alert_episodes": n_eps, "false_alert_episodes": n_false,
         "false_alerts_per_year": n_false / years if years else np.nan,
         "episode_precision": n_true / n_eps if n_eps else np.nan}
    m["warned_share"] = m["warned"] / m["n_events"] if m["n_events"] else np.nan
    return m


def evaluate(a: pd.Series, events: pd.DataFrame, windows=WINDOWS) -> tuple[list[dict], list[pd.DataFrame],
                                                                           list[pd.DataFrame]]:
    """Both conventions for every window; returns metric rows and event / episode tables."""
    rows, ev_tabs, ep_tabs = [], [], []
    x = a.fillna(0).astype(int).to_numpy()
    expo = exposure(x)
    for w in windows:
        mp, tp = permissive_metrics(x, events, w)
        ms, ts, es = strict_metrics(x, events, w)
        rows.append({"window": w, "convention": "permissive", **expo, **mp})
        rows.append({"window": w, "convention": "strict", **expo, **ms})
        ev_tabs += [tp.assign(window=w, convention="permissive"), ts.assign(window=w, convention="strict")]
        ep_tabs.append(es.assign(window=w))
    return rows, ev_tabs, ep_tabs


def exposure_matched_threshold(score: pd.Series, train_mask, share: float) -> float:
    """Train-only threshold giving the requested Train alert share (no Validation information)."""
    return float(np.nanquantile(score[train_mask].to_numpy(float), 1 - share))


def ranking_metrics(score: pd.Series, labels: pd.Series) -> dict:
    ok = labels.notna() & score.notna()
    y, s = labels[ok].to_numpy(int), score[ok].to_numpy(float)
    both = 0 < y.sum() < len(y)
    return {"n_days": int(ok.sum()), "base_rate": float(y.mean()) if len(y) else np.nan,
            "pr_auc": float(average_precision_score(y, s)) if both else np.nan,
            "roc_auc": float(roc_auc_score(y, s)) if both else np.nan}


def interval_statement(lo: float, hi: float, what: str) -> str:
    """Wording that never treats an interval containing zero as equivalence."""
    if lo > 0:
        return f"{what}: the 95% interval is above zero ({lo:+.3f} to {hi:+.3f})."
    if hi < 0:
        return f"{what}: the 95% interval is below zero ({lo:+.3f} to {hi:+.3f})."
    return (f"{what}: the 95% interval ({lo:+.3f} to {hi:+.3f}) contains zero - the difference is not resolved by "
            "these data. This is NOT evidence that the two scores are equivalent; the interval is also compatible "
            f"with differences as large as {max(abs(lo), abs(hi)):.3f}.")


# --------------------------------------------------------------------------------------
# Exposure budgets under ties / saturation (no tie-breaking)
# --------------------------------------------------------------------------------------
EXPOSURE_TOLERANCE = 0.005   # |achieved - target| Train share allowed for "exposure-matched" (0.5 percentage point)


def exposure_audit(score: pd.Series, train_mask, val_mask, target: float) -> dict:
    """Threshold rule `score >= c` with c = Train quantile(1 - target). With ties, the achievable Train exposures are
    the step values share(score >= v); no tie-breaking (chronological or random) is used to reach the target."""
    tr = score[train_mask].to_numpy(float)
    tr = tr[np.isfinite(tr)]
    va = score[val_mask].to_numpy(float)
    va = va[np.isfinite(va)]
    thr = float(np.nanquantile(tr, 1 - target))
    achieved = float((tr >= thr).mean())
    below = float((tr > thr).mean())            # next attainable exposure below the achieved one
    tmax = float(tr.max())
    out = {"target_train_share": target, "threshold": thr, "threshold_repr": repr(thr),
           "n_train": int(len(tr)), "n_train_tied_at_threshold": int((tr == thr).sum()),
           "n_train_at_maximum": int((tr == tmax).sum()), "train_maximum_repr": repr(tmax),
           "n_train_exactly_one": int((tr == 1.0).sum()),
           "achieved_train_share": achieved, "attainable_share_just_below": below,
           "validation_share": float((va >= thr).mean()) if len(va) else np.nan,
           "n_validation_tied_at_threshold": int((va == thr).sum()),
           "gap_pp": 100 * (achieved - target)}
    matched = abs(achieved - target) <= EXPOSURE_TOLERANCE
    out["exposure_matched"] = bool(matched)
    ties = out["n_train_tied_at_threshold"]
    cause = (f"because {ties} Train days are tied at the threshold {thr!r}" if ties > 1 else
             f"outside the {EXPOSURE_TOLERANCE:.1%} tolerance at threshold {thr!r}")
    out["exposure_status"] = ("EXPOSURE-MATCHED" if matched else
                              f"NOT EXPOSURE-MATCHED: target {target:.2%} not attained; the attainable Train shares "
                              f"around it are {below:.2%} and {achieved:.2%} {cause}")
    return out


# --------------------------------------------------------------------------------------
# Signal lead versus executable protection lead
# --------------------------------------------------------------------------------------
def protection_columns(t: pd.DataFrame, r: pd.Series) -> pd.DataFrame:
    """For warned events: the alert episode starts at s = onset - lead (signal after the close of s). Under the
    conservative execution convention the position changes at the close of s+1 and first earns return_(s+2).
      executable_lead_sessions = onset - (s + 1) = lead - 1  (returns r_(s+2)..r_onset are protected before onset)
      executable_warning       = executable_lead_sessions >= 1
      drawdown_at_signal / _at_execution = SPY drawdown from its running split peak at the close of s / s+1
      spy_return_execution_to_onset = log return from the close of s+1 to the close of onset (the part of the
                                      pre-onset decline that a position change at s+1 could have avoided)"""
    t = t.copy()
    lvl = r.cumsum().to_numpy()
    dd = lvl - np.maximum.accumulate(lvl)
    n = len(lvl)
    lead = t["lead_days"]
    t["signal_lead_sessions"] = lead
    t["executable_lead_sessions"] = lead - 1
    t["executable_warning"] = (lead - 1 >= 1).fillna(False).astype(bool)
    s = (t["onset_pos"] - lead)
    ok = lead.notna()
    t["drawdown_at_signal_pct"] = np.nan
    t["drawdown_at_execution_pct"] = np.nan
    t["spy_return_execution_to_onset_pct"] = np.nan
    for i in t.index[ok]:
        si, o = int(s[i]), int(t.at[i, "onset_pos"])
        t.at[i, "drawdown_at_signal_pct"] = 100 * np.expm1(dd[si])
        if si + 1 < n:
            t.at[i, "drawdown_at_execution_pct"] = 100 * np.expm1(dd[si + 1])
            t.at[i, "spy_return_execution_to_onset_pct"] = 100 * np.expm1(lvl[o] - lvl[si + 1]) if si + 1 <= o else np.nan
    return t


# --------------------------------------------------------------------------------------
# Independent invariant check of the strict matching
# --------------------------------------------------------------------------------------
def verify_strict(a, events: pd.DataFrame, window: int, table: pd.DataFrame, ep_table: pd.DataFrame,
                  late: int = LATE_WINDOW) -> list[str]:
    """Re-derive the strict-matching invariants from the raw alert series; returns violations (empty = verified)."""
    a_ = np.asarray(a, int)
    eps = episodes(a_)
    v = []
    if len(ep_table) != len(eps) or [(int(x), int(y)) for x, y in zip(ep_table["start_pos"], ep_table["end_pos_exclusive"])] \
            != [(s, f) for s, f in eps]:
        v.append("episode list differs from the raw alert series")
        return v
    used = table["matched_episode"].dropna().astype(int)
    if used.duplicated().any():
        v.append("an episode is matched to more than one event")
    ev = _events_sorted(events)
    taken: set[int] = set()
    for (_, row), e in zip(table.iterrows(), ev.itertuples()):
        o = int(e.onset_pos)
        free = [i for i in range(len(eps)) if i not in taken]
        early = [i for i in free if o - window <= eps[i][0] <= o - 1]
        lt = [i for i in free if o <= eps[i][0] <= o + late]
        if row["status"] == "warned":
            m = int(row["matched_episode"])
            if m not in early:
                v.append(f"onset {o}: warned by an episode that does not start in [onset-{window}, onset-1]")
            elif eps[m][0] != max(eps[i][0] for i in early):
                v.append(f"onset {o}: matched episode is not the most recent start")
            if row["lead_days"] != o - eps[m][0] or row["lead_days"] < 1:
                v.append(f"onset {o}: lead inconsistent")
            taken.add(m)
        elif row["status"] == "detected_late":
            m = int(row["matched_episode"])
            if early:
                v.append(f"onset {o}: late although an unmatched early episode exists")
            if m not in lt or eps[m][0] != min(eps[i][0] for i in lt):
                v.append(f"onset {o}: late match is not the earliest start in [onset, onset+{late}]")
            taken.add(m)
        else:
            if early or lt:
                v.append(f"onset {o}: missed although an unmatched episode starts in the window")
        if a_[o] and (o == 0 or not a_[o - 1]) and not any(a_[max(o - window, 0):o]) and row["status"] == "warned":
            v.append(f"onset {o}: onset-day alert counted as early warning")
    return v
