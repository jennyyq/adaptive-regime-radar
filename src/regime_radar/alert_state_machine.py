"""Stage 6: operational alert state machine (GREEN / AMBER / ORANGE / RED / RECOVERY).

Thresholds are quantiles of the EWS on Train (fixed before Validation is examined):
    AMBER  entry q80, exit q70        ORANGE entry q90, exit q85        RED entry q97, exit q93
Rules (evaluated each day with information through that day only):
  * Escalation to level L: today's EWS >= entry(L) AND EWS >= entry(L) on >= 3 of the last 5 sessions
    (today included). Levels may be skipped (GREEN -> ORANGE is allowed); the highest confirmed level wins.
  * Emergency: RED immediately (no confirmation) if severe independent evidence is observed today:
    (p_stress >= 0.90 and level-stress percentile >= 0.95) or (1-day AR(1) market innovation percentile
    >= 0.995 and level-stress percentile >= 0.90).
  * De-escalation (hysteresis): RED -> ORANGE when EWS < exit(RED) on >= 3 of the last 5 sessions;
    ORANGE -> RECOVERY when EWS < exit(ORANGE) on >= 3 of 5; AMBER -> GREEN when EWS < exit(AMBER) on >= 3 of 5.
  * RECOVERY -> GREEN after 10 consecutive sessions with EWS < exit(AMBER); RECOVERY can re-escalate by the
    normal entry rules. RECOVERY otherwise persists.
Observed HMM states and alert levels are different objects: the machine never reads the HMM argmax state.
"""

from __future__ import annotations

STAGE6_CODE_VERSION = "stage6-2026-10-21"  # must match scripts/08_build_early_warning.py

from dataclasses import dataclass

import numpy as np
import pandas as pd

LEVELS = ("GREEN", "AMBER", "ORANGE", "RED", "RECOVERY")
SEVERITY = {"GREEN": 0, "RECOVERY": 1, "AMBER": 1, "ORANGE": 2, "RED": 3}
ALERT_COLORS = {"GREEN": "#1baf7a", "AMBER": "#eda100", "ORANGE": "#eb6834", "RED": "#c93131", "RECOVERY": "#6aa4e6"}
QUANTILES = {"amber_in": 0.80, "amber_out": 0.70, "orange_in": 0.90, "orange_out": 0.85, "red_in": 0.97,
             "red_out": 0.93}
CONFIRM_WINDOW, CONFIRM_MIN = 5, 3
RECOVERY_CLEAR_DAYS = 10
EMERGENCY = {"p_stress": 0.90, "confirmation_with_p_stress": 0.95, "innovation_pct": 0.995,
             "confirmation_with_innovation": 0.90}


@dataclass
class Thresholds:
    amber_in: float
    amber_out: float
    orange_in: float
    orange_out: float
    red_in: float
    red_out: float

    @classmethod
    def from_train(cls, ews_train: pd.Series) -> "Thresholds":
        v = ews_train.dropna().to_numpy()
        return cls(**{k: float(np.quantile(v, q)) for k, q in QUANTILES.items()})

    def to_dict(self) -> dict:
        return dict(vars(self))


def _count(window) -> int:
    return int(sum(window))


def run_state_machine(ews: pd.Series, thresholds: Thresholds, p_stress: pd.Series | None = None,
                      confirmation: pd.Series | None = None, innovation_pct: pd.Series | None = None,
                      hysteresis: bool = True) -> pd.DataFrame:
    """Sequential, past-only alert levels. With hysteresis=False the level is the threshold band of today's
    EWS (no confirmation, no exit thresholds, no RECOVERY) - used only for the ablation."""
    th = thresholds
    e = ews.to_numpy(float)
    n = len(e)
    entry = {"AMBER": th.amber_in, "ORANGE": th.orange_in, "RED": th.red_in}
    level, reasons = [], []
    state, below_amber_run = "GREEN", 0
    hist = {k: [] for k in ("amber_in", "orange_in", "red_in", "amber_out", "orange_out", "red_out")}
    for t in range(n):
        v = e[t]
        if not np.isfinite(v):
            level.append(state)
            reasons.append("no score (insufficient history)")
            continue
        for k in hist:
            thr = getattr(th, k)
            hist[k].append(v >= thr if k.endswith("_in") else v < thr)
            hist[k] = hist[k][-CONFIRM_WINDOW:]
        if not hysteresis:
            state = ("RED" if v >= th.red_in else "ORANGE" if v >= th.orange_in else "AMBER" if v >= th.amber_in
                     else "GREEN")
            level.append(state)
            reasons.append(f"{state}: EWS {v:.3f} band (no hysteresis)")
            continue
        reason = None
        ps = p_stress.iloc[t] if p_stress is not None else np.nan
        cf = confirmation.iloc[t] if confirmation is not None else np.nan
        ip = innovation_pct.iloc[t] if innovation_pct is not None else np.nan
        emergency = ((np.nan_to_num(ps) >= EMERGENCY["p_stress"] and np.nan_to_num(cf) >= EMERGENCY["confirmation_with_p_stress"])
                     or (np.nan_to_num(ip) >= EMERGENCY["innovation_pct"]
                         and np.nan_to_num(cf) >= EMERGENCY["confirmation_with_innovation"]))
        confirmed = [lvl for lvl in ("RED", "ORANGE", "AMBER")
                     if v >= entry[lvl] and _count(hist[f"{lvl.lower()}_in"]) >= CONFIRM_MIN]
        target = confirmed[0] if confirmed else None
        if emergency and SEVERITY[state] < 3:
            state, reason = "RED", f"RED (emergency): p_stress {ps:.2f}, level stress pct {cf:.2f}, innovation pct {ip:.3f}"
        elif target and SEVERITY[target] > SEVERITY[state if state != "RECOVERY" else "GREEN"]:
            reason = f"{target}: EWS {v:.3f} >= {entry[target]:.3f} on {_count(hist[target.lower() + '_in'])}/5 sessions"
            state = target
        elif state == "RED" and _count(hist["red_out"]) >= CONFIRM_MIN:
            state, reason = "ORANGE", f"ORANGE (de-escalation): EWS < {th.red_out:.3f} on {_count(hist['red_out'])}/5"
        elif state == "ORANGE" and _count(hist["orange_out"]) >= CONFIRM_MIN:
            state, reason = "RECOVERY", f"RECOVERY: EWS < {th.orange_out:.3f} on {_count(hist['orange_out'])}/5"
        elif state == "AMBER" and _count(hist["amber_out"]) >= CONFIRM_MIN:
            state, reason = "GREEN", f"GREEN: EWS < {th.amber_out:.3f} on {_count(hist['amber_out'])}/5"
        below_amber_run = below_amber_run + 1 if v < th.amber_out else 0
        if state == "RECOVERY" and reason is None and below_amber_run >= RECOVERY_CLEAR_DAYS:
            state, reason = "GREEN", f"GREEN: EWS < {th.amber_out:.3f} for {RECOVERY_CLEAR_DAYS} sessions after stress"
        if reason is None:
            reason = f"{state} maintained: EWS {v:.3f}"
        level.append(state)
        reasons.append(reason)
    return pd.DataFrame({"alert_level": level, "alert_rule": reasons}, index=ews.index)


def alert_flag(levels: pd.Series, minimum: str = "AMBER") -> pd.Series:
    """1 when the alert is at least `minimum` in severity (RECOVERY counts as below AMBER)."""
    need = SEVERITY[minimum]
    return levels.map(lambda s: int(s != "RECOVERY" and SEVERITY[s] >= need))
