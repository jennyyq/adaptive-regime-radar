"""Stage 6.1 -> Stage 7 handoff: frozen, role-separated daily signal files and a guarded loader.

Two roles are handed to Stage 7. They live in SEPARATE files with role-prefixed column names, so the simple regime
baseline can never be silently mistaken for the full Early Warning System (or vice versa):

  baseline_regime   baseline_p_non_normal         = P(S2) + P(S3), frozen HMM hmm_std_diag_k3, causal filter
                    baseline_alert_ge_q80         = 1[baseline_p_non_normal >= frozen Train 80th percentile]
  challenger_ews    challenger_ews_score          = frozen Stage 6 EWS (ranking score in [0, 1])
                    challenger_alert_level        = GREEN / AMBER / ORANGE / RED / RECOVERY
                    challenger_alert_ge_amber / _ge_orange = 1[level >= AMBER] / 1[level >= ORANGE] (RECOVERY = 0)

Execution convention (frozen, conservative; the ONLY convention carried by the files):
  signal_date t                   information through the US close of t
  signal_available_after          t 16:00 America/New_York (ISO 8601 with UTC offset); never usable during session t
  execution_date                  t+1: orders are executed at the CLOSE of t+1
  execution_time                  t+1 16:00 America/New_York
  first_holding_return_date       t+2: first close-to-close return earned, return_(t+2) = log(P_t+2 / P_t+1)
The Stage 3 protocol text (return_(t+1)) is optimistic by one session and is not represented in the files.

Floats are written with 17 significant digits, so every value - including probabilities saturated at or within
1e-16 of 1.0 - round-trips exactly and the frozen threshold rules can be reproduced bit-for-bit from the files.

`load_stage7_inputs(role, split)` verifies the file hash recorded in the handoff manifest, the exact column set,
the role prefix and the READY status of the handoff, and refuses columns of the other role.
"""

from __future__ import annotations

STAGE61_CODE_VERSION = "stage6.1-2026-10-21b"  # must match scripts/08b_audit_early_warning.py

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROLES = ("baseline_regime", "challenger_ews")
TIMING_COLUMNS = ("signal_available_after", "execution_date", "execution_time", "first_holding_return_date")
ROLE_COLUMNS = {
    "baseline_regime": ("baseline_p_non_normal", "baseline_alert_ge_q80"),
    "challenger_ews": ("challenger_ews_score", "challenger_alert_level", "challenger_alert_ge_amber",
                       "challenger_alert_ge_orange"),
}
ROLE_PREFIX = {"baseline_regime": "baseline_", "challenger_ews": "challenger_"}
HANDOFF_DIRNAME = "stage7_handoff"
MANIFEST_NAME = "stage7_handoff_manifest.json"
FLOAT_FORMAT = "%.17g"
MARKET_TZ = "America/New_York"
MARKET_CLOSE = "16:00"
EXECUTION_CONVENTION = {
    "name": "conservative_close_t_plus_1",
    "signal": "computed after the US close of t from information through the close of t",
    "execution": "at the close of session t+1 (column execution_date / execution_time)",
    "first_holding_return": "return_(t+2) = log(P_(t+2) / P_(t+1)) (column first_holding_return_date)",
    "forbidden": "crediting a date-t signal with return_t or return_(t+1)",
    "stage3_protocol_text": "ARR-stage3-protocol-v1.2 credits return_(t+1); that assumes execution at the close of t "
                            "with information computed at that close and is optimistic by one session. It is not "
                            "the Stage 7 convention; if ever shown, it must be labelled as an optimistic sensitivity.",
}


class HandoffError(ValueError):
    """Raised when Stage 7 inputs do not match the frozen handoff."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close_timestamp(dates) -> pd.Series:
    """16:00 America/New_York on each date, as ISO 8601 with the UTC offset (DST-aware)."""
    d = pd.DatetimeIndex(dates)
    ts = (d + pd.Timedelta(hours=16)).tz_localize(MARKET_TZ)
    return pd.Series([t.isoformat() for t in ts], index=d)


def timing_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Timing columns from the Train+Validation calendar. The Test calendar is never read, so the last Validation
    rows have no execution / holding date (blank)."""
    idx = pd.DatetimeIndex(index)
    s = pd.Series(idx, index=idx)
    nxt, nxt2 = s.shift(-1), s.shift(-2)
    f = pd.DataFrame(index=idx)
    f["signal_available_after"] = close_timestamp(idx).to_numpy()
    f["execution_date"] = nxt.dt.strftime("%Y-%m-%d")
    exe = pd.Series(np.nan, index=idx, dtype=object)
    ok = nxt.notna()
    exe[ok] = close_timestamp(pd.DatetimeIndex(nxt[ok])).to_numpy()
    f["execution_time"] = exe
    f["first_holding_return_date"] = nxt2.dt.strftime("%Y-%m-%d")
    return f


def build_role_frames(daily: pd.DataFrame, score: pd.Series, levels: pd.Series, baseline_threshold: float
                      ) -> dict[str, pd.DataFrame]:
    """Role-separated frames. `daily` must hold p_non_normal and split."""
    from .alert_state_machine import alert_flag

    t = timing_frame(daily.index)
    p = daily["p_non_normal"]
    base = t.copy()
    base.insert(0, "split", daily["split"])
    base["baseline_p_non_normal"] = p
    base["baseline_alert_ge_q80"] = (p >= baseline_threshold).astype(int)
    ch = t.copy()
    ch.insert(0, "split", daily["split"])
    ch["challenger_ews_score"] = score
    ch["challenger_alert_level"] = levels
    ch["challenger_alert_ge_amber"] = alert_flag(levels, "AMBER")
    ch["challenger_alert_ge_orange"] = alert_flag(levels, "ORANGE")
    return {"baseline_regime": base, "challenger_ews": ch}


def write_handoff(frames: dict[str, pd.DataFrame], out_dir: Path) -> dict:
    """Write one file per role and split with exact (17-digit) floats; returns {file name: sha256}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for role, f in frames.items():
        for split in ("train", "validation"):
            path = out_dir / f"stage7_{role}_{split}.csv"
            f[f["split"] == split].drop(columns="split").to_csv(path, index_label="signal_date",
                                                                date_format="%Y-%m-%d", float_format=FLOAT_FORMAT)
            hashes[path.name] = _sha256(path)
    return hashes


def expected_columns(role: str) -> list[str]:
    if role not in ROLES:
        raise HandoffError(f"unknown role {role!r}; expected one of {ROLES}")
    return list(TIMING_COLUMNS) + list(ROLE_COLUMNS[role])


def load_stage7_inputs(role: str, split: str, early_warning_dir: Path, columns: list[str] | None = None,
                       require_ready: bool = True) -> pd.DataFrame:
    """The only supported way for Stage 7 to read Stage 6 signals.

    * `role` must be 'baseline_regime' or 'challenger_ews'; there is no combined role;
    * the handoff manifest must be READY (unless require_ready=False, used only by the Stage 6.1 audit itself);
    * the file hash must equal the one in the handoff manifest;
    * the column set must be exactly the frozen one; every signal column carries the role prefix;
    * requesting a column of the other role raises HandoffError.
    Test data are never available here (only train / validation files exist)."""
    if split not in ("train", "validation"):
        raise HandoffError("only 'train' and 'validation' signals exist in the Stage 6 handoff")
    cols = expected_columns(role)
    hdir = Path(early_warning_dir) / HANDOFF_DIRNAME
    manifest = json.loads((hdir / MANIFEST_NAME).read_text(encoding="utf-8"))
    if require_ready and manifest.get("handoff_status") != "READY":
        raise HandoffError(f"handoff status is {manifest.get('handoff_status')!r}, not READY: "
                           f"{manifest.get('handoff_status_reason', '')}")
    name = f"stage7_{role}_{split}.csv"
    path = hdir / name
    recorded = manifest["files"].get(name, {}).get("sha256")
    if recorded is None or recorded != _sha256(path):
        raise HandoffError(f"{name}: hash does not match the handoff manifest (file changed or not frozen)")
    frame = pd.read_csv(path, index_col="signal_date", parse_dates=["signal_date"], float_precision="round_trip")
    if list(frame.columns) != cols:
        raise HandoffError(f"{name}: columns {list(frame.columns)} differ from the frozen set {cols}")
    other = [r for r in ROLES if r != role][0]
    if columns is not None:
        bad = [c for c in columns if c.startswith(ROLE_PREFIX[other]) or c in ROLE_COLUMNS[other]]
        if bad:
            raise HandoffError(f"columns {bad} belong to role {other!r}; load that role explicitly")
        unknown = [c for c in columns if c not in cols]
        if unknown:
            raise HandoffError(f"columns {unknown} are not part of role {role!r}")
        frame = frame[columns]
    frame.attrs["stage7_role"] = role
    return frame


def assert_single_role(frame: pd.DataFrame) -> str:
    """Guard for Stage 7 code: a signal frame must carry columns of exactly one role."""
    roles = {r for r in ROLES for c in frame.columns if c.startswith(ROLE_PREFIX[r])}
    if len(roles) != 1:
        raise HandoffError(f"signal frame mixes or lacks roles: {sorted(roles) or 'none'}")
    role = roles.pop()
    if frame.attrs.get("stage7_role") not in (None, role):
        raise HandoffError("frame attrs and column prefixes disagree on the role")
    return role


def describe_columns(baseline_threshold: float, thresholds: dict) -> dict:
    """Column dictionary for the manifest: meaning, availability and whether it is actionable."""
    timing = {
        "signal_available_after": "t 16:00 America/New_York; the signal may not be used before this time",
        "execution_date": "t+1; execute at the close of this session",
        "execution_time": "t+1 16:00 America/New_York",
        "first_holding_return_date": "t+2; the first close-to-close return a position based on signal t may earn",
    }
    return {
        "timing (both roles)": timing,
        "baseline_regime": {
            "baseline_p_non_normal": {"meaning": "P(S2) + P(S3), causal forward-filtered probability of the frozen HMM",
                                      "type": "float64 in [0, 1], 17 significant digits",
                                      "use": "actionable (permitted portfolio input)"},
            "baseline_alert_ge_q80": {"meaning": f"1 if baseline_p_non_normal >= {baseline_threshold!r} "
                                                 "(Train 80th percentile; the rule evaluated in Stage 6)",
                                      "type": "binary", "use": "actionable (permitted portfolio input)"},
        },
        "challenger_ews": {
            "challenger_ews_score": {"meaning": "frozen Stage 6 Early Warning Score (ranking score, NOT a probability)",
                                     "type": "float64 in [0, 1]; empty in the first warm-up sessions",
                                     "use": "actionable (permitted portfolio input)"},
            "challenger_alert_level": {"meaning": "frozen 5-level state machine output",
                                       "type": "GREEN / AMBER / ORANGE / RED / RECOVERY",
                                       "thresholds_repr": {k: repr(v) for k, v in thresholds.items()},
                                       "use": "actionable (permitted portfolio input)"},
            "challenger_alert_ge_amber": {"meaning": "1 if level in {AMBER, ORANGE, RED}", "type": "binary",
                                          "use": "actionable (permitted portfolio input)"},
            "challenger_alert_ge_orange": {"meaning": "1 if level in {ORANGE, RED}", "type": "binary",
                                           "use": "actionable (permitted portfolio input)"},
        },
    }


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj
