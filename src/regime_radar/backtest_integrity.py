"""Stage 8: pre-Test lock, gated (evaluation-only) Test loader, Test-data integrity checks and the Test access log.

The reserved Test split is opened ONLY through `load_test_split`, which refuses to read anything unless
  * data/processed/backtesting/stage8_pretest_lock.json exists and authorizes Test access (all preflight checks PASS);
  * every code, configuration and frozen-artifact hash recorded in the lock still matches the files on disk;
  * the Test file hashes match the Stage 3 split manifest.
Every successful access is appended to test_access_log.json. The Stage 4-7 loaders (regime_exploration,
stage7_handoff) are not modified and still refuse the Test split.
"""

from __future__ import annotations

STAGE8_CODE_VERSION = "stage8-2026-10-21"  # must match scripts/10_backtest_portfolios.py

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

LOCK_NAME = "stage8_pretest_lock.json"
ACCESS_LOG_NAME = "test_access_log.json"
LOCK_VERSION = "stage8-lock-v1"
TEST_FILES = ("test_features.csv", "test_returns.csv")
MODEL_FILES = ("train_features.csv", "train_returns.csv", "validation_features.csv", "validation_returns.csv")


class TestAccessError(RuntimeError):
    """Raised when the Test split may not be opened."""


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(obj) -> str:
    return json.dumps(jsonable(obj), sort_keys=True, separators=(",", ":"))


def fingerprint(frozen: dict) -> str:
    return hashlib.sha256(canonical(frozen).encode("utf-8")).hexdigest()


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
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj.date()) if obj == obj.normalize() else obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    return obj


# --------------------------------------------------------------------------------------
# Hash snapshot
# --------------------------------------------------------------------------------------
def code_files(root: Path) -> list[Path]:
    """Every project module (Stages 0-8) and the Stage 8 script: any change after the lock invalidates it."""
    files = sorted((root / "src" / "regime_radar").glob("*.py"))
    files.append(root / "scripts" / "10_backtest_portfolios.py")
    return files


def snapshot(root: Path, extra_files: dict[str, Path]) -> dict:
    """Hashes of code, configs and frozen artifacts (relative paths as keys)."""
    out = {"code": {}, "artifacts": {}}
    for p in code_files(root):
        if p.exists():
            out["code"][str(p.relative_to(root))] = sha256(p)
    for name, p in extra_files.items():
        out["artifacts"][name] = sha256(p) if Path(p).exists() else None
    return out


def compare_snapshot(locked: dict, current: dict) -> list[str]:
    problems = []
    for part in ("code", "artifacts"):
        a, b = locked.get(part, {}), current.get(part, {})
        for k in sorted(set(a) | set(b)):
            if a.get(k) != b.get(k):
                problems.append(f"{part}:{k} {'missing' if b.get(k) is None else 'changed'}")
    return problems


# --------------------------------------------------------------------------------------
# Lock
# --------------------------------------------------------------------------------------
def build_lock(frozen: dict, preflight: dict, authorized: bool, reason: str) -> dict:
    return {"lock_version": LOCK_VERSION, "code_version": STAGE8_CODE_VERSION,
            "created_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
            "fingerprint": fingerprint(frozen), "frozen": jsonable(frozen), "preflight": jsonable(preflight),
            "test_data_read_before_lock": False,
            "test_access_authorized": bool(authorized), "authorization_reason": reason}


def write_json(obj: dict, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(jsonable(obj), indent=2, sort_keys=False), encoding="utf-8")


def verify_lock(lock: dict, current_snapshot: dict) -> list[str]:
    problems = []
    if lock.get("lock_version") != LOCK_VERSION:
        problems.append(f"lock version {lock.get('lock_version')!r} != {LOCK_VERSION!r}")
    if not lock.get("test_access_authorized"):
        problems.append("lock does not authorize Test access: " + str(lock.get("authorization_reason", "")))
    if fingerprint(lock.get("frozen", {})) != lock.get("fingerprint"):
        problems.append("lock fingerprint does not match its frozen content (lock edited)")
    problems += compare_snapshot(lock.get("frozen", {}).get("hashes", {}), current_snapshot)
    return problems


# --------------------------------------------------------------------------------------
# Gated Test loader
# --------------------------------------------------------------------------------------
@dataclass
class TestSplit:
    features: pd.DataFrame
    returns: pd.DataFrame
    files_read: list[str] = field(default_factory=list)
    hashes: dict = field(default_factory=dict)


def _read(path: Path) -> pd.DataFrame:
    f = pd.read_csv(path, index_col="date", parse_dates=["date"])
    f.index = pd.DatetimeIndex(f.index).astype("datetime64[ns]")
    f.index.name = "date"
    return f


def load_test_split(splits_dir: Path, lock_path: Path, current_snapshot: dict) -> TestSplit:
    """The ONLY Stage 8 entry point to the reserved Test split (evaluation only)."""
    lock_path = Path(lock_path)
    if not lock_path.exists():
        raise TestAccessError(f"{lock_path.name} not found: run the Stage 8 preflight first")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    problems = verify_lock(lock, current_snapshot)
    if problems:
        raise TestAccessError("pre-Test lock invalid: " + "; ".join(problems[:10]))
    manifest = json.loads((Path(splits_dir) / "split_manifest.json").read_text(encoding="utf-8"))
    out, hashes = {}, {}
    for name in TEST_FILES:
        path = Path(splits_dir) / name
        expected = manifest.get("split_files", {}).get(name)
        digest = sha256(path)
        if expected is None or digest != expected:
            raise TestAccessError(f"{name}: hash differs from the Stage 3 split manifest (Test file changed)")
        hashes[name] = digest
        out[name] = _read(path)
    return TestSplit(out["test_features.csv"], out["test_returns.csv"], list(TEST_FILES), hashes)


def append_access_log(path: Path, entry: dict) -> list[dict]:
    path = Path(path)
    log = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    log.append(jsonable(entry))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(log), indent=2), encoding="utf-8")
    return log


def read_access_log(path: Path) -> list[dict]:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


# --------------------------------------------------------------------------------------
# Test-data integrity (run immediately after the first read)
# --------------------------------------------------------------------------------------
def test_data_checks(test: TestSplit, manifest: dict, last_model_date: pd.Timestamp, feature_columns: list[str],
                     return_columns: list[str], model_feature_columns: list[str],
                     model_return_columns: list[str]) -> list[tuple[str, str, str]]:
    """(name, status, detail) rows. `model_*_columns` are the column lists of the Train split files."""
    rows = []
    f, r = test.features, test.returns
    spec = next((s for s in manifest.get("splits", []) if s.get("name") == "test"), {})
    ok_cols = (list(f.columns) == list(model_feature_columns) and list(r.columns) == list(model_return_columns)
               and all(c in f.columns for c in feature_columns) and all(c in r.columns for c in return_columns))
    rows.append(("Test factor order and ETF universe identical to Train/Validation", "PASS" if ok_cols else "FAIL",
                 f"factors {list(f.columns)}; return columns {list(r.columns)}"))
    chron = f.index.is_monotonic_increasing and r.index.is_monotonic_increasing and not f.index.has_duplicates \
        and not r.index.has_duplicates and f.index.equals(r.index)
    rows.append(("Test dates strictly chronological, unique, identical in factor and return files",
                 "PASS" if chron else "FAIL", f"{len(f)} sessions {f.index.min().date()}..{f.index.max().date()}"))
    no_overlap = f.index.min() > last_model_date
    rows.append(("No overlap with Validation; Test starts after the last Validation session",
                 "PASS" if no_overlap else "FAIL", f"last Validation {last_model_date.date()}, first Test {f.index.min().date()}"))
    same = (str(f.index.min().date()) == spec.get("actual_first_date") and str(f.index.max().date()) ==
            spec.get("actual_last_date") and len(f) == spec.get("rows"))
    rows.append(("Test sessions match the frozen Stage 3 split manifest (membership, first / last date, rows)",
                 "PASS" if same else "FAIL", f"manifest {spec.get('actual_first_date')}..{spec.get('actual_last_date')} "
                 f"({spec.get('rows')} rows)"))
    miss_f = int(f[feature_columns].isna().sum().sum())
    miss_r = int(r[return_columns].isna().sum().sum())
    finite = bool(np.isfinite(r[return_columns].to_numpy(float)).all())
    rows.append(("No missing or non-finite Test factors / returns (no filling)", "PASS" if not miss_f and not miss_r and finite
                 else "FAIL", f"missing factors {miss_f}, missing returns {miss_r}"))
    ex = pd.to_datetime([e["date"] for e in manifest.get("excluded_dates_in_range", [])])
    gap = ex[(ex > last_model_date) & (ex <= f.index.max())]
    rows.append(("No excluded session between the last Validation date and the Test end (no omitted holding returns)",
                 "PASS" if len(gap) == 0 else "FAIL", f"{len(gap)} excluded sessions in the window"
                 + (f": {', '.join(str(d.date()) for d in gap[:5])}" if len(gap) else "")))
    rows.append(("Return definition unchanged (daily log return of adjusted closes)",
                 "PASS" if "log" in str(manifest.get("return_definition", "")).lower() and "adjusted" in
                 str(manifest.get("return_definition", "")).lower() else "FAIL", str(manifest.get("return_definition"))))
    big = (r[return_columns].abs() > 0.5).sum().sum()
    rows.append(("No implausible daily log return (|r| > 0.5) in the Test ETF returns", "PASS" if big == 0 else "WARN",
                 f"{int(big)} values"))
    return rows
