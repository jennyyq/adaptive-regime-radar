"""Project paths and a tiny .env loader (no python-dotenv dependency)."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = ROOT / "configs" / "project.json"

RAW = ROOT / "data" / "raw"
RAW_YAHOO = RAW / "yahoo"
RAW_FRED = RAW / "fred"
MANIFEST = RAW / "manifest.json"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"


def ensure_dirs() -> None:
    for directory in (RAW_YAHOO, RAW_FRED, PROCESSED, REPORTS):
        directory.mkdir(parents=True, exist_ok=True)


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Read KEY=VALUE lines from .env into os.environ without overriding existing values."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def safe_name(ticker: str) -> str:
    """File-system safe name: ^VIX -> IDX_VIX."""
    return ticker.replace("^", "IDX_").replace("/", "_").replace("=", "_")
