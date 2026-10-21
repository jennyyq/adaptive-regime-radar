"""Build initial market features from the processed adjusted-close panel."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regime_radar.features import build_initial_features  # noqa: E402
from regime_radar.paths import PROCESSED  # noqa: E402


def main() -> int:
    source = PROCESSED / "market_adj_close.csv"
    if not source.exists():
        raise SystemExit("Run scripts/01_download_data.py and scripts/02_build_panels.py first")
    close = pd.read_csv(source, index_col="date", parse_dates=["date"])
    features = build_initial_features(close)
    output = PROCESSED / "initial_features.csv"
    features.to_csv(output)
    print(f"Saved {len(features)} dates x {len(features.columns)} columns to {output.relative_to(ROOT)}")
    print("Initial features only; the five-factor composites and regime models come next")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
