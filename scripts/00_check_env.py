"""Check the Python interpreter, packages, FRED key and network access before downloading."""

from __future__ import annotations

import os
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regime_radar.paths import load_dotenv  # noqa: E402

REQUIRED = {"numpy": "1.26", "pandas": "2.1", "yfinance": "0.2.54", "matplotlib": "3.7"}


def version_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(".")[:3] if part.isdigit())


def main() -> int:
    ok = True
    print(f"Python      : {sys.version.split()[0]}  ({sys.executable})")
    if sys.version_info < (3, 10):
        print("  !! Python 3.10+ is required")
        ok = False

    for package, minimum in REQUIRED.items():
        try:
            installed = metadata.version(package)
            flag = "" if version_tuple(installed) >= version_tuple(minimum) else f"  !! need >= {minimum}"
            ok &= not flag
            print(f"{package:<12}: {installed}{flag}")
        except metadata.PackageNotFoundError:
            print(f"{package:<12}: MISSING  -> {sys.executable} -m pip install {package}")
            ok = False

    load_dotenv()
    key = os.environ.get("FRED_API_KEY")
    print(f"FRED key    : {'found' if key else 'not set (optional; enables first-release vintages)'}")

    if "--offline" in sys.argv or not ok:
        return 0 if ok else 1

    print("\nNetwork checks ...")
    try:
        from regime_radar.sources.yahoo import download_ticker

        frame = download_ticker("SPY", "2024-01-02", "2024-01-31", retries=2)
        print(f"Yahoo       : OK ({len(frame)} SPY rows for Jan 2024)")
    except Exception as exc:
        print(f"Yahoo       : FAILED ({exc})")
        ok = False
    try:
        from regime_radar.sources.fred import download_series

        frame, source = download_series("DGS10", {"availability": {"rule": "business_days"}}, "2024-01-01", "2024-01-31")
        print(f"FRED        : OK ({len(frame)} DGS10 rows via {source})")
    except Exception as exc:
        print(f"FRED        : FAILED ({exc})")
        ok = False

    print("\nReady." if ok else "\nFix the items marked above, then rerun.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
