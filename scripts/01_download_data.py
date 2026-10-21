"""Download raw daily market data (Yahoo Finance) and macro series (FRED/ALFRED).

Writes one CSV per series to data/raw/yahoo and data/raw/fred plus data/raw/manifest.json.
Failures of optional series are reported but do not stop the run.

Examples
  python3 scripts/01_download_data.py                  # everything
  python3 scripts/01_download_data.py --source yahoo   # market data only
  python3 scripts/01_download_data.py --skip-existing  # resume after a rate limit
  python3 scripts/01_download_data.py --only SPY ^VIX DGS10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regime_radar.config import load_config, required_fred, required_yahoo, yahoo_tickers  # noqa: E402
from regime_radar.download import download_fred, download_yahoo, read_manifest, write_manifest  # noqa: E402
from regime_radar.paths import CONFIG_FILE, load_dotenv  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=CONFIG_FILE)
    parser.add_argument("--source", choices=["all", "yahoo", "fred"], default="all")
    parser.add_argument("--only", nargs="+", metavar="ID", help="subset of tickers / FRED ids")
    parser.add_argument("--start", help="override start_date (YYYY-MM-DD)")
    parser.add_argument("--end", help="override end_date, inclusive (YYYY-MM-DD)")
    parser.add_argument("--skip-existing", action="store_true", help="keep files already downloaded")
    args = parser.parse_args()

    load_dotenv()
    config = load_config(args.config)
    start = args.start or config["start_date"]
    end = args.end or config.get("end_date")
    manifest = read_manifest()
    manifest.update({"start_date": start, "end_date": end})
    failed_required: list[str] = []

    if args.source in ("all", "yahoo"):
        tickers = yahoo_tickers(config)
        if args.only:
            tickers = [t for t in tickers if t in args.only]
        results = download_yahoo(tickers, start, end, args.skip_existing)
        manifest.setdefault("yahoo", {}).update(results)
        failed_required += [t for t, r in results.items() if r["status"] == "failed" and t in required_yahoo(config)]

    if args.source in ("all", "fred"):
        series = config.get("fred_series", {})
        if args.only:
            series = {k: v for k, v in series.items() if k in args.only}
        results = download_fred(series, start, end, args.skip_existing)
        manifest.setdefault("fred", {}).update(results)
        failed_required += [s for s, r in results.items() if r["status"] == "failed" and s in required_fred(config)]

    write_manifest(manifest)
    failed = [k for src in ("yahoo", "fred") for k, r in manifest.get(src, {}).items() if r.get("status") == "failed"]
    print(f"\nManifest: data/raw/manifest.json | failed: {failed or 'none'}")
    if failed_required:
        print(f"Required series failed: {failed_required}. Rerun with --skip-existing to retry only missing files.")
        return 1
    print("Next: python3 scripts/02_build_panels.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
