"""Build aligned panels and the data-quality report from data/raw.

Outputs (data/processed/)
  market_<field>.csv            date x ticker for open/high/low/close/adj_close/volume (no filling)
  investment_prices.csv         9 ETFs, adjusted, from the first date all trade
  investment_log_returns.csv    daily log returns, missing kept as NaN
  macro_pit.csv                 FRED values as known on each trading day
  macro_pit_obs_date.csv        which observation each macro value refers to (audit trail)
  macro_release_log.csv         every observation with its availability date and method
Reports (reports/)
  data_quality.md, yahoo_coverage.csv, fred_coverage.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regime_radar.config import load_config, required_yahoo, yahoo_tickers  # noqa: E402
from regime_radar.download import load_fred_raw, load_yahoo_raw, read_manifest  # noqa: E402
from regime_radar.panels import investment_panel, macro_panel, market_tables, trading_calendar  # noqa: E402
from regime_radar.paths import CONFIG_FILE, PROCESSED, REPORTS, ensure_dirs  # noqa: E402
from regime_radar.quality import fred_quality, markdown_report, yahoo_quality  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=CONFIG_FILE)
    args = parser.parse_args()

    ensure_dirs()
    config = load_config(args.config)
    raw = load_yahoo_raw(yahoo_tickers(config))
    missing_required = sorted(required_yahoo(config) - set(raw))
    if missing_required:
        raise SystemExit(f"Missing raw files for {missing_required}. Run scripts/01_download_data.py first.")

    calendar = trading_calendar(raw, config.get("calendar_ticker", "SPY"), config["start_date"])
    tables = market_tables(raw, calendar)
    for field, table in tables.items():
        table.to_csv(PROCESSED / f"market_{field}.csv")

    prices, log_returns = investment_panel(tables["adj_close"], config["investment_etfs"])
    prices.to_csv(PROCESSED / "investment_prices.csv")
    log_returns.to_csv(PROCESSED / "investment_log_returns.csv")

    specs = config.get("fred_series", {})
    fred_raw = load_fred_raw(list(specs))
    values, obs_dates, release_log = macro_panel(fred_raw, specs, calendar)
    values.to_csv(PROCESSED / "macro_pit.csv")
    obs_dates.to_csv(PROCESSED / "macro_pit_obs_date.csv")
    if not release_log.empty:
        release_log.to_csv(PROCESSED / "macro_release_log.csv", index=False, date_format="%Y-%m-%d")

    yahoo_q = yahoo_quality(raw, calendar, required_yahoo(config))
    fred_q = fred_quality(release_log, obs_dates, specs, calendar)
    yahoo_q.to_csv(REPORTS / "yahoo_coverage.csv")
    fred_q.to_csv(REPORTS / "fred_coverage.csv")
    report = markdown_report(yahoo_q, fred_q, prices, log_returns, read_manifest())
    (REPORTS / "data_quality.md").write_text(report, encoding="utf-8")

    print(f"Trading calendar ({config.get('calendar_ticker', 'SPY')}): "
          f"{calendar.min().date()} -> {calendar.max().date()}, {len(calendar)} days")
    print(f"Investment common sample: {prices.index.min().date()} -> {prices.index.max().date()} "
          f"({len(prices)} rows); missing returns: {int(log_returns.isna().sum().sum())}")
    print(f"Macro point-in-time panel: {values.shape[1]} series")
    print("Saved data/processed/* and reports/data_quality.md")
    print("Next: python3 scripts/03_build_features.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
