"""Build the five economic risk factors from the processed panels, validate them, and report.

Inputs  (data/processed/, produced by 02_build_panels.py)
  market_adj_close.csv, market_close.csv, market_volume.csv,
  macro_pit.csv, macro_pit_obs_date.csv, macro_release_log.csv, investment_prices.csv
Outputs
  data/processed/risk_factors.csv             five composite factors
  data/processed/factor_components.csv        signed, winsorized robust z per component
  data/processed/factor_components_raw.csv    raw component signals (natural units)
  reports/factor_diagnostics.md
  reports/factor_validation.csv
  reports/figures/five_factor_timeseries.png, factor_correlation_heatmap.png, factor_event_windows.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regime_radar.config import load_config  # noqa: E402
from regime_radar.factor_diagnostics import markdown_report, save_figures  # noqa: E402
from regime_radar.factors import FACTORS, build_factors, load_factor_inputs, validate  # noqa: E402
from regime_radar.paths import CONFIG_FILE, PROCESSED, REPORTS  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=CONFIG_FILE)
    parser.add_argument("--no-figures", action="store_true", help="skip matplotlib figures")
    args = parser.parse_args()

    config = load_config(args.config)
    required = ["market_adj_close.csv", "market_close.csv", "market_volume.csv", "investment_prices.csv"]
    missing = [name for name in required if not (PROCESSED / name).exists()]
    if missing:
        raise SystemExit(f"Missing {missing}. Run scripts/02_build_panels.py first.")

    sectors = config.get("auxiliary_tickers", {}).get("sector_breadth", [])
    inputs = load_factor_inputs(PROCESSED, sectors)
    raw, components, factors = build_factors(inputs)

    sample_start = pd.read_csv(PROCESSED / "investment_prices.csv", index_col="date",
                               parse_dates=["date"]).index.min()
    obs_path = PROCESSED / "macro_pit_obs_date.csv"
    obs_dates = pd.read_csv(obs_path, index_col="date", parse_dates=["date"]) if obs_path.exists() else None
    daily_series = [s for s, spec in config.get("fred_series", {}).items() if spec.get("frequency") == "daily"]
    checks = validate(inputs, raw, components, factors, obs_dates, daily_series, sample_start)

    factors.to_csv(PROCESSED / "risk_factors.csv", float_format="%.6f")
    components.to_csv(PROCESSED / "factor_components.csv", float_format="%.6f")
    raw.to_csv(PROCESSED / "factor_components_raw.csv", float_format="%.8g")
    pd.DataFrame([vars(c) for c in checks]).to_csv(REPORTS / "factor_validation.csv", index=False)
    report = markdown_report(raw, components, factors, checks, sample_start, inputs.release_log)
    (REPORTS / "factor_diagnostics.md").write_text(report, encoding="utf-8")

    figures = []
    if not args.no_figures:
        try:
            figures = save_figures(factors.loc[sample_start:], REPORTS / "figures")
        except ImportError:
            print("matplotlib not installed: figures skipped "
                  f"({sys.executable} -m pip install matplotlib)")

    print(f"Factors: {factors.index.min().date()} -> {factors.index.max().date()}, {len(factors)} dates")
    in_sample = factors.loc[sample_start:]
    for f, label in FACTORS.items():
        first = factors[f].first_valid_index()
        print(f"  {label:<27} first valid {first.date() if first is not None else 'never'}  "
              f"in-sample coverage {in_sample[f].notna().mean():6.1%}")
    print("Validation:")
    for c in checks:
        print(f"  [{c.status}] {c.name}: {c.detail}")
    print("Saved data/processed/risk_factors.csv, factor_components.csv, factor_components_raw.csv")
    print("Saved reports/factor_diagnostics.md, reports/factor_validation.csv"
          + (", reports/figures/*.png" if figures else ""))
    return 1 if any(c.status == "FAIL" for c in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
