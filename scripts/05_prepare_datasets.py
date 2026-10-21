"""Stage 3: build chronological train / validation / test datasets and validate them. Nothing is fitted.

Inputs  (data/processed/, from scripts 02 and 04)
  risk_factors.csv, factor_components.csv, investment_log_returns.csv, investment_prices.csv,
  market_adj_close.csv, market_close.csv, market_volume.csv, macro_pit.csv,
  macro_pit_obs_date.csv, macro_release_log.csv
Nothing is fitted and no forward-return label is created.
Outputs
  data/splits/{train,validation,test}_{features,returns}.csv
  data/splits/split_manifest.json
  reports/dataset_split_audit.md, reports/dataset_validation.csv, reports/research_protocol.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regime_radar import dataset_validation as dv  # noqa: E402
from regime_radar.config import load_config  # noqa: E402
from regime_radar.dataset import (  # noqa: E402
    build_manifest, build_splits, load_model_data, render_protocol, split_specs, write_manifest, write_split_files,
)
from regime_radar.factors import load_factor_inputs  # noqa: E402
from regime_radar.paths import CONFIG_FILE, MANIFEST, PROCESSED, REPORTS  # noqa: E402

SPLITS_DIR = ROOT / "data" / "splits"


def dv_read(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col="date", parse_dates=["date"])
    frame.index = frame.index.astype("datetime64[ns]")
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=CONFIG_FILE)
    args = parser.parse_args()

    needed = ["risk_factors.csv", "factor_components.csv", "investment_log_returns.csv", "investment_prices.csv",
              "market_adj_close.csv", "macro_pit.csv", "macro_pit_obs_date.csv", "macro_release_log.csv"]
    missing = [n for n in needed if not (PROCESSED / n).exists()]
    if missing:
        raise SystemExit(f"Missing {missing}. Run scripts 02 and 04 first.")

    config = load_config(args.config)
    etfs = config["investment_etfs"]
    specs = split_specs(config)
    data = load_model_data(PROCESSED)
    result = build_splits(data.features, data.returns, specs, etfs)

    components = dv_read(PROCESSED / "factor_components.csv")
    obs_dates = dv_read(PROCESSED / "macro_pit_obs_date.csv")
    macro_pit = dv_read(PROCESSED / "macro_pit.csv")
    adj_close = dv_read(PROCESSED / "market_adj_close.csv")
    initial_path = PROCESSED / "initial_features.csv"
    initial = dv_read(initial_path) if initial_path.exists() else None
    release_log = pd.read_csv(PROCESSED / "macro_release_log.csv", parse_dates=["date", "available_date"])
    factor_inputs = load_factor_inputs(PROCESSED, config.get("auxiliary_tickers", {}).get("sector_breadth", []))
    raw_manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    fred_sources = {k: v.get("source", "unknown") for k, v in raw_manifest.get("fred", {}).items()}
    train_name = specs[0].name
    train_dates = result.features[train_name].index

    # ---------------- A/B. integrity, alignment, fill ----------------
    checks = [dv.check_index(n, f) for n, f in (("risk_factors", data.features), ("investment_log_returns",
              data.returns), ("investment_prices", data.prices), ("factor_components", components))]
    checks += dv.check_partitions(result)
    checks += dv.check_schema(result)
    checks += dv.check_alignment(result, data.calendar)
    checks += dv.check_first_eligible_return(adj_close, data.prices, data.returns, result, data.calendar, etfs)
    checks += dv.check_returns_reproducible(data.prices, data.returns, etfs)
    checks.append(dv.check_no_backfill(data.features, components))

    # ---------------- C. causality on the actual inputs ----------------
    def session_on_or_before(date: str) -> pd.Timestamp | None:
        eligible = data.calendar[data.calendar <= pd.Timestamp(date)]
        return eligible[-1] if len(eligible) and eligible[-1] > data.calendar[300] else None

    boundaries = [result.features[s.name].index.max() for s in specs[:-1]]
    interior = [session_on_or_before(d) for d in ("2006-06-30", "2011-12-30", "2017-12-29", "2023-06-30")]
    truncation_cutoffs = sorted({c for c in boundaries + interior if c is not None and c < data.calendar.max()})
    print(f"Prefix-invariance rebuilds: {len(truncation_cutoffs)} truncations + {len(boundaries)} perturbations ...")
    prefix_checks, prefix_table = dv.prefix_invariance_check(factor_inputs, data.features, components,
                                                             truncation_cutoffs, boundaries)
    checks += prefix_checks
    checks.append(dv.initial_features_prefix_check(adj_close, initial, truncation_cutoffs))
    macro_checks, macro_prefix_table = dv.macro_prefix_check(release_log, data.calendar, macro_pit, truncation_cutoffs)
    checks += macro_checks

    # ---------------- D. point-in-time audit ----------------
    pit_checks, pit_table, pit_verified = dv.macro_pit_audit(obs_dates, release_log, result.labels,
                                                             config["fred_series"], fred_sources)
    checks += pit_checks

    # ---------------- E. composition stability ----------------
    periods = {"warm-up": (data.calendar.min(), result.usable_dates.min() - pd.Timedelta(days=1))}
    periods.update({s.name: (result.features[s.name].index.min(), result.features[s.name].index.max())
                    for s in specs})
    composition = dv.composition_by_period(components, periods)
    changes = dv.composition_changes(components, result.usable_dates, data.features, result.labels, train_dates)
    sensitivity = dv.composition_sensitivity(components, data.features, train_dates)
    component_avail = dv.component_availability(components, result.labels)
    checks += dv.check_composition(sensitivity, changes)

    # ---------------- F. train-only redundancy ----------------
    redundancy_checks, redundancy = dv.train_redundancy(result.features[train_name])
    checks += redundancy_checks

    availability = dv.feature_availability(data.features, result.labels, components)
    hy = components["liquidity_credit__hy_oas"].reindex(result.usable_dates)
    hy_oas_coverage = pd.DataFrame({
        "coverage_pct": hy.groupby(result.labels).apply(lambda s: round(100 * s.notna().mean(), 1)),
    }).reindex([s.name for s in specs])
    hy_oas_coverage.index.name = "split"

    # ---------------- outputs ----------------
    split_files = write_split_files(result, SPLITS_DIR)
    first_valid = {f: data.features[f].first_valid_index().date().isoformat() for f in result.feature_columns}
    manifest = build_manifest(result, ROOT, PROCESSED, split_files, raw_manifest, dv.summarize(checks), first_valid)
    write_manifest(manifest, SPLITS_DIR / "split_manifest.json")

    protocol = render_protocol(result, manifest, etfs, dv.pit_protocol_summary(pit_verified))
    (REPORTS / "research_protocol.md").write_text(protocol, encoding="utf-8")
    pd.DataFrame([vars(c) for c in checks]).to_csv(REPORTS / "dataset_validation.csv", index=False)
    report = dv.render_audit_report(result, data.calendar, checks, availability, pit_table, pit_verified,
                                    hy_oas_coverage, component_avail, composition, changes, sensitivity,
                                    redundancy, prefix_table, macro_prefix_table, manifest)
    (REPORTS / "dataset_split_audit.md").write_text(report, encoding="utf-8")

    # ---------------- console summary ----------------
    print(f"Usable investment sample: {result.usable_dates.min().date()} -> {result.usable_dates.max().date()} "
          f"({len(result.usable_dates)} rows)")
    for s in manifest["splits"]:
        print(f"  {s['name']:<10} {s['actual_first_date']} -> {s['actual_last_date']}  rows {s['rows']:>5}")
    if not result.excluded.empty:
        for reason, group in result.excluded.groupby("reason", sort=False):
            span = f"{group['date'].min().date()}" + (f"..{group['date'].max().date()}" if len(group) > 1 else "")
            print(f"  excluded {len(group):>4} session(s) {span}: {reason}")
    print("Checks:")
    for c in checks:
        print(f"  [{c.status}] {c.name}: {c.detail}")
    summary = dv.summarize(checks)
    print("Summary: " + ", ".join(f"{summary[s]} {s}" for s in dv.STATUSES)
          + " (PASS = validated on the local files hashed in split_manifest.json)")
    print("Saved data/splits/*.csv, data/splits/split_manifest.json")
    print("Saved reports/dataset_split_audit.md, reports/dataset_validation.csv, reports/research_protocol.md")
    return 1 if summary["FAIL"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
