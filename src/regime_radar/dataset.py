"""Stage 3: chronological train / validation / test datasets for regime modelling.

Nothing is fitted here. The module only
  * selects the usable investment sample (dates with all five factors AND all nine ETF returns),
  * assigns each date to exactly one calendar-date partition (never shuffled),
  * writes features and returns to separate files, plus a manifest and a frozen protocol.

Timing convention (inherited from the factor layer)
  factor_t  : known after the US close of day t (macro values point-in-time, z-scores past-only)
  return_t  : log(P_t / P_{t-1}), the SAME-DAY close-to-close return of each ETF
  A decision formed from factor_t can first be executed in session t+1, so it earns return_{t+1}.
  No forward-looking target is created in this stage.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from .factors import COMPONENT_CLIP, COMPONENTS, DEFAULT_MIN_PERIODS, DEFAULT_WINDOW, FACTORS, MIN_COMPONENT_SHARE

PROTOCOL_VERSION = "ARR-stage3-protocol-v1.2"
PROTOCOL_AMENDMENTS = [
    {"version": "v1.0", "change": "Initial stage-3 protocol: chronological splits, timing, frozen features."},
    {"version": "v1.1", "change": "Clarified test-period exposure (descriptive inspection vs fitting vs selection vs "
                                  "evaluation); distinguished static holdout from sequential walk-forward OOS; added "
                                  "PIT categories A-D and verified-coverage dates; strengthened prefix-invariance "
                                  "tests; added change-control rule. Splits, data and factor formulas unchanged."},
    {"version": "v1.2", "change": "Accepted the effective first train row 2007-10-30 (determined by factor "
                                  "availability, see ACCEPTED_FIRST_TRAIN_ROW). Decision taken before any model was "
                                  "fitted and without test-period outcomes. Split boundaries, data and factor "
                                  "formulas unchanged."},
]
# Effective first train row accepted by the researcher (protocol v1.2). The configured train start stays
# 2007-05-30; the first usable row is later because a factor is not yet available. Validation compares the
# data against this record and warns if a re-download changes it.
ACCEPTED_FIRST_TRAIN_ROW = {
    "date": "2007-10-30",
    "binding_factor": "liquidity_credit",
    "excluded_sessions": 106,
    "reason": "Liquidity / Credit Stress needs >= 2 of its 4 components. Before 2007-10-30 only BAA10Y is available: "
              "the HYG Amihud component starts 2007-10-30, HYG/LQD 2008-01-09, and NFCI has no real-time value before "
              "its 2011 ALFRED vintages. Dates are excluded, not filled.",
    "accepted_on": "2026-10-20",
    "accepted_before_model_fitting": True,
}
TEST_PERIOD_EXPOSURE = {
    "descriptive_historical_inspection": "occurred (factor time series and 2008/2020/2022 event-window charts)",
    "causal_feature_computation": "done; permitted",
    "model_fitting": "not done",
    "hyperparameter_selection": "not done; Train/Validation only",
    "final_oos_evaluation": "not done; 2020-latest is the designated OOS period",
}
FEATURE_COLUMNS = list(FACTORS)
RANDOM_SEED = 20240531  # fixed seed for every later stochastic model (k-means init, GMM/HMM EM starts)


@dataclass(frozen=True)
class SplitSpec:
    name: str
    start: str
    end: str | None  # None = latest completed trading day in the data


DEFAULT_SPLITS = (
    SplitSpec("train", "2007-05-30", "2015-12-31"),
    SplitSpec("validation", "2016-01-01", "2019-12-31"),
    SplitSpec("test", "2020-01-01", None),
)


def split_specs(config: dict | None = None) -> tuple[SplitSpec, ...]:
    """Splits from config['splits'] if present, else the frozen defaults; validated for order/contiguity."""
    specs = DEFAULT_SPLITS
    if config and config.get("splits"):
        specs = tuple(SplitSpec(s["name"], s["start"], s.get("end")) for s in config["splits"])
    for i, spec in enumerate(specs):
        if spec.end is not None and pd.Timestamp(spec.end) < pd.Timestamp(spec.start):
            raise ValueError(f"Split {spec.name}: end before start")
        if spec.end is None and i != len(specs) - 1:
            raise ValueError("Only the last split may have an open end")
        if i:
            previous = specs[i - 1]
            if previous.end is None or pd.Timestamp(spec.start) != pd.Timestamp(previous.end) + pd.Timedelta(days=1):
                raise ValueError(f"Splits {previous.name} and {spec.name} must be contiguous calendar ranges")
    if len({s.name for s in specs}) != len(specs):
        raise ValueError("Split names must be unique")
    return specs


def assign_partitions(dates: pd.DatetimeIndex, specs: tuple[SplitSpec, ...]) -> pd.Series:
    """Label each date with the split whose inclusive calendar range contains it (NaN if none)."""
    labels = pd.Series(pd.NA, index=dates, dtype="object")
    for spec in specs:
        end = pd.Timestamp(spec.end) if spec.end else dates.max()
        inside = (dates >= pd.Timestamp(spec.start)) & (dates <= end)
        if labels[inside].notna().any():
            raise ValueError(f"Overlapping split ranges at {spec.name}")
        labels[inside] = spec.name
    return labels


# --------------------------------------------------------------------------------------
# Loading and splitting
# --------------------------------------------------------------------------------------
@dataclass
class ModelData:
    features: pd.DataFrame          # risk_factors.csv on the SPY calendar
    returns: pd.DataFrame           # investment_log_returns.csv
    prices: pd.DataFrame            # investment_prices.csv
    calendar: pd.DatetimeIndex      # SPY master calendar (market_adj_close.csv index)


def _read_dated(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col="date", parse_dates=["date"])
    frame.index = pd.DatetimeIndex(frame.index).astype("datetime64[ns]")
    frame.index.name = "date"
    return frame


def load_model_data(processed: Path) -> ModelData:
    adj = _read_dated(processed / "market_adj_close.csv")
    return ModelData(
        features=_read_dated(processed / "risk_factors.csv"),
        returns=_read_dated(processed / "investment_log_returns.csv"),
        prices=_read_dated(processed / "investment_prices.csv"),
        calendar=adj.index,
    )


@dataclass
class SplitResult:
    specs: tuple[SplitSpec, ...]
    feature_columns: list[str]
    return_columns: list[str]
    usable_dates: pd.DatetimeIndex
    excluded: pd.DataFrame                       # date, reason (dates inside the split range not used)
    labels: pd.Series                            # usable date -> split name
    features: dict[str, pd.DataFrame] = field(default_factory=dict)
    returns: dict[str, pd.DataFrame] = field(default_factory=dict)

    def counts(self) -> dict[str, int]:
        return {name: len(frame) for name, frame in self.features.items()}


def build_splits(
    features: pd.DataFrame,
    returns: pd.DataFrame,
    specs: tuple[SplitSpec, ...],
    return_columns: list[str],
    feature_columns: list[str] = FEATURE_COLUMNS,
) -> SplitResult:
    """Align features and returns on common dates; every exclusion is recorded with a reason.

    Nothing is filled: a date is usable only if all feature and return columns are present.
    """
    missing_f = [c for c in feature_columns if c not in features.columns]
    missing_r = [c for c in return_columns if c not in returns.columns]
    if missing_f or missing_r:
        raise ValueError(f"Missing columns: features {missing_f}, returns {missing_r}")

    start = pd.Timestamp(specs[0].start)
    end = pd.Timestamp(specs[-1].end) if specs[-1].end else max(features.index.max(), returns.index.max())
    candidates = features.index.union(returns.index)
    candidates = candidates[(candidates >= start) & (candidates <= end)]

    f = features.reindex(candidates)[feature_columns]
    r = returns.reindex(candidates)[return_columns]
    reasons = pd.Series("", index=candidates, dtype="object")
    reasons[~candidates.isin(returns.index)] = "no return row (e.g. first price date: return undefined)"
    reasons[(reasons == "") & ~candidates.isin(features.index)] = "no feature row"
    reasons[(reasons == "") & r.isna().any(axis=1)] = "missing ETF return"
    missing_factor = (reasons == "") & f.isna().any(axis=1)
    reasons[missing_factor] = [
        "missing factor value: " + ", ".join(c for c in feature_columns if pd.isna(f.at[d, c]))
        for d in candidates[missing_factor.to_numpy()]
    ]
    reasons[(reasons == "") & ~np.isfinite(f.to_numpy(float)).all(axis=1)] = "non-finite factor value"

    usable = candidates[reasons == ""]
    excluded = reasons[reasons != ""].rename("reason").rename_axis("date").reset_index()
    labels = assign_partitions(usable, specs)
    if labels.isna().any():
        raise ValueError("Usable dates outside every split range")

    result = SplitResult(specs, list(feature_columns), list(return_columns), usable, excluded, labels)
    for spec in specs:
        dates = usable[(labels == spec.name).to_numpy()]
        result.features[spec.name] = f.loc[dates]
        result.returns[spec.name] = r.loc[dates]
    return result


# --------------------------------------------------------------------------------------
# Outputs: split files, manifest, protocol
# --------------------------------------------------------------------------------------
def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_split_files(result: SplitResult, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name in result.features:
        for kind, frame in (("features", result.features[name]), ("returns", result.returns[name])):
            path = out_dir / f"{name}_{kind}.csv"
            frame.to_csv(path, date_format="%Y-%m-%d", float_format="%.10g")
            written.append(path)
    return written


def _versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for package in ("numpy", "pandas", "yfinance", "matplotlib"):
        try:
            out[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            out[package] = "not installed"
    return out


def build_manifest(
    result: SplitResult,
    root: Path,
    processed: Path,
    split_files: list[Path],
    raw_manifest: dict,
    checks_summary: dict[str, int],
    factor_first_valid: dict[str, str],
) -> dict:
    src = root / "src" / "regime_radar"
    inputs = ["risk_factors.csv", "investment_log_returns.csv", "factor_components.csv",
              "macro_pit.csv", "macro_release_log.csv", "market_adj_close.csv"]
    return {
        "protocol_version": PROTOCOL_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "splits": [
            {
                **asdict(spec),
                "actual_first_date": result.features[spec.name].index.min().date().isoformat(),
                "actual_last_date": result.features[spec.name].index.max().date().isoformat(),
                "rows": len(result.features[spec.name]),
            }
            for spec in result.specs
        ],
        "total_rows": len(result.usable_dates),
        "excluded_dates_in_range": result.excluded.assign(date=result.excluded["date"].dt.strftime("%Y-%m-%d"))
        .to_dict(orient="records"),
        "feature_columns": result.feature_columns,
        "return_columns": result.return_columns,
        "return_definition": "daily log return log(P_t/P_{t-1}) from adjusted close (dividends and splits); same-day",
        "timing": {
            "signal_formation": "after the US market close of date t",
            "earliest_execution": "next trading session (t+1)",
            "feature_t_pairs_with": "return_t is contemporaneous; a strategy using feature_t earns return_{t+1}",
            "targets": "none created in this stage",
            "backtest_rule": "never evaluate a date-t decision with return_t",
        },
        "test_period_exposure": TEST_PERIOD_EXPOSURE,
        "effective_first_train_row": {**ACCEPTED_FIRST_TRAIN_ROW,
                                      "observed_in_data": result.usable_dates.min().date().isoformat(),
                                      "matches_accepted": result.usable_dates.min()
                                      == pd.Timestamp(ACCEPTED_FIRST_TRAIN_ROW["date"])},
        "protocol_amendments": PROTOCOL_AMENDMENTS,
        "feature_provenance": {
            "source": "data/processed/risk_factors.csv (scripts/04_build_factors.py)",
            "factor_construction": {
                "module": "src/regime_radar/factors.py",
                "module_sha256": file_sha256(src / "factors.py"),
                "standardization": f"past-only rolling median/MAD, window {DEFAULT_WINDOW} "
                                   f"(min {DEFAULT_MIN_PERIODS}); CPI components 756 (min 252)",
                "component_clip": COMPONENT_CLIP,
                "composite": f"equal-weight mean of signed components, >= {MIN_COMPONENT_SHARE:.0%} present",
                "components": {f: [c.name for c in COMPONENTS if c.factor == f and c.in_composite] for f in FACTORS},
                "monitor_only": [c.name for c in COMPONENTS if not c.in_composite],
            },
            "factor_first_valid_date": factor_first_valid,
            "input_sha256": {name: file_sha256(processed / name) for name in inputs},
        },
        "missing_data_policy": [
            "no forward-fill or back-fill of prices, returns or factors",
            "a date is usable only if all 5 factors and all 9 ETF returns are present and finite",
            "excluded dates inside the split range are listed in excluded_dates_in_range",
            "macro values enter only after their (first-release or rule-based) availability date",
        ],
        "warm_up": f"sessions before {result.usable_dates.min().date()} are used only for causal rolling "
                   "statistics; never an evaluation row",
        "split_files": {p.name: file_sha256(p) for p in split_files},
        "data_sources": {
            "raw_download_utc": raw_manifest.get("updated_utc"),
            "raw_download_versions": raw_manifest.get("versions"),
            "fred_sources": {k: v.get("source") for k, v in raw_manifest.get("fred", {}).items()},
        },
        "environment": _versions(),
        "random_seed_policy": {
            "seed": RANDOM_SEED,
            "rule": "every stochastic estimator uses this seed; robustness reported over seeds seed..seed+9",
        },
        "validation_summary": checks_summary,
        "command": " ".join([Path(sys.argv[0]).name] + sys.argv[1:]),
    }


def write_manifest(manifest: dict, path: Path) -> None:
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def render_protocol(result: SplitResult, manifest: dict, investment_etfs: list[str], pit_lines: list[str]) -> str:
    """Research protocol (Markdown). `pit_lines` summarise verified first-release coverage from the audit."""
    rows = "\n".join(
        f"| {s['name']} | {s['start']} | {s['end'] or 'latest'} | {s['actual_first_date']} | "
        f"{s['actual_last_date']} | {s['rows']} |" for s in manifest["splits"]
    )
    components = "\n".join(
        f"- **{FACTORS[f]}** (`{f}`): {', '.join(names)}"
        for f, names in manifest["feature_provenance"]["factor_construction"]["components"].items()
    )
    env = manifest["environment"]
    pit = "\n".join(pit_lines)
    first_row = result.usable_dates.min().date().isoformat()
    accepted = ACCEPTED_FIRST_TRAIN_ROW
    accepted_status = ("matching the current data" if first_row == accepted["date"]
                       else f"which DIFFERS from the current data ({first_row}); review before modelling")
    train_frame = result.features[result.specs[0].name]
    train_rows, train_last = len(train_frame), train_frame.index.max().date().isoformat()
    lead = result.excluded[result.excluded["date"] < result.usable_dates.min()]
    factor_gaps = lead[lead["reason"].str.startswith("missing factor value")]
    if factor_gaps.empty:
        first_row_note = "no session after the first return was excluded."
    else:
        missing = sorted(set(", ".join(factor_gaps["reason"].str.replace("missing factor value: ", "", regex=False))
                             .split(", ")))
        first_row_note = (f"{len(factor_gaps)} earlier sessions are excluded because {', '.join(missing)} "
                          "is not yet available (no fill).")
    amendments = "\n".join(f"| {a['version']} | {a['change']} |" for a in PROTOCOL_AMENDMENTS)
    return f"""# Research protocol: Adaptive Regime Radar

Protocol version: **{PROTOCOL_VERSION}** (generated {manifest['created_utc'][:10]}).
This document fixes the data, timing, evaluation and change-control rules before any regime model is fitted.

## 1. Universe and data

- Investment universe (9 ETFs): {', '.join(investment_etfs)}. The universe was chosen ex post; all nine funds
  survive to today (a design-level limitation, disclosed rather than corrected).
- Master calendar: SPY trading dates. Sessions before the first train row are warm-up, used only for causal
  rolling statistics.
- Sources: Yahoo Finance (adjusted daily prices via yfinance {(manifest['data_sources']['raw_download_versions'] or {}).get('yfinance', 'unknown')}),
  FRED / ALFRED (sources per series in `data/splits/split_manifest.json`).
  Raw download time (UTC): {manifest['data_sources']['raw_download_utc']}.
- Environment: Python {env['python']}, numpy {env['numpy']}, pandas {env['pandas']}, matplotlib {env['matplotlib']}.

## 2. Features (model inputs, frozen)

Five composite factors from `src/regime_radar/factors.py` (sha256 `{manifest['feature_provenance']['factor_construction']['module_sha256']}`):

{components}

Construction is frozen: past-only rolling median/MAD z-scores (252 days; CPI components 756), economic signs,
clip +/-5, equal weights. No additional feature engineering, feature selection, PCA or full-sample
standardisation is part of the dataset. Any later scaler or projection is a model component, fitted on
training information only.

## 3. Splits

| split | configured start | configured end | first date | last date | rows |
|---|---|---|---|---|---|
{rows}

- Chronological, mutually exclusive, collectively exhaustive over the usable sample; never shuffled.
- The train split is configured from 2007-05-30, the first date on which all nine ETFs have a price (BIL is the
  binding fund). The first return is the next session, because a return requires the previous close. The first
  train row is {first_row}, the first session on which a return exists and all five factors are available;
  {first_row_note}
- Accepted effective train start (protocol v1.2): **{accepted['date']}**, {accepted_status}. Reason:
  {accepted['reason']} The researcher accepted this on {accepted['accepted_on']}, before any model was fitted.
  The effective train sample is {train_rows} rows ({first_row} to {train_last}).
- Rolling statistics are not restarted at split boundaries: earlier partitions (and the warm-up) may inform
  later dates, never the reverse. Prefix-invariance tests in `reports/dataset_split_audit.md` check this.

## 4. Timing convention

| quantity | definition | known at |
|---|---|---|
| feature_t | factor value from prices up to the close of t and macro values published by t | after the US close of t |
| return_t | log(P_t / P_t-1), adjusted close | after the US close of t (historical by then) |
| decision_t | position chosen from feature_t | after the close of t; earliest execution in session t+1 |
| earned return | return_(t+1) for close-to-close execution of decision_t | after the close of t+1 |

return_t is not a future outcome relative to feature_t: both are known at the close of t. A backtest must
never credit decision_t with return_t. Execution timing other than next-close (e.g. next open, weekly
rebalancing) must be specified before testing and applied identically to every strategy. No forward-return
labels are created in stage 3.

## 5. Information exposure and leakage taxonomy (status at this version)

| stage | description | status for the 2020-latest test period |
|---|---|---|
| 1. Descriptive historical inspection | looking at historical factor values, charts, summary tables | **Occurred**: factor time series and the 2008 / 2020 / 2022 event-window charts specified in the original research plan were viewed. |
| 2. Causal feature computation | computing features with rolling, past-only transformations on all dates | **Done; permitted.** Values on test dates use only information available at each date. |
| 3. Model fitting | estimating any parameter of a regime or portfolio model | **Not done.** No K-Means, GMM, HMM or other model has been fitted. |
| 4. Hyperparameter / specification selection | choosing model class, number of states, windows, thresholds | **Not done.** Must use Train / Validation evidence only. |
| 5. Final out-of-sample evaluation | measuring investment performance of the frozen methodology | **Not done.** 2020-latest remains the designated out-of-sample period. |

Stage-1 exposure also includes the descriptive statistics in `reports/factor_diagnostics.md` (factor
correlations, largest readings and event-window summaries computed over the full sample, including 2020+).
None of these uses asset returns or strategy outcomes.

Descriptive inspection (stage 1) is not by itself look-ahead bias, training leakage or test-set optimisation:
no parameter was estimated and no choice was scored on test-period outcomes. It does mean that the test period
is not a never-inspected, blind holdout, and the final report must say so. The factor definitions were written
before the first real-data run of `scripts/04_build_factors.py` and have not been changed in response to its
output; this ordering is the researcher's attestation and is not machine-verifiable (the hash above records the
current module, not its history).

## 6. Evaluation design

**Static holdout evaluation** fits the selected specification once on information up to 2019-12-31 and
applies it unchanged to 2020-latest.

**Sequential walk-forward out-of-sample evaluation** re-estimates the model at each simulated decision date
using only observations available before that decision. A model deciding in 2023 may therefore be estimated
on data that include 2020-2022. This is permitted, and is the primary evaluation in the original research
plan (rolling window of about five years, monthly re-estimation; exact parameters to be frozen in stage 4
using Train / Validation), provided that:

1. the estimation scheme (expanding or rolling window, window length, refit frequency, initialisation,
   seeds, label-matching rule) is fixed before any test-period outcome is computed;
2. hyperparameter-selection procedures are designed and frozen using Train / Validation only; if a selection
   rule is re-run inside walk-forward, it is the same pre-specified rule applied to past data only;
3. no observation dated after the decision (or not yet published at that time) enters the estimate;
4. no part of the methodology is changed in response to out-of-sample results.

Prohibited at all times: choosing or revising factors, components, weights, windows, model class, number of
regimes, thresholds, constraints or execution rules because of test-period performance. If a defect is found
after test results are seen, the correction, its motivation and both sets of results must be reported, and
the corrected run is no longer a clean out-of-sample result.

Supervised targets, if introduced later, require purging of overlapping target horizons (and an embargo equal
to the horizon) in cross-validation. No arbitrary 252- or 756-day embargo is imposed for unsupervised daily
regime estimation.

## 7. Point-in-time status of macro inputs

Categories: A = first-release value with ALFRED vintage date; B = first-release value with rule-based date;
C = later-vintage (revised or retrospectively estimated) value with rule-based date; D = daily market series
used one US business day after observation, current vintage.

{pit}

- Daily FRED series (Treasury yields, breakevens, BAA10Y, HY OAS) are category D: a conservative timing rule,
  not a recorded publication timestamp.
- Category C values satisfy the timing rule but are not verifiable as point-in-time; a valid release-date lag
  does not show that a revised value was historically available. NFCI history before its 2011 real-time
  releases is a retrospective estimate.
- Committed sensitivity analysis: the selected model will also be estimated on train dates from the latest
  category-A start above, and with NFCI / CPI components excluded where they are category C; differences
  will be reported.

## 8. Missing data

No forward- or back-filling. A date enters a partition only if all five factors and all nine ETF returns are
present and finite; excluded dates are listed in the manifest. A factor value requires at least half of its
components on that date; component-set changes over time are documented in the audit report.

## 9. Random seeds

All stochastic estimators use seed **{RANDOM_SEED}**; stability is reported across seeds {RANDOM_SEED}-{RANDOM_SEED + 9}.

## 10. Change control

| version | change |
|---|---|
{amendments}

Amendments may clarify documentation or add validation. Any change to splits, data, factor formulas or
evaluation rules requires a new version entry stating whether test-period information was seen before the
change.
"""
