"""Pre-modelling validation for the chronological splits (stage 3).

Every check returns `Check(name, status, detail)`. Status semantics:

  PASS     validated against the data actually passed in (when run by scripts/05, the local dataset
           whose SHA-256 hashes are recorded in data/splits/split_manifest.json)
  WARN     methodological concern, or a historical assumption that cannot be verified from the data
  FAIL     confirmed inconsistency or validation failure
  NOT RUN  the validation could not be executed against the real dataset (missing input or data)

A PASS produced by the unit tests on synthetic data is NOT a validation of the real dataset, and no
finite set of checks proves the absence of every possible form of leakage. Test-period information
used here is limited to integrity, coverage, availability and alignment; no performance statistics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .dataset import ACCEPTED_FIRST_TRAIN_ROW, FEATURE_COLUMNS, SplitResult
from .factors import COMPONENTS, FACTORS, Check, FactorInputs, build_factors
from .features import build_initial_features
from .pit import align_to_calendar

PASS, WARN, FAIL, NOT_RUN = "PASS", "WARN", "FAIL", "NOT RUN"
STATUSES = (PASS, WARN, FAIL, NOT_RUN)
STATUS_DEFINITIONS = {
    PASS: "Validated against the data passed to the check (for scripts/05: the local dataset with the hashes in "
          "split_manifest.json).",
    WARN: "Methodological concern or historical assumption that cannot be verified from the available data.",
    FAIL: "Confirmed inconsistency or validation failure.",
    NOT_RUN: "Could not be executed against the real dataset (input or historical data not available).",
}

SERIALIZATION_TOLERANCE = 1e-6   # risk_factors / factor_components CSVs keep 6 decimals (max rounding 5e-7)
EXACT_TOLERANCE = 1e-10          # in-memory rebuilds must agree to floating-point precision
DOCUMENTED_FIRST_RETURN = "2007-05-31"

# Which macro series feed which composite component (for the PIT audit)
MACRO_USAGE = {
    "DGS2": "interest_rate: ust2y_change_63d",
    "DGS10": "interest_rate: ust10y_change_63d, ust10y_realized_vol_21d",
    "T10Y2Y": "interest_rate: curve_10y2y",
    "T5YIE": "inflation: breakeven_5y",
    "T10YIE": "inflation: breakeven_5y (fallback only)",
    "CPIAUCSL": "inflation: cpi_yoy",
    "CPILFESL": "inflation: core_cpi_yoy",
    "BAA10Y": "liquidity_credit: baa_10y_spread",
    "NFCI": "liquidity_credit: nfci",
    "BAMLH0A0HYM2": "monitor only (hy_oas)",
    "DGS3MO": "not used",
    "T10Y3M": "not used",
    "DFII10": "not used",
}

PIT_CATEGORIES = {
    "A": "First-release value; availability date = ALFRED first-vintage date (date-level, verified).",
    "B": "First-release value; availability date approximated by a publication rule.",
    "C": "Value from a later vintage (revised, or retrospectively estimated before the series' first vintage); "
         "availability date approximated by a publication rule. Not verifiable as point-in-time.",
    "D": "Daily market series; availability = observation date + 1 US business day (conservative rule); "
         "current-vintage value; no publication timestamp recorded.",
}


def pit_category(method: str) -> str:
    if method == "vintage:first_release":
        return "A"
    if method == "fallback:missing_vintage":
        return "B"
    if method in ("fallback:pre_vintage_history", "fallback:no_vintage_data"):
        return "C"
    if method == "rule:business_days":
        return "D"
    if isinstance(method, str) and method.startswith("rule:"):
        return "C"  # non-daily series on a rule with current-vintage values
    return "unclassified"


def summarize(checks: list[Check]) -> dict[str, int]:
    out = {s: 0 for s in STATUSES}
    for c in checks:
        out[c.status] = out.get(c.status, 0) + 1
    return out


def _cols(factor: str, composite_only: bool = True) -> list[str]:
    return [f"{c.factor}__{c.name}" for c in COMPONENTS if c.factor == factor and (c.in_composite or not composite_only)]


# --------------------------------------------------------------------------------------
# A. Index, partition, schema and alignment integrity
# --------------------------------------------------------------------------------------
def check_index(name: str, frame: pd.DataFrame) -> Check:
    idx = frame.index
    problems = []
    if not isinstance(idx, pd.DatetimeIndex):
        problems.append("index is not a DatetimeIndex")
    else:
        if idx.tz is not None:
            problems.append("timezone-aware index")
        if not idx.is_monotonic_increasing:
            problems.append("not monotonic increasing")
        if idx.has_duplicates:
            problems.append(f"{int(idx.duplicated().sum())} duplicated dates")
        if (idx != idx.normalize()).any():
            problems.append("intraday timestamps present")
    return Check(f"Index integrity: {name}", FAIL if problems else PASS,
                 "; ".join(problems) or f"{len(idx)} unique, increasing, date-only timestamps")


def check_partitions(result: SplitResult) -> list[Check]:
    checks = []
    frames = [result.features[s.name] for s in result.specs]
    all_dates = pd.DatetimeIndex(np.concatenate([f.index.values for f in frames]))
    overlap = int(all_dates.duplicated().sum())
    checks.append(Check("Partitions mutually exclusive", PASS if overlap == 0 else FAIL,
                        f"{overlap} dates appear in more than one partition"))
    union = pd.DatetimeIndex(sorted(set(all_dates)))
    missing = result.usable_dates.difference(union)
    extra = union.difference(result.usable_dates)
    checks.append(Check("Partitions collectively exhaustive over usable sample",
                        PASS if missing.empty and extra.empty else FAIL,
                        f"{len(result.usable_dates)} usable dates; {len(missing)} unassigned; {len(extra)} unexpected"))
    ordered = all(frames[i].index.max() < frames[i + 1].index.min() for i in range(len(frames) - 1))
    within = all(
        f.index.min() >= pd.Timestamp(s.start) and (s.end is None or f.index.max() <= pd.Timestamp(s.end))
        for f, s in zip(frames, result.specs)
    )
    checks.append(Check("Partitions chronological and inside configured boundaries",
                        PASS if ordered and within else FAIL,
                        " < ".join(f"{s.name} [{f.index.min().date()}..{f.index.max().date()}]"
                                   for f, s in zip(frames, result.specs))))
    empty = [s.name for s, f in zip(result.specs, frames) if f.empty]
    checks.append(Check("No empty partition", PASS if not empty else FAIL, f"empty: {empty or 'none'}"))
    return checks


def check_schema(result: SplitResult) -> list[Check]:
    f_cols = {name: list(f.columns) for name, f in result.features.items()}
    r_cols = {name: list(r.columns) for name, r in result.returns.items()}
    same_f = all(cols == result.feature_columns for cols in f_cols.values())
    same_r = all(cols == result.return_columns for cols in r_cols.values())
    return [
        Check("Identical factor schema across splits (no target / forward-return columns)", PASS if same_f else FAIL,
              f"{len(result.feature_columns)} columns: {', '.join(result.feature_columns)}"),
        Check("Identical ETF return schema across splits", PASS if same_r else FAIL,
              f"{len(result.return_columns)} columns: {', '.join(result.return_columns)}"),
    ]


def check_alignment(result: SplitResult, calendar: pd.DatetimeIndex) -> list[Check]:
    checks = []
    mismatched = [n for n in result.features if not result.features[n].index.equals(result.returns[n].index)]
    checks.append(Check("Feature and return dates identical within each split (contemporaneous storage)",
                        PASS if not mismatched else FAIL, f"mismatched: {mismatched or 'none'}"))
    off_calendar = result.usable_dates.difference(calendar)
    checks.append(Check("All split dates are SPY master-calendar sessions",
                        PASS if off_calendar.empty else FAIL, f"{len(off_calendar)} off-calendar dates"))
    first, last = result.usable_dates.min(), result.usable_dates.max()
    span = calendar[(calendar >= first) & (calendar <= last)]
    gaps = span.difference(result.usable_dates)
    checks.append(Check("No master-calendar session missing inside the usable span",
                        PASS if gaps.empty else WARN,
                        f"{len(gaps)} sessions missing" + (f" (first: {gaps[0].date()})" if len(gaps) else "")))
    excluded = result.excluded
    reasons = excluded["reason"].value_counts().to_dict() if not excluded.empty else {}
    unexplained = int((excluded["reason"] == "").sum()) if not excluded.empty else 0
    interior = int((excluded["date"] > first).sum()) if not excluded.empty else 0  # gaps after the first usable row
    checks.append(Check("Excluded dates inside split range are explained (leading block only)",
                        PASS if not unexplained and not interior else WARN,
                        "; ".join(f"{k}: {v}" for k, v in reasons.items()) or "none"))
    frames = list(result.features.values()) + list(result.returns.values())
    bad = int(sum((~np.isfinite(frame.to_numpy(float))).sum() for frame in frames))
    checks.append(Check("No NaN / +/-inf in split files", PASS if bad == 0 else FAIL, f"{bad} non-finite values"))
    return checks


def check_first_eligible_return(
    adj_close: pd.DataFrame,
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    result: SplitResult,
    calendar: pd.DatetimeIndex,
    etfs: list[str],
    documented: str = DOCUMENTED_FIRST_RETURN,
    accepted_first_row: str = ACCEPTED_FIRST_TRAIN_ROW["date"],
) -> list[Check]:
    """Two separate facts:

    1. return convention: first return = session after the first common price date (BIL-bound);
    2. first usable modelling row: the first date where all five factors AND all returns exist. It may be
       later than the first return if a factor is still warming up. That is a sample-length decision, not an
       inconsistency: PASS when it equals the start accepted in the protocol, WARN when the data disagree
       with the accepted start (e.g. after a re-download), FAIL only if it precedes the first return.
    """
    first_listing = {t: adj_close[t].first_valid_index() for t in etfs if t in adj_close}
    binding_date = max(first_listing.values())
    binding = [t for t, d in first_listing.items() if d == binding_date]
    first_price = prices.index.min()
    later = calendar[calendar > first_price]
    expected = later[0] if len(later) else pd.NaT
    first_return, first_usable = returns.index.min(), result.usable_dates.min()

    detail = (f"first common price date {first_price.date()} (binding ETF: {', '.join(binding)}); "
              f"first return {first_return.date()} (next session {expected.date()})")
    if first_price != binding_date or first_return != expected:
        ret = Check("First eligible return date", FAIL, detail)
    elif first_return != pd.Timestamp(documented):
        ret = Check("First eligible return date", WARN, detail + f"; differs from protocol-documented {documented}")
    else:
        ret = Check("First eligible return date", PASS, detail + f"; matches protocol-documented {documented}")

    name = "First usable modelling row (all factors and returns present)"
    accepted = pd.Timestamp(accepted_first_row)
    if first_usable < first_return:
        return [ret, Check(name, FAIL, f"first usable row {first_usable.date()} precedes the first return")]
    if first_usable == first_return:
        detail = f"{first_usable.date()} = first return date"
    else:
        gap = result.excluded[(result.excluded["date"] >= first_return) & (result.excluded["date"] < first_usable)]
        missing = (gap["reason"].str.replace("missing factor value: ", "", regex=False)
                   .str.split(", ").explode().value_counts())
        detail = (f"{first_usable.date()}: {len(gap)} sessions after the first return "
                  f"({first_return.date()}..{(first_usable - pd.Timedelta(days=1)).date()}) are excluded because "
                  + ", ".join(f"{k} is missing on {v} sessions" for k, v in missing.items()))
    if first_usable == accepted:
        usable = Check(name, PASS, detail + f"; matches the effective train start accepted in the protocol "
                                            f"({accepted.date()}); configured boundary unchanged")
    else:
        usable = Check(name, WARN, detail + f"; DIFFERS from the accepted effective train start {accepted.date()} "
                                            "- review before modelling")
    return [ret, usable]


# --------------------------------------------------------------------------------------
# B. Fill / reproducibility checks
# --------------------------------------------------------------------------------------
def check_returns_reproducible(prices: pd.DataFrame, returns: pd.DataFrame, columns: list[str]) -> list[Check]:
    """Returns must equal same-day log price differences of unfilled prices (not shifted, not filled)."""
    p = prices[columns]
    recomputed = np.log(p).diff().iloc[1:]
    common = recomputed.index.intersection(returns.index)
    diff = float(np.nanmax(np.abs(recomputed.loc[common].to_numpy() - returns.loc[common, columns].to_numpy())))
    nan_prices = int(p.isna().sum().sum())
    stale = {}
    for col in columns:
        zero = recomputed[col].eq(0)
        if zero.any():
            runs = zero.groupby((~zero).cumsum()).sum()
            run = int(runs.max())
            if run >= 5:
                block = zero[(~zero).cumsum() == runs.idxmax()]
                block = block[block]
                stale[col] = f"{run} ({block.index.min().date()}..{block.index.max().date()})"
    return [
        Check("Returns = same-day log price differences log(P_t/P_t-1) (no shift / fill)",
              PASS if diff < 1e-10 else FAIL, f"max abs difference {diff:.1e} over {len(common)} dates"),
        Check("No missing investment prices (nothing to forward-fill)", PASS if nan_prices == 0 else WARN,
              f"{nan_prices} missing prices in the investment panel"),
        Check("No stale-price runs (>= 5 consecutive zero returns)", WARN if stale else PASS,
              "; ".join(f"{k}: longest run {v}" for k, v in stale.items())
              + ("; zero daily returns can be genuine for a T-bill fund (BIL) when yields are near zero and "
                 "prices are quoted to two decimals - check the dates against a second source"
                 if "BIL" in stale else "") if stale else "none"),
    ]


def check_no_backfill(factors: pd.DataFrame, components: pd.DataFrame) -> Check:
    """A factor value must never exist on a date where none of its components exists (would imply fill)."""
    issues = []
    for f in FACTORS:
        orphan = factors[f].notna() & components[_cols(f)].notna().sum(axis=1).eq(0)
        if orphan.any():
            issues.append(f"{f}: {int(orphan.sum())} dates")
    return Check("No factor value without underlying components (no fill)", PASS if not issues else FAIL,
                 "; ".join(issues) or "every factor value is backed by >= 1 component on the same date")


# --------------------------------------------------------------------------------------
# C. Causality: prefix invariance on the actual inputs
# --------------------------------------------------------------------------------------
def perturb_future(inputs: FactorInputs, cutoff: pd.Timestamp, seed: int = 0) -> FactorInputs:
    """Same information up to `cutoff`; every later observation replaced by a randomly altered value."""
    rng = np.random.default_rng(seed)

    def alter(frame: pd.DataFrame, kind: str) -> pd.DataFrame:
        out = frame.copy()
        after = out.index > cutoff
        n = int(after.sum())
        if n == 0 or out.shape[1] == 0:
            return out
        values = out.loc[after].to_numpy(dtype=float)
        if kind == "price":
            values = values * np.exp(np.cumsum(rng.normal(0, 0.05, values.shape), axis=0))
        elif kind == "volume":
            values = values * rng.uniform(0.2, 5.0, values.shape)
        else:
            values = values + rng.normal(0, 1.0, values.shape)
        out.loc[after] = values
        return out

    log = inputs.release_log.copy()
    if not log.empty:
        later = pd.to_datetime(log["available_date"]) > cutoff
        log.loc[later, "value"] = log.loc[later, "value"] * (1 + rng.normal(0, 0.05, int(later.sum())))
    return FactorInputs(alter(inputs.adj_close, "price"), alter(inputs.close, "price"),
                        alter(inputs.volume, "volume"), alter(inputs.macro, "level"), log, inputs.sector_tickers)


def _compare(reference: pd.DataFrame, candidate: pd.DataFrame) -> tuple[float, int]:
    a = reference.to_numpy(float)
    b = candidate.reindex(index=reference.index, columns=reference.columns).to_numpy(float)
    nan_mismatch = int((np.isnan(a) != np.isnan(b)).sum())
    both = ~np.isnan(a) & ~np.isnan(b)
    return (float(np.abs(a[both] - b[both]).max()) if both.any() else 0.0), nan_mismatch


def prefix_invariance_check(
    inputs: FactorInputs,
    stored_factors: pd.DataFrame,
    stored_components: pd.DataFrame,
    truncation_cutoffs: list[pd.Timestamp],
    perturbation_cutoffs: list[pd.Timestamp],
) -> tuple[list[Check], pd.DataFrame]:
    """Decompose any discrepancy into serialization error vs genuine dependence on later data.

    1. stored CSV vs in-memory full rebuild          -> only rounding (<= SERIALIZATION_TOLERANCE) allowed
    2. truncated rebuild vs full rebuild, up to c    -> must be identical (removing future data)
    3. future-perturbed rebuild vs full rebuild      -> must be identical (changing future data)
    """
    _, full_comps, full_facs = build_factors(inputs)
    rows = []
    for kind, stored, rebuilt in (("factors", stored_factors, full_facs), ("components", stored_components, full_comps)):
        diff, nan_mm = _compare(stored, rebuilt)
        rows.append({"test": "serialization", "information_set": "full sample", "output": kind,
                     "dates_compared": len(stored), "max_abs_diff": diff, "nan_pattern_mismatches": nan_mm})
    for test, cutoffs in (("truncation", truncation_cutoffs), ("perturbation", perturbation_cutoffs)):
        for i, cutoff in enumerate(cutoffs):
            variant = inputs.truncate(cutoff) if test == "truncation" else perturb_future(inputs, cutoff, seed=i)
            _, comps, facs = build_factors(variant)
            for kind, full, other in (("factors", full_facs, facs), ("components", full_comps, comps)):
                ref = full.loc[:cutoff]
                diff, nan_mm = _compare(ref, other)
                rows.append({"test": test, "information_set": f"cutoff {cutoff.date()}", "output": kind,
                             "dates_compared": len(ref), "max_abs_diff": diff, "nan_pattern_mismatches": nan_mm})
    table = pd.DataFrame(rows)

    def ok(test: str, tol: float) -> tuple[bool, pd.DataFrame]:
        sub = table[table["test"] == test]
        return bool((sub["max_abs_diff"] <= tol).all() and (sub["nan_pattern_mismatches"] == 0).all()), sub

    ok_ser, ser = ok("serialization", SERIALIZATION_TOLERANCE)
    ok_tr, tr = ok("truncation", EXACT_TOLERANCE)
    ok_pe, pe = ok("perturbation", EXACT_TOLERANCE)
    checks = [
        Check("Stored factor files reproduce from current code and inputs (serialization only)",
              PASS if ok_ser else FAIL,
              f"max abs diff {ser['max_abs_diff'].max():.1e} (6-decimal CSV; tolerance {SERIALIZATION_TOLERANCE:g}); "
              f"NaN-pattern mismatches {int(ser['nan_pattern_mismatches'].sum())}"),
        Check("Prefix invariance: removing later data leaves history unchanged",
              PASS if ok_tr else FAIL,
              f"{len(truncation_cutoffs)} cutoffs ({', '.join(str(c.date()) for c in truncation_cutoffs)}); "
              f"max abs diff {tr['max_abs_diff'].max():.1e}; NaN-pattern mismatches "
              f"{int(tr['nan_pattern_mismatches'].sum())}"),
        Check("Prefix invariance: altering later data leaves history unchanged",
              PASS if ok_pe else FAIL,
              f"{len(perturbation_cutoffs)} cutoffs ({', '.join(str(c.date()) for c in perturbation_cutoffs)}); "
              f"max abs diff {pe['max_abs_diff'].max():.1e}; NaN-pattern mismatches "
              f"{int(pe['nan_pattern_mismatches'].sum())}"),
    ]
    return checks, table


def initial_features_prefix_check(adj_close: pd.DataFrame, stored: pd.DataFrame | None,
                                  cutoffs: list[pd.Timestamp]) -> Check:
    """Stage-1 initial features (features.py) must also be prefix invariant."""
    name = "Prefix invariance: initial features (features.py)"
    if stored is None:
        return Check(name, NOT_RUN, "data/processed/initial_features.csv not found")
    full = build_initial_features(adj_close)
    worst, nan_mm = 0.0, 0
    for cutoff in cutoffs:
        diff, mm = _compare(full.loc[:cutoff], build_initial_features(adj_close.loc[:cutoff]))
        worst, nan_mm = max(worst, diff), nan_mm + mm
    ser, ser_mm = _compare(stored, full)
    status = PASS if worst <= EXACT_TOLERANCE and nan_mm == 0 and ser <= 1e-9 and ser_mm == 0 else FAIL
    return Check(name, status, f"truncated vs full max abs diff {worst:.1e}, NaN mismatches {nan_mm}; "
                               f"stored vs rebuilt {ser:.1e}")


def macro_prefix_check(release_log: pd.DataFrame, calendar: pd.DatetimeIndex, stored_macro: pd.DataFrame,
                       cutoffs: list[pd.Timestamp]) -> tuple[list[Check], pd.DataFrame]:
    """Re-align every macro series from the release log, using only releases available by each cutoff.

    This tests the PIT alignment itself (the factor-level truncation test reuses the stored macro panel).
    """
    log = release_log.copy()
    log["date"] = pd.to_datetime(log["date"])
    log["available_date"] = pd.to_datetime(log["available_date"])
    rows = []
    full_aligned = {}
    for series in stored_macro.columns:
        obs = log[log["series_id"] == series][["date", "value", "available_date"]]
        if obs.empty:
            continue
        full_aligned[series] = align_to_calendar(obs, calendar)["value"].reindex(calendar)
        diff, mm = _compare(stored_macro[[series]], full_aligned[series].to_frame(series))
        rows.append({"test": "reproduce stored macro_pit", "series": series, "cutoff": "full",
                     "max_abs_diff": diff, "nan_pattern_mismatches": mm})
        for cutoff in cutoffs:
            visible = obs[obs["available_date"] <= cutoff]
            cal = calendar[calendar <= cutoff]
            realigned = (align_to_calendar(visible, cal)["value"].reindex(cal) if not visible.empty
                         else pd.Series(np.nan, index=cal))
            diff, mm = _compare(full_aligned[series].loc[:cutoff].to_frame(series), realigned.to_frame(series))
            rows.append({"test": "releases known by cutoff", "series": series, "cutoff": str(cutoff.date()),
                         "max_abs_diff": diff, "nan_pattern_mismatches": mm})
    table = pd.DataFrame(rows)
    if table.empty:
        return [Check("Macro PIT alignment reproducible and prefix invariant", NOT_RUN, "no macro release log")], table
    repro = table[table["test"] == "reproduce stored macro_pit"]
    prefix = table[table["test"] == "releases known by cutoff"]
    ok_r = bool((repro["max_abs_diff"] <= 1e-9).all() and (repro["nan_pattern_mismatches"] == 0).all())
    ok_p = bool((prefix["max_abs_diff"] <= EXACT_TOLERANCE).all() and (prefix["nan_pattern_mismatches"] == 0).all())
    return [
        Check("Stored macro_pit reproduces from the release log", PASS if ok_r else FAIL,
              f"{len(repro)} series; max abs diff {repro['max_abs_diff'].max():.1e}"),
        Check("Macro PIT alignment uses only releases known by each cutoff", PASS if ok_p else FAIL,
              f"{len(cutoffs)} cutoffs x {len(repro)} series; max abs diff {prefix['max_abs_diff'].max():.1e}; "
              f"NaN mismatches {int(prefix['nan_pattern_mismatches'].sum())}"),
    ], table


# --------------------------------------------------------------------------------------
# D. Macro point-in-time audit (timing + vintage category)
# --------------------------------------------------------------------------------------
def check_macro_published(obs_dates: pd.DataFrame, release_log: pd.DataFrame, dates: pd.DatetimeIndex) -> Check:
    """Every macro value used on trade date t must have availability_date <= t (no unpublished observation)."""
    log = release_log.copy()
    log["date"] = pd.to_datetime(log["date"])
    log["available_date"] = pd.to_datetime(log["available_date"])
    obs = obs_dates.apply(pd.to_datetime).reindex(dates)
    trade = obs.index.to_series()
    violations, unmatched = {}, {}
    for series in obs.columns:
        avail = log[log["series_id"] == series].drop_duplicates("date", keep="last").set_index("date")["available_date"]
        used = obs[series]
        mapped = used.map(avail)
        unmatched[series] = int((used.notna() & mapped.isna()).sum())
        violations[series] = int((mapped > trade).sum())
    bad = {k: v for k, v in violations.items() if v}
    um = {k: v for k, v in unmatched.items() if v}
    status = PASS if not bad and not um else FAIL
    return Check("Macro values used only after their availability date", status,
                 f"{sum(violations.values())} trade-day values used before availability"
                 + (f" ({bad})" if bad else "") + f"; {sum(unmatched.values())} values not traceable to the release log"
                 + (f" ({um})" if um else ""))


def macro_pit_audit(
    obs_dates: pd.DataFrame,
    release_log: pd.DataFrame,
    labels: pd.Series,
    specs: dict[str, dict],
    fred_sources: dict[str, str] | None = None,
) -> tuple[list[Check], pd.DataFrame, pd.DataFrame]:
    """Classify each trade day's macro value into PIT categories A-D, per split.

    Returns (checks, method/category table, verified-coverage table).
    """
    checks = []
    obs = obs_dates.apply(pd.to_datetime).reindex(labels.index)
    trade = obs.index.to_series()

    future = {c: int((obs[c] > trade).sum()) for c in obs.columns}
    daily = [s for s, sp in specs.items() if sp.get("frequency") == "daily" and s in obs.columns]
    same_day = {c: int((obs[c] >= trade).sum()) for c in daily}
    checks.append(Check("No macro observation dated after the trade date", PASS if not any(future.values())
                        else FAIL, f"{sum(future.values())} violations"))
    checks.append(Check("Daily FRED values lag their observation date by >= 1 session",
                        PASS if not any(same_day.values()) else FAIL, f"{sum(same_day.values())} violations"))
    checks.append(check_macro_published(obs_dates, release_log, labels.index))

    log = release_log.copy()
    log["date"] = pd.to_datetime(log["date"])
    rows, coverage = [], []
    splits = list(dict.fromkeys(labels.dropna()))
    for series in obs.columns:
        method_map = log[log["series_id"] == series].drop_duplicates("date").set_index("date")["availability_method"]
        used = obs[series].map(method_map).fillna("no value")
        category = used.map(lambda m: "none" if m == "no value" else pit_category(m))
        for split in splits:
            in_split = (labels == split).to_numpy()
            counts = pd.DataFrame({"method": used[in_split], "category": category[in_split]}).value_counts(normalize=True)
            for (method, cat), share in counts.items():
                rows.append({"series": series, "used_in": MACRO_USAGE.get(series, "-"), "split": split,
                             "category": cat, "availability_method": method,
                             "share_of_trade_days_pct": round(100 * share, 1)})
        if specs.get(series, {}).get("availability", {}).get("rule") == "vintage":
            not_a = category[category != "A"]                     # C, B or no value at all
            verified_from = (category.index[category.index > not_a.index.max()].min() if len(not_a)
                             else category.index.min())
            row = {
                "series": series, "used_in": MACRO_USAGE.get(series, "-"),
                "download_source": (fred_sources or {}).get(series, "unknown"),
                "category_A_from": verified_from.date().isoformat() if pd.notna(verified_from) else "never",
                "split_of_A_start": labels.get(verified_from, "-") if pd.notna(verified_from) else "-",
            }
            for s in splits:
                cats = category[(labels == s).to_numpy()]
                row[f"{s}_pct_A"] = round(100 * (cats == "A").mean(), 1)
                row[f"{s}_pct_BC"] = round(100 * cats.isin(["B", "C", "unclassified"]).mean(), 1)
                row[f"{s}_pct_no_value"] = round(100 * (cats == "none").mean(), 1)
            coverage.append(row)
    table = pd.DataFrame(rows)
    verified = pd.DataFrame(coverage)

    name = "CPI / Core CPI / NFCI: every value used is a verified first release (category A)"
    if verified.empty:
        checks.append(Check(name, NOT_RUN, "no vintage-rule series in the dataset"))
    else:
        bc_cols = [c for c in verified.columns if c.endswith("_pct_BC")]
        any_bc = bool((verified[bc_cols] > 0).any().any())
        parts = []
        for _, r in verified.iterrows():
            shares = ", ".join(f"{s} A {r[f'{s}_pct_A']}% / B-C {r[f'{s}_pct_BC']}% / no value "
                               f"{r[f'{s}_pct_no_value']}%" for s in splits)
            parts.append(f"{r['series']}: category A from {r['category_A_from']} ({shares})")
        detail = "; ".join(parts)
        if any_bc:
            detail += "; B/C values are used on some trade days and are not verifiable as point-in-time"
        no_value = [r["series"] for _, r in verified.iterrows()
                    if any(r[f"{s}_pct_no_value"] > 0 for s in splits)]
        if no_value:
            detail += (f"; {', '.join(no_value)} has no real-time value on part of the sample: the component is "
                       "absent there (not filled), which shortens its history instead of using unverifiable values")
        checks.append(Check(name, WARN if any_bc else PASS, detail))
    checks.append(Check("Daily FRED series timing (category D) is a conservative assumption", WARN,
                        "availability = observation date + 1 US business day; no publication timestamps recorded; "
                        "values are current vintage"))
    checks.append(Check("Historical revisions of daily FRED series checked against ALFRED vintages", NOT_RUN,
                        "daily-series vintages were not downloaded"))
    checks.append(Check("Release time of day versus signal formation (after US close)", WARN,
                        "availability is date-level only; scheduled 08:30 ET releases (CPI, NFCI) precede the "
                        "16:00 ET close, and daily series are used one session later, but exact timestamps are "
                        "not in the data"))
    return checks, table, verified


# --------------------------------------------------------------------------------------
# E. Component composition over time
# --------------------------------------------------------------------------------------
def component_availability(components: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    rows = []
    splits = list(dict.fromkeys(labels.dropna()))
    for c in COMPONENTS:
        col = f"{c.factor}__{c.name}"
        s = components[col]
        first = s.first_valid_index()
        row = {"factor": c.factor, "component": c.name, "in_composite": "yes" if c.in_composite else "monitor only",
               "first_valid": first.date().isoformat() if first is not None else "never"}
        for split in splits:
            row[f"{split}_available_pct"] = round(100 * s.reindex(labels.index[(labels == split).to_numpy()])
                                                  .notna().mean(), 1)
        rows.append(row)
    return pd.DataFrame(rows).set_index("component")


def composition_by_period(components: pd.DataFrame, periods: dict[str, tuple]) -> pd.DataFrame:
    rows = []
    for f in FACTORS:
        cols = [c for c in COMPONENTS if c.factor == f and c.in_composite]
        for period, (start, end) in periods.items():
            block = components.loc[start:end, [f"{c.factor}__{c.name}" for c in cols]]
            if block.empty:
                continue
            active = block.notna().sum(axis=1)
            partial = [c.name for c in cols if 0 < block[f"{c.factor}__{c.name}"].notna().mean() < 1]
            absent = [c.name for c in cols if block[f"{c.factor}__{c.name}"].notna().mean() == 0]
            rows.append({
                "factor": f, "period": period, "components_defined": len(cols),
                "min_active": int(active.min()), "max_active": int(active.max()),
                "full_composition_pct": round(100 * (active == len(cols)).mean(), 1),
                "partially_available": ", ".join(
                    f"{n} ({block[f'{f}__{n}'].notna().mean():.0%})" for n in partial) or "-",
                "absent": ", ".join(absent) or "-",
            })
    return pd.DataFrame(rows)


def composition_changes(
    components: pd.DataFrame,
    dates: pd.DatetimeIndex,
    factors: pd.DataFrame | None = None,
    labels: pd.Series | None = None,
    train_dates: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Dates where a factor's active component set changes, with the jump attributable to the change.

    inclusion_effect = mean(active set today) - mean(previous active set, evaluated today), i.e. the part of
    today's factor value caused purely by the composition change. Compared with the 99th percentile of
    absolute daily factor moves measured on TRAIN dates only.
    """
    rows = []
    p99 = (factors.loc[train_dates].diff().abs().quantile(0.99)
           if factors is not None and train_dates is not None else None)
    for f in FACTORS:
        cols = _cols(f)
        values = components.loc[dates, cols]
        active = values.notna()
        changed = active.ne(active.shift()).any(axis=1)
        changed.iloc[0] = False
        for date in changed[changed].index:
            prev = active.index[active.index.get_loc(date) - 1]
            added = [c.split("__")[1] for c in cols if active.at[date, c] and not active.at[prev, c]]
            removed = [c.split("__")[1] for c in cols if not active.at[date, c] and active.at[prev, c]]
            row = {"factor": f, "date": date.date().isoformat(), "added": ", ".join(added) or "-",
                   "removed": ", ".join(removed) or "-", "active_after": int(active.loc[date].sum())}
            if labels is not None:
                row["split"] = labels.get(date, "-")
            if p99 is not None:
                # additions are evaluated today (added components exist today), removals on the previous
                # session (removed components existed then); `common` = components active on both dates
                today, before = values.loc[date], values.loc[prev]
                common = [c for c in cols if active.at[prev, c] and active.at[date, c]]
                if common:
                    effect = ((today[active.loc[date]].mean() - today[common].mean())
                              + (before[common].mean() - before[active.loc[prev]].mean()))
                else:
                    effect = np.nan
                row["inclusion_effect"] = round(float(effect), 3) if pd.notna(effect) else np.nan
                row["factor_move"] = round(float(factors.at[date, f] - factors.at[prev, f]), 3) \
                    if pd.notna(factors.at[date, f]) and pd.notna(factors.at[prev, f]) else np.nan
                row["train_p99_abs_daily_move"] = round(float(p99[f]), 3)
            rows.append(row)
    columns = ["factor", "date", "added", "removed", "active_after"]
    if labels is not None:
        columns.append("split")
    if p99 is not None:
        columns += ["inclusion_effect", "factor_move", "train_p99_abs_daily_move"]
    return pd.DataFrame(rows, columns=columns)


def composition_sensitivity(components: pd.DataFrame, factors: pd.DataFrame, train_dates: pd.DatetimeIndex
                            ) -> pd.DataFrame:
    """Train only: compare each factor with a version restricted to components active on the first train date."""
    rows = []
    first = train_dates[0]
    for f in FACTORS:
        cols = _cols(f)
        block = components.loc[train_dates, cols]
        early = [c for c in cols if pd.notna(block.at[first, c])]
        late = [c for c in cols if c not in early]
        if not late:
            continue
        full_rows = block.notna().all(axis=1)
        restricted = block[early].mean(axis=1)
        a, b = factors.loc[train_dates, f][full_rows], restricted[full_rows]
        change_date = full_rows[full_rows].index.min()
        rows.append({
            "factor": f,
            "late_components": ", ".join(c.split("__")[1] for c in late),
            "full_set_from": change_date.date().isoformat() if pd.notna(change_date) else "never",
            "corr_full_vs_early_subset": round(float(a.corr(b)), 3) if len(a) > 2 else np.nan,
            "mean_abs_gap": round(float((a - b).abs().mean()), 3) if len(a) else np.nan,
        })
    return pd.DataFrame(rows)


def check_composition(sensitivity: pd.DataFrame, changes: pd.DataFrame) -> list[Check]:
    checks = []
    has_effect = "inclusion_effect" in changes.columns
    if "split" in changes.columns:
        later, train = changes[changes["split"] != "train"], changes[changes["split"] == "train"]
    else:  # no split labels: treat every change as a potential post-train change
        later, train = changes, changes.iloc[0:0]

    def describe(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "none"
        parts = [f"{r.factor} {r.date} +[{r.added}] -[{r.removed}]"
                 + (f" effect {r.inclusion_effect}" if has_effect else "") for r in frame.head(6).itertuples()]
        return f"{len(frame)} changes: " + "; ".join(parts) + (" ..." if len(frame) > 6 else "")

    checks.append(Check("Factor composition stable in validation and test", PASS if later.empty else WARN,
                        describe(later)))
    large = (train[train["inclusion_effect"].abs() > train["train_p99_abs_daily_move"]]
             if has_effect and not train.empty else train.iloc[0:0])
    weak = sensitivity[sensitivity["corr_full_vs_early_subset"] < 0.8] if not sensitivity.empty else sensitivity
    status = WARN if (not large.empty or not weak.empty) else PASS
    detail = describe(train)
    if not sensitivity.empty:
        detail += "; corr(full set, early subset): " + ", ".join(
            f"{r.factor} {r.corr_full_vs_early_subset}" for r in sensitivity.itertuples())
    if not large.empty:
        detail += f"; {len(large)} inclusion effects exceed the train p99 daily move"
    checks.append(Check("Composition changes inside train create no artificial jump", status, detail))
    return checks


# --------------------------------------------------------------------------------------
# F. Train-only redundancy diagnostics
# --------------------------------------------------------------------------------------
def train_redundancy(train: pd.DataFrame) -> tuple[list[Check], dict[str, pd.DataFrame]]:
    pearson = train.corr(method="pearson")
    spearman = train.corr(method="spearman")
    std = train.std()
    iqr = train.quantile(0.75) - train.quantile(0.25)
    mode_share = train.apply(lambda s: s.round(6).value_counts(normalize=True).iloc[0])
    eig = np.linalg.eigvalsh(pearson.to_numpy())
    singular = eig.min() <= 1e-10
    if singular:  # exact linear dependence: VIF and condition number are infinite
        vif = pd.Series(np.inf, index=train.columns)
        condition = float("inf")
    else:
        vif = pd.Series(np.diag(np.linalg.inv(pearson.to_numpy())), index=train.columns)
        condition = float(np.sqrt(eig.max() / eig.min()))
    off = pearson.where(~np.eye(len(pearson), dtype=bool)).abs()
    max_pair = off.stack().idxmax()
    duplicates = [(a, b) for a in train for b in train if a < b and (train[a] - train[b]).abs().max() < 1e-12]

    stats = pd.DataFrame({"std": std, "iqr": iqr, "largest_value_share": mode_share, "vif": vif}).round(3)
    checks = [
        Check("Train: no near-constant factor", PASS if (std > 1e-6).all() and (mode_share < 0.5).all() else FAIL,
              "min std " + f"{std.min():.3f}; max share of a single value {mode_share.max():.1%}"),
        Check("Train: no duplicate factor columns", PASS if not duplicates else FAIL, f"{duplicates or 'none'}"),
        Check("Train: no severe multicollinearity (VIF < 10, |rho| < 0.9)",
              FAIL if singular else (PASS if vif.max() < 10 and off.max().max() < 0.9 else WARN),
              f"max VIF {vif.max():.2f} ({vif.idxmax()}); max |Pearson| {off.max().max():.2f} "
              f"({max_pair[0]} vs {max_pair[1]}); condition number {condition:.2f}"),
    ]
    return checks, {"pearson": pearson.round(2), "spearman": spearman.round(2), "stats": stats}


def feature_availability(features: pd.DataFrame, labels: pd.Series, components: pd.DataFrame) -> pd.DataFrame:
    """Per factor: first valid date and number of active components on each partition's first date."""
    rows = []
    starts = {split: labels[labels == split].index.min() for split in labels.unique()}
    for f in FEATURE_COLUMNS:
        cols = _cols(f)
        row = {"factor": f, "first_valid": features[f].first_valid_index().date().isoformat()}
        for split, start in starts.items():
            row[f"active_components_on_{split}_start"] = f"{int(components.loc[start, cols].notna().sum())}/{len(cols)}"
        rows.append(row)
    return pd.DataFrame(rows).set_index("factor")


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------
def pit_protocol_summary(verified: pd.DataFrame) -> list[str]:
    """Short PIT status lines for the research protocol."""
    if verified.empty:
        return ["- No vintage-rule macro series in the dataset."]
    splits = [c[: -len("_pct_A")] for c in verified.columns if c.endswith("_pct_A")]
    lines = []
    for _, r in verified.iterrows():
        shares = "; ".join(
            f"{s}: A {r[f'{s}_pct_A']}%"
            + (f", B/C {r[f'{s}_pct_BC']}%" if r.get(f"{s}_pct_BC", 0) else "")
            + (f", no value {r[f'{s}_pct_no_value']}%" if r.get(f"{s}_pct_no_value", 0) else "")
            for s in splits)
        lines.append(f"- `{r['series']}` ({r['used_in']}; source {r['download_source']}): verified first-release "
                     f"(category A) from {r['category_A_from']}. Share of trade days - {shares}.")
    return lines


def render_audit_report(
    result: SplitResult,
    calendar: pd.DatetimeIndex,
    checks: list[Check],
    availability: pd.DataFrame,
    pit_table: pd.DataFrame,
    pit_verified: pd.DataFrame,
    hy_oas_coverage: pd.DataFrame,
    component_avail: pd.DataFrame,
    composition: pd.DataFrame,
    changes: pd.DataFrame,
    sensitivity: pd.DataFrame,
    redundancy: dict[str, pd.DataFrame],
    prefix_table: pd.DataFrame,
    macro_prefix_table: pd.DataFrame,
    manifest: dict,
) -> str:
    from .quality import _md_table

    status = summarize(checks)
    by_status = {s: [c for c in checks if c.status == s] for s in STATUSES}
    lines = ["# Dataset split audit (stage 3)", ""]
    lines.append(f"Protocol `{manifest['protocol_version']}`; generated {manifest['created_utc']} UTC. Checks: "
                 + ", ".join(f"{status[s]} {s}" for s in STATUSES) + ".")

    lines += ["", "## 0. Scope of this audit", ""]
    lines += [
        "- The checks below were executed on the files whose SHA-256 hashes are listed in "
        "`data/splits/split_manifest.json` (`feature_provenance.input_sha256`, `split_files`). A PASS applies to "
        "those files only. PASS results from `tests/` use synthetic data and are not evidence about this dataset.",
        "- A finite set of checks cannot prove the absence of every form of leakage. The prefix-invariance tests "
        "establish that, for the tested cutoffs, no stored value depends on later rows of the processed inputs. "
        "They cannot detect information that is already embedded in the inputs themselves (e.g. revised or "
        "retrospectively estimated macro history, ex-post universe selection); those are covered in sections 5-6.",
        "",
        "Status definitions:",
        "",
    ]
    lines += [f"- **{s}**: {d}" for s, d in STATUS_DEFINITIONS.items()]

    lines += ["", "## 1. Dataset overview and timing convention", ""]
    total = len(result.usable_dates)
    lines += [
        f"- SPY master calendar: {calendar.min().date()} to {calendar.max().date()} ({len(calendar)} sessions).",
        f"- Usable investment sample: {result.usable_dates.min().date()} to {result.usable_dates.max().date()} "
        f"({total} dates with all 5 factors and all 9 ETF returns). Earlier sessions are warm-up only.",
        f"- Features (model inputs): {', '.join(result.feature_columns)}.",
        f"- Returns (stored in separate files): daily log returns of {', '.join(result.return_columns)}.",
        "",
        "| quantity | definition | known at |",
        "|---|---|---|",
        "| feature_t | factor value computed from prices up to the close of t and macro values published by t | "
        "after the US close of t (signal formation) |",
        "| return_t | log(P_t / P_t-1), adjusted close, same day | after the US close of t (historical, not a future "
        "outcome) |",
        "| decision_t | position chosen from feature_t | after the close of t; earliest execution in session t+1 |",
        "| earned return | return_(t+1) for a close-to-close implementation of decision_t | after the close of t+1 |",
        "",
        "Feature and return files share the same date index for storage only. Evaluating a date-t signal against "
        "the date-t return would credit the strategy with a return realised before the signal existed; the "
        "backtest must pair decision_t with return_(t+1) (or the return of the actual execution convention). "
        "No forward-return labels are created in this stage.",
    ]

    lines += ["", "## 2. Split summary and actual observation counts", ""]
    split_rows = pd.DataFrame([{
        "split": s["name"], "configured_start": s["start"], "configured_end": s["end"] or "latest",
        "first_date": s["actual_first_date"], "last_date": s["actual_last_date"], "rows": s["rows"],
        "share_pct": round(100 * s["rows"] / total, 1),
    } for s in manifest["splits"]]).set_index("split")
    lines.append(_md_table(split_rows))
    lines += ["", f"Total: {total} rows. Dates inside the configured range but not used:"]
    if result.excluded.empty:
        lines.append("none.")
    else:
        lines += ["", _md_table(result.excluded.assign(date=result.excluded["date"].dt.date).set_index("date"))]
    for check_name, label in (("First eligible return date", "First eligible return"),
                              ("First usable modelling row (all factors and returns present)", "First usable row")):
        found = next((c for c in checks if c.name == check_name), None)
        if found:
            lines += ["", f"{label}: [{found.status}] {found.detail}."]
    lines += ["", f"The configured train start ({result.specs[0].start}) is unchanged. The first train row is the "
                  f"first session on which a return exists AND all five factors are available "
                  f"({result.usable_dates.min().date()})."]
    lines.append("Rolling statistics are not restarted at split boundaries; earlier history (including the "
                 "2004-2007 warm-up) may inform later rows, never the reverse (section 6).")

    lines += ["", "## 3. Validation checks", ""]
    lines.append(_md_table(pd.DataFrame([vars(c) for c in checks]).set_index("name")))

    lines += ["", "## 4. Research-stage exposure of the test period", ""]
    lines += [
        "| stage | test-period (2020+) status |",
        "|---|---|",
        "| 1. Descriptive historical inspection | **Occurred**: factor time series and 2008 / 2020 / 2022 event-window "
        "charts (event windows specified in the original research plan) were viewed. Not a blind holdout. |",
        "| 2. Causal feature computation | **Permitted and done**: factors on test dates use only information "
        "available at each date (sections 5-6). |",
        "| 3. Model fitting | **Not done**: no K-Means, GMM, HMM or other regime model has been fitted. |",
        "| 4. Hyperparameter / specification selection | **Not done**; must use Train/Validation only. |",
        "| 5. Final out-of-sample evaluation | **Not done**; 2020-latest remains the designated OOS period. |",
        "",
        "Viewing historical charts is not, by itself, look-ahead bias, training leakage or test-set optimisation. "
        "It does mean the test period is not a never-inspected holdout; see `reports/research_protocol.md`.",
    ]

    lines += ["", "## 5. Point-in-time audit of macro inputs", ""]
    lines.append("Categories:")
    lines += [""] + [f"- **{k}**: {v}" for k, v in PIT_CATEGORIES.items()]
    lines += [
        "",
        "A release-date lag establishes WHEN a value is assumed known; it does not establish that the value itself "
        "(if revised later) was historically available. Category C values pass the timing checks but are not "
        "verifiable as point-in-time.",
        "",
        "Verified first-release coverage of the vintage-rule series (share of usable trade days in category A, in "
        "category B or C, and with no value at all):",
        "",
        _md_table(pit_verified.set_index("series")) if not pit_verified.empty else "none.",
        "",
        "Share of trade days by series, category and availability method:",
        "",
    ]
    if not pit_table.empty:
        pivot = pit_table.pivot_table(index=["series", "category", "availability_method"], columns="split",
                                      values="share_of_trade_days_pct", aggfunc="first").fillna(0)
        pivot = pivot[[s.name for s in result.specs if s.name in pivot.columns]]
        pivot.index = [f"{a} / {b} / {c}" for a, b, c in pivot.index]
        pivot.index.name = "series / category / method"
        lines.append(_md_table(pivot))
    lines += [
        "",
        "Notes:",
        "",
        "- **NFCI**: the Chicago Fed re-estimates the index every week. Real-time releases archived in ALFRED start "
        "in 2011; earlier values are retrospective estimates produced with later data. Where ALFRED records those "
        "back-filled values with their (2011) vintage date, the pipeline treats them as unknown before that date: "
        "NFCI then has 'no value' on earlier trade days, the component is absent (not filled) and enters the "
        "Liquidity / Credit factor only after its 126-observation z-score window has filled. Where back-filled "
        "values were instead given rule-based dates, they appear above as category C and are not verifiable as "
        "point-in-time.",
        "- **Missing component vs. missing factor**: a factor needs at least half of its components. If NFCI and the "
        "HYG-based components are all unavailable, Liquidity / Credit Stress is missing and the date is excluded "
        "from the splits (section 2) rather than filled.",
        "- **CPI / Core CPI**: year-over-year changes are computed from the release log and become usable only after "
        "both months are published. For category A months both legs are first-release values, which were public by "
        "then (feasible in real time, although a contemporaneous analyst would have used the revised t-12 value).",
        "- **Daily series (D)**: Treasury yields, breakevens, BAA10Y and HY OAS are used one US business day after "
        "their observation date; the issuing sources publish them by the evening of the observation date. Values "
        "are current vintage; revisions to these market series are believed rare but were not checked.",
        "- `scripts/04` reports 'Macro values point-in-time: PASS'. That check verifies observation-date timing only, "
        "not value vintages.",
        "",
        "Recommended PIT-verified sensitivity analysis (later stages, decided now): re-estimate the selected model "
        "(a) on train dates from the latest `category_A_from` date above, and/or (b) with the Liquidity / Credit and "
        "Inflation factors recomputed without NFCI / CPI components before their category-A start, and report "
        "whether regime assignments in validation change materially.",
        "",
        "HY OAS (monitor only, not in any composite) coverage by split:",
        "",
        _md_table(hy_oas_coverage),
    ]

    lines += ["", "## 6. Causality and prefix invariance", ""]
    lines += [
        "Inspected implementation (rolling windows are trailing; no `center=True`, negative shifts, back-fills or "
        "full-sample statistics):",
        "",
        "| element | implementation | information used for date t |",
        "|---|---|---|",
        "| Robust z-score | median / MAD of x[t-W..t-1] (factors.rolling_robust_z) | past W values; x_t only in the "
        "numerator |",
        "| Volatility, downside vol, Amihud | trailing rolling windows ending at t | returns up to t |",
        "| Momentum / relative strength | log-price differences, shift(21), shift(252) | prices up to t |",
        "| Rolling correlation | trailing 63-day pairwise correlation | returns up to t |",
        "| Rate / spread changes | 63-day differences of the PIT macro panel | values published by t |",
        "| CPI YoY | ratio of two first-release values, usable at the later availability date | releases by t |",
        "| Macro alignment | merge_asof backward on availability date (pit.align_to_calendar) | releases by t |",
        "| Composite | mean of components present on t, >= 50% required | same-day components |",
        "| Missing values | no fill; NaN propagates; zero MAD -> window std -> NaN | - |",
        "",
        "Empirical tests on the actual processed inputs:",
        "",
    ]
    shown = prefix_table.assign(max_abs_diff=prefix_table["max_abs_diff"].map(lambda v: f"{v:.1e}"))
    lines.append(_md_table(shown.set_index("test")))
    lines += [
        "",
        "Interpretation: `serialization` differences (<= 5e-7) come from the 6-decimal CSV format; `truncation` and "
        "`perturbation` rebuilds are compared in memory and must agree exactly.",
    ]
    if not macro_prefix_table.empty:
        agg = macro_prefix_table.groupby("test").agg(series=("series", "nunique"), max_abs_diff=("max_abs_diff", "max"),
                                                     nan_pattern_mismatches=("nan_pattern_mismatches", "sum"))
        agg["max_abs_diff"] = agg["max_abs_diff"].map(lambda v: f"{v:.1e}")
        lines += ["", "Macro PIT alignment re-run from the release log:", "", _md_table(agg)]
    lines += [
        "",
        "Not covered by these tests (inherent in the inputs): category C macro values (section 5); Yahoo adjusted "
        "closes are rescaled when later distributions occur, which leaves price ratios within any window ending at t "
        "unchanged but makes stored price levels download-date dependent; the nine-ETF universe was chosen ex post "
        "(all funds survive to today).",
    ]

    lines += ["", "## 7. Factor composition stability", ""]
    lines.append("Component availability (share of usable dates with a value) by split:")
    lines += ["", _md_table(component_avail)]
    lines += ["", "Composition by period:", "", _md_table(composition.set_index("factor"))]
    lines += ["", "Composition changes inside the usable sample. `inclusion_effect` = the part of the factor's "
              "move caused only by the change of component set (additions measured on the change date, removals on "
              "the previous session, both against the components active on both dates); the threshold is the 99th "
              "percentile of absolute daily factor moves on train dates only:", ""]
    lines.append(_md_table(changes.set_index("factor")) if not changes.empty else "none.")
    lines += ["", "Train-only comparison of each factor with its early-component subset:", ""]
    lines.append(_md_table(sensitivity.set_index("factor")) if not sensitivity.empty else "no composition change in train.")
    lines += [
        "",
        "Component definitions and weights are unchanged; nothing here was selected using test outcomes. "
        "Recommended robustness checks for the modelling stage (decided now, before modelling): (i) refit on train "
        "dates after the last composition change; (ii) refit with factors restricted to components available from "
        "the first train date; (iii) check whether estimated regime transitions coincide with composition-change "
        "dates; (iv) the PIT sensitivity in section 5.",
    ]

    lines += ["", "## 8. Train-only correlation diagnostics", ""]
    lines.append("Computed on the train partition only. Nothing is re-weighted, removed or transformed on the basis "
                 "of these numbers.")
    lines += ["", "Pearson:", "", _md_table(redundancy["pearson"])]
    lines += ["", "Spearman:", "", _md_table(redundancy["spearman"])]
    lines += ["", "Dispersion and variance inflation:", "", _md_table(redundancy["stats"])]

    lines += ["", "## 9. Readiness assessment", ""]
    lines.append("**Verified on this dataset (PASS):** " + "; ".join(c.name for c in by_status[PASS]) + ".")
    lines += ["", "**Unverifiable assumptions and concerns (WARN):**", ""]
    lines += [f"- {c.name}: {c.detail}" for c in by_status[WARN]] or ["- none"]
    lines += ["", "**Not executed (NOT RUN):**", ""]
    lines += [f"- {c.name}: {c.detail}" for c in by_status[NOT_RUN]] or ["- none"]
    lines += ["", "**Failures (FAIL):**", ""]
    lines += [f"- {c.name}: {c.detail}" for c in by_status[FAIL]] or ["- none"]
    lines.append("")
    if by_status[FAIL]:
        lines.append("Verdict: **NOT READY** - resolve the FAIL checks before any model is fitted.")
    else:
        lines.append("Verdict: **ready for train-only K-Means / GMM exploration**, with the WARN items carried as "
                     "documented limitations and the PIT / composition sensitivity analyses scheduled above.")
    return "\n".join(lines) + "\n"
