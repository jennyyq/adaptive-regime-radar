"""Economic risk-factor layer: five interpretable composites built from standardized components.

Pipeline for every component
  1. raw signal x_t computed only from data known at the close of day t
     (prices on the SPY calendar; FRED values already point-in-time aligned)
  2. robust z-score against the PREVIOUS `window` observations (t-window .. t-1):
         z_t = (x_t - median_{t-1}) / (1.4826 * MAD_{t-1})
  3. economic sign applied (+1 / -1) so that, after signing,
         stress factors: higher = more risk / stress
         Equity Trend : higher = stronger / healthier equity trend
  4. winsorized at +/- COMPONENT_CLIP so no single component can dominate
Factor value = equal-weight mean of its available signed components, required to
have at least half of its composite components (rounded up) present on that day.

No weights are estimated from data, so there is nothing to fit on any sample.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from .pit import align_to_calendar

# --------------------------------------------------------------------------------------
# Settings (kept consistent with the existing project: 252-day window, 126 minimum)
# --------------------------------------------------------------------------------------
DEFAULT_WINDOW = 252
DEFAULT_MIN_PERIODS = 126
# Monthly CPI changes value only ~12 times per 252 days, so a 1-year MAD is unstable
# (frequently near zero). CPI components therefore use a 3-year window.
MONTHLY_WINDOW = 756
MONTHLY_MIN_PERIODS = 252
COMPONENT_CLIP = 5.0
MAD_SCALE = 1.4826
MIN_COMPONENT_SHARE = 0.5

FACTORS = {
    "equity_trend": "Equity Trend",
    "interest_rate": "Interest Rate Conditions",
    "inflation": "Inflation Pressure",
    "volatility": "Volatility Stress",
    "liquidity_credit": "Liquidity / Credit Stress",
}
FACTOR_DIRECTION = {
    "equity_trend": "higher = stronger, broader equity trend (risk-on)",
    "interest_rate": "higher = tighter rates / larger rate shock",
    "inflation": "higher = more inflation pressure",
    "volatility": "higher = more volatility stress",
    "liquidity_credit": "higher = more liquidity / credit stress",
}

TRADING_DAYS = 252
RISK_ASSETS_FOR_CORRELATION = ["SPY", "EFA", "EEM", "HYG", "DBC"]


@dataclass(frozen=True)
class Component:
    name: str
    factor: str
    sign: int
    definition: str
    source: str
    sign_rationale: str
    window: int = DEFAULT_WINDOW
    min_periods: int = DEFAULT_MIN_PERIODS
    in_composite: bool = True


COMPONENTS: list[Component] = [
    # ---------------- Equity Trend (higher = healthier) ----------------
    Component("spy_mom_63d", "equity_trend", +1,
              "SPY 63-day log return", "Yahoo SPY adj close",
              "Positive medium-term momentum = healthier trend"),
    Component("spy_mom_12_1", "equity_trend", +1,
              "SPY 12-1 month momentum: log(P[t-21] / P[t-252])", "Yahoo SPY adj close",
              "Long-horizon trend, skipping the last month to avoid short-term reversal"),
    Component("sector_breadth_200d", "equity_trend", +1,
              "Share of 9 SPDR sectors above their 200-day average (10-day mean)", "Yahoo XL* adj close",
              "More sectors in uptrend = broader participation"),
    Component("equal_vs_cap_weight_63d", "equity_trend", +1,
              "63-day change of log(RSP / SPY)", "Yahoo RSP, SPY",
              "Equal-weight outperforming cap-weight = rally is broad, not narrow"),
    Component("small_vs_large_63d", "equity_trend", +1,
              "63-day change of log(IWM / SPY)", "Yahoo IWM, SPY",
              "Small caps leading = risk appetite and domestic growth confidence"),
    # ---------------- Interest Rate Conditions (higher = tighter / rate shock) ----------------
    Component("ust2y_change_63d", "interest_rate", +1,
              "63-day change in 2Y Treasury yield (pp)", "FRED DGS2 (PIT)",
              "Rising front-end yields = policy tightening priced in"),
    Component("ust10y_change_63d", "interest_rate", +1,
              "63-day change in 10Y Treasury yield (pp)", "FRED DGS10 (PIT)",
              "Rising long yields = duration losses and higher discount rates"),
    Component("curve_10y2y", "interest_rate", -1,
              "10Y minus 2Y Treasury spread (pp)", "FRED T10Y2Y (PIT)",
              "Flatter / inverted curve = restrictive policy stance (sign flipped)"),
    Component("ust10y_realized_vol_21d", "interest_rate", +1,
              "21-day realized volatility of daily 10Y yield changes, annualized (pp)", "FRED DGS10 (PIT)",
              "Volatile rates = rate shock / bond-market stress"),
    # ---------------- Inflation Pressure (higher = more inflation) ----------------
    Component("cpi_yoy", "inflation", +1,
              "Headline CPI year-over-year % (as first published, by release date)", "FRED/ALFRED CPIAUCSL",
              "Higher realized inflation",
              window=MONTHLY_WINDOW, min_periods=MONTHLY_MIN_PERIODS),
    Component("core_cpi_yoy", "inflation", +1,
              "Core CPI year-over-year % (as first published, by release date)", "FRED/ALFRED CPILFESL",
              "Higher persistent (ex food & energy) inflation",
              window=MONTHLY_WINDOW, min_periods=MONTHLY_MIN_PERIODS),
    Component("breakeven_5y", "inflation", +1,
              "5-year breakeven inflation (pp); 10-year used if 5-year missing", "FRED T5YIE (PIT)",
              "Higher market-implied inflation expectations"),
    Component("tip_vs_ief_63d", "inflation", +1,
              "63-day change of log(TIP / IEF)", "Yahoo TIP, IEF",
              "Inflation-linked bonds beating nominals = breakevens rising"),
    Component("commodity_mom_126d", "inflation", +1,
              "DBC 126-day log return", "Yahoo DBC",
              "Rising commodity prices feed input-cost inflation"),
    # ---------------- Volatility Stress (higher = more stress) ----------------
    Component("vix_log", "volatility", +1,
              "log(VIX)", "Yahoo ^VIX",
              "Higher implied equity volatility"),
    Component("spy_realized_vol_21d", "volatility", +1,
              "SPY 21-day realized volatility, annualized", "Yahoo SPY",
              "Higher realized volatility"),
    Component("spy_downside_vol_63d", "volatility", +1,
              "SPY 63-day downside semi-deviation, annualized", "Yahoo SPY",
              "Losses concentrated on down days"),
    Component("vix_term_structure", "volatility", +1,
              "log(VIX / VIX3M); > 0 means inverted (backwardation)", "Yahoo ^VIX, ^VIX3M",
              "Near-term fear above 3-month fear = acute stress"),
    Component("risk_asset_correlation_63d", "volatility", +1,
              "Average pairwise 63-day correlation of SPY, EFA, EEM, HYG, DBC daily returns", "Yahoo",
              "Risk assets moving together = diversification failure"),
    # ---------------- Liquidity / Credit Stress (higher = more stress) ----------------
    Component("baa_10y_spread", "liquidity_credit", +1,
              "Moody's Baa yield minus 10Y Treasury (pp)", "FRED BAA10Y (PIT)",
              "Wider corporate spreads = more credit risk premium"),
    Component("hyg_vs_lqd_63d", "liquidity_credit", -1,
              "63-day change of log(HYG / LQD)", "Yahoo HYG, LQD",
              "High yield underperforming investment grade = credit stress (sign flipped)"),
    Component("nfci", "liquidity_credit", +1,
              "Chicago Fed National Financial Conditions Index", "FRED/ALFRED NFCI (PIT)",
              "Positive / rising NFCI = tighter financial conditions"),
    Component("hyg_illiquidity_21d", "liquidity_credit", +1,
              "log of 21-day mean Amihud ratio |r| / dollar volume for HYG", "Yahoo HYG close, volume",
              "Larger price impact per dollar traded = poorer credit-market liquidity"),
    Component("hy_oas", "liquidity_credit", +1,
              "ICE BofA US High Yield OAS (pp)", "FRED BAMLH0A0HYM2 (PIT)",
              "Wider HY spreads = credit stress. MONITOR ONLY: FRED serves ~3 years of history",
              in_composite=False),
]
COMPONENT_BY_NAME = {c.name: c for c in COMPONENTS}


# --------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------
@dataclass
class FactorInputs:
    adj_close: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    macro: pd.DataFrame
    release_log: pd.DataFrame
    sector_tickers: list[str] = field(default_factory=list)

    @property
    def calendar(self) -> pd.DatetimeIndex:
        return self.adj_close.index

    def truncate(self, cutoff: pd.Timestamp) -> "FactorInputs":
        """Information set at the close of `cutoff` (used by the leakage test)."""
        log = self.release_log
        if not log.empty:
            log = log[pd.to_datetime(log["available_date"]) <= cutoff]
        return FactorInputs(
            self.adj_close.loc[:cutoff], self.close.loc[:cutoff], self.volume.loc[:cutoff],
            self.macro.loc[:cutoff], log, self.sector_tickers,
        )


def load_factor_inputs(processed: Path, sector_tickers: list[str]) -> FactorInputs:
    def wide(name: str) -> pd.DataFrame:
        return pd.read_csv(processed / name, index_col="date", parse_dates=["date"])

    log_path = processed / "macro_release_log.csv"
    release_log = (pd.read_csv(log_path, parse_dates=["date", "available_date"])
                   if log_path.exists() else pd.DataFrame(columns=["series_id", "date", "value", "available_date"]))
    macro_path = processed / "macro_pit.csv"
    adj = wide("market_adj_close.csv")
    macro = wide("macro_pit.csv") if macro_path.exists() else pd.DataFrame(index=adj.index)
    return FactorInputs(adj, wide("market_close.csv"), wide("market_volume.csv"), macro, release_log,
                        list(sector_tickers))


# --------------------------------------------------------------------------------------
# Robust standardization
# --------------------------------------------------------------------------------------
def rolling_robust_z(
    series: pd.Series,
    window: int = DEFAULT_WINDOW,
    min_periods: int = DEFAULT_MIN_PERIODS,
    clip: float | None = COMPONENT_CLIP,
) -> pd.Series:
    """Robust z-score of x_t against the previous `window` observations only.

    center_t = median(x[t-window .. t-1]),  scale_t = 1.4826 * MAD over the same window
    (MAD measured around that window's own median). Observation t never enters its own
    center/scale, and nothing after t is used. If MAD is zero (e.g. a flat or discrete
    series) the window's standard deviation is used instead; if that is also zero the
    result is NaN. Missing x_t gives NaN; windows with < min_periods valid values give NaN.
    """
    values = series.to_numpy(dtype=float)
    n = len(values)
    if n == 0:
        return series.astype(float)
    past = np.concatenate([[np.nan], values[:-1]])                    # x_{t-1} at position t
    padded = np.concatenate([np.full(window - 1, np.nan), past])
    windows = sliding_window_view(padded, window)                     # row t = x[t-window .. t-1]
    count = np.sum(~np.isnan(windows), axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)      # all-NaN rows at the start
        center = np.nanmedian(windows, axis=1)
        mad = np.nanmedian(np.abs(windows - center[:, None]), axis=1)
        std = np.nanstd(windows, axis=1, ddof=1)
    scale = MAD_SCALE * mad
    tiny = 1e-12 * np.maximum(1.0, np.abs(center))
    scale = np.where(scale > tiny, scale, np.where(std > tiny, std, np.nan))
    z = (values - center) / scale
    z[(count < min_periods) | ~np.isfinite(z)] = np.nan
    if clip is not None:
        z = np.clip(z, -clip, clip)
    return pd.Series(z, index=series.index, name=series.name)


# --------------------------------------------------------------------------------------
# Raw component signals
# --------------------------------------------------------------------------------------
def _col(frame: pd.DataFrame, name: str) -> pd.Series:
    if name in frame.columns:
        return frame[name].astype(float)
    return pd.Series(np.nan, index=frame.index, name=name)


def _log_price(adj: pd.DataFrame, ticker: str) -> pd.Series:
    price = _col(adj, ticker)
    return np.log(price.where(price > 0))


def pit_yoy(release_log: pd.DataFrame, series_id: str, calendar: pd.DatetimeIndex) -> pd.Series:
    """Year-over-year % change of a monthly series, known only once BOTH months were published."""
    empty = pd.Series(np.nan, index=calendar)
    if release_log.empty or series_id not in set(release_log["series_id"]):
        return empty
    log = release_log[release_log["series_id"] == series_id].dropna(subset=["value"]).copy()
    log["date"] = pd.to_datetime(log["date"])
    log["available_date"] = pd.to_datetime(log["available_date"])
    log = log.drop_duplicates("date", keep="last").set_index("date").sort_index()
    prior = log.reindex(log.index - pd.DateOffset(months=12))
    yoy = 100 * (log["value"].to_numpy() / prior["value"].to_numpy() - 1)
    available = pd.concat(
        [log["available_date"].reset_index(drop=True), prior["available_date"].reset_index(drop=True)], axis=1
    ).max(axis=1, skipna=False)
    obs = pd.DataFrame({"date": log.index, "value": yoy, "available_date": available.to_numpy()})
    obs = obs.dropna(subset=["value", "available_date"])
    if obs.empty:
        return empty
    return align_to_calendar(obs, calendar)["value"].reindex(calendar)


def _sector_breadth(adj: pd.DataFrame, sectors: list[str]) -> pd.Series:
    prices = adj.reindex(columns=[s for s in sectors if s in adj.columns])
    if prices.shape[1] == 0:
        return pd.Series(np.nan, index=adj.index)
    average = prices.rolling(200, min_periods=200).mean()
    valid = prices.notna() & average.notna()
    above = (prices > average) & valid
    n_valid = valid.sum(axis=1)
    share = (above.sum(axis=1) / n_valid).where(n_valid >= min(6, prices.shape[1]))
    return share.rolling(10, min_periods=5).mean()


def _average_pairwise_correlation(returns: pd.DataFrame, window: int = 63, min_periods: int = 50) -> pd.Series:
    cols = list(returns.columns)
    pairs = [returns[a].rolling(window, min_periods=min_periods).corr(returns[b])
             for i, a in enumerate(cols) for b in cols[i + 1:]]
    if not pairs:
        return pd.Series(np.nan, index=returns.index)
    stacked = pd.concat(pairs, axis=1)
    return stacked.mean(axis=1).where(stacked.notna().sum(axis=1) >= 3)


def raw_components(inputs: FactorInputs) -> pd.DataFrame:
    """Unstandardized component signals on the trading calendar (natural units, natural sign)."""
    adj, close, volume, macro = inputs.adj_close, inputs.close, inputs.volume, inputs.macro
    cal = inputs.calendar
    macro = macro.reindex(cal)
    lp = {t: _log_price(adj, t) for t in ["SPY", "RSP", "IWM", "TIP", "IEF", "DBC", "HYG", "LQD"]}
    spy_ret = lp["SPY"].diff()

    builders: dict[str, Callable[[], pd.Series]] = {
        "spy_mom_63d": lambda: lp["SPY"].diff(63),
        "spy_mom_12_1": lambda: lp["SPY"].shift(21) - lp["SPY"].shift(252),
        "sector_breadth_200d": lambda: _sector_breadth(adj, inputs.sector_tickers),
        "equal_vs_cap_weight_63d": lambda: (lp["RSP"] - lp["SPY"]).diff(63),
        "small_vs_large_63d": lambda: (lp["IWM"] - lp["SPY"]).diff(63),
        "ust2y_change_63d": lambda: _col(macro, "DGS2").diff(63),
        "ust10y_change_63d": lambda: _col(macro, "DGS10").diff(63),
        "curve_10y2y": lambda: _col(macro, "T10Y2Y"),
        "ust10y_realized_vol_21d": lambda: _col(macro, "DGS10").diff().rolling(21, min_periods=15).std()
        * math.sqrt(TRADING_DAYS),
        "cpi_yoy": lambda: pit_yoy(inputs.release_log, "CPIAUCSL", cal),
        "core_cpi_yoy": lambda: pit_yoy(inputs.release_log, "CPILFESL", cal),
        "breakeven_5y": lambda: _col(macro, "T5YIE").combine_first(_col(macro, "T10YIE")),
        "tip_vs_ief_63d": lambda: (lp["TIP"] - lp["IEF"]).diff(63),
        "commodity_mom_126d": lambda: lp["DBC"].diff(126),
        "vix_log": lambda: np.log(_col(adj, "^VIX").where(lambda s: s > 0)),
        "spy_realized_vol_21d": lambda: spy_ret.rolling(21, min_periods=21).std() * math.sqrt(TRADING_DAYS),
        "spy_downside_vol_63d": lambda: np.sqrt(spy_ret.clip(upper=0).pow(2).rolling(63, min_periods=50).mean())
        * math.sqrt(TRADING_DAYS),
        "vix_term_structure": lambda: np.log(_col(adj, "^VIX") / _col(adj, "^VIX3M").where(lambda s: s > 0)),
        "risk_asset_correlation_63d": lambda: _average_pairwise_correlation(
            pd.DataFrame({t: _log_price(adj, t).diff() for t in RISK_ASSETS_FOR_CORRELATION})),
        "baa_10y_spread": lambda: _col(macro, "BAA10Y"),
        "hyg_vs_lqd_63d": lambda: (lp["HYG"] - lp["LQD"]).diff(63),
        "nfci": lambda: _col(macro, "NFCI"),
        "hyg_illiquidity_21d": lambda: np.log(
            (lp["HYG"].diff().abs() / (_col(close, "HYG") * _col(volume, "HYG")).where(lambda s: s > 0) * 1e9)
            .rolling(21, min_periods=15).mean().where(lambda s: s > 0)),
        "hy_oas": lambda: _col(macro, "BAMLH0A0HYM2"),
    }
    raw = pd.DataFrame({name: builders[name]() for name in COMPONENT_BY_NAME}, index=cal)
    raw = raw.replace([np.inf, -np.inf], np.nan)
    raw.index.name = "date"
    return raw


# --------------------------------------------------------------------------------------
# Standardize, sign, composite
# --------------------------------------------------------------------------------------
def standardized_components(raw: pd.DataFrame) -> pd.DataFrame:
    """Signed, winsorized robust z-scores. Column names: '<factor>__<component>'."""
    out = {}
    for comp in COMPONENTS:
        z = rolling_robust_z(raw[comp.name], comp.window, comp.min_periods, clip=COMPONENT_CLIP)
        out[f"{comp.factor}__{comp.name}"] = comp.sign * z
    frame = pd.DataFrame(out, index=raw.index)
    frame.index.name = "date"
    return frame


def composite_factors(components: pd.DataFrame, min_share: float = MIN_COMPONENT_SHARE) -> pd.DataFrame:
    factors = {}
    for factor in FACTORS:
        cols = [f"{c.factor}__{c.name}" for c in COMPONENTS if c.factor == factor and c.in_composite]
        block = components[cols]
        required = math.ceil(min_share * len(cols))
        factors[factor] = block.mean(axis=1).where(block.notna().sum(axis=1) >= required)
    frame = pd.DataFrame(factors, index=components.index)
    frame.index.name = "date"
    return frame


def build_factors(inputs: FactorInputs) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (raw components, signed standardized components, five composite factors)."""
    raw = raw_components(inputs)
    comps = standardized_components(raw)
    return raw, comps, composite_factors(comps)


# --------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------
@dataclass
class Check:
    name: str
    status: str  # PASS / WARN / FAIL
    detail: str


def component_dominance(components: pd.DataFrame) -> pd.DataFrame:
    """Average share of |signed z| each component contributes, and its correlation with the factor."""
    factors = composite_factors(components)
    rows = []
    for factor in FACTORS:
        cols = [f"{c.factor}__{c.name}" for c in COMPONENTS if c.factor == factor and c.in_composite]
        block = components[cols]
        valid_rows = factors[factor].notna()
        absolute = block[valid_rows].abs()
        share = absolute.div(absolute.sum(axis=1).replace(0, np.nan), axis=0).mean()
        for col in cols:
            rest = block.drop(columns=col).mean(axis=1)
            rows.append({
                "factor": factor,
                "component": col.split("__", 1)[1],
                "avg_abs_share": round(float(share[col]), 3) if pd.notna(share[col]) else np.nan,
                "corr_with_factor": round(float(block[col].corr(factors[factor])), 3),
                "corr_with_other_components": round(float(block[col].corr(rest)), 3),
            })
    return pd.DataFrame(rows)


def validate(
    inputs: FactorInputs,
    raw: pd.DataFrame,
    components: pd.DataFrame,
    factors: pd.DataFrame,
    macro_obs_dates: pd.DataFrame | None,
    daily_macro_series: list[str],
    sample_start: pd.Timestamp,
    cutoffs: list[pd.Timestamp] | None = None,
) -> list[Check]:
    checks: list[Check] = []

    # 1. finiteness
    n_inf = sum(int(np.isinf(df.to_numpy(dtype=float)).sum()) for df in (raw, components, factors))
    checks.append(Check("No +/-inf values", "PASS" if n_inf == 0 else "FAIL", f"{n_inf} infinite values"))

    # 2. leakage: rebuild on truncated information sets and compare history
    cal = inputs.calendar
    if cutoffs is None:
        cutoffs = [cal[int(len(cal) * q)] for q in (0.45, 0.75)]
    worst = 0.0
    for cutoff in cutoffs:
        _, comps_cut, fac_cut = build_factors(inputs.truncate(cutoff))
        for full, cut in ((components, comps_cut), (factors, fac_cut)):
            a = full.loc[:cutoff].to_numpy(dtype=float)
            b = cut.reindex(full.loc[:cutoff].index).to_numpy(dtype=float)
            mismatch_nan = int((np.isnan(a) != np.isnan(b)).sum())
            diff = np.nanmax(np.abs(a - b)) if np.isfinite(a - b).any() else 0.0
            worst = max(worst, float(diff), float(mismatch_nan))
    checks.append(Check(
        "No future leakage (truncated rebuild identical)",
        "PASS" if worst < 1e-9 else "FAIL",
        f"cutoffs {', '.join(c.date().isoformat() for c in cutoffs)}; max abs difference {worst:.2e}",
    ))

    # 3. point-in-time macro
    if macro_obs_dates is not None and not macro_obs_dates.empty:
        obs = macro_obs_dates.apply(pd.to_datetime)
        trade = obs.index.to_series()
        future = int(sum((obs[c] > trade).sum() for c in obs.columns))
        same_day = int(sum((obs[c] >= trade).sum() for c in daily_macro_series if c in obs.columns))
        status = "PASS" if future == 0 and same_day == 0 else "FAIL"
        checks.append(Check("Macro values point-in-time", status,
                            f"{future} values with observation date after trade date; "
                            f"{same_day} daily FRED values used on their own observation date"))
    else:
        checks.append(Check("Macro values point-in-time", "WARN", "macro_pit_obs_date.csv not available"))

    # 4. coverage inside the investment sample
    in_sample = factors.loc[sample_start:]
    coverage = in_sample.notna().mean()
    low = coverage[coverage < 0.95]
    checks.append(Check(
        "Factor coverage >= 95% in investment sample",
        "PASS" if low.empty else "WARN",
        "; ".join(f"{k}={v:.1%}" for k, v in coverage.items()),
    ))

    # 5. single-component dominance
    dom = component_dominance(components.loc[sample_start:])
    multi = dom.groupby("factor").filter(lambda g: len(g) >= 3)
    heavy = multi[multi["avg_abs_share"] > 0.5]
    checks.append(Check(
        "No component > 50% of its factor's absolute signal",
        "PASS" if heavy.empty else "WARN",
        "max share " + "; ".join(f"{f}={g['avg_abs_share'].max():.2f}" for f, g in dom.groupby("factor")),
    ))

    # 6. factors are distinct
    corr = in_sample.corr()
    off = corr.where(~np.eye(len(corr), dtype=bool)).abs()
    max_corr = float(off.max().max())
    pair = off.stack().idxmax() if off.notna().any().any() else ("-", "-")
    status = "PASS" if max_corr < 0.85 else ("WARN" if max_corr < 0.95 else "FAIL")
    checks.append(Check("Factors not near-identical (|rho| < 0.85)", status,
                        f"max |rho| = {max_corr:.2f} ({pair[0]} vs {pair[1]})"))
    return checks
