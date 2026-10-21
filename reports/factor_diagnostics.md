# Factor diagnostics: five economic risk factors

Calendar: 2004-01-02 to 2026-10-19 (5735 trading days). Investment sample starts 2007-05-30; statistics marked 'in sample' use dates from that point on.

## Validation checks

| name | status | detail |
|---|---|---|
| No +/-inf values | PASS | 0 infinite values |
| No future leakage (truncated rebuild identical) | PASS | cutoffs 2014-04-03, 2021-02-03; max abs difference 0.00e+00 |
| Macro values point-in-time | PASS | 0 values with observation date after trade date; 0 daily FRED values used on their own observation date |
| Factor coverage >= 95% in investment sample | PASS | equity_trend=100.0%; interest_rate=100.0%; inflation=100.0%; volatility=100.0%; liquidity_credit=97.8% |
| No component > 50% of its factor's absolute signal | PASS | max share equity_trend=0.24; inflation=0.22; interest_rate=0.32; liquidity_credit=0.31; volatility=0.22 |
| Factors not near-identical (|rho| < 0.85) | PASS | max |rho| = 0.71 (equity_trend vs liquidity_credit) |

## Construction

- Each component is standardized with a rolling robust z-score against the previous 252 trading days (minimum 126 observations): `z_t = (x_t - median[t-W..t-1]) / (1.4826 * MAD[t-W..t-1])`. Day t never enters its own center or scale; no future data are used; no full-sample mean/std anywhere.
- Monthly CPI components use a 756-day window (minimum 252) because a monthly series changes only ~12 times per year and a 1-year MAD is unstable.
- If MAD is zero the window standard deviation is used; if that is also zero the value is missing.
- Components are signed so that higher = more stress (Equity Trend: higher = healthier), then winsorized at +/-5.
- Factor = equal-weight mean of available signed components; a factor needs at least 50% of its composite components (rounded up) on that day. No weights are estimated.
- Macro inputs come from the point-in-time panel (`macro_pit.csv`); CPI year-over-year is computed from the release log and becomes usable only once both months were published.

## Factor definitions

| factor | name | direction | composite components |
|---|---|---|---|
| equity_trend | Equity Trend | higher = stronger, broader equity trend (risk-on) | spy_mom_63d, spy_mom_12_1, sector_breadth_200d, equal_vs_cap_weight_63d, small_vs_large_63d |
| interest_rate | Interest Rate Conditions | higher = tighter rates / larger rate shock | ust2y_change_63d, ust10y_change_63d, curve_10y2y, ust10y_realized_vol_21d |
| inflation | Inflation Pressure | higher = more inflation pressure | cpi_yoy, core_cpi_yoy, breakeven_5y, tip_vs_ief_63d, commodity_mom_126d |
| volatility | Volatility Stress | higher = more volatility stress | vix_log, spy_realized_vol_21d, spy_downside_vol_63d, vix_term_structure, risk_asset_correlation_63d |
| liquidity_credit | Liquidity / Credit Stress | higher = more liquidity / credit stress | baa_10y_spread, hyg_vs_lqd_63d, nfci, hyg_illiquidity_21d |

## Components and sign conventions

| component | factor | definition | source | sign | window | in_composite | raw_first_valid | z_first_valid | z_missing_in_sample_pct |
|---|---|---|---|---|---|---|---|---|---|
| spy_mom_63d | equity_trend | SPY 63-day log return | Yahoo SPY adj close | +1 | 252 | yes | 2004-04-02 | 2004-10-04 | 0.0 |
| spy_mom_12_1 | equity_trend | SPY 12-1 month momentum: log(P[t-21] / P[t-252]) | Yahoo SPY adj close | +1 | 252 | yes | 2005-01-03 | 2005-07-05 | 0.0 |
| sector_breadth_200d | equity_trend | Share of 9 SPDR sectors above their 200-day average (10-day mean) | Yahoo XL* adj close | +1 | 252 | yes | 2004-10-22 | 2005-04-25 | 0.0 |
| equal_vs_cap_weight_63d | equity_trend | 63-day change of log(RSP / SPY) | Yahoo RSP, SPY | +1 | 252 | yes | 2004-04-02 | 2004-10-04 | 0.0 |
| small_vs_large_63d | equity_trend | 63-day change of log(IWM / SPY) | Yahoo IWM, SPY | +1 | 252 | yes | 2004-04-02 | 2004-10-04 | 0.0 |
| ust2y_change_63d | interest_rate | 63-day change in 2Y Treasury yield (pp) | FRED DGS2 (PIT) | +1 | 252 | yes | 2004-04-05 | 2004-10-05 | 0.0 |
| ust10y_change_63d | interest_rate | 63-day change in 10Y Treasury yield (pp) | FRED DGS10 (PIT) | +1 | 252 | yes | 2004-04-05 | 2004-10-05 | 0.0 |
| curve_10y2y | interest_rate | 10Y minus 2Y Treasury spread (pp) | FRED T10Y2Y (PIT) | -1 | 252 | yes | 2004-01-05 | 2004-07-07 | 0.0 |
| ust10y_realized_vol_21d | interest_rate | 21-day realized volatility of daily 10Y yield changes, annualized (pp) | FRED DGS10 (PIT) | +1 | 252 | yes | 2004-01-27 | 2004-07-28 | 0.0 |
| cpi_yoy | inflation | Headline CPI year-over-year % (as first published, by release date) | FRED/ALFRED CPIAUCSL | +1 | 756 | yes | 2005-02-23 | 2006-02-23 | 0.0 |
| core_cpi_yoy | inflation | Core CPI year-over-year % (as first published, by release date) | FRED/ALFRED CPILFESL | +1 | 756 | yes | 2005-02-23 | 2006-02-23 | 0.0 |
| breakeven_5y | inflation | 5-year breakeven inflation (pp); 10-year used if 5-year missing | FRED T5YIE (PIT) | +1 | 252 | yes | 2004-01-05 | 2004-07-07 | 0.0 |
| tip_vs_ief_63d | inflation | 63-day change of log(TIP / IEF) | Yahoo TIP, IEF | +1 | 252 | yes | 2004-04-02 | 2004-10-04 | 0.0 |
| commodity_mom_126d | inflation | DBC 126-day log return | Yahoo DBC | +1 | 252 | yes | 2006-08-07 | 2007-02-07 | 0.0 |
| vix_log | volatility | log(VIX) | Yahoo ^VIX | +1 | 252 | yes | 2004-01-02 | 2004-07-06 | 0.0 |
| spy_realized_vol_21d | volatility | SPY 21-day realized volatility, annualized | Yahoo SPY | +1 | 252 | yes | 2004-02-03 | 2004-08-04 | 0.0 |
| spy_downside_vol_63d | volatility | SPY 63-day downside semi-deviation, annualized | Yahoo SPY | +1 | 252 | yes | 2004-03-16 | 2004-09-15 | 0.0 |
| vix_term_structure | volatility | log(VIX / VIX3M); > 0 means inverted (backwardation) | Yahoo ^VIX, ^VIX3M | +1 | 252 | yes | 2006-07-17 | 2007-01-17 | 0.0 |
| risk_asset_correlation_63d | volatility | Average pairwise 63-day correlation of SPY, EFA, EEM, HYG, DBC daily returns | Yahoo | +1 | 252 | yes | 2004-03-16 | 2004-09-15 | 0.0 |
| baa_10y_spread | liquidity_credit | Moody's Baa yield minus 10Y Treasury (pp) | FRED BAA10Y (PIT) | +1 | 252 | yes | 2004-01-05 | 2004-07-07 | 0.0 |
| hyg_vs_lqd_63d | liquidity_credit | 63-day change of log(HYG / LQD) | Yahoo HYG, LQD | -1 | 252 | yes | 2007-07-11 | 2008-01-09 | 3.2 |
| nfci | liquidity_credit | Chicago Fed National Financial Conditions Index | FRED/ALFRED NFCI (PIT) | +1 | 252 | yes | 2011-06-01 | 2011-11-29 | 23.3 |
| hyg_illiquidity_21d | liquidity_credit | log of 21-day mean Amihud ratio |r| / dollar volume for HYG | Yahoo HYG close, volume | +1 | 252 | yes | 2007-05-02 | 2007-10-30 | 2.2 |
| hy_oas | liquidity_credit | ICE BofA US High Yield OAS (pp) | FRED BAMLH0A0HYM2 (PIT) | +1 | 252 | monitor only | 2023-10-24 | 2024-04-25 | 87.2 |

Sign rationale:

- `spy_mom_63d` (+1): Positive medium-term momentum = healthier trend
- `spy_mom_12_1` (+1): Long-horizon trend, skipping the last month to avoid short-term reversal
- `sector_breadth_200d` (+1): More sectors in uptrend = broader participation
- `equal_vs_cap_weight_63d` (+1): Equal-weight outperforming cap-weight = rally is broad, not narrow
- `small_vs_large_63d` (+1): Small caps leading = risk appetite and domestic growth confidence
- `ust2y_change_63d` (+1): Rising front-end yields = policy tightening priced in
- `ust10y_change_63d` (+1): Rising long yields = duration losses and higher discount rates
- `curve_10y2y` (-1): Flatter / inverted curve = restrictive policy stance (sign flipped)
- `ust10y_realized_vol_21d` (+1): Volatile rates = rate shock / bond-market stress
- `cpi_yoy` (+1): Higher realized inflation
- `core_cpi_yoy` (+1): Higher persistent (ex food & energy) inflation
- `breakeven_5y` (+1): Higher market-implied inflation expectations
- `tip_vs_ief_63d` (+1): Inflation-linked bonds beating nominals = breakevens rising
- `commodity_mom_126d` (+1): Rising commodity prices feed input-cost inflation
- `vix_log` (+1): Higher implied equity volatility
- `spy_realized_vol_21d` (+1): Higher realized volatility
- `spy_downside_vol_63d` (+1): Losses concentrated on down days
- `vix_term_structure` (+1): Near-term fear above 3-month fear = acute stress
- `risk_asset_correlation_63d` (+1): Risk assets moving together = diversification failure
- `baa_10y_spread` (+1): Wider corporate spreads = more credit risk premium
- `hyg_vs_lqd_63d` (-1): High yield underperforming investment grade = credit stress (sign flipped)
- `nfci` (+1): Positive / rising NFCI = tighter financial conditions
- `hyg_illiquidity_21d` (+1): Larger price impact per dollar traded = poorer credit-market liquidity
- `hy_oas` (+1): Wider HY spreads = credit stress. MONITOR ONLY: FRED serves ~3 years of history

## Factor summary statistics

| factor | first_valid | missing_full_pct | missing_in_sample_pct | mean | std | p05 | median | p95 | min | max |
|---|---|---|---|---|---|---|---|---|---|---|
| equity_trend | 2004-10-04 | 3.3 | 0.0 | -0.2 | 1.08 | -2.15 | -0.03 | 1.23 | -4.84 | 4.23 |
| interest_rate | 2004-07-28 | 2.5 | 0.0 | 0.08 | 0.83 | -1.22 | -0.01 | 1.53 | -2.66 | 3.63 |
| inflation | 2006-02-23 | 9.4 | 0.0 | -0.02 | 1.04 | -1.88 | -0.08 | 1.67 | -3.8 | 3.65 |
| volatility | 2004-09-15 | 3.1 | 0.0 | 0.15 | 1.27 | -1.36 | -0.14 | 2.93 | -2.92 | 4.96 |
| liquidity_credit | 2007-10-30 | 16.8 | 2.2 | -0.06 | 1.13 | -1.38 | -0.33 | 2.21 | -2.41 | 5.0 |

## Correlations between factors (in sample)

|  | equity_trend | interest_rate | inflation | volatility | liquidity_credit |
|---|---|---|---|---|---|
| equity_trend | 1.0 | 0.0 | 0.26 | -0.64 | -0.71 |
| interest_rate | 0.0 | 1.0 | 0.45 | 0.0 | -0.01 |
| inflation | 0.26 | 0.45 | 1.0 | -0.35 | -0.38 |
| volatility | -0.64 | 0.0 | -0.35 | 1.0 | 0.69 |
| liquidity_credit | -0.71 | -0.01 | -0.38 | 0.69 | 1.0 |

## Component contribution (in sample)

`avg_abs_share` = average share of the factor's absolute signal coming from the component; `corr_with_other_components` = correlation with the mean of the factor's other components.

| component | factor | avg_abs_share | corr_with_factor | corr_with_other_components |
|---|---|---|---|---|
| spy_mom_63d | equity_trend | 0.179 | 0.738 | 0.568 |
| spy_mom_12_1 | equity_trend | 0.238 | 0.568 | 0.281 |
| sector_breadth_200d | equity_trend | 0.212 | 0.803 | 0.586 |
| equal_vs_cap_weight_63d | equity_trend | 0.186 | 0.594 | 0.362 |
| small_vs_large_63d | equity_trend | 0.184 | 0.593 | 0.394 |
| ust2y_change_63d | interest_rate | 0.227 | 0.782 | 0.472 |
| ust10y_change_63d | interest_rate | 0.219 | 0.589 | 0.261 |
| curve_10y2y | interest_rate | 0.317 | 0.507 | -0.007 |
| ust10y_realized_vol_21d | interest_rate | 0.238 | 0.43 | 0.032 |
| cpi_yoy | inflation | 0.199 | 0.611 | 0.332 |
| core_cpi_yoy | inflation | 0.223 | 0.614 | 0.333 |
| breakeven_5y | inflation | 0.202 | 0.776 | 0.606 |
| tip_vs_ief_63d | inflation | 0.17 | 0.597 | 0.392 |
| commodity_mom_126d | inflation | 0.205 | 0.672 | 0.439 |
| vix_log | volatility | 0.195 | 0.928 | 0.88 |
| spy_realized_vol_21d | volatility | 0.196 | 0.891 | 0.817 |
| spy_downside_vol_63d | volatility | 0.2 | 0.888 | 0.795 |
| vix_term_structure | volatility | 0.189 | 0.693 | 0.563 |
| risk_asset_correlation_63d | volatility | 0.221 | 0.682 | 0.51 |
| baa_10y_spread | liquidity_credit | 0.312 | 0.835 | 0.53 |
| hyg_vs_lqd_63d | liquidity_credit | 0.23 | 0.539 | 0.276 |
| nfci | liquidity_credit | 0.251 | 0.828 | 0.629 |
| hyg_illiquidity_21d | liquidity_credit | 0.262 | 0.71 | 0.467 |

## Largest historical readings

Highest values for stress factors; lowest values for Equity Trend. One date per 63-trading-day episode.

**Equity Trend**

| date | value |
|---|---|
| 2020-04-21 | -4.84 |
| 2011-10-03 | -3.72 |
| 2008-11-25 | -3.5 |
| 2015-09-24 | -3.08 |
| 2008-01-17 | -2.69 |

**Interest Rate Conditions**

| date | value |
|---|---|
| 2022-03-29 | 3.63 |
| 2021-12-08 | 2.53 |
| 2026-05-26 | 2.33 |
| 2026-09-29 | 2.29 |
| 2013-07-15 | 2.18 |

**Inflation Pressure**

| date | value |
|---|---|
| 2022-03-21 | 3.65 |
| 2021-11-17 | 3.06 |
| 2026-05-18 | 2.17 |
| 2021-05-12 | 2.16 |
| 2016-05-18 | 1.9 |

**Volatility Stress**

| date | value |
|---|---|
| 2020-03-12 | 4.96 |
| 2008-10-15 | 4.63 |
| 2018-02-08 | 4.43 |
| 2011-08-18 | 4.34 |
| 2025-04-07 | 4.19 |

**Liquidity / Credit Stress**

| date | value |
|---|---|
| 2020-03-25 | 5.0 |
| 2008-10-27 | 3.61 |
| 2018-12-24 | 3.24 |
| 2015-01-06 | 3.11 |
| 2022-05-20 | 2.84 |

## Diagnostic event windows (not regime labels)

Percentile = share of all historical factor values below the window mean. Both tails are shown: e.g. Interest Rate Conditions typically falls (easing) in growth shocks and rises in rate shocks.

| event | factor | window_mean | window_min | min_date | window_max | max_date | mean_percentile_vs_history |
|---|---|---|---|---|---|---|---|
| 2008 Global Financial Crisis | equity_trend | -1.42 | -3.5 | 2008-11-25 | 0.32 | 2008-09-19 | 13.0 |
| 2008 Global Financial Crisis | interest_rate | -0.11 | -0.81 | 2008-11-13 | 0.85 | 2009-03-24 | 44.0 |
| 2008 Global Financial Crisis | inflation | -2.32 | -3.78 | 2008-12-16 | -0.6 | 2008-09-02 | 3.0 |
| 2008 Global Financial Crisis | volatility | 2.43 | -0.39 | 2008-09-03 | 4.63 | 2008-10-15 | 93.0 |
| 2008 Global Financial Crisis | liquidity_credit | 1.56 | -0.01 | 2009-03-27 | 3.61 | 2008-10-27 | 89.0 |
| 2020 COVID shock | equity_trend | -3.12 | -4.84 | 2020-04-21 | 0.39 | 2020-02-20 | 2.0 |
| 2020 COVID shock | interest_rate | -1.02 | -1.71 | 2020-04-23 | 0.36 | 2020-02-19 | 7.0 |
| 2020 COVID shock | inflation | -1.81 | -2.97 | 2020-04-21 | 0.67 | 2020-02-19 | 5.0 |
| 2020 COVID shock | volatility | 3.84 | -0.1 | 2020-02-19 | 4.96 | 2020-03-12 | 98.0 |
| 2020 COVID shock | liquidity_credit | 3.18 | -0.65 | 2020-02-21 | 5.0 | 2020-03-25 | 98.0 |
| 2022 inflation / rates shock | equity_trend | -1.19 | -2.42 | 2022-02-07 | 0.12 | 2022-09-19 | 17.0 |
| 2022 inflation / rates shock | interest_rate | 1.68 | -0.32 | 2022-09-15 | 3.63 | 2022-03-29 | 96.0 |
| 2022 inflation / rates shock | inflation | 1.62 | -0.18 | 2022-09-30 | 3.65 | 2022-03-21 | 95.0 |
| 2022 inflation / rates shock | volatility | 1.07 | -0.26 | 2022-08-18 | 2.54 | 2022-02-23 | 81.0 |
| 2022 inflation / rates shock | liquidity_credit | 1.31 | -0.31 | 2022-01-25 | 2.84 | 2022-05-20 | 87.0 |

## Data limitations

- **HY OAS (BAMLH0A0HYM2)**: starts 2023-10-24. FRED now serves only a rolling ~3-year window, so it is monitor-only and excluded from the Liquidity / Credit composite (BAA10Y and HYG/LQD carry the credit-spread signal over the full sample).
- **NFCI** release dates: vintage:first_release=802, fallback:pre_vintage_history=1. 'fallback' rows use rule-based release dates and current-vintage values. NFCI is re-estimated every week, so current-vintage history is smoother than what was known in real time.
- **CPI** release dates: vintage:first_release=273. 'fallback' rows use rule-based release dates and current-vintage values. CPI revisions are small (mainly seasonal factors), so the effect is minor.
- **Core CPI** release dates: vintage:first_release=273. 'fallback' rows use rule-based release dates and current-vintage values. Core CPI revisions are small (mainly seasonal factors), so the effect is minor.
- **VIX term structure** starts 2006-07-17 (VIX3M history); before that Volatility Stress averages the remaining components.
- **HYG-based components** start 2007-07-11; Liquidity / Credit Stress uses BAA10Y and NFCI before that.
- **HYG vs LQD** also contains a duration difference (LQD is longer duration), so in rising-rate periods it can understate credit stress.
- **Rolling 1-year standardization** measures stress relative to the trailing year: in a long stress period (e.g. 2008-2009) z-scores drift back toward zero as the window absorbs the stress. Keep this in mind when interpreting persistence in the regime model.
- Statistics in this report cover all valid dates; the investment sample starts 2007-05-30.
