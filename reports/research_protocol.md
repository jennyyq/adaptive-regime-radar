# Research protocol: Adaptive Regime Radar

Protocol version: **ARR-stage3-protocol-v1.2** (generated 2026-10-20).
This document fixes the data, timing, evaluation and change-control rules before any regime model is fitted.

## 1. Universe and data

- Investment universe (9 ETFs): SPY, EFA, EEM, IEF, TIP, HYG, GLD, DBC, BIL. The universe was chosen ex post; all nine funds
  survive to today (a design-level limitation, disclosed rather than corrected).
- Master calendar: SPY trading dates. Sessions before the first train row are warm-up, used only for causal
  rolling statistics.
- Sources: Yahoo Finance (adjusted daily prices via yfinance 1.7.0),
  FRED / ALFRED (sources per series in `data/splits/split_manifest.json`).
  Raw download time (UTC): 2026-10-20T14:59:38+00:00.
- Environment: Python 3.12.0, numpy 2.2.6, pandas 3.0.6, matplotlib 3.11.2.

## 2. Features (model inputs, frozen)

Five composite factors from `src/regime_radar/factors.py` (sha256 `28414cd58b03be827a7f5635f8b005b6cecbc87bc7f0ac6795fb3be7c38601c6`):

- **Equity Trend** (`equity_trend`): spy_mom_63d, spy_mom_12_1, sector_breadth_200d, equal_vs_cap_weight_63d, small_vs_large_63d
- **Interest Rate Conditions** (`interest_rate`): ust2y_change_63d, ust10y_change_63d, curve_10y2y, ust10y_realized_vol_21d
- **Inflation Pressure** (`inflation`): cpi_yoy, core_cpi_yoy, breakeven_5y, tip_vs_ief_63d, commodity_mom_126d
- **Volatility Stress** (`volatility`): vix_log, spy_realized_vol_21d, spy_downside_vol_63d, vix_term_structure, risk_asset_correlation_63d
- **Liquidity / Credit Stress** (`liquidity_credit`): baa_10y_spread, hyg_vs_lqd_63d, nfci, hyg_illiquidity_21d

Construction is frozen: past-only rolling median/MAD z-scores (252 days; CPI components 756), economic signs,
clip +/-5, equal weights. No additional feature engineering, feature selection, PCA or full-sample
standardisation is part of the dataset. Any later scaler or projection is a model component, fitted on
training information only.

## 3. Splits

| split | configured start | configured end | first date | last date | rows |
|---|---|---|---|---|---|
| train | 2007-05-30 | 2015-12-31 | 2007-10-30 | 2015-12-31 | 2058 |
| validation | 2016-01-01 | 2019-12-31 | 2016-01-04 | 2019-12-31 | 1006 |
| test | 2020-01-01 | latest | 2020-01-02 | 2026-10-19 | 1708 |

- Chronological, mutually exclusive, collectively exhaustive over the usable sample; never shuffled.
- The train split is configured from 2007-05-30, the first date on which all nine ETFs have a price (BIL is the
  binding fund). The first return is the next session, because a return requires the previous close. The first
  train row is 2007-10-30, the first session on which a return exists and all five factors are available;
  106 earlier sessions are excluded because liquidity_credit is not yet available (no fill).
- Accepted effective train start (protocol v1.2): **2007-10-30**, matching the current data. Reason:
  Liquidity / Credit Stress needs >= 2 of its 4 components. Before 2007-10-30 only BAA10Y is available: the HYG Amihud component starts 2007-10-30, HYG/LQD 2008-01-09, and NFCI has no real-time value before its 2011 ALFRED vintages. Dates are excluded, not filled. The researcher accepted this on 2026-10-20, before any model was fitted.
  The effective train sample is 2058 rows (2007-10-30 to 2015-12-31).
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

- `NFCI` (liquidity_credit: nfci; source fred_api_first_release): verified first-release (category A) from 2011-06-08. Share of trade days - train: A 55.9%, B/C 0.2%, no value 43.9%; validation: A 100.0%; test: A 100.0%.
- `CPIAUCSL` (inflation: cpi_yoy; source fred_api_first_release): verified first-release (category A) from 2007-10-30. Share of trade days - train: A 100.0%; validation: A 100.0%; test: A 100.0%.
- `CPILFESL` (inflation: core_cpi_yoy; source fred_api_first_release): verified first-release (category A) from 2007-10-30. Share of trade days - train: A 100.0%; validation: A 100.0%; test: A 100.0%.

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

All stochastic estimators use seed **20240531**; stability is reported across seeds 20240531-20240540.

## 10. Change control

| version | change |
|---|---|
| v1.0 | Initial stage-3 protocol: chronological splits, timing, frozen features. |
| v1.1 | Clarified test-period exposure (descriptive inspection vs fitting vs selection vs evaluation); distinguished static holdout from sequential walk-forward OOS; added PIT categories A-D and verified-coverage dates; strengthened prefix-invariance tests; added change-control rule. Splits, data and factor formulas unchanged. |
| v1.2 | Accepted the effective first train row 2007-10-30 (determined by factor availability, see ACCEPTED_FIRST_TRAIN_ROW). Decision taken before any model was fitted and without test-period outcomes. Split boundaries, data and factor formulas unchanged. |

Amendments may clarify documentation or add validation. Any change to splits, data, factor formulas or
evaluation rules requires a new version entry stating whether test-period information was seen before the
change.
