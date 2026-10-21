# Dataset split audit (stage 3)

Protocol `ARR-stage3-protocol-v1.2`; generated 2026-10-20T15:02:17+00:00 UTC. Checks: 34 PASS, 4 WARN, 0 FAIL, 1 NOT RUN.

## 0. Scope of this audit

- The checks below were executed on the files whose SHA-256 hashes are listed in `data/splits/split_manifest.json` (`feature_provenance.input_sha256`, `split_files`). A PASS applies to those files only. PASS results from `tests/` use synthetic data and are not evidence about this dataset.
- A finite set of checks cannot prove the absence of every form of leakage. The prefix-invariance tests establish that, for the tested cutoffs, no stored value depends on later rows of the processed inputs. They cannot detect information that is already embedded in the inputs themselves (e.g. revised or retrospectively estimated macro history, ex-post universe selection); those are covered in sections 5-6.

Status definitions:

- **PASS**: Validated against the data passed to the check (for scripts/05: the local dataset with the hashes in split_manifest.json).
- **WARN**: Methodological concern or historical assumption that cannot be verified from the available data.
- **FAIL**: Confirmed inconsistency or validation failure.
- **NOT RUN**: Could not be executed against the real dataset (input or historical data not available).

## 1. Dataset overview and timing convention

- SPY master calendar: 2004-01-02 to 2026-10-19 (5735 sessions).
- Usable investment sample: 2007-10-30 to 2026-10-19 (4772 dates with all 5 factors and all 9 ETF returns). Earlier sessions are warm-up only.
- Features (model inputs): equity_trend, interest_rate, inflation, volatility, liquidity_credit.
- Returns (stored in separate files): daily log returns of SPY, EFA, EEM, IEF, TIP, HYG, GLD, DBC, BIL.

| quantity | definition | known at |
|---|---|---|
| feature_t | factor value computed from prices up to the close of t and macro values published by t | after the US close of t (signal formation) |
| return_t | log(P_t / P_t-1), adjusted close, same day | after the US close of t (historical, not a future outcome) |
| decision_t | position chosen from feature_t | after the close of t; earliest execution in session t+1 |
| earned return | return_(t+1) for a close-to-close implementation of decision_t | after the close of t+1 |

Feature and return files share the same date index for storage only. Evaluating a date-t signal against the date-t return would credit the strategy with a return realised before the signal existed; the backtest must pair decision_t with return_(t+1) (or the return of the actual execution convention). No forward-return labels are created in this stage.

## 2. Split summary and actual observation counts

| split | configured_start | configured_end | first_date | last_date | rows | share_pct |
|---|---|---|---|---|---|---|
| train | 2007-05-30 | 2015-12-31 | 2007-10-30 | 2015-12-31 | 2058 | 43.1 |
| validation | 2016-01-01 | 2019-12-31 | 2016-01-04 | 2019-12-31 | 1006 | 21.1 |
| test | 2020-01-01 | latest | 2020-01-02 | 2026-10-19 | 1708 | 35.8 |

Total: 4772 rows. Dates inside the configured range but not used:

| date | reason |
|---|---|
| 2007-05-30 | no return row (e.g. first price date: return undefined) |
| 2007-05-31 | missing factor value: liquidity_credit |
| 2007-06-01 | missing factor value: liquidity_credit |
| 2007-06-04 | missing factor value: liquidity_credit |
| 2007-06-05 | missing factor value: liquidity_credit |
| 2007-06-06 | missing factor value: liquidity_credit |
| 2007-06-07 | missing factor value: liquidity_credit |
| 2007-06-08 | missing factor value: liquidity_credit |
| 2007-06-11 | missing factor value: liquidity_credit |
| 2007-06-12 | missing factor value: liquidity_credit |
| 2007-06-13 | missing factor value: liquidity_credit |
| 2007-06-14 | missing factor value: liquidity_credit |
| 2007-06-15 | missing factor value: liquidity_credit |
| 2007-06-18 | missing factor value: liquidity_credit |
| 2007-06-19 | missing factor value: liquidity_credit |
| 2007-06-20 | missing factor value: liquidity_credit |
| 2007-06-21 | missing factor value: liquidity_credit |
| 2007-06-22 | missing factor value: liquidity_credit |
| 2007-06-25 | missing factor value: liquidity_credit |
| 2007-06-26 | missing factor value: liquidity_credit |
| 2007-06-27 | missing factor value: liquidity_credit |
| 2007-06-28 | missing factor value: liquidity_credit |
| 2007-06-29 | missing factor value: liquidity_credit |
| 2007-07-02 | missing factor value: liquidity_credit |
| 2007-07-03 | missing factor value: liquidity_credit |
| 2007-07-05 | missing factor value: liquidity_credit |
| 2007-07-06 | missing factor value: liquidity_credit |
| 2007-07-09 | missing factor value: liquidity_credit |
| 2007-07-10 | missing factor value: liquidity_credit |
| 2007-07-11 | missing factor value: liquidity_credit |
| 2007-07-12 | missing factor value: liquidity_credit |
| 2007-07-13 | missing factor value: liquidity_credit |
| 2007-07-16 | missing factor value: liquidity_credit |
| 2007-07-17 | missing factor value: liquidity_credit |
| 2007-07-18 | missing factor value: liquidity_credit |
| 2007-07-19 | missing factor value: liquidity_credit |
| 2007-07-20 | missing factor value: liquidity_credit |
| 2007-07-23 | missing factor value: liquidity_credit |
| 2007-07-24 | missing factor value: liquidity_credit |
| 2007-07-25 | missing factor value: liquidity_credit |
| 2007-07-26 | missing factor value: liquidity_credit |
| 2007-07-27 | missing factor value: liquidity_credit |
| 2007-07-30 | missing factor value: liquidity_credit |
| 2007-07-31 | missing factor value: liquidity_credit |
| 2007-08-01 | missing factor value: liquidity_credit |
| 2007-08-02 | missing factor value: liquidity_credit |
| 2007-08-03 | missing factor value: liquidity_credit |
| 2007-08-06 | missing factor value: liquidity_credit |
| 2007-08-07 | missing factor value: liquidity_credit |
| 2007-08-08 | missing factor value: liquidity_credit |
| 2007-08-09 | missing factor value: liquidity_credit |
| 2007-08-10 | missing factor value: liquidity_credit |
| 2007-08-13 | missing factor value: liquidity_credit |
| 2007-08-14 | missing factor value: liquidity_credit |
| 2007-08-15 | missing factor value: liquidity_credit |
| 2007-08-16 | missing factor value: liquidity_credit |
| 2007-08-17 | missing factor value: liquidity_credit |
| 2007-08-20 | missing factor value: liquidity_credit |
| 2007-08-21 | missing factor value: liquidity_credit |
| 2007-08-22 | missing factor value: liquidity_credit |
| 2007-08-23 | missing factor value: liquidity_credit |
| 2007-08-24 | missing factor value: liquidity_credit |
| 2007-08-27 | missing factor value: liquidity_credit |
| 2007-08-28 | missing factor value: liquidity_credit |
| 2007-08-29 | missing factor value: liquidity_credit |
| 2007-08-30 | missing factor value: liquidity_credit |
| 2007-08-31 | missing factor value: liquidity_credit |
| 2007-09-04 | missing factor value: liquidity_credit |
| 2007-09-05 | missing factor value: liquidity_credit |
| 2007-09-06 | missing factor value: liquidity_credit |
| 2007-09-07 | missing factor value: liquidity_credit |
| 2007-09-10 | missing factor value: liquidity_credit |
| 2007-09-11 | missing factor value: liquidity_credit |
| 2007-09-12 | missing factor value: liquidity_credit |
| 2007-09-13 | missing factor value: liquidity_credit |
| 2007-09-14 | missing factor value: liquidity_credit |
| 2007-09-17 | missing factor value: liquidity_credit |
| 2007-09-18 | missing factor value: liquidity_credit |
| 2007-09-19 | missing factor value: liquidity_credit |
| 2007-09-20 | missing factor value: liquidity_credit |
| 2007-09-21 | missing factor value: liquidity_credit |
| 2007-09-24 | missing factor value: liquidity_credit |
| 2007-09-25 | missing factor value: liquidity_credit |
| 2007-09-26 | missing factor value: liquidity_credit |
| 2007-09-27 | missing factor value: liquidity_credit |
| 2007-09-28 | missing factor value: liquidity_credit |
| 2007-10-01 | missing factor value: liquidity_credit |
| 2007-10-02 | missing factor value: liquidity_credit |
| 2007-10-03 | missing factor value: liquidity_credit |
| 2007-10-04 | missing factor value: liquidity_credit |
| 2007-10-05 | missing factor value: liquidity_credit |
| 2007-10-08 | missing factor value: liquidity_credit |
| 2007-10-09 | missing factor value: liquidity_credit |
| 2007-10-10 | missing factor value: liquidity_credit |
| 2007-10-11 | missing factor value: liquidity_credit |
| 2007-10-12 | missing factor value: liquidity_credit |
| 2007-10-15 | missing factor value: liquidity_credit |
| 2007-10-16 | missing factor value: liquidity_credit |
| 2007-10-17 | missing factor value: liquidity_credit |
| 2007-10-18 | missing factor value: liquidity_credit |
| 2007-10-19 | missing factor value: liquidity_credit |
| 2007-10-22 | missing factor value: liquidity_credit |
| 2007-10-23 | missing factor value: liquidity_credit |
| 2007-10-24 | missing factor value: liquidity_credit |
| 2007-10-25 | missing factor value: liquidity_credit |
| 2007-10-26 | missing factor value: liquidity_credit |
| 2007-10-29 | missing factor value: liquidity_credit |

First eligible return: [PASS] first common price date 2007-05-30 (binding ETF: BIL); first return 2007-05-31 (next session 2007-05-31); matches protocol-documented 2007-05-31.

First usable row: [PASS] 2007-10-30: 106 sessions after the first return (2007-05-31..2007-10-29) are excluded because liquidity_credit is missing on 106 sessions; matches the effective train start accepted in the protocol (2007-10-30); configured boundary unchanged.

The configured train start (2007-05-30) is unchanged. The first train row is the first session on which a return exists AND all five factors are available (2007-10-30).
Rolling statistics are not restarted at split boundaries; earlier history (including the 2004-2007 warm-up) may inform later rows, never the reverse (section 6).

## 3. Validation checks

| name | status | detail |
|---|---|---|
| Index integrity: risk_factors | PASS | 5735 unique, increasing, date-only timestamps |
| Index integrity: investment_log_returns | PASS | 4878 unique, increasing, date-only timestamps |
| Index integrity: investment_prices | PASS | 4879 unique, increasing, date-only timestamps |
| Index integrity: factor_components | PASS | 5735 unique, increasing, date-only timestamps |
| Partitions mutually exclusive | PASS | 0 dates appear in more than one partition |
| Partitions collectively exhaustive over usable sample | PASS | 4772 usable dates; 0 unassigned; 0 unexpected |
| Partitions chronological and inside configured boundaries | PASS | train [2007-10-30..2015-12-31] < validation [2016-01-04..2019-12-31] < test [2020-01-02..2026-10-19] |
| No empty partition | PASS | empty: none |
| Identical factor schema across splits (no target / forward-return columns) | PASS | 5 columns: equity_trend, interest_rate, inflation, volatility, liquidity_credit |
| Identical ETF return schema across splits | PASS | 9 columns: SPY, EFA, EEM, IEF, TIP, HYG, GLD, DBC, BIL |
| Feature and return dates identical within each split (contemporaneous storage) | PASS | mismatched: none |
| All split dates are SPY master-calendar sessions | PASS | 0 off-calendar dates |
| No master-calendar session missing inside the usable span | PASS | 0 sessions missing |
| Excluded dates inside split range are explained (leading block only) | PASS | missing factor value: liquidity_credit: 106; no return row (e.g. first price date: return undefined): 1 |
| No NaN / +/-inf in split files | PASS | 0 non-finite values |
| First eligible return date | PASS | first common price date 2007-05-30 (binding ETF: BIL); first return 2007-05-31 (next session 2007-05-31); matches protocol-documented 2007-05-31 |
| First usable modelling row (all factors and returns present) | PASS | 2007-10-30: 106 sessions after the first return (2007-05-31..2007-10-29) are excluded because liquidity_credit is missing on 106 sessions; matches the effective train start accepted in the protocol (2007-10-30); configured boundary unchanged |
| Returns = same-day log price differences log(P_t/P_t-1) (no shift / fill) | PASS | max abs difference 1.0e-16 over 4878 dates |
| No missing investment prices (nothing to forward-fill) | PASS | 0 missing prices in the investment panel |
| No stale-price runs (>= 5 consecutive zero returns) | WARN | BIL: longest run 18 (2011-07-25..2011-08-17); zero daily returns can be genuine for a T-bill fund (BIL) when yields are near zero and prices are quoted to two decimals - check the dates against a second source |
| No factor value without underlying components (no fill) | PASS | every factor value is backed by >= 1 component on the same date |
| Stored factor files reproduce from current code and inputs (serialization only) | PASS | max abs diff 5.0e-07 (6-decimal CSV; tolerance 1e-06); NaN-pattern mismatches 0 |
| Prefix invariance: removing later data leaves history unchanged | PASS | 6 cutoffs (2006-06-30, 2011-12-30, 2015-12-31, 2017-12-29, 2019-12-31, 2023-06-30); max abs diff 0.0e+00; NaN-pattern mismatches 0 |
| Prefix invariance: altering later data leaves history unchanged | PASS | 2 cutoffs (2015-12-31, 2019-12-31); max abs diff 0.0e+00; NaN-pattern mismatches 0 |
| Prefix invariance: initial features (features.py) | PASS | truncated vs full max abs diff 0.0e+00, NaN mismatches 0; stored vs rebuilt 8.9e-16 |
| Stored macro_pit reproduces from the release log | PASS | 13 series; max abs diff 0.0e+00 |
| Macro PIT alignment uses only releases known by each cutoff | PASS | 6 cutoffs x 13 series; max abs diff 0.0e+00; NaN mismatches 0 |
| No macro observation dated after the trade date | PASS | 0 violations |
| Daily FRED values lag their observation date by >= 1 session | PASS | 0 violations |
| Macro values used only after their availability date | PASS | 0 trade-day values used before availability; 0 values not traceable to the release log |
| CPI / Core CPI / NFCI: every value used is a verified first release (category A) | WARN | NFCI: category A from 2011-06-08 (train A 55.9% / B-C 0.2% / no value 43.9%, validation A 100.0% / B-C 0.0% / no value 0.0%, test A 100.0% / B-C 0.0% / no value 0.0%); CPIAUCSL: category A from 2007-10-30 (train A 100.0% / B-C 0.0% / no value 0.0%, validation A 100.0% / B-C 0.0% / no value 0.0%, test A 100.0% / B-C 0.0% / no value 0.0%); CPILFESL: category A from 2007-10-30 (train A 100.0% / B-C 0.0% / no value 0.0%, validation A 100.0% / B-C 0.0% / no value 0.0%, test A 100.0% / B-C 0.0% / no value 0.0%); B/C values are used on some trade days and are not verifiable as point-in-time; NFCI has no real-time value on part of the sample: the component is absent there (not filled), which shortens its history instead of using unverifiable values |
| Daily FRED series timing (category D) is a conservative assumption | WARN | availability = observation date + 1 US business day; no publication timestamps recorded; values are current vintage |
| Historical revisions of daily FRED series checked against ALFRED vintages | NOT RUN | daily-series vintages were not downloaded |
| Release time of day versus signal formation (after US close) | WARN | availability is date-level only; scheduled 08:30 ET releases (CPI, NFCI) precede the 16:00 ET close, and daily series are used one session later, but exact timestamps are not in the data |
| Factor composition stable in validation and test | PASS | none |
| Composition changes inside train create no artificial jump | PASS | 2 changes: liquidity_credit 2008-01-09 +[hyg_vs_lqd_63d] -[-] effect 0.349; liquidity_credit 2011-11-29 +[nfci] -[-] effect 0.092; corr(full set, early subset): liquidity_credit 0.894 |
| Train: no near-constant factor | PASS | min std 0.654; max share of a single value 0.6% |
| Train: no duplicate factor columns | PASS | none |
| Train: no severe multicollinearity (VIF < 10, |rho| < 0.9) | PASS | max VIF 3.18 (liquidity_credit); max |Pearson| 0.76 (volatility vs liquidity_credit); condition number 3.74 |

## 4. Research-stage exposure of the test period

| stage | test-period (2020+) status |
|---|---|
| 1. Descriptive historical inspection | **Occurred**: factor time series and 2008 / 2020 / 2022 event-window charts (event windows specified in the original research plan) were viewed. Not a blind holdout. |
| 2. Causal feature computation | **Permitted and done**: factors on test dates use only information available at each date (sections 5-6). |
| 3. Model fitting | **Not done**: no K-Means, GMM, HMM or other regime model has been fitted. |
| 4. Hyperparameter / specification selection | **Not done**; must use Train/Validation only. |
| 5. Final out-of-sample evaluation | **Not done**; 2020-latest remains the designated OOS period. |

Viewing historical charts is not, by itself, look-ahead bias, training leakage or test-set optimisation. It does mean the test period is not a never-inspected holdout; see `reports/research_protocol.md`.

## 5. Point-in-time audit of macro inputs

Categories:

- **A**: First-release value; availability date = ALFRED first-vintage date (date-level, verified).
- **B**: First-release value; availability date approximated by a publication rule.
- **C**: Value from a later vintage (revised, or retrospectively estimated before the series' first vintage); availability date approximated by a publication rule. Not verifiable as point-in-time.
- **D**: Daily market series; availability = observation date + 1 US business day (conservative rule); current-vintage value; no publication timestamp recorded.

A release-date lag establishes WHEN a value is assumed known; it does not establish that the value itself (if revised later) was historically available. Category C values pass the timing checks but are not verifiable as point-in-time.

Verified first-release coverage of the vintage-rule series (share of usable trade days in category A, in category B or C, and with no value at all):

| series | used_in | download_source | category_A_from | split_of_A_start | train_pct_A | train_pct_BC | train_pct_no_value | validation_pct_A | validation_pct_BC | validation_pct_no_value | test_pct_A | test_pct_BC | test_pct_no_value |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NFCI | liquidity_credit: nfci | fred_api_first_release | 2011-06-08 | train | 55.9 | 0.2 | 43.9 | 100.0 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 |
| CPIAUCSL | inflation: cpi_yoy | fred_api_first_release | 2007-10-30 | train | 100.0 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 |
| CPILFESL | inflation: core_cpi_yoy | fred_api_first_release | 2007-10-30 | train | 100.0 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 |

Share of trade days by series, category and availability method:

| series / category / method | train | validation | test |
|---|---|---|---|
| BAA10Y / D / rule:business_days | 100.0 | 100.0 | 100.0 |
| BAMLH0A0HYM2 / D / rule:business_days | 0.0 | 0.0 | 43.9 |
| BAMLH0A0HYM2 / none / no value | 100.0 | 100.0 | 56.1 |
| CPIAUCSL / A / vintage:first_release | 100.0 | 100.0 | 100.0 |
| CPILFESL / A / vintage:first_release | 100.0 | 100.0 | 100.0 |
| DFII10 / D / rule:business_days | 100.0 | 100.0 | 100.0 |
| DGS10 / D / rule:business_days | 100.0 | 100.0 | 100.0 |
| DGS2 / D / rule:business_days | 100.0 | 100.0 | 100.0 |
| DGS3MO / D / rule:business_days | 100.0 | 100.0 | 100.0 |
| NFCI / A / vintage:first_release | 55.9 | 100.0 | 100.0 |
| NFCI / C / fallback:pre_vintage_history | 0.2 | 0.0 | 0.0 |
| NFCI / none / no value | 43.9 | 0.0 | 0.0 |
| T10Y2Y / D / rule:business_days | 100.0 | 100.0 | 100.0 |
| T10Y3M / D / rule:business_days | 100.0 | 100.0 | 100.0 |
| T10YIE / D / rule:business_days | 100.0 | 100.0 | 100.0 |
| T5YIE / D / rule:business_days | 100.0 | 100.0 | 100.0 |

Notes:

- **NFCI**: the Chicago Fed re-estimates the index every week. Real-time releases archived in ALFRED start in 2011; earlier values are retrospective estimates produced with later data. Where ALFRED records those back-filled values with their (2011) vintage date, the pipeline treats them as unknown before that date: NFCI then has 'no value' on earlier trade days, the component is absent (not filled) and enters the Liquidity / Credit factor only after its 126-observation z-score window has filled. Where back-filled values were instead given rule-based dates, they appear above as category C and are not verifiable as point-in-time.
- **Missing component vs. missing factor**: a factor needs at least half of its components. If NFCI and the HYG-based components are all unavailable, Liquidity / Credit Stress is missing and the date is excluded from the splits (section 2) rather than filled.
- **CPI / Core CPI**: year-over-year changes are computed from the release log and become usable only after both months are published. For category A months both legs are first-release values, which were public by then (feasible in real time, although a contemporaneous analyst would have used the revised t-12 value).
- **Daily series (D)**: Treasury yields, breakevens, BAA10Y and HY OAS are used one US business day after their observation date; the issuing sources publish them by the evening of the observation date. Values are current vintage; revisions to these market series are believed rare but were not checked.
- `scripts/04` reports 'Macro values point-in-time: PASS'. That check verifies observation-date timing only, not value vintages.

Recommended PIT-verified sensitivity analysis (later stages, decided now): re-estimate the selected model (a) on train dates from the latest `category_A_from` date above, and/or (b) with the Liquidity / Credit and Inflation factors recomputed without NFCI / CPI components before their category-A start, and report whether regime assignments in validation change materially.

HY OAS (monitor only, not in any composite) coverage by split:

| split | coverage_pct |
|---|---|
| train | 0.0 |
| validation | 0.0 |
| test | 36.5 |

## 6. Causality and prefix invariance

Inspected implementation (rolling windows are trailing; no `center=True`, negative shifts, back-fills or full-sample statistics):

| element | implementation | information used for date t |
|---|---|---|
| Robust z-score | median / MAD of x[t-W..t-1] (factors.rolling_robust_z) | past W values; x_t only in the numerator |
| Volatility, downside vol, Amihud | trailing rolling windows ending at t | returns up to t |
| Momentum / relative strength | log-price differences, shift(21), shift(252) | prices up to t |
| Rolling correlation | trailing 63-day pairwise correlation | returns up to t |
| Rate / spread changes | 63-day differences of the PIT macro panel | values published by t |
| CPI YoY | ratio of two first-release values, usable at the later availability date | releases by t |
| Macro alignment | merge_asof backward on availability date (pit.align_to_calendar) | releases by t |
| Composite | mean of components present on t, >= 50% required | same-day components |
| Missing values | no fill; NaN propagates; zero MAD -> window std -> NaN | - |

Empirical tests on the actual processed inputs:

| test | information_set | output | dates_compared | max_abs_diff | nan_pattern_mismatches |
|---|---|---|---|---|---|
| serialization | full sample | factors | 5735 | 5.0e-07 | 0 |
| serialization | full sample | components | 5735 | 5.0e-07 | 0 |
| truncation | cutoff 2006-06-30 | factors | 629 | 0.0e+00 | 0 |
| truncation | cutoff 2006-06-30 | components | 629 | 0.0e+00 | 0 |
| truncation | cutoff 2011-12-30 | factors | 2015 | 0.0e+00 | 0 |
| truncation | cutoff 2011-12-30 | components | 2015 | 0.0e+00 | 0 |
| truncation | cutoff 2015-12-31 | factors | 3021 | 0.0e+00 | 0 |
| truncation | cutoff 2015-12-31 | components | 3021 | 0.0e+00 | 0 |
| truncation | cutoff 2017-12-29 | factors | 3524 | 0.0e+00 | 0 |
| truncation | cutoff 2017-12-29 | components | 3524 | 0.0e+00 | 0 |
| truncation | cutoff 2019-12-31 | factors | 4027 | 0.0e+00 | 0 |
| truncation | cutoff 2019-12-31 | components | 4027 | 0.0e+00 | 0 |
| truncation | cutoff 2023-06-30 | factors | 4907 | 0.0e+00 | 0 |
| truncation | cutoff 2023-06-30 | components | 4907 | 0.0e+00 | 0 |
| perturbation | cutoff 2015-12-31 | factors | 3021 | 0.0e+00 | 0 |
| perturbation | cutoff 2015-12-31 | components | 3021 | 0.0e+00 | 0 |
| perturbation | cutoff 2019-12-31 | factors | 4027 | 0.0e+00 | 0 |
| perturbation | cutoff 2019-12-31 | components | 4027 | 0.0e+00 | 0 |

Interpretation: `serialization` differences (<= 5e-7) come from the 6-decimal CSV format; `truncation` and `perturbation` rebuilds are compared in memory and must agree exactly.

Macro PIT alignment re-run from the release log:

| test | series | max_abs_diff | nan_pattern_mismatches |
|---|---|---|---|
| releases known by cutoff | 13 | 0.0e+00 | 0 |
| reproduce stored macro_pit | 13 | 0.0e+00 | 0 |

Not covered by these tests (inherent in the inputs): category C macro values (section 5); Yahoo adjusted closes are rescaled when later distributions occur, which leaves price ratios within any window ending at t unchanged but makes stored price levels download-date dependent; the nine-ETF universe was chosen ex post (all funds survive to today).

## 7. Factor composition stability

Component availability (share of usable dates with a value) by split:

| component | factor | in_composite | first_valid | train_available_pct | validation_available_pct | test_available_pct |
|---|---|---|---|---|---|---|
| spy_mom_63d | equity_trend | yes | 2004-10-04 | 100.0 | 100.0 | 100.0 |
| spy_mom_12_1 | equity_trend | yes | 2005-07-05 | 100.0 | 100.0 | 100.0 |
| sector_breadth_200d | equity_trend | yes | 2005-04-25 | 100.0 | 100.0 | 100.0 |
| equal_vs_cap_weight_63d | equity_trend | yes | 2004-10-04 | 100.0 | 100.0 | 100.0 |
| small_vs_large_63d | equity_trend | yes | 2004-10-04 | 100.0 | 100.0 | 100.0 |
| ust2y_change_63d | interest_rate | yes | 2004-10-05 | 100.0 | 100.0 | 100.0 |
| ust10y_change_63d | interest_rate | yes | 2004-10-05 | 100.0 | 100.0 | 100.0 |
| curve_10y2y | interest_rate | yes | 2004-07-07 | 100.0 | 100.0 | 100.0 |
| ust10y_realized_vol_21d | interest_rate | yes | 2004-07-28 | 100.0 | 100.0 | 100.0 |
| cpi_yoy | inflation | yes | 2006-02-23 | 100.0 | 100.0 | 100.0 |
| core_cpi_yoy | inflation | yes | 2006-02-23 | 100.0 | 100.0 | 100.0 |
| breakeven_5y | inflation | yes | 2004-07-07 | 100.0 | 100.0 | 100.0 |
| tip_vs_ief_63d | inflation | yes | 2004-10-04 | 100.0 | 100.0 | 100.0 |
| commodity_mom_126d | inflation | yes | 2007-02-07 | 100.0 | 100.0 | 100.0 |
| vix_log | volatility | yes | 2004-07-06 | 100.0 | 100.0 | 100.0 |
| spy_realized_vol_21d | volatility | yes | 2004-08-04 | 100.0 | 100.0 | 100.0 |
| spy_downside_vol_63d | volatility | yes | 2004-09-15 | 100.0 | 100.0 | 100.0 |
| vix_term_structure | volatility | yes | 2007-01-17 | 100.0 | 100.0 | 100.0 |
| risk_asset_correlation_63d | volatility | yes | 2004-09-15 | 100.0 | 100.0 | 100.0 |
| baa_10y_spread | liquidity_credit | yes | 2004-07-07 | 100.0 | 100.0 | 100.0 |
| hyg_vs_lqd_63d | liquidity_credit | yes | 2008-01-09 | 97.7 | 100.0 | 100.0 |
| nfci | liquidity_credit | yes | 2011-11-29 | 50.0 | 100.0 | 100.0 |
| hyg_illiquidity_21d | liquidity_credit | yes | 2007-10-30 | 100.0 | 100.0 | 100.0 |
| hy_oas | liquidity_credit | monitor only | 2024-04-25 | 0.0 | 0.0 | 36.5 |

Composition by period:

| factor | period | components_defined | min_active | max_active | full_composition_pct | partially_available | absent |
|---|---|---|---|---|---|---|---|
| equity_trend | warm-up | 5 | 0 | 5 | 60.7 | spy_mom_63d (80%), spy_mom_12_1 (61%), sector_breadth_200d (66%), equal_vs_cap_weight_63d (80%), small_vs_large_63d (80%) | - |
| equity_trend | train | 5 | 5 | 5 | 100.0 | - | - |
| equity_trend | validation | 5 | 5 | 5 | 100.0 | - | - |
| equity_trend | test | 5 | 5 | 5 | 100.0 | - | - |
| interest_rate | warm-up | 4 | 0 | 4 | 80.3 | ust2y_change_63d (80%), ust10y_change_63d (80%), curve_10y2y (87%), ust10y_realized_vol_21d (85%) | - |
| interest_rate | train | 4 | 4 | 4 | 100.0 | - | - |
| interest_rate | validation | 4 | 4 | 4 | 100.0 | - | - |
| interest_rate | test | 4 | 4 | 4 | 100.0 | - | - |
| inflation | warm-up | 5 | 0 | 5 | 19.1 | cpi_yoy (44%), core_cpi_yoy (44%), breakeven_5y (87%), tip_vs_ief_63d (80%), commodity_mom_126d (19%) | - |
| inflation | train | 5 | 5 | 5 | 100.0 | - | - |
| inflation | validation | 5 | 5 | 5 | 100.0 | - | - |
| inflation | test | 5 | 5 | 5 | 100.0 | - | - |
| volatility | warm-up | 5 | 0 | 5 | 20.7 | vix_log (87%), spy_realized_vol_21d (85%), spy_downside_vol_63d (82%), vix_term_structure (21%), risk_asset_correlation_63d (82%) | - |
| volatility | train | 5 | 5 | 5 | 100.0 | - | - |
| volatility | validation | 5 | 5 | 5 | 100.0 | - | - |
| volatility | test | 5 | 5 | 5 | 100.0 | - | - |
| liquidity_credit | warm-up | 4 | 0 | 1 | 0.0 | baa_10y_spread (87%) | hyg_vs_lqd_63d, nfci, hyg_illiquidity_21d |
| liquidity_credit | train | 4 | 2 | 4 | 50.0 | hyg_vs_lqd_63d (98%), nfci (50%) | - |
| liquidity_credit | validation | 4 | 4 | 4 | 100.0 | - | - |
| liquidity_credit | test | 4 | 4 | 4 | 100.0 | - | - |

Composition changes inside the usable sample. `inclusion_effect` = the part of the factor's move caused only by the change of component set (additions measured on the change date, removals on the previous session, both against the components active on both dates); the threshold is the 99th percentile of absolute daily factor moves on train dates only:

| factor | date | added | removed | active_after | split | inclusion_effect | factor_move | train_p99_abs_daily_move |
|---|---|---|---|---|---|---|---|---|
| liquidity_credit | 2008-01-09 | hyg_vs_lqd_63d | - | 3 | train | 0.349 | 0.488 | 0.418 |
| liquidity_credit | 2011-11-29 | nfci | - | 4 | train | 0.092 | 0.055 | 0.418 |

Train-only comparison of each factor with its early-component subset:

| factor | late_components | full_set_from | corr_full_vs_early_subset | mean_abs_gap |
|---|---|---|---|---|
| liquidity_credit | hyg_vs_lqd_63d, nfci | 2011-11-29 | 0.894 | 0.413 |

Component definitions and weights are unchanged; nothing here was selected using test outcomes. Recommended robustness checks for the modelling stage (decided now, before modelling): (i) refit on train dates after the last composition change; (ii) refit with factors restricted to components available from the first train date; (iii) check whether estimated regime transitions coincide with composition-change dates; (iv) the PIT sensitivity in section 5.

## 8. Train-only correlation diagnostics

Computed on the train partition only. Nothing is re-weighted, removed or transformed on the basis of these numbers.

Pearson:

|  | equity_trend | interest_rate | inflation | volatility | liquidity_credit |
|---|---|---|---|---|---|
| equity_trend | 1.0 | 0.03 | 0.32 | -0.73 | -0.74 |
| interest_rate | 0.03 | 1.0 | 0.02 | 0.01 | -0.02 |
| inflation | 0.32 | 0.02 | 1.0 | -0.56 | -0.54 |
| volatility | -0.73 | 0.01 | -0.56 | 1.0 | 0.76 |
| liquidity_credit | -0.74 | -0.02 | -0.54 | 0.76 | 1.0 |

Spearman:

|  | equity_trend | interest_rate | inflation | volatility | liquidity_credit |
|---|---|---|---|---|---|
| equity_trend | 1.0 | -0.0 | 0.24 | -0.66 | -0.72 |
| interest_rate | -0.0 | 1.0 | 0.02 | 0.04 | 0.08 |
| inflation | 0.24 | 0.02 | 1.0 | -0.48 | -0.42 |
| volatility | -0.66 | 0.04 | -0.48 | 1.0 | 0.72 |
| liquidity_credit | -0.72 | 0.08 | -0.42 | 0.72 | 1.0 |

Dispersion and variance inflation:

|  | std | iqr | largest_value_share | vif |
|---|---|---|---|---|
| equity_trend | 1.095 | 1.292 | 0.0 | 2.829 |
| interest_rate | 0.654 | 0.902 | 0.0 | 1.005 |
| inflation | 0.956 | 1.082 | 0.006 | 1.65 |
| volatility | 1.29 | 1.381 | 0.001 | 3.124 |
| liquidity_credit | 1.111 | 1.39 | 0.0 | 3.185 |

## 9. Readiness assessment

**Verified on this dataset (PASS):** Index integrity: risk_factors; Index integrity: investment_log_returns; Index integrity: investment_prices; Index integrity: factor_components; Partitions mutually exclusive; Partitions collectively exhaustive over usable sample; Partitions chronological and inside configured boundaries; No empty partition; Identical factor schema across splits (no target / forward-return columns); Identical ETF return schema across splits; Feature and return dates identical within each split (contemporaneous storage); All split dates are SPY master-calendar sessions; No master-calendar session missing inside the usable span; Excluded dates inside split range are explained (leading block only); No NaN / +/-inf in split files; First eligible return date; First usable modelling row (all factors and returns present); Returns = same-day log price differences log(P_t/P_t-1) (no shift / fill); No missing investment prices (nothing to forward-fill); No factor value without underlying components (no fill); Stored factor files reproduce from current code and inputs (serialization only); Prefix invariance: removing later data leaves history unchanged; Prefix invariance: altering later data leaves history unchanged; Prefix invariance: initial features (features.py); Stored macro_pit reproduces from the release log; Macro PIT alignment uses only releases known by each cutoff; No macro observation dated after the trade date; Daily FRED values lag their observation date by >= 1 session; Macro values used only after their availability date; Factor composition stable in validation and test; Composition changes inside train create no artificial jump; Train: no near-constant factor; Train: no duplicate factor columns; Train: no severe multicollinearity (VIF < 10, |rho| < 0.9).

**Unverifiable assumptions and concerns (WARN):**

- No stale-price runs (>= 5 consecutive zero returns): BIL: longest run 18 (2011-07-25..2011-08-17); zero daily returns can be genuine for a T-bill fund (BIL) when yields are near zero and prices are quoted to two decimals - check the dates against a second source
- CPI / Core CPI / NFCI: every value used is a verified first release (category A): NFCI: category A from 2011-06-08 (train A 55.9% / B-C 0.2% / no value 43.9%, validation A 100.0% / B-C 0.0% / no value 0.0%, test A 100.0% / B-C 0.0% / no value 0.0%); CPIAUCSL: category A from 2007-10-30 (train A 100.0% / B-C 0.0% / no value 0.0%, validation A 100.0% / B-C 0.0% / no value 0.0%, test A 100.0% / B-C 0.0% / no value 0.0%); CPILFESL: category A from 2007-10-30 (train A 100.0% / B-C 0.0% / no value 0.0%, validation A 100.0% / B-C 0.0% / no value 0.0%, test A 100.0% / B-C 0.0% / no value 0.0%); B/C values are used on some trade days and are not verifiable as point-in-time; NFCI has no real-time value on part of the sample: the component is absent there (not filled), which shortens its history instead of using unverifiable values
- Daily FRED series timing (category D) is a conservative assumption: availability = observation date + 1 US business day; no publication timestamps recorded; values are current vintage
- Release time of day versus signal formation (after US close): availability is date-level only; scheduled 08:30 ET releases (CPI, NFCI) precede the 16:00 ET close, and daily series are used one session later, but exact timestamps are not in the data

**Not executed (NOT RUN):**

- Historical revisions of daily FRED series checked against ALFRED vintages: daily-series vintages were not downloaded

**Failures (FAIL):**

- none

Verdict: **ready for train-only K-Means / GMM exploration**, with the WARN items carried as documented limitations and the PIT / composition sensitivity analyses scheduled above.
