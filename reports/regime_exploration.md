# Stage 4: exploratory regime models (K-Means and Gaussian mixtures)

Generated 2026-10-21 03:26 UTC from the local Train and Validation splits (stage-3 protocol ARR-stage3-protocol-v1.2). Checks: 21 PASS. The test split was not read.

## Stage 4 decision (executive summary)

- Decision date: 2026-10-21; primary K = **2**; reference model **gmm_diag_k2** (probabilistic), baseline **kmeans_k2** (hard clustering); alternatives carried into the HMM search: K = 3, 4.
- Data used: Train for fitting, Validation for comparison; the test split was not read.

Basis:

- Predefined rules select K=2 in both families (GMM: best validation log-likelihood and most parsimonious; K-Means: best validation silhouette).
- K=2 is the most stable solution (mean pairwise ARI 1.00 across 10 seeds for both families) and the GMM and K-Means K=2 partitions agree (ARI 0.82 Train, 0.67 Validation).
- K=3 and K=4 are not significantly worse on Validation (Newey-West t < 2) and stay in the HMM search range.
- K=3 as the primary model was rejected under criteria fixed before the K=3 profiles were inspected: the K-Means rates state is not stable between Train and Validation, and the GMM third state is a stress-intensity split, so the third dimension is model-dependent.
- Elbow diagnostic (supporting, not decisive): the primary Kneedle elbow is computed over the candidate range K=1-6, with every other range reported as sensitivity; the figures from this run are in the evidence list below and in sections 8-9.

Consistency of the recorded decision with this run:

- [PASS] Decision: reference model equals the rule-based GMM preference: decision gmm_diag_k2; rule output gmm_diag_k2
- [PASS] Decision: baseline equals the rule-based K-Means preference: decision kmeans_k2; rule output kmeans_k2
- [PASS] Decision: primary and alternative K inside the validation-supported range: decision K [2, 3, 4]; not-significantly-worse range [2, 3, 4]
- [PASS] Decision: state names match the data (S2 = higher stress in Train and Validation): gmm_diag_k2 train: stress score S1 -1.31 < S2 4.21; gmm_diag_k2 validation: stress score S1 -1.37 < S2 3.36; kmeans_k2 train: stress score S1 -1.12 < S2 4.63; kmeans_k2 validation: stress score S1 -1.03 < S2 3.98

State naming: S2 is the state with the higher train stress score (volatility + liquidity_credit - equity_trend). Names are descriptive; the stage-4 rule label 'Liquidity Crisis' was not adopted because the state covers about a third of train days, not only crisis episodes.

**State profiles, shares and persistence (reference model):**

| state | split | state_name | share | equity_trend | interest_rate | inflation | volatility | liquidity_credit | episodes | median_run_days | mean_run_days |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 | train | Normal / Low stress | 0.65 | 0.19 | 0.12 | 0.02 | -0.48 | -0.63 | 19 | 5.00 | 70.89 |
| S2 | train | Stress / Risk-off | 0.35 | -1.45 | 0.10 | -1.02 | 1.54 | 1.22 | 20 | 6.50 | 35.55 |
| S1 | validation | Normal / Low stress | 0.73 | 0.25 | 0.17 | 0.48 | -0.44 | -0.68 | 10 | 19.50 | 73.10 |
| S2 | validation | Stress / Risk-off | 0.27 | -0.88 | 0.27 | -0.25 | 1.69 | 0.78 | 10 | 14.00 | 27.50 |

**Empirical daily transition probabilities** (implied expected duration = 1 / (1 - p_stay)):

| model_id | split | from | to_S1 | to_S2 | implied_expected_duration_days |
|---|---|---|---|---|---|
| gmm_diag_k2 | train | S1 | 0.986 | 0.014 | 70.895 |
| gmm_diag_k2 | train | S2 | 0.027 | 0.973 | 37.368 |
| gmm_diag_k2 | validation | S1 | 0.988 | 0.012 | 81.111 |
| gmm_diag_k2 | validation | S2 | 0.036 | 0.964 | 27.500 |
| kmeans_k2 | train | S1 | 0.989 | 0.011 | 90.125 |
| kmeans_k2 | train | S2 | 0.028 | 0.972 | 36.176 |
| kmeans_k2 | validation | S1 | 0.984 | 0.016 | 62.615 |
| kmeans_k2 | validation | S2 | 0.068 | 0.932 | 14.692 |

**S2 runs of at least 10 trading days** (14 episodes; Train and Validation):

| split | state | start | end | trading_days | mean_posterior | baseline_agreement |
|---|---|---|---|---|---|---|
| train | S2 | 2007-10-30 | 2008-02-22 | 79 | 0.993 | 0.937 |
| train | S2 | 2008-03-06 | 2008-03-31 | 17 | 0.93 | 0.235 |
| train | S2 | 2008-09-09 | 2009-03-17 | 131 | 0.987 | 0.969 |
| train | S2 | 2010-05-13 | 2010-08-05 | 59 | 0.988 | 0.949 |
| train | S2 | 2010-08-09 | 2010-09-01 | 18 | 0.936 | 1.0 |
| train | S2 | 2011-07-26 | 2011-12-28 | 109 | 0.985 | 0.936 |
| train | S2 | 2014-09-19 | 2015-04-06 | 136 | 0.976 | 0.919 |
| train | S2 | 2015-06-29 | 2015-11-30 | 108 | 0.988 | 0.898 |
| train | S2 | 2015-12-07 | 2015-12-31 | 18 | 0.969 | 0.556 |
| validation | S2 | 2016-01-04 | 2016-03-09 | 46 | 0.986 | 0.804 |
| validation | S2 | 2018-02-02 | 2018-05-22 | 76 | 0.997 | 0.579 |
| validation | S2 | 2018-10-10 | 2019-02-26 | 94 | 0.987 | 0.957 |
| validation | S2 | 2019-05-28 | 2019-06-21 | 19 | 0.803 | 0.158 |
| validation | S2 | 2019-08-14 | 2019-09-12 | 21 | 0.963 | 0.762 |

**Annualized mean ETF return by state (%; descriptive, not tradable for same_day):**

| split / horizon / state | SPY | EFA | EEM | HYG | IEF | TIP | GLD | DBC | BIL |
|---|---|---|---|---|---|---|---|---|---|
| train / next_session / S1 | 14.5 | 7.8 | 3.3 | 9.4 | 1.1 | 3.4 | 2.8 | 3.9 | 0.1 |
| train / next_session / S2 | -11.2 | -18.4 | -18.8 | -5.1 | 13.5 | 4.0 | 4.3 | -34.3 | 0.5 |
| train / same_day / S1 | 23.9 | 17.0 | 13.6 | 13.4 | 0.6 | 3.8 | 3.6 | 8.3 | 0.1 |
| train / same_day / S2 | -29.3 | -36.1 | -38.9 | -12.8 | 14.5 | 3.3 | 2.5 | -43.4 | 0.6 |
| validation / next_session / S1 | 12.3 | 6.3 | 5.6 | 6.0 | 1.7 | 3.0 | -0.5 | 6.9 | 1.0 |
| validation / next_session / S2 | 17.7 | 11.2 | 26.4 | 11.8 | 6.4 | 4.7 | 31.5 | 1.1 | 1.6 |
| validation / same_day / S1 | 19.9 | 12.7 | 13.6 | 8.0 | 0.1 | 2.3 | -0.1 | 10.7 | 1.0 |
| validation / same_day / S2 | -4.0 | -7.4 | 2.4 | 5.8 | 10.9 | 6.8 | 31.7 | -9.5 | 1.6 |

Reading of the evidence (computed from this run):

- S2 has weaker equity trend (-1.45 vs +0.19), higher volatility (+1.54 vs -0.48) and credit stress (+1.22 vs -0.63) on Train: a risk-off profile; it covers 35% of Train days, so it is broader than crisis episodes alone.
- Interest-rate factor S1 +0.12 vs S2 +0.10; inflation factor S1 +0.02 vs S2 -1.02.
- Treasuries (IEF) same-day on Train: S2 +14.5% vs S1 +0.6% annualized (flight to quality).
- Same-day differences are partly mechanical, because the factors are built from the same prices.
- SPY next-session return after S2 days: Train -11.2%, Validation +17.7% annualized; the sign reverses between periods, so the state is descriptive and carries no stable information about the next day's equity return.
- Elbow (supporting, not decisive): the largest inertia reduction is from K=1 to K=2 (40%); the Kneedle elbow over the candidate range 1-6 is K=2, and over 1-10 it is K=3 (margin 0.017 over K=4), so the curve has no sharp knee.
- Known limitation: K=2 represents the risk-on / risk-off axis only. Interest-rate conditions are almost identical in the two states and inflation pressure is lower in the stress state (disinflationary stress), so a rates or inflation shock in which bonds and equities fall together has no dedicated state. The HMM stage tests K=3 for this reason.


## 1. Research objectives

1. How many statistically distinguishable regimes exist in the five-factor space?
2. Are the regimes economically interpretable and stable (across seeds and into Validation)?
3. Do probabilistic memberships (GMM) add information beyond hard clustering (K-Means)?

Candidate state counts K = 2, 3, 4, 5, 6. No state count was fixed in advance and no historical crisis labels were used.

## 2. Dataset and temporal split

| split | first date | last date | rows | role in stage 4 |
|---|---|---|---|---|
| Train | 2007-10-30 | 2015-12-31 | 2058 | fitting (scaler, centroids, mixture parameters) |
| Validation | 2016-01-04 | 2019-12-31 | 1006 | comparison and selection only (predict / score) |
| Test | 2020-01-02 onwards | - | - | not read in stage 4 |

Split files are verified against the SHA-256 hashes in `data/splits/split_manifest.json`.

## 3. Input factors

| factor | name | direction |
|---|---|---|
| equity_trend | Equity Trend | higher = healthier equity trend |
| interest_rate | Interest Rate Conditions | higher = more stress |
| inflation | Inflation Pressure | higher = more stress |
| volatility | Volatility Stress | higher = more stress |
| liquidity_credit | Liquidity / Credit Stress | higher = more stress |

Definitions are frozen (stage 2); the factors.py hash is checked against the stage-3 manifest.

## 4. Pre-modelling checks and preprocessing

| name | status | detail |
|---|---|---|
| Five factor columns present in the frozen order | PASS | train ['equity_trend', 'interest_rate', 'inflation', 'volatility', 'liquidity_credit']; validation ['equity_trend', 'interest_rate', 'inflation', 'volatility', 'liquidity_credit'] |
| train factors: chronological, unique dates, finite | PASS | 2058 rows 2007-10-30..2015-12-31 |
| validation factors: chronological, unique dates, finite | PASS | 1006 rows 2016-01-04..2019-12-31 |
| Validation starts after train ends (no overlap) | PASS | train ends 2015-12-31, validation starts 2016-01-04 |
| Row counts match split_manifest.json | PASS | train 2058 (manifest 2058), validation 1006 (manifest 1006) |
| train returns aligned with factors | PASS | 9 ETF columns |
| validation returns aligned with factors | PASS | 9 ETF columns |
| Factor definitions unchanged since stage 3 (factors.py hash) | PASS | manifest 28414cd58b03..., current 28414cd58b03... |
| Test split never read | PASS | files read: split_manifest.json, train_features.csv, train_returns.csv, validation_features.csv, validation_returns.csv |
| Train/validation files match split_manifest.json hashes | PASS | mismatched: none |
| Scaler fitted on train only | PASS | train mean / train std (ddof=0); fitted on {'split': 'train', 'first_date': '2007-10-30', 'last_date': '2015-12-31', 'rows': 2058} |
| Fitted parameters independent of validation data | PASS | refit of gmm_diag_k2, kmeans_k2 with validation replaced by noise; max parameter difference 2.2e-16 (tolerance 1e-09, floating-point rounding); train label mismatches 0 |
| Same seed reproduces the same fit | PASS | max parameter difference 2.2e-16 (tolerance 1e-09); train label mismatches 0 |
| State ids within 0..K-1 for every candidate | PASS | 0 invalid ids |
| GMM posterior probabilities finite and sum to one | PASS | max |sum - 1| = 3.6e-15 |
| No output date beyond the validation period | PASS | last classified date 2019-12-31 |
| Non-temporal models: no smoothing, no use of later observations | PASS | each day is classified from its own factor vector with train-fitted parameters |
| Decision: reference model equals the rule-based GMM preference | PASS | decision gmm_diag_k2; rule output gmm_diag_k2 |
| Decision: baseline equals the rule-based K-Means preference | PASS | decision kmeans_k2; rule output kmeans_k2 |
| Decision: primary and alternative K inside the validation-supported range | PASS | decision K [2, 3, 4]; not-significantly-worse range [2, 3, 4] |
| Decision: state names match the data (S2 = higher stress in Train and Validation) | PASS | gmm_diag_k2 train: stress score S1 -1.31 < S2 4.21; gmm_diag_k2 validation: stress score S1 -1.37 < S2 3.36; kmeans_k2 train: stress score S1 -1.12 < S2 4.63; kmeans_k2 validation: stress score S1 -1.03 < S2 3.98 |

Train-only distribution of the factors:

|  | mean | std | median | mad_scaled | iqr | skew | excess_kurtosis | p01 | p99 | min | max | share_abs_gt_3 | variance_share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| equity_trend | -0.375 | 1.095 | -0.160 | 0.875 | 1.292 | -0.614 | 0.379 | -3.338 | 1.823 | -3.725 | 2.712 | 0.028 | 0.220 |
| interest_rate | 0.112 | 0.653 | 0.049 | 0.656 | 0.902 | 0.140 | 0.036 | -1.442 | 1.642 | -2.122 | 2.180 | 0.000 | 0.079 |
| inflation | -0.342 | 0.956 | -0.266 | 0.805 | 1.082 | -0.847 | 0.933 | -3.093 | 1.497 | -3.778 | 1.692 | 0.019 | 0.168 |
| volatility | 0.215 | 1.289 | -0.151 | 0.940 | 1.381 | 1.410 | 1.641 | -1.471 | 4.263 | -1.924 | 4.632 | 0.062 | 0.306 |
| liquidity_credit | 0.008 | 1.111 | -0.291 | 0.948 | 1.390 | 0.951 | 0.380 | -1.668 | 3.182 | -1.967 | 3.612 | 0.017 | 0.227 |

**Scaling decision (rule fixed before results):** train std ratio max/min = 1.97 (threshold 1.25); max |mean|/std = 0.36 (threshold 0.25); largest variance share 31% (volatility). Apply train-fitted standardization so that no factor dominates Euclidean distance.

Scaler parameters (Train only): equity_trend: mean -0.375, std 1.095, interest_rate: mean 0.112, std 0.653, inflation: mean -0.342, std 0.956, volatility: mean 0.215, std 1.289, liquidity_credit: mean 0.008, std 1.111

GMM log-likelihoods are computed in the scaled space; the Jacobian constant is identical for all candidates, so likelihood comparisons are unaffected.

## 5. K-Means methodology

- Fitted on Train for each K with 20 k-means++ initializations and seeds 20240531..20240540; the first seed is the reference fit.
- Validation days are assigned with `predict()` to the fixed Train centroids; nothing is refitted.
- Metrics: inertia, silhouette, Calinski-Harabasz, Davies-Bouldin, state populations, switching and run lengths. Distances are not probabilities; AIC/BIC are not defined for K-Means and are not used.
- Elbow method: K-Means inertia for K = 1-10 on Train (K=1 = total sum of squares) and the validation inertia of the same Train centroids; diagnostic only. The primary Kneedle elbow is computed over K = 1-6, the same range as the candidate comparison; the elbow for every other range is reported as a sensitivity check.

## 6. Gaussian mixture methodology

- Covariance types: full, diag; 5 initializations per seed; reg_covar = 0.0001; convergence warnings captured and recorded.
- Validation: `predict_proba` and `score_samples` with Train parameters. Posteriors sum to one (checked).
- The posterior p(state | x_t) is a static mixture membership computed from day t alone. It is not an HMM filtered probability p(state_t | x_1..x_t) and carries no persistence.

## 7. Selection metrics and predefined rules

- 1. Eligibility (all candidates): every train state >= 3% of days; mean pairwise ARI across 10 seeds >= 0.80; GMM: every seed converged. A state below 1% of Validation days is FLAGGED as collapsed but does not exclude the candidate: Validation (2016-2019) contains no episode comparable to 2008, so excluding such states would select against crisis regimes by construction. Out-of-sample adequacy is judged by validation likelihood.
- 2. GMM preference: among eligible GMMs, take the best validation average log-likelihood. A candidate is 'not significantly worse' if the mean daily log-likelihood shortfall versus the best has a Newey-West t-statistic (21 lags) below 2. Prefer the not-significantly-worse candidate with the fewest parameters (ties: smaller K). Train BIC/AIC are reported for reference, not used to override validation evidence.
- 3. K-Means preference: among eligible K-Means, the smallest K whose validation silhouette is within 0.01 of the best. Silhouette is known to favour small K, so this is a baseline indication only. Inertia always falls with K and is not a selection criterion; AIC/BIC are not used for K-Means.
- 4. Conflicts: if the K-Means and GMM preferences differ, both are reported as leading candidates; the GMM preference is the reference for the HMM stage because it is likelihood-based and probabilistic.
- 5. Economic interpretability, switching frequency and asset behaviour are reported for every candidate but are not used to rank them, so that no candidate is chosen for producing attractive historical plots.
- 6. If no candidate is eligible, no preferred model is declared and the leading candidates are reported.

Thresholds: min_train_state_share = 0.03, min_validation_state_share = 0.01, min_mean_ari_train = 0.8, gmm_require_all_seeds_converged = True, nw_lags = 21, nw_t_threshold = 2.0, kmeans_silhouette_tolerance = 0.01, flag_median_run_days = 2, profile_shift_threshold = 1.0.

## 8-9. Candidate state counts: Train and Validation results

**K-Means**

| model_id | train_inertia_per_obs | train_silhouette | train_calinski_harabasz | train_davies_bouldin | validation_silhouette | validation_davies_bouldin | validation_to_train_distortion | train_shares | validation_shares |
|---|---|---|---|---|---|---|---|---|---|
| kmeans_k2 | 2.992 | 0.408 | 1379.465 | 1.134 | 0.346 | 1.380 | 1.240 | S1=0.701;S2=0.299 | S1=0.810;S2=0.190 |
| kmeans_k3 | 2.463 | 0.274 | 1058.646 | 1.292 | 0.289 | 1.331 | 1.343 | S1=0.523;S2=0.267;S3=0.210 | S1=0.683;S2=0.220;S3=0.097 |
| kmeans_k4 | 2.098 | 0.276 | 946.882 | 1.297 | 0.230 | 1.893 | 1.488 | S1=0.479;S2=0.248;S3=0.190;S4=0.083 | S1=0.579;S2=0.252;S3=0.161;S4=0.008 |
| kmeans_k5 | 1.827 | 0.247 | 891.173 | 1.323 | 0.207 | 1.633 | 1.558 | S1=0.278;S2=0.300;S3=0.187;S4=0.150;S5=0.086 | S1=0.240;S2=0.445;S3=0.169;S4=0.140;S5=0.006 |
| kmeans_k6 | 1.588 | 0.263 | 881.894 | 1.249 | 0.245 | 1.757 | 1.740 | S1=0.292;S2=0.257;S3=0.185;S4=0.119;S5=0.067;S6=0.081 | S1=0.280;S2=0.420;S3=0.169;S4=0.071;S5=0.053;S6=0.007 |

**K-Means elbow method (diagnostic, K = 1-10, Train fit)**

| k | train_inertia | train_inertia_per_obs | validation_inertia_per_obs | train_explained_share | pct_reduction_vs_previous_k | validation_pct_reduction_vs_previous_k | kneedle_elbow |
|---|---|---|---|---|---|---|---|
| 1 | 10290.000 | 5.000 | 5.042 | 0.000 |  |  | False |
| 2 | 6158.188 | 2.992 | 3.711 | 0.402 | 40.154 | 26.403 | True |
| 3 | 5068.184 | 2.463 | 3.306 | 0.507 | 17.700 | 10.911 | False |
| 4 | 4318.138 | 2.098 | 3.121 | 0.580 | 14.799 | 5.591 | False |
| 5 | 3760.507 | 1.827 | 2.847 | 0.635 | 12.914 | 8.803 | False |
| 6 | 3267.849 | 1.588 | 2.763 | 0.682 | 13.101 | 2.933 | False |
| 7 | 2939.740 | 1.428 | 2.667 | 0.714 | 10.040 | 3.490 | False |
| 8 | 2677.159 | 1.301 | 2.331 | 0.740 | 8.932 | 12.581 | False |
| 9 | 2469.383 | 1.200 | 2.341 | 0.760 | 7.761 | -0.410 | False |
| 10 | 2286.714 | 1.111 | 2.215 | 0.778 | 7.397 | 5.389 | False |

Primary Kneedle elbow (maximum distance from the chord, K = 1-6): **K = 2**. The range matches the stage-4 candidate set (K = 2-6); K = 7-10 were never candidates but would still move the chord, so they are shown as sensitivity rather than used for the primary reading.

*Kneedle range sensitivity*

| k_range | max_k | kneedle_elbow | runner_up | margin_vs_runner_up | primary_range |
|---|---|---|---|---|---|
| 1-4 | 4 | 2 | 3 | 0.151 | False |
| 1-5 | 5 | 2 | 3 | 0.083 | False |
| 1-6 | 6 | 2 | 3 | 0.045 | True |
| 1-7 | 7 | 2 | 3 | 0.018 | False |
| 1-8 | 8 | 3 | 2 | 0.000 | False |
| 1-9 | 9 | 3 | 2 | 0.014 | False |
| 1-10 | 10 | 3 | 4 | 0.017 | False |

The elbow depends on the range: K = 3 over 1-10 (margin 0.017 over K = 4). The inertia curve is smooth with no sharp knee, so the elbow alone does not separate the leading candidates. The elbow is a visual heuristic for the point of diminishing returns; it is reported as supporting evidence and does not override the predefined selection rules.

**Gaussian mixtures**

| model_id | n_parameters | converged | n_iter | train_bic | train_aic | train_avg_loglik | validation_avg_loglik | validation_mean_max_posterior | validation_mean_entropy | validation_share_max_posterior_below_0.6 | train_shares | validation_shares |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gmm_full_k2 | 41 | True | 36 | 22670.496 | 22439.687 | -5.432 | -6.858 | 0.884 | 0.391 | 0.075 | S1=0.590;S2=0.410 | S1=0.586;S2=0.414 |
| gmm_full_k3 | 62 | True | 20 | 21794.189 | 21445.161 | -5.180 | -6.942 | 0.900 | 0.222 | 0.058 | S1=0.550;S2=0.274;S3=0.176 | S1=0.605;S2=0.275;S3=0.119 |
| gmm_full_k4 | 83 | True | 48 | 21248.429 | 20781.182 | -5.009 | -7.261 | 0.870 | 0.215 | 0.091 | S1=0.322;S2=0.385;S3=0.109;S4=0.184 | S1=0.231;S2=0.566;S3=0.082;S4=0.122 |
| gmm_full_k5 | 104 | True | 57 | 20656.052 | 20070.585 | -4.826 | -8.457 | 0.888 | 0.161 | 0.073 | S1=0.267;S2=0.419;S3=0.149;S4=0.116;S5=0.049 | S1=0.127;S2=0.689;S3=0.078;S4=0.106;S5=0.000 |
| gmm_full_k6 | 125 | True | 34 | 20052.017 | 19348.331 | -4.640 | -9.292 | 0.853 | 0.194 | 0.123 | S1=0.256;S2=0.334;S3=0.084;S4=0.143;S5=0.135;S6=0.049 | S1=0.154;S2=0.413;S3=0.116;S4=0.112;S5=0.205;S6=0.000 |
| gmm_diag_k2 | 21 | True | 8 | 24269.312 | 24151.093 | -5.857 | -6.655 | 0.967 | 0.116 | 0.021 | S1=0.655;S2=0.345 | S1=0.727;S2=0.273 |
| gmm_diag_k3 | 32 | True | 9 | 23341.840 | 23161.696 | -5.612 | -6.890 | 0.951 | 0.111 | 0.025 | S1=0.611;S2=0.307;S3=0.082 | S1=0.662;S2=0.320;S3=0.018 |
| gmm_diag_k4 | 43 | True | 48 | 22691.827 | 22449.759 | -5.433 | -6.981 | 0.886 | 0.188 | 0.075 | S1=0.277;S2=0.367;S3=0.276;S4=0.080 | S1=0.325;S2=0.391;S3=0.269;S4=0.015 |
| gmm_diag_k5 | 54 | True | 51 | 22273.997 | 21970.004 | -5.311 | -7.001 | 0.889 | 0.158 | 0.077 | S1=0.276;S2=0.370;S3=0.258;S4=0.052;S5=0.044 | S1=0.323;S2=0.394;S3=0.271;S4=0.012;S5=0.000 |
| gmm_diag_k6 | 65 | True | 33 | 21806.030 | 21440.113 | -5.177 | -7.064 | 0.897 | 0.137 | 0.067 | S1=0.173;S2=0.197;S3=0.286;S4=0.246;S5=0.053;S6=0.044 | S1=0.067;S2=0.343;S3=0.320;S4=0.258;S5=0.012;S6=0.000 |

Convergence warnings: none.

## 10. Regime stability

Switching and durations (reference seed; durations in trading days):

| model_id | train_switches_per_year | train_median_duration_days | train_mean_duration_days | train_share_single_day_runs | validation_switches_per_year | validation_median_duration_days |
|---|---|---|---|---|---|---|
| kmeans_k2 | 4.04 | 9.50 | 60.53 | 0.18 | 6.51 | 4.00 |
| kmeans_k3 | 10.16 | 4.00 | 24.50 | 0.26 | 12.52 | 5.00 |
| kmeans_k4 | 12.98 | 4.00 | 19.23 | 0.26 | 14.78 | 4.00 |
| kmeans_k5 | 16.65 | 4.00 | 15.02 | 0.26 | 22.04 | 3.00 |
| kmeans_k6 | 16.90 | 3.00 | 14.81 | 0.27 | 16.78 | 4.50 |
| gmm_full_k2 | 8.57 | 4.00 | 28.99 | 0.25 | 17.79 | 3.00 |
| gmm_full_k3 | 10.65 | 4.50 | 23.39 | 0.27 | 16.28 | 4.00 |
| gmm_full_k4 | 16.90 | 3.00 | 14.81 | 0.29 | 26.55 | 3.00 |
| gmm_full_k5 | 17.27 | 3.00 | 14.49 | 0.24 | 21.54 | 3.00 |
| gmm_full_k6 | 20.69 | 3.00 | 12.11 | 0.31 | 34.32 | 3.00 |
| gmm_diag_k2 | 4.65 | 5.00 | 52.77 | 0.15 | 4.76 | 14.50 |
| gmm_diag_k3 | 7.59 | 5.00 | 32.67 | 0.27 | 8.77 | 6.00 |
| gmm_diag_k4 | 15.55 | 4.00 | 16.08 | 0.23 | 17.03 | 4.00 |
| gmm_diag_k5 | 16.78 | 4.00 | 14.91 | 0.25 | 17.03 | 4.00 |
| gmm_diag_k6 | 15.18 | 4.00 | 16.46 | 0.27 | 19.04 | 2.00 |

Initialization sensitivity across 10 seeds (ARI and label-aligned agreement are permutation invariant):

| model_id | family | k | covariance_type | n_seeds | train_mean_ari | train_min_ari | train_mean_aligned_agreement | validation_mean_ari | validation_min_ari | validation_mean_aligned_agreement | converged_share | inertia_relative_range | lower_bound_range | validation_avg_loglik_range |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kmeans_k2 | kmeans | 2 | - | 10 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |  |  |
| kmeans_k3 | kmeans | 3 | - | 10 | 0.992 | 0.971 | 0.997 | 0.989 | 0.956 | 0.997 | 1.000 | 0.000 |  |  |
| kmeans_k4 | kmeans | 4 | - | 10 | 0.989 | 0.957 | 0.996 | 0.979 | 0.934 | 0.993 | 1.000 | 0.000 |  |  |
| kmeans_k5 | kmeans | 5 | - | 10 | 0.970 | 0.932 | 0.990 | 0.970 | 0.932 | 0.989 | 1.000 | 0.000 |  |  |
| kmeans_k6 | kmeans | 6 | - | 10 | 0.994 | 0.981 | 0.998 | 0.996 | 0.985 | 0.998 | 1.000 | 0.000 |  |  |
| gmm_full_k2 | gmm | 2 | full | 10 | 1.000 | 1.000 | 1.000 | 0.999 | 0.996 | 1.000 | 1.000 |  | 0.000 | 0.000 |
| gmm_full_k3 | gmm | 3 | full | 10 | 0.871 | 0.680 | 0.924 | 0.827 | 0.554 | 0.903 | 1.000 |  | 0.028 | 0.890 |
| gmm_full_k4 | gmm | 4 | full | 10 | 0.611 | 0.285 | 0.764 | 0.542 | 0.173 | 0.759 | 1.000 |  | 0.043 | 0.703 |
| gmm_full_k5 | gmm | 5 | full | 10 | 0.491 | 0.254 | 0.701 | 0.341 | 0.125 | 0.648 | 1.000 |  | 0.130 | 1.469 |
| gmm_full_k6 | gmm | 6 | full | 10 | 0.507 | 0.324 | 0.723 | 0.360 | 0.137 | 0.633 | 1.000 |  | 0.107 | 1.861 |
| gmm_diag_k2 | gmm | 2 | diag | 10 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |  | 0.000 | 0.000 |
| gmm_diag_k3 | gmm | 3 | diag | 10 | 0.943 | 0.714 | 0.967 | 0.932 | 0.658 | 0.959 | 1.000 |  | 0.015 | 0.182 |
| gmm_diag_k4 | gmm | 4 | diag | 10 | 0.855 | 0.580 | 0.928 | 0.826 | 0.493 | 0.908 | 1.000 |  | 0.034 | 0.124 |
| gmm_diag_k5 | gmm | 5 | diag | 10 | 0.786 | 0.502 | 0.850 | 0.796 | 0.497 | 0.903 | 1.000 |  | 0.029 | 0.186 |
| gmm_diag_k6 | gmm | 6 | diag | 10 | 0.682 | 0.440 | 0.793 | 0.697 | 0.416 | 0.855 | 1.000 |  | 0.019 | 0.243 |

## 11. Economic interpretation

Hypothesis labels come from fixed rules applied to Train factor profiles:

- **Liquidity Crisis**: volatility >= 1.0 and liquidity_credit >= 1.0 and equity_trend <= -0.5
- **Inflation / Rates Shock**: max(inflation, interest_rate) >= 0.75 and volatility < 1.0
- **Fragile / Transition**: equity_trend < 0 and max(volatility, liquidity_credit) >= 0.25 and not Liquidity Crisis
- **Growth / Risk-on**: equity_trend >= 0.25 and volatility <= 0 and liquidity_credit <= 0
- Otherwise, or if several rules match, the state stays **Neutral** and keeps its S# id.

**gmm_diag_k2**

| state | split | n_obs | share | equity_trend | interest_rate | inflation | volatility | liquidity_credit | hypothesis_label | label_rule | train_validation_profile_distance | generalization_flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 | train | 1347 | 0.65 | 0.19 | 0.12 | 0.02 | -0.48 | -0.63 | Neutral | no rule matched | 0.47 | - |
| S1 | validation | 731 | 0.73 | 0.25 | 0.17 | 0.48 | -0.44 | -0.68 | Neutral | no rule matched | 0.47 | - |
| S2 | train | 711 | 0.35 | -1.45 | 0.10 | -1.02 | 1.54 | 1.22 | Liquidity Crisis (hypothesis) | Liquidity Crisis | 1.08 | profile shift |
| S2 | validation | 275 | 0.27 | -0.88 | 0.27 | -0.25 | 1.69 | 0.78 | Liquidity Crisis (hypothesis) | Liquidity Crisis | 1.08 | profile shift |

**kmeans_k2**

| state | split | n_obs | share | equity_trend | interest_rate | inflation | volatility | liquidity_credit | hypothesis_label | label_rule | train_validation_profile_distance | generalization_flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 | train | 1443 | 0.70 | 0.14 | 0.14 | -0.00 | -0.41 | -0.57 | Neutral | no rule matched | 0.50 | - |
| S1 | validation | 815 | 0.81 | 0.19 | 0.20 | 0.45 | -0.24 | -0.60 | Neutral | no rule matched | 0.50 | - |
| S2 | train | 615 | 0.30 | -1.58 | 0.04 | -1.14 | 1.69 | 1.36 | Liquidity Crisis (hypothesis) | Liquidity Crisis | 0.87 | - |
| S2 | validation | 191 | 0.19 | -1.14 | 0.18 | -0.47 | 1.76 | 1.08 | Liquidity Crisis (hypothesis) | Liquidity Crisis | 0.87 | - |

All candidates' profiles are in `data/processed/regime_exploration/regime_factor_profiles.csv`.

## 12. Conditional ETF behaviour (descriptive)

`same_day` groups return_t by the state of day t (both known only at the close of t: not tradable). `next_session` groups return_(t+1) by the state of day t within the same split; it is closer to an implementable timing but remains in-sample for Train. Neither is evidence of predictive power.

gmm_diag_k2, train, same_day: annualized mean return (%)

| state | n_days | BIL | DBC | EEM | EFA | GLD | HYG | IEF | SPY | TIP |
|---|---|---|---|---|---|---|---|---|---|---|
| S1 | 1347 | 0.1 | 8.3 | 13.6 | 17.0 | 3.6 | 13.4 | 0.6 | 23.9 | 3.8 |
| S2 | 711 | 0.6 | -43.4 | -38.9 | -36.1 | 2.5 | -12.8 | 14.5 | -29.3 | 3.3 |

gmm_diag_k2, train, same_day: annualized volatility (%)

| state | BIL | DBC | EEM | EFA | GLD | HYG | IEF | SPY | TIP |
|---|---|---|---|---|---|---|---|---|---|
| S1 | 0.4 | 17.6 | 21.8 | 18.0 | 17.4 | 8.1 | 6.8 | 14.0 | 5.8 |
| S2 | 1.0 | 26.5 | 51.2 | 37.7 | 25.3 | 20.2 | 8.7 | 32.2 | 8.9 |

gmm_diag_k2, validation, same_day: annualized mean return (%)

| state | n_days | BIL | DBC | EEM | EFA | GLD | HYG | IEF | SPY | TIP |
|---|---|---|---|---|---|---|---|---|---|---|
| S1 | 731 | 1.0 | 10.7 | 13.6 | 12.7 | -0.1 | 8.0 | 0.1 | 19.9 | 2.3 |
| S2 | 275 | 1.6 | -9.5 | 2.4 | -7.4 | 31.7 | 5.8 | 10.9 | -4.0 | 6.8 |

gmm_diag_k2, validation, same_day: annualized volatility (%)

| state | BIL | DBC | EEM | EFA | GLD | HYG | IEF | SPY | TIP |
|---|---|---|---|---|---|---|---|---|---|
| S1 | 0.3 | 14.0 | 15.8 | 12.0 | 11.9 | 4.9 | 4.7 | 8.9 | 3.8 |
| S2 | 0.2 | 16.3 | 23.3 | 16.9 | 12.6 | 7.3 | 5.0 | 19.9 | 3.8 |

gmm_diag_k2, train, next_session: annualized mean return (%)

| state | n_days | BIL | DBC | EEM | EFA | GLD | HYG | IEF | SPY | TIP |
|---|---|---|---|---|---|---|---|---|---|---|
| S1 | 1347 | 0.1 | 3.9 | 3.3 | 7.8 | 2.8 | 9.4 | 1.1 | 14.5 | 3.4 |
| S2 | 710 | 0.5 | -34.3 | -18.8 | -18.4 | 4.3 | -5.1 | 13.5 | -11.2 | 4.0 |

gmm_diag_k2, validation, next_session: annualized mean return (%)

| state | n_days | BIL | DBC | EEM | EFA | GLD | HYG | IEF | SPY | TIP |
|---|---|---|---|---|---|---|---|---|---|---|
| S1 | 730 | 1.0 | 6.9 | 5.6 | 6.3 | -0.5 | 6.0 | 1.7 | 12.3 | 3.0 |
| S2 | 275 | 1.6 | 1.1 | 26.4 | 11.2 | 31.5 | 11.8 | 6.4 | 17.7 | 4.7 |

## 13. Model comparison

| model_id | k | covariance_type | n_parameters | train_bic | train_aic | train_avg_loglik | validation_avg_loglik | train_minus_validation_loglik | mean_loglik_shortfall | nw_t_vs_best | eligible | ineligible_reasons | not_significantly_worse |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gmm_full_k2 | 2 | full | 41 | 22670.4957 | 22439.6866 | -5.4319 | -6.8584 | 1.4265 | 0.2029 | 0.7952 | True | - | True |
| gmm_full_k3 | 3 | full | 62 | 21794.1891 | 21445.1607 | -5.1801 | -6.9419 | 1.7618 | 0.2864 | 1.2472 | True | - | True |
| gmm_full_k4 | 4 | full | 83 | 21248.4294 | 20781.1817 | -5.0085 | -7.2606 | 2.2521 | 0.6051 | 2.5655 | False | seed ARI 0.61 < 0.8 | False |
| gmm_full_k5 | 5 | full | 104 | 20656.0521 | 20070.5851 | -4.8257 | -8.4573 | 3.6316 | 1.8018 | 2.3064 | False | seed ARI 0.49 < 0.8 | False |
| gmm_full_k6 | 6 | full | 125 | 20052.0175 | 19348.3312 | -4.6400 | -9.2923 | 4.6523 | 2.6368 | 3.0655 | False | seed ARI 0.51 < 0.8 | False |
| gmm_diag_k2 | 2 | diag | 21 | 24269.3123 | 24151.0930 | -5.8574 | -6.6555 | 0.7980 | 0.0000 | 0.0000 | True | - | True |
| gmm_diag_k3 | 3 | diag | 32 | 23341.8395 | 23161.6959 | -5.6117 | -6.8903 | 1.2786 | 0.2348 | 1.2304 | True | - | True |
| gmm_diag_k4 | 4 | diag | 43 | 22691.8270 | 22449.7589 | -5.4334 | -6.9811 | 1.5477 | 0.3257 | 1.5096 | True | - | True |
| gmm_diag_k5 | 5 | diag | 54 | 22273.9966 | 21970.0042 | -5.3115 | -7.0008 | 1.6893 | 0.3453 | 1.5521 | False | seed ARI 0.79 < 0.8 | False |
| gmm_diag_k6 | 6 | diag | 65 | 21806.0300 | 21440.1131 | -5.1774 | -7.0642 | 1.8868 | 0.4087 | 1.7413 | False | seed ARI 0.68 < 0.8 | False |

Best validation log-likelihood: gmm_diag_k2. Not significantly worse (NW t < 2.0): gmm_diag_k2, gmm_diag_k3, gmm_full_k2, gmm_diag_k4, gmm_full_k3. Most parsimonious: gmm_diag_k2.

`train_minus_validation_loglik` is the in-sample minus out-of-sample average log-likelihood; a gap that widens with K indicates components fitted to train-specific structure.

| model_id | k | train_inertia | train_silhouette | train_calinski_harabasz | train_davies_bouldin | validation_silhouette | validation_to_train_distortion | train_min_state_share | validation_min_state_share | eligible | ineligible_reasons |
|---|---|---|---|---|---|---|---|---|---|---|---|
| kmeans_k2 | 2 | 6158.188 | 0.408 | 1379.465 | 1.134 | 0.346 | 1.240 | 0.299 | 0.190 | True | - |
| kmeans_k3 | 3 | 5068.184 | 0.274 | 1058.646 | 1.292 | 0.289 | 1.343 | 0.210 | 0.097 | True | - |
| kmeans_k4 | 4 | 4318.138 | 0.276 | 946.882 | 1.297 | 0.230 | 1.488 | 0.083 | 0.008 | True | - |
| kmeans_k5 | 5 | 3760.507 | 0.247 | 891.173 | 1.323 | 0.207 | 1.558 | 0.086 | 0.006 | True | - |
| kmeans_k6 | 6 | 3267.849 | 0.263 | 881.894 | 1.249 | 0.245 | 1.740 | 0.067 | 0.007 | True | - |

Best validation silhouette 0.346; candidates within 0.01: kmeans_k2; smallest K: kmeans_k2.

Train BIC minimum (reference only): gmm_full_k6.
Train BIC is minimised by gmm_full_k6 (K=6); validation likelihood is maximised by gmm_diag_k2 (K=2). Under the predefined rule the validation evidence prevails: additional components improve the in-sample fit but not the out-of-sample fit, i.e. part of the in-sample gain is train-specific.
Agreement between the preferred GMM and K-Means assignments: ARI 0.82 (Train), 0.67 (Validation); same K: True.

Eligibility of every candidate:

| model_id | eligible | ineligible_reasons | excessive_switching_flag | validation_collapsed_states |
|---|---|---|---|---|
| kmeans_k2 | True | - | False | 0 |
| kmeans_k3 | True | - | False | 0 |
| kmeans_k4 | True | - | False | 1 |
| kmeans_k5 | True | - | False | 1 |
| kmeans_k6 | True | - | False | 1 |
| gmm_full_k2 | True | - | False | 0 |
| gmm_full_k3 | True | - | False | 0 |
| gmm_full_k4 | False | seed ARI 0.61 < 0.8 | False | 0 |
| gmm_full_k5 | False | seed ARI 0.49 < 0.8 | False | 1 |
| gmm_full_k6 | False | seed ARI 0.51 < 0.8 | False | 1 |
| gmm_diag_k2 | True | - | False | 0 |
| gmm_diag_k3 | True | - | False | 0 |
| gmm_diag_k4 | True | - | False | 0 |
| gmm_diag_k5 | False | seed ARI 0.79 < 0.8 | False | 1 |
| gmm_diag_k6 | False | seed ARI 0.68 < 0.8 | False | 1 |

## 14. Preferred exploratory candidate

- **GMM: gmm_diag_k2** under the predefined rules (validation average log-likelihood -6.655; 21 parameters).
- K-Means: **kmeans_k2**.
- Candidates not significantly worse than the best on Validation span K = 2, 3, 4. The rule picks the most parsimonious one, but the evidence does not reject the larger models in this range; they are the leading alternatives, not discarded models.

**K=3 alternative: is there a distinct rates / inflation state?** Criteria fixed on 2026-10-20, before the K=3 profiles were inspected: (c1) a state other than the most stressed one has train interest_rate or inflation >= 0.5; (c2) its validation share >= 5%; (c3) train-validation profile distance < 1.0; (c4) seed mean ARI >= 0.9.

| model_id | candidate_state | train_interest_rate | train_inflation | train_volatility | validation_share | profile_distance | train_mean_ari | c1_rates_or_inflation_state | c2_validation_share | c3_profile_stable | c4_seed_stability | criteria_met | accepted |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gmm_diag_k3 | S2 | 0.13 | -0.79 | 0.86 | 0.32 | 0.76 | 0.94 | False | True | True | True | 3/4 | False |
| gmm_full_k3 | S2 | 0.35 | -0.58 | 0.44 | 0.28 | 0.62 | 0.87 | False | True | True | False | 2/4 | False |
| kmeans_k3 | S2 | 0.81 | -0.51 | 0.15 | 0.22 | 1.48 | 0.99 | True | True | False | True | 3/4 | False |

- Result: no K=3 candidate meets all four criteria; K=3 remains an alternative inside the HMM search range, not the primary hypothesis.
- Validation (2016-2019) contains no stress episode comparable to 2008, so the validation likelihood cannot reward a separate crisis component. A small preferred K therefore partly reflects the calm validation period and should not be read as proof that only that many regimes exist.
- Figures show gmm_diag_k2 and kmeans_k2 (stage-4 decision (K=2)). The recorded decision is in the executive summary at the top of this report.
- This is an exploratory baseline. The final state count is decided in the Sticky HMM stage, using Train/Validation only, with the candidates above as the starting range.

## 15. Limitations and risks

- K-Means and GMM treat days as exchangeable: they ignore the time order, so regime persistence is neither modelled nor imposed, and day-to-day flicker is expected near state boundaries.
- GMM memberships are static; they are not forward-filtered state probabilities and must not be described as such.
- Train (2007-10 to 2015) contains one major crisis; rare stress states are estimated from few episodes, so their parameters are uncertain.
- Factors are rolling 252-day z-scores: persistent stress decays toward zero inside the window, which can split a long crisis into several states.
- PIT limitations from stage 3 remain: NFCI enters Liquidity / Credit only from 2011-11; daily FRED series rely on a one-day publication lag; the ETF universe was chosen ex post.
- Conditional ETF statistics are descriptive and in-sample; a statistically good classification does not imply better portfolio performance.
- Validation (2016-2019) is a relatively calm period; states that are rare in Validation are hard to assess out of sample.

## 16. Implications for the Sticky HMM stage

- HMM search range: K = 2, 3, 4 (the candidates not significantly worse on Validation), fixed now, before any HMM is fitted; the HMM is not restricted to the static models' preferred K.
- The observed switching frequency and median run lengths of the static models are the benchmark that a sticky transition matrix should improve on without losing validation likelihood.
- The HMM must output forward-filtered probabilities p(state_t | x_1..x_t) for decisions; smoothed probabilities may only be used for description.
- Keep the Train-fitted scaler (or refit it inside each walk-forward window with the same rule), the seed policy and the label-ordering convention, and compare HMM states with the GMM states via ARI.
- Carry the PIT and composition sensitivity analyses committed in the stage-3 protocol.
