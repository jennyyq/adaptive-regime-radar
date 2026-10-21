# Stage 5.5 - Autoregressive HMM robustness and predictive calibration

Train 2007-10-30..2015-12-31 (2058 rows; scored from row 2); Validation 2016-01-04..2019-12-31 (1006 rows). Test not read. Checks: 25 PASS.

**Status of the evidence:** Validation was already used for Stage 5 model selection. This experiment is a robustness check on the development sample, not an independent holdout. The Stage 5 model is frozen and was not refitted.

## Decision

- Verdict: **challenger_density** -> model for Stage 6: **frozen_hmm_k3 for regimes; ar_hmm_k3 as predictive-density challenger**.
- Pre-specified criteria: 1_gain_vs_frozen PASS; 2_gain_vs_ar1 PASS; 3_residual_dependence PASS; 4_coverage90 PASS; 5_stability_and_persistence FAIL; 6_stress_state FAIL.

## 1. Research question and design

Can autoregressive emissions fix the temporal-dependence misspecification of the frozen HMM (within-state residual autocorrelation ~0.97, 90% interval coverage ~81%) while keeping stable, interpretable regimes? The comparison separates three sources of predictive information:

- autoregression alone: C (AR(1)) vs the i.i.d. Gaussian reference;
- latent states alone: A (frozen HMM) vs the i.i.d. Gaussian reference;
- states on top of autoregression: B (AR-HMM) vs C. Only this last comparison measures what the latent regimes add once the obvious persistence of the factors is modelled.

## 2. Models and likelihood

- A: frozen Stage 5 `hmm_std_diag_k3` (parameters read from its JSON, no refit).
- B: AR-HMM K=3: X_t = mu_k + Phi_k (X_t-1 - mu_k) + e_t, e_t ~ N(0, Sigma_k), Phi_k and Sigma_k diagonal, state-specific mu_k, Phi_k, Sigma_k, transition matrix A.
- C: single-regime Gaussian AR(1) with diagonal Phi and Sigma (B with K=1; closed-form least squares).
- D: AR-HMM K=2 (robustness only). Reference: i.i.d. Gaussian with Train mean and variance.
- Conditional likelihood: the first Train row is used only as a lag; the chain starts at t=2 with pi. p(x_t | x_1..x_t-1) = sum_k P(Z_t=k | x_1..x_t-1) N(x_t; c_k + Phi_k x_t-1, Sigma_k).
- Estimation: EM on Train; M-step = per-state weighted least squares of x_t on (1, x_t-1) with |phi| <= 0.99 and innovation variance max(s2, 0.0001) (exact constrained optimum, so EM is monotone; checked for every start and by a Q-function optimality test); 10 seeds x 5 starts, tol 1e-06/obs, max 1000 iterations.
- Filtering: log space; the predicted distribution and the score of day t are computed before x_t is used. Validation continues from the last Train filtered distribution with the last Train row as first lag.
- Scoring: identical dates for every model (Train rows 2..T, all Validation rows); A's Train day 1 is filtered but not scored.

## 3. Adoption criteria (fixed before results)

- Replace the frozen HMM for Stage 6 only if ALL hold: (1) Validation log score B - A >= 0.10 nats/day with NW t >= 2; (2) B - C > 0 with NW t >= 2, i.e. the latent states add predictive value beyond a single-regime AR(1); (3) mean |lag-1 autocorrelation| of Validation quantile residuals at most half of A's; (4) Validation 90% interval coverage within [0.85, 0.95] and closer to 0.90 than A's; (5) >= 80% of seeds converge, EM monotone, seed ARI >= 0.80, every Train state >= 3% of days, Validation switches per year at most twice A's; (6) a stress state that meets the Stage 4 Liquidity Crisis rule.
- If (1)-(4) hold but (5) or (6) fails: keep the frozen HMM for regimes; the AR-HMM is a predictive-density challenger only.
- If (2) fails: the gain comes from autoregression, not from latent states; keep the frozen HMM for regime identification and report AR(1) as the density benchmark.
- Otherwise (mixed evidence): retain the frozen HMM; the AR-HMM remains a robustness challenger.
- Validation (2016-2019) was already used in Stage 5. This is a robustness experiment on the development sample, not a new holdout; the specification (K=3 primary, K=2 robustness, diagonal Phi and Sigma, |phi| <= 0.99) was fixed in advance and is not tuned on these results.

Thresholds: min_validation_gain_vs_frozen = 0.1, min_validation_gain_vs_ar1 = 0.0, nw_t = 2.0, max_residual_acf1_ratio = 0.5, coverage90_band = (0.85, 0.95), coverage90_must_improve = True, min_converged_seed_share = 0.8, min_mean_pairwise_ari_train = 0.8, min_train_state_share = 0.03, max_switch_ratio_vs_frozen = 2.0, stress_state_rule = highest-stress state satisfies the Stage 4 'Liquidity Crisis' rule on its Train profile (volatility >= 1.0, liquidity_credit >= 1.0, equity_trend <= -0.5).

## 4. Convergence and initialization

| model_id | n_seeds | n_init_per_seed | converged_share | all_monotone | n_starts | n_starts_with_decrease | max_objective_decrease | empty_state_events | loglik_per_obs_best | loglik_per_obs_range | best_basin_share | single_start_best_basin_share | train_mean_pairwise_ari | train_min_pairwise_ari | validation_mean_pairwise_ari | mean_phi_bound_share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ar_hmm_k3 | 10 | 5 | 1.000 | True | 50 | 0 | 0.000 | 0 | 3.185 | 0.000 | 1.000 | 0.820 | 0.622 | 0.320 | 0.592 | 0.367 |
| ar_hmm_k2 | 10 | 5 | 1.000 | True | 50 | 0 | 0.000 | 0 | 3.077 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.300 |

Estimation amendments (the first-run outcome is recorded, not deleted):

- stage5.5 v1 -> v1.1 (2026-10-21): The v1 M-step set the innovation variance to s2 + 1e-4 (weighted residual variance plus a constant) and floored pi at 1e-8. s2 + 1e-4 is not the maximizer of the EM Q-function, so EM was not guaranteed to increase the likelihood. In the first real run some AR-HMM K=3 starts decreased their objective (all_monotone = False), and no check failed because only convergence of the reference fit was tested. Fix: Variance update replaced by the exact constrained maximizer max(s2, 1e-4); pi = gamma_1 (exact). Every start's trajectory is kept; any decrease above MONOTONE_ABS_TOL = 1e-7 nats now FAILS the run, and a Q-function optimality check verifies the M-step on Train. Audit: scripts/07c_audit_ar_hmm_em.py. First run: v1 (real data): ar_hmm_k3 Train 3.184 / Validation 3.354 nats per day, verdict challenger_density; criteria 5 and 6 failed. All Stage 5.5 results are re-estimated with the exact M-step; the frozen Stage 5 HMM is unaffected (it is not refitted).

## 5. Predictive log scores

| model_id | k | n_parameters | train_avg_log_score | validation_avg_log_score | train_validation_gap | converged | phi_bound_share |
|---|---|---|---|---|---|---|---|
| frozen_hmm_k3 | 3 | 38 | -4.739 | -5.885 | 1.146 | True |  |
| ar_hmm_k3 | 3 | 53 | 3.185 | 3.418 | -0.234 | True | 0.400 |
| ar1 | 1 | 15 | 2.604 | 3.014 | -0.410 | True | 0.600 |
| ar_hmm_k2 | 2 | 33 | 3.077 | 3.360 | -0.282 | True | 0.300 |
| iid_gaussian | 1 | 10 | -7.094 | -7.116 | 0.022 | True |  |

Paired daily differences on Validation (Newey-West, 21 lags):

| model | versus | mean_diff | nw_t | ci95_low | ci95_high | share_days_better |
|---|---|---|---|---|---|---|
| frozen_hmm_k3 | ar1 | -8.899 | -24.104 | -9.623 | -8.175 | 0.010 |
| ar_hmm_k3 | frozen_hmm_k3 | 9.303 | 24.760 | 8.567 | 10.040 | 0.999 |
| ar_hmm_k3 | ar1 | 0.404 | 6.722 | 0.286 | 0.522 | 0.608 |
| ar1 | frozen_hmm_k3 | 8.899 | 24.104 | 8.175 | 9.623 | 0.990 |
| ar_hmm_k2 | frozen_hmm_k3 | 9.245 | 24.626 | 8.509 | 9.980 | 0.999 |
| ar_hmm_k2 | ar1 | 0.346 | 6.302 | 0.238 | 0.453 | 0.628 |
| iid_gaussian | frozen_hmm_k3 | -1.231 | -5.305 | -1.686 | -0.776 | 0.210 |
| iid_gaussian | ar1 | -10.130 | -38.640 | -10.644 | -9.616 | 0.011 |

Decomposition of the Validation log score (per day, relative to the i.i.d. Gaussian reference):

- latent states only (A - ref): +1.231
- autoregression only (C - ref): +10.130
- autoregression + states (B - ref): +10.534
- states beyond autoregression (B - C): +0.404

## 6. Residual temporal dependence

Quantile residuals r = Phi^-1(PIT) of the one-step predictive mixture (no hindsight states). Under a correct model they are i.i.d. N(0,1). within_state_resid_acf1 repeats the Stage 5 metric (filtered argmax state).

| model_id | split | mean_acf_lag1 | mean_acf_lag5 | mean_acf_lag21 | mean_sq_acf_lag1 | max_ljung_box_p21 | mean_qres_std | mean_within_state_acf1 |
|---|---|---|---|---|---|---|---|---|
| ar1 | train | 0.018 | -0.020 | -0.030 | 0.165 | 0.003 | 0.996 | 0.018 |
| ar1 | validation | 0.036 | 0.032 | -0.012 | 0.073 | 0.088 | 0.908 | 0.038 |
| ar_hmm_k2 | train | 0.014 | -0.015 | -0.021 | 0.113 | 0.010 | 0.979 | 0.024 |
| ar_hmm_k2 | validation | 0.020 | 0.025 | -0.007 | 0.040 | 0.090 | 0.942 | 0.023 |
| ar_hmm_k3 | train | 0.007 | -0.010 | -0.017 | 0.108 | 0.002 | 0.985 | 0.012 |
| ar_hmm_k3 | validation | 0.019 | 0.027 | -0.006 | 0.019 | 0.044 | 0.950 | 0.025 |
| frozen_hmm_k3 | train | 0.954 | 0.805 | 0.450 | 0.884 | 0.000 | 0.967 | 0.963 |
| frozen_hmm_k3 | validation | 0.963 | 0.840 | 0.557 | 0.925 | 0.000 | 1.109 | 0.975 |
| iid_gaussian | train | 0.986 | 0.934 | 0.727 | 0.968 | 0.000 | 1.000 | 0.987 |
| iid_gaussian | validation | 0.987 | 0.935 | 0.708 | 0.974 | 0.000 | 0.937 | 0.988 |

By factor (Validation):

| model_id | factor | qres_acf_lag1 | qres_acf_lag5 | qres_acf_lag21 | qres_sq_acf_lag1 | ljung_box_q21 | ljung_box_p21 | qres_mean | qres_std | pearson_innovation_std | within_state_resid_acf1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| frozen_hmm_k3 | equity_trend | 0.964 | 0.818 | 0.450 | 0.913 | 10105.739 | 0.000 | 0.292 | 0.871 | 0.869 | 0.978 |
| frozen_hmm_k3 | interest_rate | 0.980 | 0.897 | 0.593 | 0.947 | 13381.333 | 0.000 | 0.167 | 1.301 | 1.306 | 0.984 |
| frozen_hmm_k3 | inflation | 0.992 | 0.955 | 0.850 | 0.988 | 17814.587 | 0.000 | 0.807 | 1.079 | 1.069 | 0.995 |
| frozen_hmm_k3 | volatility | 0.917 | 0.727 | 0.389 | 0.817 | 8248.011 | 0.000 | 0.006 | 1.114 | 1.128 | 0.938 |
| frozen_hmm_k3 | liquidity_credit | 0.963 | 0.804 | 0.502 | 0.959 | 10437.220 | 0.000 | -0.335 | 1.180 | 1.153 | 0.980 |
| ar_hmm_k3 | equity_trend | 0.012 | 0.049 | 0.112 | 0.057 | 45.657 | 0.001 | 0.045 | 0.907 | 0.886 | 0.022 |
| ar_hmm_k3 | interest_rate | -0.034 | -0.029 | -0.090 | -0.010 | 33.173 | 0.044 | 0.001 | 0.925 | 0.906 | -0.059 |
| ar_hmm_k3 | inflation | 0.134 | 0.021 | 0.055 | -0.049 | 40.286 | 0.007 | 0.098 | 0.969 | 0.960 | 0.119 |
| ar_hmm_k3 | volatility | -0.031 | 0.047 | -0.038 | 0.101 | 33.498 | 0.041 | 0.017 | 0.956 | 1.006 | 0.047 |
| ar_hmm_k3 | liquidity_credit | 0.015 | 0.049 | -0.069 | -0.004 | 41.824 | 0.004 | -0.042 | 0.994 | 0.966 | -0.003 |
| ar1 | equity_trend | 0.026 | 0.037 | 0.110 | 0.100 | 44.759 | 0.002 | 0.033 | 0.865 | 0.865 | 0.026 |
| ar1 | interest_rate | -0.033 | -0.029 | -0.094 | 0.048 | 41.015 | 0.006 | -0.002 | 0.884 | 0.884 | -0.033 |
| ar1 | inflation | 0.153 | 0.024 | 0.064 | 0.032 | 47.624 | 0.001 | 0.085 | 0.915 | 0.915 | 0.153 |
| ar1 | volatility | -0.019 | 0.068 | -0.070 | 0.140 | 30.217 | 0.088 | -0.014 | 0.942 | 0.964 | -0.006 |
| ar1 | liquidity_credit | 0.051 | 0.060 | -0.067 | 0.043 | 43.689 | 0.003 | -0.035 | 0.935 | 0.935 | 0.051 |
| ar_hmm_k2 | equity_trend | 0.012 | 0.049 | 0.111 | 0.060 | 44.673 | 0.002 | 0.022 | 0.897 | 0.886 | 0.019 |
| ar_hmm_k2 | interest_rate | -0.047 | -0.033 | -0.088 | 0.010 | 32.997 | 0.046 | 0.002 | 0.921 | 0.910 | -0.076 |
| ar_hmm_k2 | inflation | 0.146 | 0.023 | 0.055 | 0.011 | 43.581 | 0.003 | 0.083 | 0.952 | 0.944 | 0.161 |
| ar_hmm_k2 | volatility | -0.028 | 0.044 | -0.042 | 0.101 | 30.079 | 0.090 | 0.018 | 0.946 | 1.036 | 0.052 |
| ar_hmm_k2 | liquidity_credit | 0.018 | 0.042 | -0.072 | 0.016 | 34.562 | 0.032 | -0.024 | 0.993 | 0.968 | -0.039 |
| iid_gaussian | equity_trend | 0.985 | 0.924 | 0.655 | 0.965 | 14729.039 | 0.000 | 0.287 | 0.764 | 0.764 | 0.988 |
| iid_gaussian | interest_rate | 0.978 | 0.898 | 0.585 | 0.948 | 13297.333 | 0.000 | 0.134 | 1.099 | 1.099 | 0.980 |
| iid_gaussian | inflation | 0.996 | 0.974 | 0.881 | 0.991 | 18887.306 | 0.000 | 0.650 | 0.927 | 0.927 | 0.996 |
| iid_gaussian | volatility | 0.984 | 0.925 | 0.662 | 0.982 | 14943.198 | 0.000 | -0.057 | 0.992 | 0.992 | 0.984 |
| iid_gaussian | liquidity_credit | 0.993 | 0.956 | 0.754 | 0.986 | 16762.602 | 0.000 | -0.262 | 0.902 | 0.902 | 0.993 |

## 7. Predictive calibration

Coverage of central intervals of the full mixture predictive distribution (equivalent to PIT inside [(1-p)/2, (1+p)/2]); widths in standardized units (sharpness); CRPS per factor (lower = better).

| model_id | split | n | pit_mean | pit_std | pit_ks | crps | coverage_50 | mean_width_50 | coverage_80 | mean_width_80 | coverage_90 | mean_width_90 | coverage_95 | mean_width_95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| frozen_hmm_k3 | train | 10285 | 0.495 | 0.288 | 0.020 | 0.382 | 0.482 | 0.925 | 0.817 | 1.761 | 0.922 | 2.264 | 0.964 | 2.704 |
| frozen_hmm_k3 | validation | 5030 | 0.551 | 0.310 | 0.088 | 0.457 | 0.443 | 0.900 | 0.712 | 1.711 | 0.815 | 2.200 | 0.892 | 2.627 |
| ar_hmm_k3 | train | 10285 | 0.501 | 0.279 | 0.022 | 0.079 | 0.537 | 0.171 | 0.825 | 0.352 | 0.908 | 0.488 | 0.952 | 0.626 |
| ar_hmm_k3 | validation | 5030 | 0.507 | 0.277 | 0.029 | 0.074 | 0.535 | 0.168 | 0.831 | 0.345 | 0.918 | 0.478 | 0.962 | 0.617 |
| ar1 | train | 10285 | 0.494 | 0.259 | 0.066 | 0.081 | 0.613 | 0.207 | 0.855 | 0.393 | 0.918 | 0.505 | 0.946 | 0.601 |
| ar1 | validation | 5030 | 0.499 | 0.253 | 0.067 | 0.075 | 0.625 | 0.207 | 0.870 | 0.393 | 0.935 | 0.505 | 0.960 | 0.601 |
| ar_hmm_k2 | train | 10285 | 0.499 | 0.277 | 0.029 | 0.080 | 0.548 | 0.176 | 0.824 | 0.354 | 0.908 | 0.483 | 0.950 | 0.617 |
| ar_hmm_k2 | validation | 5030 | 0.504 | 0.275 | 0.030 | 0.074 | 0.549 | 0.172 | 0.830 | 0.346 | 0.918 | 0.472 | 0.961 | 0.604 |
| iid_gaussian | train | 10285 | 0.494 | 0.280 | 0.033 | 0.555 | 0.526 | 1.349 | 0.835 | 2.563 | 0.903 | 3.290 | 0.937 | 3.920 |
| iid_gaussian | validation | 5030 | 0.531 | 0.286 | 0.058 | 0.566 | 0.509 | 1.349 | 0.791 | 2.563 | 0.894 | 3.290 | 0.956 | 3.920 |

By factor (Validation, 90%):

| model_id | equity_trend | inflation | interest_rate | liquidity_credit | volatility |
|---|---|---|---|---|---|
| ar1 | 0.942 | 0.924 | 0.935 | 0.925 | 0.947 |
| ar_hmm_k2 | 0.927 | 0.913 | 0.925 | 0.896 | 0.930 |
| ar_hmm_k3 | 0.927 | 0.914 | 0.923 | 0.901 | 0.927 |
| frozen_hmm_k3 | 0.914 | 0.711 | 0.774 | 0.829 | 0.847 |
| iid_gaussian | 0.962 | 0.805 | 0.852 | 0.921 | 0.929 |

This is calibration of observed factor forecasts. Without observed regime labels, no model's P(stress) can be called a calibrated crisis probability.

## 8. Probability extremeness and regime quality

| model_id | train_shares | train_min_state_share | train_switches_per_year | train_median_duration_days | train_mean_max_filtered_prob | train_mean_normalized_entropy | train_share_uncertain_days | validation_shares | validation_min_state_share | validation_switches_per_year | validation_median_duration_days | validation_mean_max_filtered_prob | validation_mean_normalized_entropy | validation_share_uncertain_days | self_transitions | expected_durations_days | validation_implied_switches_per_year | validation_realized_to_implied_switch_ratio | stress_states |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| frozen_hmm_k3 | S1 41% / S2 34% / S3 25% | 0.248 | 4.778 | 46.500 | 0.981 | 0.043 | 0.011 | S1 45% / S2 34% / S3 21% | 0.208 | 6.763 | 18.000 | 0.977 | 0.053 | 0.010 | 0.9881 / 0.9783 / 0.9879 | 84.0 / 46.1 / 83.0 | 3.855 | 1.754 | S3 |
| ar_hmm_k3 | S1 62% / S2 24% / S3 14% | 0.141 | 88.819 | 2.000 | 0.840 | 0.369 | 0.271 | S1 68% / S2 20% / S3 12% | 0.117 | 86.421 | 2.000 | 0.829 | 0.394 | 0.307 | 0.8118 / 0.6281 / 0.3489 | 5.3 / 2.7 / 1.5 | 74.960 | 1.153 | S2 S3 |
| ar_hmm_k2 | S1 78% / S2 22% | 0.222 | 58.192 | 2.000 | 0.914 | 0.308 | 0.160 | S1 83% / S2 17% | 0.169 | 56.863 | 2.000 | 0.921 | 0.300 | 0.124 | 0.8576 / 0.5909 | 7.0 / 2.4 | 49.963 | 1.138 | S2 |

## 9. Economic interpretation of the AR-HMM states

| model_id | state | split | share | equity_trend | interest_rate | inflation | volatility | liquidity_credit | is_stress_state | hypothesis_label | train_validation_profile_distance |
|---|---|---|---|---|---|---|---|---|---|---|---|
| frozen_hmm_k3 | S1 | train | 0.41 | 0.39 | -0.05 | 0.15 | -0.71 | -0.90 | False | Growth / Risk-on (hypothesis) | 0.78 |
| frozen_hmm_k3 | S1 | validation | 0.45 | 0.48 | 0.32 | 0.81 | -0.72 | -1.06 | False | Growth / Risk-on (hypothesis) | 0.78 |
| frozen_hmm_k3 | S2 | train | 0.34 | -0.36 | 0.43 | -0.33 | 0.10 | -0.04 | False | Neutral | 0.53 |
| frozen_hmm_k3 | S2 | validation | 0.34 | -0.21 | -0.01 | -0.09 | 0.18 | 0.02 | False | Neutral | 0.53 |
| frozen_hmm_k3 | S3 | train | 0.25 | -1.65 | -0.05 | -1.17 | 1.90 | 1.58 | True | Liquidity Crisis (hypothesis) | 1.32 |
| frozen_hmm_k3 | S3 | validation | 0.21 | -1.00 | 0.27 | -0.28 | 1.96 | 0.92 | True | Liquidity Crisis (hypothesis) | 1.32 |
| ar_hmm_k3 | S1 | train | 0.62 | -0.23 | 0.10 | -0.24 | 0.03 | -0.17 | False | Neutral | 0.64 |
| ar_hmm_k3 | S1 | validation | 0.68 | -0.01 | 0.14 | 0.31 | -0.05 | -0.38 | False | Neutral | 0.64 |
| ar_hmm_k3 | S2 | train | 0.24 | -0.44 | 0.17 | -0.39 | 0.17 | 0.09 | True | Neutral | 0.92 |
| ar_hmm_k3 | S2 | validation | 0.20 | 0.05 | 0.28 | 0.33 | 0.15 | -0.17 | True | Neutral | 0.92 |
| ar_hmm_k3 | S3 | train | 0.14 | -0.89 | 0.08 | -0.71 | 1.09 | 0.65 | True | Fragile / Transition (hypothesis) | 1.04 |
| ar_hmm_k3 | S3 | validation | 0.12 | -0.57 | 0.42 | 0.03 | 1.22 | 0.11 | True | Fragile / Transition (hypothesis) | 1.04 |
| ar_hmm_k2 | S1 | train | 0.78 | -0.28 | 0.10 | -0.24 | 0.04 | -0.14 | False | Neutral | 0.66 |
| ar_hmm_k2 | S1 | validation | 0.83 | 0.00 | 0.16 | 0.32 | -0.03 | -0.36 | False | Neutral | 0.66 |
| ar_hmm_k2 | S2 | train | 0.22 | -0.70 | 0.15 | -0.72 | 0.83 | 0.52 | True | Fragile / Transition (hypothesis) | 1.03 |
| ar_hmm_k2 | S2 | validation | 0.17 | -0.38 | 0.39 | 0.10 | 0.98 | 0.07 | True | Fragile / Transition (hypothesis) | 1.03 |

AR coefficients and innovation s.d. (original units) of the AR models (Train estimates):

| model_id | state | phi_equity_trend | phi_interest_rate | phi_inflation | phi_volatility | phi_liquidity_credit | innov_sd_equity_trend | innov_sd_interest_rate | innov_sd_inflation | innov_sd_volatility | innov_sd_liquidity_credit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ar_hmm_k3 | S1 | 0.990 | 0.975 | 0.990 | 0.975 | 0.990 | 0.123 | 0.097 | 0.053 | 0.126 | 0.077 |
| ar_hmm_k3 | S2 | 0.946 | 0.958 | 0.990 | 0.963 | 0.963 | 0.130 | 0.237 | 0.124 | 0.192 | 0.122 |
| ar_hmm_k3 | S3 | 0.990 | 0.973 | 0.990 | 0.953 | 0.990 | 0.187 | 0.173 | 0.081 | 0.382 | 0.163 |
| ar_hmm_k2 | S1 | 0.982 | 0.976 | 0.989 | 0.971 | 0.985 | 0.125 | 0.114 | 0.065 | 0.139 | 0.084 |
| ar_hmm_k2 | S2 | 0.990 | 0.955 | 0.990 | 0.972 | 0.990 | 0.202 | 0.248 | 0.124 | 0.377 | 0.186 |

State matching with the frozen HMM (label-invariant ARI; economic matching by Train profile distance):

| model_id | reference | train_ari | validation_ari | profile_matching | train_stress_overlap_recall | train_stress_overlap_precision | validation_p_stress_corr |
|---|---|---|---|---|---|---|---|
| ar_hmm_k3 | frozen_hmm_k3 | 0.049 | 0.046 | S1<->S1 (distance 1.28); S2<->S2 (distance 0.31); S3<->S3 (distance 1.52) | 0.536 | 0.352 | 0.201 |
| ar_hmm_k2 | frozen_hmm_k3 | 0.049 | 0.048 | S2<->S1 (distance 0.37); S3<->S2 (distance 1.84) | 0.384 | 0.430 | 0.224 |

## 10. Causality and leakage checks

| name | status | detail |
|---|---|---|
| Frozen Stage 5 model is the documented baseline | PASS | Stage 5 manifest selected hmm_std_diag_k3; expected hmm_std_diag_k3 |
| Five factor columns present in the frozen order | PASS | train ['equity_trend', 'interest_rate', 'inflation', 'volatility', 'liquidity_credit']; validation ['equity_trend', 'interest_rate', 'inflation', 'volatility', 'liquidity_credit'] |
| train factors: chronological, unique dates, finite | PASS | 2058 rows 2007-10-30..2015-12-31 |
| validation factors: chronological, unique dates, finite | PASS | 1006 rows 2016-01-04..2019-12-31 |
| Validation starts after train ends (no overlap) | PASS | train ends 2015-12-31, validation starts 2016-01-04 |
| Row counts match split_manifest.json | PASS | train 2058 (manifest 2058), validation 1006 (manifest 1006) |
| train returns aligned with factors | PASS | 9 ETF columns |
| validation returns aligned with factors | PASS | 9 ETF columns |
| Factor definitions unchanged since stage 3 (factors.py hash) | PASS | manifest 28414cd58b03..., current 28414cd58b03... |
| Split files match split_manifest.json hashes | PASS | 4 files |
| Train-only scaler identical to the frozen model's preprocessing | PASS | train mean / train std (ddof=0) |
| Frozen HMM reproduced from its saved parameters (no refit) | PASS | Train log-likelihood -9755.889562 vs saved -9755.889562; Validation log score -5.884769 vs Stage 5 table -5.884770 (table stored to 6 significant digits) |
| Filtered probabilities sum to one (all models) | PASS | 1.8e-15 |
| AR-HMM transition rows sum to one | PASS | 0.0e+00 |
| Filter log scores reproduce the AR-HMM EM log-likelihood | PASS | 0.0e+00 |
| Validation continues the Train recursion (lag and state) without reset | PASS | max difference vs filtering the joined sequence 0.0e+00 |
| Future Validation changes leave earlier filtered probabilities unchanged | PASS | max change before cut 0.0e+00 |
| Predicted state probability and score of day t use only data through t-1 | PASS | changing x_t leaves P(Z_t | x_1..x_t-1) unchanged (max change 0.0e+00) |
| AR-HMM estimates depend on Train only and are reproducible | PASS | refit of the reference seed: |log-likelihood difference| 0.0e+00 |
| No test file read | PASS | split_manifest.json, train_features.csv, train_returns.csv, validation_features.csv, validation_returns.csv |
| AR reference fits converged | PASS | all converged |
| EM objective monotone for every start (all seeds x starts, tolerance 1e-07 nats) | PASS | ar1: 0/1 starts with decrease > 1e-07 (max 0.0e+00); ar_hmm_k3: 0/50 starts with decrease > 1e-07 (max 0.0e+00); ar_hmm_k2: 0/50 starts with decrease > 1e-07 (max 0.0e+00) |
| M-step is the exact constrained maximizer of the EM Q-function (Train) | PASS | ar1: max Q gain of any feasible coordinate move -5.1e-06; ar_hmm_k3: max Q gain of any feasible coordinate move -8.4e-07; ar_hmm_k2: max Q gain of any feasible coordinate move -1.3e-06 |
| Stage 5 outputs unchanged (data/processed/hmm hashes before == after) | PASS | 50 files compared |
| No library warnings | PASS | none |

## 11. Limitations

- Validation reused from Stage 5 (development sample); conclusions are robustness evidence, not a holdout test.
- The factors are rolling-window statistics (21-252 day windows). Overlapping windows induce moving-average dependence at many lags that an AR(1) cannot fully remove; residual autocorrelation at lags 5 and 21 shows how much remains.
- Diagonal Phi and Sigma ignore cross-factor dynamics and contemporaneous correlation.
- |phi| <= 0.99 keeps state means identified; bound hits are reported (phi_bound_share).
- Better density forecasts of the factors do not imply better crisis prediction or portfolio returns.

## 12. Recommendation for Stage 6

- frozen_hmm_k3 for regimes; ar_hmm_k3 as predictive-density challenger (verdict challenger_density).
- Stage 5 selection results and files are unchanged (hash check above).

## Figures

- `reports/figures/ar_hmm/ar_hmm_model_comparison.png`
- `reports/figures/ar_hmm/predictive_interval_coverage.png`
- `reports/figures/ar_hmm/residual_autocorrelation_comparison.png`
- `reports/figures/ar_hmm/predictive_pit_diagnostics.png`
- `reports/figures/ar_hmm/hmm_vs_ar_hmm_probabilities.png`
- `reports/figures/ar_hmm/ar_hmm_regime_factor_profiles.png`
- `reports/figures/ar_hmm/train_validation_generalization.png`
