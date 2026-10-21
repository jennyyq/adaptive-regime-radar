# Stage 5 - Standard and Sticky Gaussian HMM regime detection

Train 2007-10-30..2015-12-31 (2058 rows); Validation 2016-01-04..2019-12-31 (1006 rows). Stage 3 protocol ARR-stage3-protocol-v1.2. Test split not read. Checks: 23 PASS.

## Summary

- Selected HMM: **hmm_std_diag_k3** (standard, K=3, diag covariance).
- Rule outcome: K=3 chosen because smaller-K models are significantly worse on Validation; its advantage over hmm_std_diag_k2 holds in 4 of 4 Validation years. Standard model kept: no Sticky model met all of competitiveness, a >= 20% switch reduction and the detection-delay budget at this K and covariance (hmm_sticky_diag_k3_w0.10: switch reduction +0%, competitive True, delay ok True; hmm_sticky_diag_k3_w0.30: switch reduction +7%, competitive True, delay ok True; hmm_sticky_diag_k3_w1.00: switch reduction +7%, competitive True, delay ok True).
- Best Validation log score among eligible models: hmm_std_diag_k4; best overall (any eligibility): hmm_std_diag_k4; competitive set (NW t < 2.0): hmm_std_diag_k3, hmm_std_diag_k4, hmm_sticky_diag_k3_w0.10, hmm_sticky_diag_k3_w0.30, hmm_sticky_diag_k3_w1.00, hmm_sticky_diag_k4_w0.10, hmm_sticky_diag_k4_w0.30, hmm_sticky_diag_k4_w1.00.
- Selected model: Validation log score -5.885 per day, Train -4.740; Validation switches 6.8/year; self-transitions 0.9881 / 0.9783 / 0.9879.

- Versus the frozen Stage 4 GMM (diag, K=2) on Validation: mean daily log-score difference +0.771 (Newey-West t 6.02); HMM better on 75% of days; state ARI 0.41.

**Key findings (computed from this run)**

- K=2: the strongest prior (w=1.00) changes implied durations from 177.8 / 99.1 to 330.9 / 248.5 days, but Validation switching from 2.8 to 2.8 per year and the Validation log score by -0.004 per day.
- K=3: the strongest prior (w=1.00) changes implied durations from 84.0 / 46.1 / 83.0 to 153.2 / 92.2 / 197.8 days, but Validation switching from 6.8 to 6.3 per year and the Validation log score by -0.009 per day.
- K=4: the strongest prior (w=1.00) changes implied durations from 96.3 / 85.5 / 45.0 / 77.0 to 201.6 / 193.0 / 85.7 / 164.6 days, but Validation switching from 6.3 to 6.8 per year and the Validation log score by -0.003 per day.
- Why the prior matters little here: on a typical Validation day the emission log-likelihood ratio between the two most likely states is 4.3 nats (median), while the transition log-odds log(A_ii / (1 - A_ii)) average 4.2. Daily evidence from five persistent factors dominates the transition prior, so the filtered state is mostly decided by the emissions.
- Filtered probabilities are very confident (mean max probability 0.977, uncertain days 1.0% on Validation). They should not be read as calibrated crisis probabilities without a calibration check in the Early Warning stage.
- Profile shift > 1.0 between Train and Validation (selected model): S3 1.32. The state's composition changes out of sample; read its label with care.
- SPY next-session annualized mean by state (train): S1 +13.2%, S2 +15.1%, S3 -19.8% (descriptive).
- SPY next-session annualized mean by state (validation): S1 +12.6%, S2 +13.9%, S3 +16.1% (descriptive).

## 1. Research question

Does explicitly modelling regime persistence (Standard vs Sticky Gaussian HMM) give more stable and more informative causal state probabilities than static clustering, without unacceptable detection delay? Sticky is not assumed to be better; the comparison uses Validation evidence fixed in advance.

## 2. Data and temporal protocol

- Factors (frozen order): equity_trend, interest_rate, inflation, volatility, liquidity_credit. Factor definitions unchanged (factors.py hash checked).
- Preprocessing: train mean / train std (ddof=0), fitted on Train only, identical to Stage 4. train std ratio max/min = 1.97 (threshold 1.25); max |mean|/std = 0.36 (threshold 0.25); largest variance share 31% (volatility). Apply train-fitted standardization so that no factor dominates Euclidean distance.
- Train: all HMM parameters, the scaler and the reference seed. Validation: fixed parameters, causal filtering, model selection. Test (2020 onward): not loaded.
- No shuffling; one continuous Train sequence; no refit on Train + Validation in this stage.

## 3. Standard Gaussian HMM

Z_t in {1..K} follows a first-order Markov chain with P(Z_1 = k) = pi_k and A_ij = P(Z_t = j | Z_t-1 = i). Emissions X_t | Z_t = k ~ N(mu_k, Sigma_k), with Sigma_k diagonal (primary baseline) or full (controlled challenger). The likelihood is log p(x_1:T) = sum_t log sum_k P(Z_t = k | x_1:t-1) N(x_t; mu_k, Sigma_k). Covariances get reg_covar = 0.0001 on the diagonal (as in Stage 4). Assumptions: time-homogeneous transitions, conditionally independent Gaussian emissions given the state, geometric state durations.

## 4. Sticky HMM

Each row of A has a Dirichlet prior with alpha_ij = 1 + kappa 1[i = j] (finite-K analogue of the sticky HDP-HMM, Fox et al. 2011). MAP-EM maximizes log p(x | theta) + kappa sum_i log A_ii; the M-step is A_ij = (n_ij + kappa 1[i=j]) / (sum_j n_ij + kappa), with n_ij the expected transition counts. The prior acts inside every EM iteration and therefore changes the states themselves, not just their labels. kappa = w (T_train - 1) / K with the predetermined grid w in [0.1, 0.3, 1.0] (prior pseudo-transitions as a share of the average observed transitions per row). Train AIC/BIC for Sticky models use the likelihood at the MAP estimate with the nominal parameter count; the prior reduces effective complexity, so these numbers are indicative only.

## 5. Estimation and initialization

- 10 seeds (20240531..20240540). For each seed the estimator runs several EM starts (n_init_per_seed in the stability table; protocol 5, as for the Stage 4 GMM) and keeps the start with the highest Train objective. Each start begins from one K-Means run (n_init=1): means, within-cluster covariances and a smoothed K-Means transition matrix. Standard and Sticky fits with the same seed use identical starting points.
- EM until the objective changes by < 1e-06 per observation (max 1000 iterations); monotonicity of the objective is checked every iteration. The reference fit is the converged seed with the highest Train objective; the others are aligned to it by Hungarian matching of emission means.
- States are ordered by the stress score (volatility + liquidity_credit - equity_trend) of their emission means in original units; S1..SK are neutral identifiers. 'Stress states' = states whose emission stress score exceeds the Train average (fixed rule); P(stress) is the sum of their filtered probabilities.

## 6. Filtering versus smoothing

- Decision quantities: predicted P(Z_t | x_1:t-1) and filtered P(Z_t | x_1:t), computed in log space with log-sum-exp, saved for Validation (predicted and filtered) and Train (filtered).
- Validation starts from the last Train filtered distribution propagated by A; this equals filtering the concatenated sequence and uses no later Validation data (checked below).
- Smoothed probabilities and Viterbi paths use the whole sequence. They appear only as a diagnostic ('hindsight revision share' = Validation days whose smoothed argmax differs from the filtered argmax).
- **Train-period filtered states use parameters estimated with data through 2015. They are retrospective in-sample analysis, not live predictions available in 2008.**

## 7. Model-selection criteria (fixed before results)

- 1. Eligibility: >= 80% of the 10 seeds converge and EM is monotone; every Train state >= 3% of days (filtered argmax); mean pairwise ARI of Train filtered assignments across seeds >= 0.80. A state below 1% of Validation days, a self-transition above 0.999 or a median run of <= 2 days is FLAGGED, not excluded.
- 2. Validation evidence: the primary score is the average one-step-ahead predictive log density log p(x_t | x_1..x_t-1) on Validation, with parameters fixed at their Train estimates and the recursion continued from the last Train filtered distribution. The best eligible model defines the benchmark; a model is 'competitive' if its mean daily shortfall has a Newey-West t-statistic (21 lags) below 2.
- 3. Parsimony: among competitive models choose the smallest K, then diagonal over full covariance. If that K is larger than the K of the best eligible smaller model (i.e. the smaller K is significantly worse), the larger K must also beat it in at least 3 of the 4 Validation calendar years (phase consistency); otherwise the smaller K is kept.
- 4. Persistence vs delay (same K and covariance): a Sticky model is preferred to the Standard model only if it is competitive, reduces Validation switches per year by >= 20%, and respects the delay budget on Train+Validation: (a) versus factor-based stress onsets, the median days until P(stress) >= 0.5 rise by <= 5 and at most 10 percentage points more onsets go undetected within 63 days; (b) on the same sequence, the Sticky model enters the Standard model's stress episodes with a median lag of <= 5 days and misses <= 10% of them. Among qualifying Sticky models, the highest Validation log score wins. Otherwise the Standard model is kept.
- 5. Train AIC/BIC, economic profiles, asset behaviour and timelines are reported for every model but do not rank models, so that no model is chosen for an attractive timeline. Crisis dates are never used.
- 6. The selected HMM is compared with the frozen Stage 4 GMM (diag, K=2) by daily predictive log scores (Newey-West t-test). Classification agreement is reported separately and is not evidence of better forecasts.

Thresholds: min_converged_seed_share = 0.8, min_train_state_share = 0.03, min_mean_pairwise_ari_train = 0.8, min_validation_state_share = 0.01, near_absorbing_self_transition = 0.999, flag_median_run_days = 2, nw_lags = 21, nw_t_threshold = 2.0, max_median_delay_increase_days = 5, max_undetected_share_increase = 0.1, min_switch_reduction = 0.2, best_basin_tolerance_per_obs = 0.001.

**Protocol amendments after the first real-data run** (the first-run outcome is recorded; nothing is deleted):

- stage5 v1 -> v1.1 (2026-10-21): Seed-stability eligibility is measured on the per-seed estimator 'best of 5 EM starts' instead of single EM starts. Reason: In v1 each seed was ONE EM start, so the pairwise ARI across seeds measured how often EM falls into a local optimum, not whether the estimator is reproducible. Stage 4 (whose 0.80 ARI threshold this rule copies) used n_init = 5 starts per seed for the GMM. In the first real run 7/10 single starts for Standard diag K=3 and 2/10 for K=4 reached the identical best optimum (parameter distance 0), yet K=3/K=4 were excluded with single-start ARI 0.725 / 0.643 although their Validation log scores were higher than K=2. First run: v1 rules selected hmm_std_diag_k2. The frequency of EM local optima is still reported (single_start_best_basin_share in hmm_initialization_stability.csv).
- stage5 v1 -> v1.1 (2026-10-21): Detection lag is signed: if the model is already in its stress states when a reference onset occurs, the lag is minus the number of days the current stress run had lasted (capped at -63). Reason: In the first real run the K=2 models were already in stress at 100% of factor-score onsets, so the v1 lag was 0 by construction and could not distinguish models.
- stage5 v1 -> v1.1 (2026-10-21): Report fixes only: '|' inside table cells replaced by '/', best eligible and best overall Validation model reported separately, emission-vs-transition evidence diagnostic added. Reason: Markdown tables were misaligned; the summary named the best eligible model as the best model.
- stage5 v1.1 -> v1.2 (2026-10-21): Phase-consistency gate for a larger K (>= 3 of 4 Validation years), probability-calibration diagnostics with an out-of-sample covariance-inflation test (chosen on 2016-2017, judged on 2018-2019), training-window sensitivity (4 predefined windows), Stage 4 GMM reference rebuilt from Train instead of unpickled, and a log of all library warnings. Reason: Requested review after the first real run: (1) K=3/K=4 scored higher on Validation than K=2, so a larger K needs evidence that the gain is not one episode; the yearly split was fixed before the yearly results were seen; (2) mean max filtered probability 0.990 on Validation needs a calibration test rather than ad hoc smoothing; (3) a scikit-learn version-mismatch warning came from unpickling the Stage 4 GMM. First run: Under v1 the selected model was hmm_std_diag_k2; best overall Validation score was hmm_std_diag_k4 (-5.803 vs -6.159 per day).

## 8. K = 2 / 3 / 4 comparison

nw_t_vs_best = Newey-West t-statistic of the mean daily shortfall versus the best ELIGIBLE model; a negative value means the model scores higher than that benchmark (possible only for ineligible models).

| model_id | n_parameters | converged | train_avg_loglik | validation_avg_loglik | nw_t_vs_best | competitive | eligible | train_bic | train_shares | validation_shares | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hmm_std_diag_k2 | 23 | True | -5.282 | -6.159 | 2.280 | False | True | 21916.369 | S1 64% / S2 36% | S1 69% / S2 31% | - |
| hmm_std_full_k2 | 43 | True | -4.862 | -6.501 | 2.851 | False | True | 20339.346 | S1 55% / S2 45% | S1 55% / S2 45% | - |
| hmm_std_diag_k3 | 38 | True | -4.740 | -5.885 | 0.750 | True | True | 19801.700 | S1 41% / S2 34% / S3 25% | S1 45% / S2 34% / S3 21% | - |
| hmm_std_full_k3 | 68 | True | -4.289 | -6.562 | 3.200 | False | False | 18174.359 | S1 43% / S2 32% / S3 25% | S1 48% / S2 26% / S3 26% | - |
| hmm_std_diag_k4 | 55 | True | -4.317 | -5.803 | 0.000 | True | True | 18188.062 | S1 23% / S2 23% / S3 31% / S4 24% | S1 13% / S2 38% / S3 31% / S4 19% | - |
| hmm_std_full_k4 | 95 | True | -3.912 | -7.459 | 3.072 | False | False | 16826.837 | S1 45% / S2 24% / S3 21% / S4 10% | S1 48% / S2 21% / S3 24% / S4 7% | - |
| hmm_sticky_diag_k2_w0.10 | 23 | True | -5.282 | -6.160 | 2.282 | False | True | 21916.549 | S1 64% / S2 36% | S1 69% / S2 31% | - |
| hmm_sticky_diag_k2_w0.30 | 23 | True | -5.282 | -6.161 | 2.285 | False | True | 21917.566 | S1 64% / S2 36% | S1 69% / S2 31% | - |
| hmm_sticky_diag_k2_w1.00 | 23 | True | -5.284 | -6.163 | 2.297 | False | True | 21922.881 | S1 64% / S2 36% | S1 68% / S2 32% | - |
| hmm_sticky_diag_k3_w0.10 | 38 | True | -4.741 | -5.886 | 0.761 | True | True | 19802.001 | S1 41% / S2 34% / S3 25% | S1 45% / S2 34% / S3 21% | - |
| hmm_sticky_diag_k3_w0.30 | 38 | True | -4.741 | -5.888 | 0.778 | True | True | 19803.778 | S1 41% / S2 34% / S3 25% | S1 45% / S2 34% / S3 21% | - |
| hmm_sticky_diag_k3_w1.00 | 38 | True | -4.743 | -5.893 | 0.825 | True | True | 19813.972 | S1 41% / S2 35% / S3 25% | S1 45% / S2 34% / S3 21% | - |
| hmm_sticky_diag_k4_w0.10 | 55 | True | -4.317 | -5.803 | 0.068 | True | True | 18188.353 | S1 23% / S2 23% / S3 30% / S4 24% | S1 13% / S2 38% / S3 30% / S4 19% | - |
| hmm_sticky_diag_k4_w0.30 | 55 | True | -4.317 | -5.804 | 0.340 | True | True | 18190.054 | S1 23% / S2 23% / S3 30% / S4 24% | S1 13% / S2 38% / S3 30% / S4 19% | - |
| hmm_sticky_diag_k4_w1.00 | 55 | True | -4.320 | -5.806 | 1.011 | True | True | 18199.679 | S1 23% / S2 23% / S3 30% / S4 24% | S1 13% / S2 38% / S3 30% / S4 19% | - |

Initialization stability:

Each seed's estimate is the best of its EM starts (n_init_per_seed). best_basin_share = share of seeds whose estimate reaches the best Train objective; single_start_best_basin_share = the same for individual EM starts (local-optimum frequency).

| model_id | n_init_per_seed | converged_share | best_basin_share | single_start_best_basin_share | objective_per_obs_range | train_mean_pairwise_ari | validation_mean_pairwise_ari | best_basin_max_mean_distance | mean_iterations |
|---|---|---|---|---|---|---|---|---|---|
| hmm_std_diag_k2 | 5 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 10.000 |
| hmm_std_full_k2 | 5 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 29.000 |
| hmm_std_diag_k3 | 5 | 1.000 | 1.000 | 0.720 | 0.000 | 0.999 | 1.000 | 0.000 | 21.300 |
| hmm_std_full_k3 | 5 | 1.000 | 0.200 | 0.040 | 0.036 | 0.727 | 0.678 | 0.000 | 35.400 |
| hmm_std_diag_k4 | 5 | 1.000 | 0.900 | 0.300 | 0.045 | 0.868 | 0.894 | 0.000 | 25.600 |
| hmm_std_full_k4 | 5 | 1.000 | 0.100 | 0.020 | 0.045 | 0.734 | 0.690 | 0.000 | 25.800 |
| hmm_sticky_diag_k2_w0.10 | 5 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 10.000 |
| hmm_sticky_diag_k2_w0.30 | 5 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 10.000 |
| hmm_sticky_diag_k2_w1.00 | 5 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 9.000 |
| hmm_sticky_diag_k3_w0.10 | 5 | 1.000 | 1.000 | 0.720 | 0.000 | 1.000 | 1.000 | 0.000 | 20.300 |
| hmm_sticky_diag_k3_w0.30 | 5 | 1.000 | 1.000 | 0.720 | 0.000 | 1.000 | 1.000 | 0.000 | 14.000 |
| hmm_sticky_diag_k3_w1.00 | 5 | 1.000 | 1.000 | 0.720 | 0.000 | 1.000 | 1.000 | 0.000 | 14.000 |
| hmm_sticky_diag_k4_w0.10 | 5 | 1.000 | 0.900 | 0.320 | 0.045 | 0.868 | 0.894 | 0.000 | 25.000 |
| hmm_sticky_diag_k4_w0.30 | 5 | 1.000 | 0.900 | 0.320 | 0.046 | 0.868 | 0.893 | 0.000 | 25.100 |
| hmm_sticky_diag_k4_w1.00 | 5 | 1.000 | 0.900 | 0.320 | 0.048 | 0.868 | 0.894 | 0.000 | 29.200 |

Sticky assessment at the selected K and covariance (rule 4):

| model_id | competitive | switch_reduction | median_delay_increase | undetected_increase | paired_median_entry_lag | paired_missed_share | paired_median_exit_lag | delay_ok | qualifies | validation_avg_loglik |
|---|---|---|---|---|---|---|---|---|---|---|
| hmm_sticky_diag_k3_w0.10 | True | 0.000 | 0.500 | 0.000 | 0.000 | 0.000 | 0.000 | True | False | -5.886 |
| hmm_sticky_diag_k3_w0.30 | True | 0.074 | 0.500 | 0.000 | 0.000 | 0.000 | 0.000 | True | False | -5.888 |
| hmm_sticky_diag_k3_w1.00 | True | 0.074 | 0.500 | 0.000 | 0.000 | 0.000 | 0.000 | True | False | -5.893 |

## 9. Transition matrices

| model_id | self_transitions | expected_durations_days | stress_states |
|---|---|---|---|
| hmm_std_diag_k2 | 0.9944 / 0.9899 | 177.8 / 99.1 | S2 |
| hmm_std_full_k2 | 0.9938 / 0.9924 | 160.6 / 130.8 | S2 |
| hmm_std_diag_k3 | 0.9881 / 0.9783 / 0.9879 | 84.0 / 46.1 / 83.0 | S3 |
| hmm_std_full_k3 | 0.9910 / 0.9789 / 0.9843 | 110.9 / 47.3 / 63.7 | S2 S3 |
| hmm_std_diag_k4 | 0.9896 / 0.9883 / 0.9778 / 0.9870 | 96.3 / 85.5 / 45.0 / 77.0 | S3 S4 |
| hmm_std_full_k4 | 0.9925 / 0.9769 / 0.9883 / 0.9952 | 134.1 / 43.2 / 85.2 / 209.6 | S2 S3 S4 |
| hmm_sticky_diag_k2_w0.10 | 0.9948 / 0.9912 | 193.7 / 114.1 | S2 |
| hmm_sticky_diag_k2_w0.30 | 0.9956 / 0.9931 | 224.8 / 144.1 | S2 |
| hmm_sticky_diag_k2_w1.00 | 0.9970 / 0.9960 | 330.9 / 248.5 | S2 |
| hmm_sticky_diag_k3_w0.10 | 0.9890 / 0.9803 / 0.9894 | 91.0 / 50.7 / 94.5 | S3 |
| hmm_sticky_diag_k3_w0.30 | 0.9905 / 0.9833 / 0.9915 | 104.8 / 60.0 / 117.5 | S3 |
| hmm_sticky_diag_k3_w1.00 | 0.9935 / 0.9892 / 0.9949 | 153.2 / 92.2 / 197.8 | S3 |
| hmm_sticky_diag_k4_w0.10 | 0.9906 / 0.9896 / 0.9796 / 0.9884 | 106.7 / 96.5 / 49.1 / 85.8 | S3 S4 |
| hmm_sticky_diag_k4_w0.30 | 0.9922 / 0.9916 / 0.9826 / 0.9903 | 127.7 / 118.4 / 57.3 / 103.5 | S3 S4 |
| hmm_sticky_diag_k4_w1.00 | 0.9950 / 0.9948 / 0.9883 / 0.9939 | 201.6 / 193.0 / 85.7 / 164.6 | S3 S4 |

Full matrices: `data/processed/hmm/hmm_transition_matrices.csv`; figure `hmm_transition_matrices.png`.

## 10. State durations and switching

| model_id | train_switches_per_year | validation_switches_per_year | train_median_duration_days | validation_median_duration_days | validation_hindsight_revision_share |
|---|---|---|---|---|---|
| hmm_std_diag_k2 | 2.20 | 2.76 | 109.00 | 51.50 | 0.03 |
| hmm_std_full_k2 | 1.71 | 6.26 | 109.00 | 21.00 | 0.07 |
| hmm_std_diag_k3 | 4.78 | 6.76 | 46.50 | 18.00 | 0.05 |
| hmm_std_full_k3 | 3.43 | 7.01 | 60.00 | 20.00 | 0.08 |
| hmm_std_diag_k4 | 4.29 | 6.26 | 45.50 | 24.00 | 0.07 |
| hmm_std_full_k4 | 3.18 | 6.76 | 74.00 | 23.00 | 0.11 |
| hmm_sticky_diag_k2_w0.10 | 2.20 | 3.26 | 109.00 | 39.50 | 0.03 |
| hmm_sticky_diag_k2_w0.30 | 2.20 | 3.26 | 109.00 | 40.00 | 0.03 |
| hmm_sticky_diag_k2_w1.00 | 2.20 | 2.76 | 110.00 | 51.50 | 0.03 |
| hmm_sticky_diag_k3_w0.10 | 4.53 | 6.76 | 49.00 | 18.50 | 0.05 |
| hmm_sticky_diag_k3_w0.30 | 4.29 | 6.26 | 49.50 | 26.50 | 0.05 |
| hmm_sticky_diag_k3_w1.00 | 4.04 | 6.26 | 52.00 | 26.50 | 0.05 |
| hmm_sticky_diag_k4_w0.10 | 4.04 | 6.26 | 50.00 | 24.00 | 0.07 |
| hmm_sticky_diag_k4_w0.30 | 4.04 | 6.26 | 50.00 | 24.00 | 0.07 |
| hmm_sticky_diag_k4_w1.00 | 4.04 | 6.76 | 50.00 | 20.50 | 0.07 |

Rare-state warnings (< 5% of a split): none.

## 11. Filtered probability behaviour

| model_id | train_mean_max_filtered_prob | validation_mean_max_filtered_prob | train_mean_normalized_entropy | validation_mean_normalized_entropy | validation_mean_abs_daily_change_p_stress | validation_p_stress_crossings_per_year | validation_share_uncertain_days | validation_mean_uncertain_spell_days | validation_median_emission_log_lr | mean_transition_log_odds |
|---|---|---|---|---|---|---|---|---|---|---|
| hmm_std_diag_k2 | 0.993 | 0.990 | 0.023 | 0.035 | 0.012 | 2.755 | 0.021 | 2.100 | 6.629 | 4.880 |
| hmm_std_full_k2 | 0.992 | 0.970 | 0.032 | 0.105 | 0.025 | 6.262 | 0.066 | 2.750 | 2.890 | 4.969 |
| hmm_std_diag_k3 | 0.981 | 0.977 | 0.043 | 0.053 | 0.009 | 2.004 | 0.010 | 1.667 | 4.328 | 4.211 |
| hmm_std_full_k3 | 0.990 | 0.974 | 0.026 | 0.063 | 0.023 | 5.260 | 0.036 | 1.714 | 2.971 | 4.225 |
| hmm_std_diag_k4 | 0.987 | 0.978 | 0.025 | 0.041 | 0.015 | 3.256 | 0.023 | 1.769 | 4.313 | 4.277 |
| hmm_std_full_k4 | 0.990 | 0.975 | 0.020 | 0.050 | 0.021 | 4.258 | 0.027 | 1.688 | 3.079 | 4.602 |
| hmm_sticky_diag_k2_w0.10 | 0.994 | 0.990 | 0.023 | 0.035 | 0.012 | 3.256 | 0.021 | 2.100 | 6.627 | 4.995 |
| hmm_sticky_diag_k2_w0.30 | 0.994 | 0.990 | 0.022 | 0.033 | 0.012 | 3.256 | 0.020 | 1.818 | 6.626 | 5.187 |
| hmm_sticky_diag_k2_w1.00 | 0.994 | 0.992 | 0.020 | 0.030 | 0.011 | 2.755 | 0.019 | 1.727 | 6.625 | 5.655 |
| hmm_sticky_diag_k3_w0.10 | 0.981 | 0.977 | 0.042 | 0.052 | 0.009 | 2.004 | 0.009 | 1.286 | 4.327 | 4.315 |
| hmm_sticky_diag_k3_w0.30 | 0.982 | 0.978 | 0.041 | 0.050 | 0.009 | 2.004 | 0.009 | 1.125 | 4.327 | 4.493 |
| hmm_sticky_diag_k3_w1.00 | 0.984 | 0.980 | 0.037 | 0.046 | 0.008 | 2.004 | 0.009 | 1.286 | 4.325 | 4.940 |
| hmm_sticky_diag_k4_w0.10 | 0.987 | 0.978 | 0.025 | 0.041 | 0.015 | 3.256 | 0.022 | 1.692 | 4.316 | 4.384 |
| hmm_sticky_diag_k4_w0.30 | 0.988 | 0.978 | 0.024 | 0.039 | 0.014 | 3.256 | 0.021 | 1.615 | 4.321 | 4.567 |
| hmm_sticky_diag_k4_w1.00 | 0.988 | 0.979 | 0.022 | 0.036 | 0.014 | 3.757 | 0.021 | 1.500 | 4.320 | 5.024 |

validation_median_emission_log_lr = median daily log-density gap between the two most likely states; mean_transition_log_odds = mean of log(A_ii / (1 - A_ii)). When the first is much larger than the second, the observation of the day decides the filtered state and a persistence prior has little room to act.

## 12. Economic interpretation (descriptive)

Selected model hmm_std_diag_k3 - mean factor values by filtered state:

|  | share | equity_trend | interest_rate | inflation | volatility | liquidity_credit | is_stress_state | hypothesis_label | train_validation_profile_distance |
|---|---|---|---|---|---|---|---|---|---|
| train S1 | 0.41 | 0.39 | -0.05 | 0.15 | -0.71 | -0.90 | False | Growth / Risk-on (hypothesis) | 0.78 |
| validation S1 | 0.45 | 0.48 | 0.32 | 0.81 | -0.72 | -1.06 | False | Growth / Risk-on (hypothesis) | 0.78 |
| train S2 | 0.34 | -0.36 | 0.43 | -0.33 | 0.10 | -0.04 | False | Neutral | 0.53 |
| validation S2 | 0.34 | -0.21 | -0.01 | -0.09 | 0.18 | 0.02 | False | Neutral | 0.53 |
| train S3 | 0.25 | -1.64 | -0.05 | -1.17 | 1.89 | 1.58 | True | Liquidity Crisis (hypothesis) | 1.32 |
| validation S3 | 0.21 | -1.00 | 0.27 | -0.28 | 1.96 | 0.92 | True | Liquidity Crisis (hypothesis) | 1.32 |

Annualized mean ETF log return by state (%). same_day pairs state_t with return_t and is NOT tradable (the state is known only at the close of t); next_session pairs state_t with return_t+1. Both are descriptive and in-sample conditional on Train parameters:

|  | DBC | GLD | HYG | IEF | SPY | TIP |
|---|---|---|---|---|---|---|
| train / next_session / S1 | 3.3 | 0.8 | 9.4 | 4.8 | 13.2 | 8.0 |
| train / next_session / S2 | -5.5 | -1.9 | 7.4 | 1.3 | 15.1 | -0.2 |
| train / next_session / S3 | -35.3 | 14.7 | -8.0 | 11.9 | -19.8 | 1.7 |
| train / same_day / S1 | 3.6 | -3.6 | 11.0 | 3.3 | 17.6 | 6.4 |
| train / same_day / S2 | -1.5 | 1.9 | 7.9 | 1.2 | 14.6 | 1.0 |
| train / same_day / S3 | -42.3 | 16.2 | -11.4 | 14.6 | -26.7 | 2.5 |
| validation / next_session / S1 | 12.2 | 5.3 | 5.8 | -1.5 | 12.6 | 1.7 |
| validation / next_session / S2 | -4.9 | 5.1 | 8.5 | 8.1 | 13.9 | 6.2 |
| validation / next_session / S3 | 7.2 | 19.7 | 9.9 | 4.1 | 16.1 | 2.6 |
| validation / same_day / S1 | 16.1 | 2.0 | 7.6 | -3.8 | 19.8 | 0.3 |
| validation / same_day / S2 | -8.3 | 12.8 | 6.7 | 10.9 | 7.9 | 7.7 |
| validation / same_day / S3 | 3.6 | 16.0 | 8.3 | 5.1 | 8.5 | 3.5 |

K=3 check for a rates/inflation state (non-stress state with the largest rates or inflation mean, Train): hmm_std_diag_k3: S2 rates +0.43, inflation -0.33, share 34%; hmm_std_full_k3: S1 rates -0.09, inflation +0.17, share 43%; hmm_sticky_diag_k3_w0.10: S2 rates +0.43, inflation -0.33, share 34%; hmm_sticky_diag_k3_w0.30: S2 rates +0.43, inflation -0.33, share 34%; hmm_sticky_diag_k3_w1.00: S2 rates +0.43, inflation -0.33, share 35%. Stage 4 criterion: >= +0.5.

## 13. Validation generalization

See the dedicated section 'Train vs Validation Generalization' below.

## 14. Comparison with K-Means / GMM

| model_id | split | hmm_avg_log_score | gmm_avg_log_score | mean_daily_difference | nw_se | nw_t | share_days_hmm_better | ari_hmm_vs_gmm_states | corr_p_stress | hmm_switches_per_year | gmm_switches_per_year | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hmm_std_diag_k3 | train | -4.740 | -5.857 | 1.117 | 0.080 | 13.955 | 0.872 | 0.423 | 0.825 | 4.776 | 4.653 | in-sample for both models |
| hmm_std_diag_k3 | validation | -5.885 | -6.655 | 0.771 | 0.128 | 6.015 | 0.754 | 0.409 | 0.871 | 6.763 | 4.759 | out-of-sample, fixed Train parameters |
| hmm_std_diag_k4 | train | -4.317 | -5.857 | 1.540 | 0.091 | 16.940 | 0.905 | 0.275 | 0.703 | 4.286 | 4.653 | in-sample for both models |
| hmm_std_diag_k4 | validation | -5.803 | -6.655 | 0.852 | 0.140 | 6.093 | 0.756 | 0.287 | 0.681 | 6.262 | 4.759 | out-of-sample, fixed Train parameters |

The daily difference compares one-step-ahead predictive densities of the same standardized observation: the GMM density ignores the past, the HMM density conditions on x_1..x_t-1 through the filtered state. This is a valid proper-scoring-rule comparison of two forecasting models; label agreement (ARI) is descriptive and is not forecasting evidence. Stage 4 K-Means has no density and is compared through the Stage 4 report only.

## 15. Stability versus detection delay

Reference onsets (rules fixed in advance): factor stress score above its Train 80% quantile (3.12), gaps < 5 days closed, episodes >= 10 days, episodes < 21 days apart merged; and Stage 4 GMM stress-state runs with the same rules. Lag = trading days from onset until P(stress) >= threshold; if the model was already above the threshold on the previous day the lag is negative (minus the length of its ongoing stress run, capped at -63): the model led the reference. Undetected = not reached within 63 days. These are descriptive references, not ground-truth regimes, and no model was tuned on them.

Onsets: factor_stress_score train: 7, factor_stress_score validation: 3, gmm_static_stress train: 6, gmm_static_stress validation: 5.

| model_id | reference | n_onsets | already_in_stress_share_p50 | median_lag_p50 | mean_lag_p50 | undetected_share_p50 | median_lag_p80 | missed_share_in_episodes | stress_share_outside_episodes |
|---|---|---|---|---|---|---|---|---|---|
| hmm_std_diag_k2 | factor_stress_score | 10 | 1.00 | -6.50 | -14.40 | 0.00 | -6.00 | 0.00 | 0.19 |
| hmm_std_diag_k2 | gmm_static_stress | 11 | 0.00 | 1.00 | 0.82 | 0.00 | 1.00 | 0.01 | 0.05 |
| hmm_std_diag_k3 | factor_stress_score | 10 | 0.50 | -1.00 | -7.80 | 0.00 | -0.50 | 0.01 | 0.06 |
| hmm_std_diag_k3 | gmm_static_stress | 11 | 0.00 | 8.00 | 12.55 | 0.00 | 9.00 | 0.26 | 0.00 |
| hmm_std_diag_k4 | factor_stress_score | 10 | 1.00 | -31.50 | -30.70 | 0.00 | -22.00 | 0.00 | 0.41 |
| hmm_std_diag_k4 | gmm_static_stress | 11 | 0.73 | -25.00 | -24.18 | 0.00 | -10.00 | 0.00 | 0.31 |
| hmm_std_full_k2 | factor_stress_score | 10 | 1.00 | -8.50 | -18.40 | 0.00 | -8.00 | 0.00 | 0.33 |
| hmm_std_full_k2 | gmm_static_stress | 11 | 0.36 | 0.00 | -6.09 | 0.00 | 0.00 | 0.01 | 0.21 |
| hmm_std_full_k3 | factor_stress_score | 10 | 1.00 | -21.00 | -26.80 | 0.00 | -21.00 | 0.00 | 0.45 |
| hmm_std_full_k3 | gmm_static_stress | 11 | 0.73 | -10.00 | -20.27 | 0.00 | -9.00 | 0.00 | 0.35 |
| hmm_std_full_k4 | factor_stress_score | 10 | 1.00 | -21.00 | -26.60 | 0.00 | -20.00 | 0.00 | 0.44 |
| hmm_std_full_k4 | gmm_static_stress | 11 | 0.73 | -10.00 | -20.09 | 0.00 | -9.00 | 0.00 | 0.33 |
| hmm_sticky_diag_k2_w0.10 | factor_stress_score | 10 | 1.00 | -6.50 | -14.40 | 0.00 | -6.00 | 0.00 | 0.19 |
| hmm_sticky_diag_k2_w0.10 | gmm_static_stress | 11 | 0.00 | 1.00 | 0.82 | 0.00 | 1.00 | 0.01 | 0.05 |
| hmm_sticky_diag_k2_w0.30 | factor_stress_score | 10 | 1.00 | -6.50 | -14.40 | 0.00 | -6.00 | 0.00 | 0.19 |
| hmm_sticky_diag_k2_w0.30 | gmm_static_stress | 11 | 0.00 | 1.00 | 0.82 | 0.00 | 1.00 | 0.01 | 0.05 |
| hmm_sticky_diag_k2_w1.00 | factor_stress_score | 10 | 1.00 | -6.00 | -14.30 | 0.00 | -6.00 | 0.00 | 0.19 |
| hmm_sticky_diag_k2_w1.00 | gmm_static_stress | 11 | 0.00 | 1.00 | 0.91 | 0.00 | 1.00 | 0.01 | 0.05 |
| hmm_sticky_diag_k3_w0.10 | factor_stress_score | 10 | 0.50 | -0.50 | -7.70 | 0.00 | -0.50 | 0.01 | 0.06 |
| hmm_sticky_diag_k3_w0.10 | gmm_static_stress | 11 | 0.00 | 8.00 | 12.64 | 0.00 | 9.00 | 0.26 | 0.00 |
| hmm_sticky_diag_k3_w0.30 | factor_stress_score | 10 | 0.50 | -0.50 | -7.50 | 0.00 | -0.50 | 0.01 | 0.06 |
| hmm_sticky_diag_k3_w0.30 | gmm_static_stress | 11 | 0.00 | 9.00 | 12.73 | 0.00 | 9.00 | 0.26 | 0.00 |
| hmm_sticky_diag_k3_w1.00 | factor_stress_score | 10 | 0.50 | -0.50 | -7.50 | 0.00 | -0.50 | 0.01 | 0.06 |
| hmm_sticky_diag_k3_w1.00 | gmm_static_stress | 11 | 0.00 | 9.00 | 12.73 | 0.00 | 9.00 | 0.26 | 0.00 |
| hmm_sticky_diag_k4_w0.10 | factor_stress_score | 10 | 1.00 | -31.00 | -30.60 | 0.00 | -22.00 | 0.00 | 0.41 |
| hmm_sticky_diag_k4_w0.10 | gmm_static_stress | 11 | 0.73 | -24.00 | -24.09 | 0.00 | -10.00 | 0.00 | 0.30 |
| hmm_sticky_diag_k4_w0.30 | factor_stress_score | 10 | 1.00 | -31.00 | -30.60 | 0.00 | -22.00 | 0.00 | 0.41 |
| hmm_sticky_diag_k4_w0.30 | gmm_static_stress | 11 | 0.73 | -24.00 | -24.09 | 0.00 | -10.00 | 0.00 | 0.30 |
| hmm_sticky_diag_k4_w1.00 | factor_stress_score | 10 | 1.00 | -30.50 | -30.50 | 0.00 | -21.50 | 0.00 | 0.41 |
| hmm_sticky_diag_k4_w1.00 | gmm_static_stress | 11 | 0.73 | -24.00 | -23.64 | 0.00 | -10.00 | 0.00 | 0.30 |

already_in_stress_share = share of onsets at which the model was already in stress (negative lags). The same-sequence comparison below is the direct Standard-vs-Sticky delay test.

**Same-sequence comparison (Sticky vs its Standard counterpart).** Episodes are the Standard model's runs of P(stress) >= 0.5 (same event rules). Entry lag = days until the Sticky P(stress) reaches 0.5, searched from 10 days before the Standard onset (negative = Sticky earlier); exit lag = days until the Sticky P(stress) falls below 0.5 relative to the Standard exit (positive = Sticky stays in stress longer).

| model_id | standard_model | split | n_standard_episodes | missed_share | median_entry_lag | mean_entry_lag | max_entry_lag | median_exit_lag | mean_exit_lag |
|---|---|---|---|---|---|---|---|---|---|
| hmm_sticky_diag_k2_w0.10 | hmm_std_diag_k2 | train | 7 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| hmm_sticky_diag_k2_w0.10 | hmm_std_diag_k2 | validation | 4 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| hmm_sticky_diag_k2_w0.10 | hmm_std_diag_k2 | pooled | 10 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| hmm_sticky_diag_k2_w0.30 | hmm_std_diag_k2 | train | 7 | 0.00 | 0.00 | 0.14 | 1.00 | 0.00 | 0.17 |
| hmm_sticky_diag_k2_w0.30 | hmm_std_diag_k2 | validation | 4 | 0.00 | 0.00 | 0.00 | 0.00 | 0.50 | 0.50 |
| hmm_sticky_diag_k2_w0.30 | hmm_std_diag_k2 | pooled | 10 | 0.00 | 0.00 | 0.10 | 1.00 | 0.00 | 0.30 |
| hmm_sticky_diag_k2_w1.00 | hmm_std_diag_k2 | train | 7 | 0.00 | 0.00 | 0.29 | 1.00 | 0.00 | 0.33 |
| hmm_sticky_diag_k2_w1.00 | hmm_std_diag_k2 | validation | 4 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.25 |
| hmm_sticky_diag_k2_w1.00 | hmm_std_diag_k2 | pooled | 10 | 0.00 | 0.00 | 0.20 | 1.00 | 1.00 | 0.70 |
| hmm_sticky_diag_k3_w0.10 | hmm_std_diag_k3 | train | 6 | 0.00 | 0.00 | 0.17 | 1.00 | 0.00 | 0.00 |
| hmm_sticky_diag_k3_w0.10 | hmm_std_diag_k3 | validation | 4 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.25 |
| hmm_sticky_diag_k3_w0.10 | hmm_std_diag_k3 | pooled | 10 | 0.00 | 0.00 | 0.10 | 1.00 | 0.00 | 0.10 |
| hmm_sticky_diag_k3_w0.30 | hmm_std_diag_k3 | train | 6 | 0.00 | 0.00 | 0.33 | 1.00 | 0.00 | 0.17 |
| hmm_sticky_diag_k3_w0.30 | hmm_std_diag_k3 | validation | 4 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.25 |
| hmm_sticky_diag_k3_w0.30 | hmm_std_diag_k3 | pooled | 10 | 0.00 | 0.00 | 0.20 | 1.00 | 0.00 | 0.20 |
| hmm_sticky_diag_k3_w1.00 | hmm_std_diag_k3 | train | 6 | 0.00 | 0.00 | 0.33 | 1.00 | 0.00 | 0.17 |
| hmm_sticky_diag_k3_w1.00 | hmm_std_diag_k3 | validation | 4 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.25 |
| hmm_sticky_diag_k3_w1.00 | hmm_std_diag_k3 | pooled | 10 | 0.00 | 0.00 | 0.20 | 1.00 | 0.00 | 0.20 |
| hmm_sticky_diag_k4_w0.10 | hmm_std_diag_k4 | train | 9 | 0.00 | 0.00 | 0.33 | 2.00 | 0.00 | 0.00 |
| hmm_sticky_diag_k4_w0.10 | hmm_std_diag_k4 | validation | 4 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| hmm_sticky_diag_k4_w0.10 | hmm_std_diag_k4 | pooled | 12 | 0.00 | 0.00 | 0.25 | 2.00 | 0.00 | 0.00 |
| hmm_sticky_diag_k4_w0.30 | hmm_std_diag_k4 | train | 9 | 0.00 | 0.00 | 0.33 | 2.00 | 0.00 | 0.00 |
| hmm_sticky_diag_k4_w0.30 | hmm_std_diag_k4 | validation | 4 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| hmm_sticky_diag_k4_w0.30 | hmm_std_diag_k4 | pooled | 12 | 0.00 | 0.00 | 0.25 | 2.00 | 0.00 | 0.00 |
| hmm_sticky_diag_k4_w1.00 | hmm_std_diag_k4 | train | 9 | 0.00 | 0.00 | 0.56 | 2.00 | 0.00 | 0.12 |
| hmm_sticky_diag_k4_w1.00 | hmm_std_diag_k4 | validation | 4 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.25 |
| hmm_sticky_diag_k4_w1.00 | hmm_std_diag_k4 | pooled | 12 | 0.00 | 0.00 | 0.42 | 2.00 | 0.00 | 0.17 |

## 16. Limitations

- Some historical NFCI values are unavailable before the 2011 real-time vintages; liquidity_credit starts 2007-10-30 and its composition changes inside Train. Daily FRED series use conservative availability lags; macro release timestamps are not fully verified (Stage 3 PIT categories).
- Factors are correlated (volatility, liquidity_credit, equity_trend), so states are driven mainly by a common stress dimension; rolling robust z-scores measure stress relative to the trailing window, not permanent economic levels.
- Gaussian emissions understate fat tails; geometric durations are a strong assumption; parameters are time-homogeneous over 2007-2015, which includes the GFC.
- Validation (2016-2019) contains no crisis comparable to 2008, so stress-state calibration and detection delay are judged largely on Train, in-sample.
- Late-Train composition sensitivity: last active-component change inside Train on 2011-11-29.

| model_id | window | start | end | fit_rows | k | converged | validation_avg_loglik | validation_shares | validation_switches_per_year | validation_ari_vs_full | max_emission_mean_shift_original_units | self_transitions | validation_mean_max_prob | rank_in_window |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hmm_std_diag_k2 | post_composition | 2011-11-29 | 2015-12-31 | 1029 | 2 | True | -6.353 | S1 69% / S2 31% | 3.256 | 0.960 | 0.534 | 0.9944 / 0.9873 | 0.988 | 1 |
| hmm_std_diag_k3 | post_composition | 2011-11-29 | 2015-12-31 | 1029 | 3 | True | -6.556 | S1 59% / S2 18% / S3 23% | 5.260 | 0.575 | 0.568 | 0.9942 / 0.9848 / 0.9830 | 0.980 | 2 |
| hmm_std_diag_k4 | post_composition | 2011-11-29 | 2015-12-31 | 1029 | 4 | True | -6.822 | S1 33% / S2 24% / S3 19% / S4 24% | 9.018 | 0.576 | 0.622 | 0.9926 / 0.9839 / 0.9842 / 0.9830 | 0.974 | 3 |

## 17. Final selected HMM configuration

- Model: **hmm_std_diag_k3**; seed 20240540; kappa 0.00; converged True after 22 iterations.
- Transition matrix (Train): S1: 0.9881, 0.0119, 0.0000; S2: 0.0144, 0.9783, 0.0073; S3: 0.0000, 0.0121, 0.9879
- Stress states: S3.
- Decision filter: emission covariance inflation c = 1 (see 'Probability calibration').
- Parameters, preprocessing, seeds and input hashes: `data/processed/hmm/models/hmm_std_diag_k3.json`; selection evidence: `hmm_selection_manifest.json`.

## 18. Readiness for the Early Warning stage

- Available for later stages: Train-estimated parameters, the causal filter, and Validation predicted / filtered probabilities. Later stages must keep the same filter and must not use smoothed or Viterbi output.
- Better regime classification does not by itself imply better portfolio returns; that requires the later out-of-sample evaluation with t+1 execution.

## Train vs Validation Generalization

Same Train parameters and scaler for both partitions; Validation continues the forward recursion from the last Train filtered distribution; both averages are one-step-ahead predictive log densities per day (the Train value is in-sample).

| model_id | family | k | covariance_type | prior_weight | train_avg_loglik | validation_avg_loglik | train_validation_gap | train_shares | validation_shares | train_switches_per_year | validation_switches_per_year | train_median_duration_days | validation_median_duration_days | train_mean_normalized_entropy | validation_mean_normalized_entropy | train_mean_max_filtered_prob | validation_mean_max_filtered_prob | converged | converged_share | initialization_stability_mean_ari | best_basin_share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hmm_std_diag_k2 | standard | 2 | diag | 0.000 | -5.282 | -6.159 | 0.877 | S1 64% / S2 36% | S1 69% / S2 31% | 2.204 | 2.755 | 109.000 | 51.500 | 0.023 | 0.035 | 0.993 | 0.990 | True | 1.000 | 1.000 | 1.000 |
| hmm_std_full_k2 | standard | 2 | full | 0.000 | -4.862 | -6.501 | 1.640 | S1 55% / S2 45% | S1 55% / S2 45% | 1.714 | 6.262 | 109.000 | 21.000 | 0.032 | 0.105 | 0.992 | 0.970 | True | 1.000 | 1.000 | 1.000 |
| hmm_std_diag_k3 | standard | 3 | diag | 0.000 | -4.740 | -5.885 | 1.144 | S1 41% / S2 34% / S3 25% | S1 45% / S2 34% / S3 21% | 4.776 | 6.763 | 46.500 | 18.000 | 0.043 | 0.053 | 0.981 | 0.977 | True | 1.000 | 0.999 | 1.000 |
| hmm_std_full_k3 | standard | 3 | full | 0.000 | -4.289 | -6.562 | 2.272 | S1 43% / S2 32% / S3 25% | S1 48% / S2 26% / S3 26% | 3.429 | 7.014 | 60.000 | 20.000 | 0.026 | 0.063 | 0.990 | 0.974 | True | 1.000 | 0.727 | 0.200 |
| hmm_std_diag_k4 | standard | 4 | diag | 0.000 | -4.317 | -5.803 | 1.486 | S1 23% / S2 23% / S3 31% / S4 24% | S1 13% / S2 38% / S3 31% / S4 19% | 4.286 | 6.262 | 45.500 | 24.000 | 0.025 | 0.041 | 0.987 | 0.978 | True | 1.000 | 0.868 | 0.900 |
| hmm_std_full_k4 | standard | 4 | full | 0.000 | -3.912 | -7.459 | 3.547 | S1 45% / S2 24% / S3 21% / S4 10% | S1 48% / S2 21% / S3 24% / S4 7% | 3.184 | 6.763 | 74.000 | 23.000 | 0.020 | 0.050 | 0.990 | 0.975 | True | 1.000 | 0.734 | 0.100 |
| hmm_sticky_diag_k2_w0.10 | sticky | 2 | diag | 0.100 | -5.282 | -6.160 | 0.878 | S1 64% / S2 36% | S1 69% / S2 31% | 2.204 | 3.256 | 109.000 | 39.500 | 0.023 | 0.035 | 0.994 | 0.990 | True | 1.000 | 1.000 | 1.000 |
| hmm_sticky_diag_k2_w0.30 | sticky | 2 | diag | 0.300 | -5.282 | -6.161 | 0.878 | S1 64% / S2 36% | S1 69% / S2 31% | 2.204 | 3.256 | 109.000 | 40.000 | 0.022 | 0.033 | 0.994 | 0.990 | True | 1.000 | 1.000 | 1.000 |
| hmm_sticky_diag_k2_w1.00 | sticky | 2 | diag | 1.000 | -5.284 | -6.163 | 0.880 | S1 64% / S2 36% | S1 68% / S2 32% | 2.204 | 2.755 | 110.000 | 51.500 | 0.020 | 0.030 | 0.994 | 0.992 | True | 1.000 | 1.000 | 1.000 |
| hmm_sticky_diag_k3_w0.10 | sticky | 3 | diag | 0.100 | -4.741 | -5.886 | 1.145 | S1 41% / S2 34% / S3 25% | S1 45% / S2 34% / S3 21% | 4.531 | 6.763 | 49.000 | 18.500 | 0.042 | 0.052 | 0.981 | 0.977 | True | 1.000 | 1.000 | 1.000 |
| hmm_sticky_diag_k3_w0.30 | sticky | 3 | diag | 0.300 | -4.741 | -5.888 | 1.147 | S1 41% / S2 34% / S3 25% | S1 45% / S2 34% / S3 21% | 4.286 | 6.262 | 49.500 | 26.500 | 0.041 | 0.050 | 0.982 | 0.978 | True | 1.000 | 1.000 | 1.000 |
| hmm_sticky_diag_k3_w1.00 | sticky | 3 | diag | 1.000 | -4.743 | -5.893 | 1.150 | S1 41% / S2 35% / S3 25% | S1 45% / S2 34% / S3 21% | 4.041 | 6.262 | 52.000 | 26.500 | 0.037 | 0.046 | 0.984 | 0.980 | True | 1.000 | 1.000 | 1.000 |
| hmm_sticky_diag_k4_w0.10 | sticky | 4 | diag | 0.100 | -4.317 | -5.803 | 1.486 | S1 23% / S2 23% / S3 30% / S4 24% | S1 13% / S2 38% / S3 30% / S4 19% | 4.041 | 6.262 | 50.000 | 24.000 | 0.025 | 0.041 | 0.987 | 0.978 | True | 1.000 | 0.868 | 0.900 |
| hmm_sticky_diag_k4_w0.30 | sticky | 4 | diag | 0.300 | -4.317 | -5.804 | 1.486 | S1 23% / S2 23% / S3 30% / S4 24% | S1 13% / S2 38% / S3 30% / S4 19% | 4.041 | 6.262 | 50.000 | 24.000 | 0.024 | 0.039 | 0.988 | 0.978 | True | 1.000 | 0.868 | 0.900 |
| hmm_sticky_diag_k4_w1.00 | sticky | 4 | diag | 1.000 | -4.320 | -5.806 | 1.486 | S1 23% / S2 23% / S3 30% / S4 24% | S1 13% / S2 38% / S3 30% / S4 19% | 4.041 | 6.763 | 50.000 | 20.500 | 0.022 | 0.036 | 0.988 | 0.979 | True | 1.000 | 0.868 | 0.900 |

- Standard diag: Train log score rises with K (K2 -5.282, K3 -4.740, K4 -4.317); Validation is highest at K=4 (K2 -6.159, K3 -5.885, K4 -5.803).
- Standard full: Train log score rises with K (K2 -4.862, K3 -4.289, K4 -3.912); Validation is highest at K=2 (K2 -6.501, K3 -6.562, K4 -7.459).
- K=2 full vs diag: Train +0.420/day, Validation -0.342/day -> full covariance fits Train better but generalizes worse (overfitting signal).
- K=3 full vs diag: Train +0.451/day, Validation -0.677/day -> full covariance fits Train better but generalizes worse (overfitting signal).
- K=4 full vs diag: Train +0.405/day, Validation -1.656/day -> full covariance fits Train better but generalizes worse (overfitting signal).
- Largest Train-Validation gap: hmm_std_full_k4 (3.547 per day); smallest: hmm_std_diag_k2 (0.877).
- K=2 Sticky vs Standard diag (Validation): w=0.10: val -0.000/day, switches 3.3 vs 2.8/yr; w=0.30: val -0.001/day, switches 3.3 vs 2.8/yr; w=1.00: val -0.004/day, switches 2.8 vs 2.8/yr.
- K=3 Sticky vs Standard diag (Validation): w=0.10: val -0.001/day, switches 6.8 vs 6.8/yr; w=0.30: val -0.003/day, switches 6.3 vs 6.8/yr; w=1.00: val -0.009/day, switches 6.3 vs 6.8/yr.
- K=4 Sticky vs Standard diag (Validation): w=0.10: val -0.000/day, switches 6.3 vs 6.3/yr; w=0.30: val -0.000/day, switches 6.3 vs 6.3/yr; w=1.00: val -0.003/day, switches 6.8 vs 6.3/yr.
- Largest Train-to-Validation factor-profile shift of any state: hmm_std_diag_k2 1.09, hmm_std_diag_k3 1.32, hmm_std_diag_k4 1.39, hmm_std_full_k2 1.34, hmm_std_full_k3 1.80, hmm_std_full_k4 1.87, hmm_sticky_diag_k2_w0.10 1.09, hmm_sticky_diag_k2_w0.30 1.10, hmm_sticky_diag_k2_w1.00 1.10, hmm_sticky_diag_k3_w0.10 1.33, hmm_sticky_diag_k3_w0.30 1.33, hmm_sticky_diag_k3_w1.00 1.33, hmm_sticky_diag_k4_w0.10 1.39, hmm_sticky_diag_k4_w0.30 1.38, hmm_sticky_diag_k4_w1.00 1.38 (Euclidean, factor units; > 1.0 flagged in Stage 4).
- A worse Validation score is not by itself rejection: 2016-2019 had lower and less varied stress than 2008-2015, so all models are expected to score differently out of sample. Overfitting is indicated when Train improves with complexity while Validation deteriorates.

## Probability calibration

Question: are filtered probabilities near 0/1 a property of well separated regimes, or a symptom of misspecification? All tests use Validation with Train parameters fixed; nothing is smoothed.

- PIT: u = predictive CDF of each factor under the one-step-ahead mixture. Calibrated -> uniform, 90% interval coverage 0.90, no autocorrelation.
- Reliability (martingale property): under a correct model E[filtered P(stress)_t | x_1..x_t-1] = predicted P(stress)_t; the realized switching rate should match the model-implied rate sum_i P(Z_t=i)(1-A_ii).
- Residual autocorrelation: lag-1 autocorrelation of standardized residuals on consecutive same-state days. The HMM assumes zero (x_t independent given Z_t).
- Recalibration test: emission covariances x c inside the filter, c in [1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0], chosen on 2016-2017, judged on 2018-2019; adopted only if the out-of-sample gain has NW t >= 2.0.

PIT summary (all five factors pooled):

| model_id | n | mean | std | ks_statistic | coverage_50 | coverage_90 | share_below_0.01 | share_above_0.99 | lag1_autocorrelation |
|---|---|---|---|---|---|---|---|---|---|
| hmm_std_diag_k3 | 5030 | 0.551 | 0.310 | 0.088 | 0.443 | 0.815 | 0.025 | 0.030 | 0.962 |
| hmm_std_diag_k2 | 5030 | 0.542 | 0.312 | 0.104 | 0.437 | 0.836 | 0.016 | 0.027 | 0.973 |
| hmm_std_diag_k4 | 5030 | 0.548 | 0.318 | 0.100 | 0.430 | 0.795 | 0.030 | 0.034 | 0.963 |

Reliability and switching:

| model_id | mean_filtered_minus_predicted | nw_t | mean_abs_update | calibration_slope | implied_switches_per_year | realized_switches_per_year | realized_to_implied_switch_ratio |
|---|---|---|---|---|---|---|---|
| hmm_std_diag_k3 | -0.000 | -0.005 | 0.013 | 1.003 | 3.855 | 6.763 | 1.754 |
| hmm_std_diag_k2 | -0.002 | -0.591 | 0.018 | 1.004 | 1.766 | 2.755 | 1.560 |
| hmm_std_diag_k4 | -0.001 | -0.342 | 0.021 | 1.005 | 3.774 | 6.262 | 1.659 |

Within-state residual lag-1 autocorrelation:

| model_id | split | equity_trend | interest_rate | inflation | volatility | liquidity_credit | mean |
|---|---|---|---|---|---|---|---|
| hmm_std_diag_k3 | train | 0.98 | 0.96 | 0.99 | 0.91 | 0.97 | 0.96 |
| hmm_std_diag_k3 | validation | 0.98 | 0.98 | 0.99 | 0.94 | 0.98 | 0.97 |
| hmm_std_diag_k2 | train | 0.98 | 0.97 | 0.99 | 0.93 | 0.98 | 0.97 |
| hmm_std_diag_k2 | validation | 0.98 | 0.98 | 1.00 | 0.94 | 0.99 | 0.98 |
| hmm_std_diag_k4 | train | 0.98 | 0.95 | 0.98 | 0.91 | 0.97 | 0.96 |
| hmm_std_diag_k4 | validation | 0.98 | 0.98 | 0.99 | 0.93 | 0.98 | 0.97 |

Covariance-inflation test:

| model_id | chosen_inflation_on_val_a | val_b_mean_gain | val_b_nw_t | adopted | decision_inflation |
|---|---|---|---|---|---|
| hmm_std_diag_k3 | 1.500 | 0.167 | 1.032 | False | 1.000 |
| hmm_std_diag_k2 | 1.500 | 0.111 | 0.885 | False | 1.000 |
| hmm_std_diag_k4 | 2.000 | 0.124 | 0.551 | False | 1.000 |

**Reading for the selected model (computed):**

- Predictive intervals: 90% coverage 81% (nominal 90%); PIT std 0.310 (uniform 0.289); intervals too narrow -> overconfident predictive density.
- Temporal dependence: mean within-state residual autocorrelation 0.97 on Validation and mean PIT autocorrelation 0.96. The conditional-independence assumption is strongly violated: consecutive days are treated as independent evidence, which pushes filtered probabilities toward 0/1.
- Reliability: mean filtered minus predicted P(stress) -0.000 (NW t -0.00); realized / model-implied switches 1.75.
- Recalibration: c* = 1.5 on 2016-17; 2018-19 gain +0.167 per day (NW t 1.03); adopted: False. The Train estimate (c = 1) stays the decision filter; the confidence of the probabilities must be read with the diagnostics above.
- Structural remedy (not implemented here, it changes the model class): an autoregressive-emission HMM (Markov-switching VAR) models x_t given x_t-1 and the state, which removes most of the temporal dependence instead of rescaling it.

## Robustness: training windows, initialization, market phases

Training windows (fixed in advance; parameters estimated on the window only, the filter then runs to the end of Train and into Validation): full 2007-10-30..2015-12-31, start_plus_2y 2009-10-30..2015-12-31, end_minus_2y 2007-10-30..2013-12-31, post_composition 2011-11-29..2015-12-31.

| window | model_id | fit_rows | converged | validation_avg_loglik | rank_in_window | validation_ari_vs_full | max_emission_mean_shift_original_units | validation_switches_per_year | validation_shares |
|---|---|---|---|---|---|---|---|---|---|
| full | hmm_std_diag_k2 | 2058 | True | -6.159 | 3 | 1.000 | 0.000 | 2.755 | S1 69% / S2 31% |
| full | hmm_std_diag_k3 | 2058 | True | -5.885 | 2 | 1.000 | 0.000 | 6.763 | S1 45% / S2 34% / S3 21% |
| full | hmm_std_diag_k4 | 2058 | True | -5.803 | 1 | 1.000 | 0.000 | 6.262 | S1 13% / S2 38% / S3 31% / S4 19% |
| start_plus_2y | hmm_std_diag_k2 | 1553 | True | -6.448 | 3 | 0.960 | 0.298 | 3.256 | S1 69% / S2 31% |
| start_plus_2y | hmm_std_diag_k3 | 1553 | True | -5.910 | 1 | 0.706 | 0.499 | 5.511 | S1 41% / S2 34% / S3 25% |
| start_plus_2y | hmm_std_diag_k4 | 1553 | True | -6.015 | 2 | 0.628 | 1.378 | 5.010 | S1 42% / S2 30% / S3 28% / S4 0% |
| end_minus_2y | hmm_std_diag_k2 | 1554 | True | -6.300 | 2 | 0.952 | 0.243 | 2.254 | S1 68% / S2 32% |
| end_minus_2y | hmm_std_diag_k3 | 1554 | True | -5.956 | 1 | 0.614 | 0.403 | 6.012 | S1 35% / S2 39% / S3 26% |
| end_minus_2y | hmm_std_diag_k4 | 1554 | True | -6.712 | 3 | 0.321 | 1.634 | 7.264 | S1 33% / S2 29% / S3 37% / S4 1% |
| post_composition | hmm_std_diag_k2 | 1029 | True | -6.353 | 1 | 0.960 | 0.534 | 3.256 | S1 69% / S2 31% |
| post_composition | hmm_std_diag_k3 | 1029 | True | -6.556 | 2 | 0.575 | 0.568 | 5.260 | S1 59% / S2 18% / S3 23% |
| post_composition | hmm_std_diag_k4 | 1029 | True | -6.822 | 3 | 0.576 | 0.622 | 9.018 | S1 33% / S2 24% / S3 19% / S4 24% |

- Best Validation model per window: end_minus_2y: hmm_std_diag_k3, full: hmm_std_diag_k4, post_composition: hmm_std_diag_k2, start_plus_2y: hmm_std_diag_k3 (changes with the window).

Initialization (EM local optima vs estimator stability):

| model_id | n_init_per_seed | single_start_best_basin_share | best_basin_share | train_mean_pairwise_ari | validation_mean_pairwise_ari | best_basin_max_mean_distance |
|---|---|---|---|---|---|---|
| hmm_std_diag_k2 | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| hmm_std_full_k2 | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| hmm_std_diag_k3 | 5 | 0.720 | 1.000 | 0.999 | 1.000 | 0.000 |
| hmm_std_full_k3 | 5 | 0.040 | 0.200 | 0.727 | 0.678 | 0.000 |
| hmm_std_diag_k4 | 5 | 0.300 | 0.900 | 0.868 | 0.894 | 0.000 |
| hmm_std_full_k4 | 5 | 0.020 | 0.100 | 0.734 | 0.690 | 0.000 |
| hmm_sticky_diag_k2_w0.10 | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| hmm_sticky_diag_k2_w0.30 | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| hmm_sticky_diag_k2_w1.00 | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| hmm_sticky_diag_k3_w0.10 | 5 | 0.720 | 1.000 | 1.000 | 1.000 | 0.000 |
| hmm_sticky_diag_k3_w0.30 | 5 | 0.720 | 1.000 | 1.000 | 1.000 | 0.000 |
| hmm_sticky_diag_k3_w1.00 | 5 | 0.720 | 1.000 | 1.000 | 1.000 | 0.000 |
| hmm_sticky_diag_k4_w0.10 | 5 | 0.320 | 0.900 | 0.868 | 0.894 | 0.000 |
| hmm_sticky_diag_k4_w0.30 | 5 | 0.320 | 0.900 | 0.868 | 0.893 | 0.000 |
| hmm_sticky_diag_k4_w1.00 | 5 | 0.320 | 0.900 | 0.868 | 0.894 | 0.000 |

Mean daily log score by market phase - Validation (out-of-sample):

| phase | gmm_diag_k2 | hmm_std_diag_k2 | hmm_std_diag_k3 | hmm_std_diag_k4 |
|---|---|---|---|---|
| 2016 | -6.913 | -6.429 | -6.138 | -6.284 |
| 2017 | -5.639 | -5.229 | -5.065 | -5.009 |
| 2018 | -7.694 | -7.039 | -6.994 | -6.974 |
| 2019 | -6.376 | -5.940 | -5.343 | -4.947 |
| high stress (score > Train median) | -7.169 | -6.583 | -6.141 | -6.217 |
| low stress (score <= Train median) | -6.158 | -5.748 | -5.637 | -5.402 |

Train (in-sample, for context only):

| phase | gmm_diag_k2 | hmm_std_diag_k2 | hmm_std_diag_k3 | hmm_std_diag_k4 |
|---|---|---|---|---|
| 2007 | -9.548 | -8.524 | -8.266 | -8.254 |
| 2008 | -7.602 | -6.988 | -6.306 | -6.006 |
| 2009 | -5.776 | -5.215 | -4.179 | -3.256 |
| 2010 | -5.042 | -4.480 | -4.412 | -3.932 |
| 2011 | -5.816 | -5.174 | -4.610 | -4.257 |
| 2012 | -3.875 | -3.436 | -2.805 | -2.386 |
| 2013 | -4.995 | -4.584 | -4.000 | -3.678 |
| 2014 | -6.050 | -5.582 | -5.326 | -4.883 |
| 2015 | -7.050 | -6.223 | -5.663 | -5.444 |
| high stress (score > Train median) | -7.257 | -6.553 | -5.789 | -5.724 |
| low stress (score <= Train median) | -4.457 | -4.011 | -3.692 | -2.910 |

- Phase consistency (rule: larger K must win in >= 3 of 4 Validation years): hmm_std_diag_k3 vs hmm_std_diag_k2: 2016 +0.291, 2017 +0.164, 2018 +0.045, 2019 +0.596 -> positive in 4 of 4; consistent.
- Stage 4 GMM reference: rebuilt on Train with scikit-learn 1.9.1 (the pickle was written with 1.6.1); pickle not unpickled across versions.

## Checks

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
| Split files match split_manifest.json hashes | PASS | 4 files verified |
| Preprocessing identical to the Stage 4 Train scaler | PASS | train mean / train std (ddof=0); Stage 4 manifest matches |
| Stage 4 GMM reference rebuilt from Train (predictions only) | PASS | refitted on Train with scikit-learn 1.9.1 (Stage 4 spec: diag, K=2, seed 20240531, n_init 5); Stage 4 pickle written by scikit-learn 1.6.1, so it is not unpickled (avoids the version-mismatch warning) |
| Training-window sensitivity | PASS | windows full 2007-10-30..2015-12-31, start_plus_2y 2009-10-30..2015-12-31, end_minus_2y 2007-10-30..2013-12-31, post_composition 2011-11-29..2015-12-31; last active-component change inside Train on 2011-11-29 |
| Calibration: covariance inflation chosen on 2016-17, tested on 2018-19 | PASS | hmm_std_diag_k3: c* = 1.5, 2018-19 gain +0.167/day (NW t 1.03); adopted False -> decision inflation 1 |
| No test file read | PASS | split_manifest.json, train_features.csv, train_returns.csv, validation_features.csv, validation_returns.csv |
| Filtered and predicted probabilities sum to one | PASS | max |sum - 1| = 1.78e-15 |
| Transition-matrix rows sum to one | PASS | max |row sum - 1| = 2.22e-16 |
| Filter log scores reproduce the EM Train log-likelihood | PASS | max relative difference 2.26e-16 |
| Reference fits converged | PASS | all 15 converged |
| EM objective monotone for every seed | PASS | all seeds monotone |
| Validation filter = continuation of the Train filter | PASS | max abs difference 0.00e+00 |
| Changing later Validation data leaves earlier filtered probabilities unchanged | PASS | selected hmm_std_diag_k3: max change before cut 0.0e+00, after cut 1.00 |
| Parameters depend on Train only (refit reproduces the estimate) | PASS | max parameter difference 0.00e+00 |
| No library warnings during fitting and scoring | PASS | none |

## Figures

- `reports/figures/hmm/hmm_model_selection.png`
- `reports/figures/hmm/hmm_transition_matrices.png`
- `reports/figures/hmm/hmm_filtered_probabilities.png`
- `reports/figures/hmm/hmm_regime_timeline.png`
- `reports/figures/hmm/hmm_regime_factor_profiles.png`
- `reports/figures/hmm/hmm_state_duration_comparison.png`
- `reports/figures/hmm/hmm_stability_vs_detection_delay.png`
- `reports/figures/hmm/hmm_vs_gmm_comparison.png`
- `reports/figures/hmm/hmm_generalization_gap.png`
- `reports/figures/hmm/hmm_probability_calibration.png`
- `reports/figures/hmm/hmm_robustness.png`
