# Stage 6 - Dual-speed market early warning system

Train 2007-10-30..2015-12-31 (2058 sessions); Validation 2016-01-04..2019-12-31 (1006 sessions). Test not read. Checks: 27 PASS.

## 1. Executive summary

- The full EWS has the highest Validation PR-AUC; it is significantly better than 1 of 5 single-signal baselines (F_bocpd).
- Retained specification: **C P(non-normal)** - the full system is not demonstrably better than a simpler method, so the simpler specification is retained.
- State machine >= AMBER on Validation: 5/7 events warned, 1 late, 1 missed; median lead 3 sessions; 1.00 false episodes/year; 21.5% of sessions on alert.
- Events (SPY drawdown reaching -5%): Train 9, Validation 7. With this few Validation events, event-level differences between systems are not statistically meaningful on their own.
- No portfolio optimization, dynamic allocation or investment backtest was run (Stage 7 scope).

## 2. Status of the evidence

- Validation (2016-2019) was used to select the Stage 5 HMM and to decide Stage 5.5. Every Validation number here is **development-sample evidence**, not an independent out-of-sample test.
- Train-period HMM probabilities use parameters estimated on all of Train: they are in-sample reconstructions, not what a live system would have shown.
- To limit tuning on the reused Validation sample, all weights, quantiles, hazards and windows were fixed in advance and every threshold is a Train quantile. No parameter was changed after seeing Validation.
- The equity_trend factor contains SPY momentum, so the SPY outcomes are not fully independent of the signals; outcomes use FORWARD returns only.

## 3. Frozen inputs and integrity

| name | status | detail |
|---|---|---|
| Frozen Stage 5 model is the documented hmm_std_diag_k3 | PASS | Stage 5 manifest selected hmm_std_diag_k3; Stage 6 uses hmm_std_diag_k3.json (as Stage 5.5 did) |
| Stage 5.5 ar1 produced by the corrected exact M-step | PASS | variance_rule floor, model_version stage5.5-2026-10-21b |
| Stage 5.5 ar_hmm_k3 produced by the corrected exact M-step | PASS | variance_rule floor, model_version stage5.5-2026-10-21b |
| Stage 5.5 run had no FAILed checks and records the estimation amendment | PASS | 25 checks; FAIL: none |
| Stage 5.5 EM audit passed | PASS | 7 audit checks |
| Five factor columns present in the frozen order | PASS | train ['equity_trend', 'interest_rate', 'inflation', 'volatility', 'liquidity_credit']; validation ['equity_trend', 'interest_rate', 'inflation', 'volatility', 'liquidity_credit'] |
| train factors: chronological, unique dates, finite | PASS | 2058 rows 2007-10-30..2015-12-31 |
| validation factors: chronological, unique dates, finite | PASS | 1006 rows 2016-01-04..2019-12-31 |
| Validation starts after train ends (no overlap) | PASS | train ends 2015-12-31, validation starts 2016-01-04 |
| Row counts match split_manifest.json | PASS | train 2058 (manifest 2058), validation 1006 (manifest 1006) |
| train returns aligned with factors | PASS | 9 ETF columns |
| validation returns aligned with factors | PASS | 9 ETF columns |
| Factor definitions unchanged since stage 3 (factors.py hash) | PASS | manifest 28414cd58b03..., current 28414cd58b03... |
| Train scaler identical to the frozen HMM preprocessing | PASS | train mean / train std (ddof=0) |
| Train scaler identical to the frozen AR(1) preprocessing | PASS | train mean / train std (ddof=0) |
| Train scaler identical to the frozen AR-HMM preprocessing | PASS | train mean / train std (ddof=0) |
| Frozen HMM reproduces its saved Train log-likelihood | PASS | -9755.889562 vs -9755.889562 |
| Frozen ar1 reproduces its saved Train log-likelihood | PASS | 5356.921082 vs 5356.921082 |
| Frozen ar_hmm_k3 reproduces its saved Train log-likelihood | PASS | 6551.423017 vs 6551.423017 |
| Changing later Validation factors leaves every earlier signal, score and alert unchanged | PASS | max change before cut: signals 0.0e+00, EWS 0.0e+00; alerts identical True |
| Investment returns are not an input of signal construction | PASS | build_signals(hmm_params, ar1_params, arhmm_params, x_train, x_validation, train_index, val_index, levels_train, levels_validation) |
| Filtered HMM probabilities sum to one (filtered, not smoothed) | PASS | 1.8e-15 |
| EWS finite after the warm-up sessions | PASS | 5 warm-up NaN |
| No test file read | PASS | split_manifest.json, train_features.csv, train_returns.csv, validation_features.csv, validation_returns.csv |
| Baselines compared on the same labelled days (up to signal warm-up) | PASS | max difference in eligible days across systems: train 5, validation 0, validation 2016-2017 0, validation 2018-2019 0 |
| Stage 5 and Stage 5.5 artifacts unchanged (hashes before == after) | PASS | 71 files compared |
| No library warnings | PASS | none |

The Stage 5.5 models were verified to come from the corrected exact variance-floor M-step (variance_rule = floor, model_version stage5.5-2026-10-21b) and their saved Train log-likelihoods were recomputed from the saved parameters before any signal was built.

## 4. Architecture: two speeds

- **Slow layer** (persistent regimes): frozen HMM K=3 filtered probabilities, entropy, probability velocity.
- **Fast layer** (abrupt shocks): AR(1) one-step innovations (what the factors did that their own persistence did not predict) and BOCPD on those innovations.
- **Confirmation layer**: factor LEVELS of equity weakness, volatility and credit stress, plus rates/inflation pressure; a fast signal without level confirmation is weighted down.
- **Decision layer**: a fixed-weight Early Warning Score and a 5-level alert state machine with confirmation and hysteresis.

Timing: every date-t value uses factors through t and models estimated on Train. Investment returns are not an argument of the signal builder; they are used only for forward outcomes.

## 5. Slow layer: regime probabilities, entropy and velocity

- p_non_normal = P(S2) + P(S3); p_stress = P(S3); regime component = 0.5 p_non_normal + 0.5 p_stress.
- Normalized entropy = -sum p log p / log 3.
- Velocity = p_t - p_(t-5); only the positive part (deterioration) enters the EWS.

| split | mean | median | share_above_0.2 | share_above_0.5 |
|---|---|---|---|---|
| train | 0.043 | 0.001 | 0.072 | 0.028 |
| validation | 0.053 | 0.002 | 0.094 | 0.035 |

## 6. Fast layer: AR(1) innovations

- z_t,d = (x_t,d - c_d - phi_d x_t-1,d) / sigma_d from the frozen Stage 5.5 AR(1) (corrected EM).
- Adverse direction: equity -1, rates +1, inflation +1, volatility +1, credit +1.
- market_innovation = (-z_equity + z_volatility + z_credit) / k_market; macro_innovation = (z_rate + z_inflation) / k_macro; k = Train standard deviation, so the correlated market factors are counted once as a channel, not three times.
- 5-session versions are rolling sums rescaled to unit Train variance; ar_innovation_stress = max(market_innovation_5d, 0).

Train normalization constants: k_market 2.21, k_macro 1.473, k_market_5d 2.402, k_macro_5d 2.5, stress_index_mean 1.105e-16, stress_index_sd 2.731, surprise_train_n 2057.

## 7. BOCPD method

- Model: y_t | segment ~ N(mu, sigma2), (mu, sigma2) ~ Normal-Inverse-Gamma; predictive is Student-t.
- Prior: mu0 = Train mean, kappa0 = 1.0, alpha0 = 2.0, beta0 set so the prior predictive variance equals the Train variance.
- Constant hazard 1/lambda with lambda in [63, 126, 252] (primary 126).
- Exact recursion, no run-length truncation; verified against brute-force enumeration in tests.
- Under a constant hazard P(r_t = 0 | y_1..t) = h identically, so the reported evidence is P(r_t < 10 | y_1..t) (a change within the last 10 sessions), compared with its prior value 0.077.
- In the EWS the BOCPD component is gated: it counts only when the 5-session market innovation is adverse (> 0), because a structural change toward calm is not a warning.

Synthetic scenario study (20 seeds each; change at session 300; prior from the first 250 sessions; alarm = recent-change probability > 0.5). It shows why BOCPD runs on innovations: a persistent AR(1) level with no change produces many spurious change points, its innovations do not; slow drifts are detected late or not at all.

| scenario | has_change | runs | false_alarm_episodes_per_252 | detection_rate | median_delay_sessions | alarm_share_after_change |
|---|---|---|---|---|---|---|
| mean_shift | True | 20 | 0.462 | 0.950 | 5.000 | 0.017 |
| variance_increase | True | 20 | 0.462 | 1.000 | 2.500 | 0.028 |
| single_outlier | False | 20 | 1.176 | 0.000 |  |  |
| gradual_drift | True | 20 | 0.462 | 0.600 | 130.000 | 0.005 |
| consecutive_shocks | True | 20 | 0.462 | 1.000 | 3.500 | 0.051 |
| ar1_no_change_on_levels | False | 20 | 9.618 | 0.000 |  |  |
| ar1_no_change_on_innovations | False | 20 | 0.798 | 0.000 |  |  |

Priors (fitted on Train):

|  | mu0 | kappa0 | alpha0 | beta0 |
|---|---|---|---|---|
| B1_level | -0.0 | 1.0 | 2.0 | 0.4998 |
| B2_innovation | -0.0 | 1.0 | 2.0 | 0.4998 |
| B3_arhmm_surprise | -0.0 | 1.0 | 2.0 | 0.4968 |
| B2_volatility | -0.0 | 1.0 | 2.0 | 0.5 |
| B2_credit | 0.0 | 1.0 | 2.0 | 0.5 |

## 8. BOCPD input and hazard comparison

B1 = factor LEVEL stress index (persistent; expected to re-detect slow trends), B2 = AR(1) market innovation (primary), B3 = AR-HMM predictive surprise (probit of the Train ECDF), plus single-channel B2 runs. Each uses its own Train 80th percentile as alert threshold.

| input | expected_run_length | split | prior_p_change_within_10d | mean_posterior | pr_auc | warned | missed | false_alerts_per_year | time_share_alerted |
|---|---|---|---|---|---|---|---|---|---|
| B1_level | 63 | train | 0.148 | 0.124 | 0.241 | 8 | 0 | 1.469 | 0.200 |
| B1_level | 63 | validation | 0.148 | 0.115 | 0.095 | 6 | 0 | 2.755 | 0.207 |
| B1_level | 126 | train | 0.077 | 0.100 | 0.242 | 8 | 0 | 1.714 | 0.200 |
| B1_level | 126 | validation | 0.077 | 0.090 | 0.096 | 6 | 0 | 2.755 | 0.210 |
| B1_level | 252 | train | 0.039 | 0.084 | 0.245 | 8 | 0 | 1.592 | 0.200 |
| B1_level | 252 | validation | 0.039 | 0.074 | 0.096 | 6 | 0 | 2.505 | 0.203 |
| B2_innovation | 63 | train | 0.148 | 0.191 | 0.209 | 9 | 0 | 3.061 | 0.200 |
| B2_innovation | 63 | validation | 0.148 | 0.201 | 0.090 | 6 | 1 | 6.513 | 0.200 |
| B2_innovation | 126 | train | 0.077 | 0.117 | 0.212 | 8 | 0 | 3.061 | 0.200 |
| B2_innovation | 126 | validation | 0.077 | 0.128 | 0.090 | 6 | 1 | 6.763 | 0.197 |
| B2_innovation | 252 | train | 0.039 | 0.072 | 0.218 | 8 | 0 | 2.694 | 0.200 |
| B2_innovation | 252 | validation | 0.039 | 0.084 | 0.090 | 6 | 1 | 6.763 | 0.195 |
| B3_arhmm_surprise | 63 | train | 0.148 | 0.163 | 0.255 | 9 | 0 | 3.796 | 0.200 |
| B3_arhmm_surprise | 63 | validation | 0.148 | 0.166 | 0.090 | 6 | 1 | 6.513 | 0.223 |
| B3_arhmm_surprise | 126 | train | 0.077 | 0.097 | 0.261 | 9 | 0 | 3.551 | 0.200 |
| B3_arhmm_surprise | 126 | validation | 0.077 | 0.093 | 0.091 | 6 | 1 | 7.014 | 0.236 |
| B3_arhmm_surprise | 252 | train | 0.039 | 0.059 | 0.265 | 9 | 0 | 3.429 | 0.200 |
| B3_arhmm_surprise | 252 | validation | 0.039 | 0.051 | 0.092 | 6 | 1 | 7.014 | 0.236 |
| B2_volatility | 126 | train | 0.077 | 0.139 | 0.244 | 9 | 0 | 2.694 | 0.200 |
| B2_volatility | 126 | validation | 0.077 | 0.171 | 0.102 | 6 | 0 | 5.511 | 0.220 |
| B2_credit | 126 | train | 0.077 | 0.102 | 0.247 | 9 | 0 | 2.816 | 0.200 |
| B2_credit | 126 | validation | 0.077 | 0.101 | 0.093 | 6 | 0 | 4.509 | 0.195 |

## 9. Economic confirmation layer

- confirmation = Train ECDF of the factor-level stress index (-equity_trend + volatility + liquidity_credit, standardized on Train).
- Channel tags for the alert explanation: equity weakness, volatility stress, credit stress (level percentile >= 0.80) and rates/inflation pressure (macro component >= 0.90).
- Ablation `no_credit_confirmation` replaces the index with equity + volatility only.

## 10. Early Warning Score

EWS_t = sum_i w_i * component_i,t; each component in [0, 1] via its Train ECDF (share of Train values strictly below the current value). Weights fixed in advance, never optimized:

| component | weight | meaning |
|---|---|---|
| regime | 0.25 | HMM non-normal / stress regime |
| confirmation | 0.25 | factor-level market stress |
| innovation | 0.2 | adverse AR(1) innovations |
| bocpd | 0.15 | structural change (BOCPD) |
| velocity | 0.1 | rising non-normal probability |
| macro | 0.05 | rates / inflation pressure |
| entropy | 0.0 | regime uncertainty |

The EWS is a ranking score, not a probability of loss.

## 11. Alert state machine

| level | entry (Train quantile) | exit (Train quantile) |
|---|---|---|
| AMBER | 0.6145 (q80) | 0.5096 (q70) |
| ORANGE | 0.7103 (q90) | 0.6729 (q85) |
| RED | 0.7945 (q97) | 0.7376 (q93) |

- Escalation needs EWS >= entry on 3 of the last 5 sessions; levels may be skipped.
- Emergency RED: {'p_stress': 0.9, 'confirmation_with_p_stress': 0.95, 'innovation_pct': 0.995, 'confirmation_with_innovation': 0.9}.
- De-escalation needs EWS < exit on 3 of 5; ORANGE/RED fall to RECOVERY, which clears after 10 consecutive sessions below the AMBER exit.
- The machine never reads the HMM argmax state: alert levels and regimes are different objects.

| level | train | validation |
|---|---|---|
| GREEN | 68.3% | 74.2% |
| AMBER | 9.8% | 9.6% |
| ORANGE | 7.5% | 9.1% |
| RED | 5.5% | 2.7% |
| RECOVERY | 8.9% | 4.4% |

## 12. Outcome definitions

- Daily label: SPY worst cumulative log return over the next 21 sessions <= log(0.95). Secondary: the strategic 9-ETF reference portfolio (configs/project.json weights) <= log(0.97).
- Forward windows never cross a split boundary (last 21 sessions of each split unlabeled) or the 2017/2018 block boundary in block-level results.
- Event: SPY drawdown from its running peak reaches -5% (onset); re-arms when it recovers above -2.5%.

Validation events:

|  | peak_date | onset_date | end_date | trough_loss_pct | censored |
|---|---|---|---|---|---|
| 0 | 2016-01-05 | 2016-01-13 | 2016-03-01 | -9.2 | False |
| 1 | 2016-06-08 | 2016-06-27 | 2016-06-29 | -5.5 | False |
| 2 | 2018-01-26 | 2018-02-05 | 2018-06-11 | -10.1 | False |
| 3 | 2018-01-26 | 2018-06-27 | 2018-07-09 | -5.2 | False |
| 4 | 2018-09-20 | 2018-10-11 | 2019-03-18 | -19.3 | False |
| 5 | 2019-05-03 | 2019-05-29 | 2019-06-07 | -6.6 | False |
| 6 | 2019-07-26 | 2019-08-05 | 2019-09-05 | -6.0 | False |

## 13. Evaluation protocol

- Warned: alert on any of the 63 sessions before onset; late: first alert within 21 sessions after onset; otherwise missed.
- False alert episode: an alert run with no onset within 63 sessions of its start and no overlap with an event period.
- Equal alert budget: every continuous baseline alerts when its score exceeds its Train 80% quantile. A (hard HMM state) is binary; I is also evaluated through the state machine.
- Daily ranking: PR-AUC; uncertainty: moving-block bootstrap (63-session blocks, 1000 valid replications) of PR-AUC differences versus the best.
- Selection rule (fixed in advance): least complex score whose block-bootstrap 95% PR-AUC interval vs the best includes 0.

## 14. Baseline comparison (systems A-I)

| system | split | pr_auc | base_rate | daily_lift | portfolio_pr_auc | time_share_alerted |
|---|---|---|---|---|---|---|
| A_hmm_hard_stress | train | 0.312 | 0.214 | 1.947 | 0.278 | 0.249 |
| A_hmm_hard_stress | validation | 0.151 | 0.111 | 1.902 | 0.082 | 0.208 |
| B_p_stress | train | 0.382 | 0.214 | 1.944 | 0.352 | 0.200 |
| B_p_stress | validation | 0.180 | 0.111 | 2.090 | 0.103 | 0.146 |
| C_p_non_normal | train | 0.354 | 0.214 | 1.956 | 0.319 | 0.204 |
| C_p_non_normal | validation | 0.181 | 0.111 | 2.021 | 0.104 | 0.151 |
| D_volatility_level | train | 0.403 | 0.214 | 2.061 | 0.386 | 0.200 |
| D_volatility_level | validation | 0.161 | 0.111 | 2.047 | 0.087 | 0.180 |
| E_ar_innovation | train | 0.259 | 0.212 | 1.389 | 0.225 | 0.200 |
| E_ar_innovation | validation | 0.153 | 0.111 | 1.868 | 0.115 | 0.180 |
| F_bocpd | train | 0.212 | 0.214 | 0.982 | 0.195 | 0.200 |
| F_bocpd | validation | 0.090 | 0.111 | 0.507 | 0.047 | 0.197 |
| G_hmm_bocpd | train | 0.304 | 0.212 | 1.437 | 0.285 | 0.200 |
| G_hmm_bocpd | validation | 0.186 | 0.111 | 1.791 | 0.117 | 0.221 |
| H_hmm_innovation_bocpd | train | 0.286 | 0.212 | 1.444 | 0.261 | 0.200 |
| H_hmm_innovation_bocpd | validation | 0.193 | 0.111 | 1.799 | 0.133 | 0.220 |
| I_full_ews_score | train | 0.311 | 0.212 | 1.536 | 0.280 | 0.200 |
| I_full_ews_score | validation | 0.205 | 0.111 | 1.859 | 0.137 | 0.208 |
| I_full_ews_state_amber | train | 0.311 | 0.212 | 1.674 | 0.280 | 0.228 |
| I_full_ews_state_amber | validation | 0.205 | 0.111 | 1.673 | 0.137 | 0.215 |
| I_full_ews_state_orange | train | 0.311 | 0.212 | 1.834 | 0.280 | 0.130 |
| I_full_ews_state_orange | validation | 0.205 | 0.111 | 2.278 | 0.137 | 0.118 |

Validation by block (stability):

| system | split | pr_auc | base_rate | n_events | warned |
|---|---|---|---|---|---|
| A_hmm_hard_stress | validation 2016-2017 | 0.038 | 0.023 | 2 | 1 |
| A_hmm_hard_stress | validation 2018-2019 | 0.219 | 0.203 | 5 | 2 |
| B_p_stress | validation 2016-2017 | 0.057 | 0.023 | 2 | 0 |
| B_p_stress | validation 2018-2019 | 0.207 | 0.203 | 5 | 1 |
| C_p_non_normal | validation 2016-2017 | 0.060 | 0.023 | 2 | 0 |
| C_p_non_normal | validation 2018-2019 | 0.217 | 0.203 | 5 | 1 |
| D_volatility_level | validation 2016-2017 | 0.065 | 0.023 | 2 | 2 |
| D_volatility_level | validation 2018-2019 | 0.199 | 0.203 | 5 | 3 |
| E_ar_innovation | validation 2016-2017 | 0.032 | 0.023 | 2 | 2 |
| E_ar_innovation | validation 2018-2019 | 0.246 | 0.203 | 5 | 5 |
| F_bocpd | validation 2016-2017 | 0.017 | 0.023 | 2 | 1 |
| F_bocpd | validation 2018-2019 | 0.160 | 0.203 | 5 | 5 |
| G_hmm_bocpd | validation 2016-2017 | 0.036 | 0.023 | 2 | 2 |
| G_hmm_bocpd | validation 2018-2019 | 0.232 | 0.203 | 5 | 5 |
| H_hmm_innovation_bocpd | validation 2016-2017 | 0.034 | 0.023 | 2 | 2 |
| H_hmm_innovation_bocpd | validation 2018-2019 | 0.254 | 0.203 | 5 | 5 |
| I_full_ews_score | validation 2016-2017 | 0.070 | 0.023 | 2 | 1 |
| I_full_ews_score | validation 2018-2019 | 0.262 | 0.203 | 5 | 5 |
| I_full_ews_state_amber | validation 2016-2017 | 0.070 | 0.023 | 2 | 1 |
| I_full_ews_state_amber | validation 2018-2019 | 0.262 | 0.203 | 5 | 4 |
| I_full_ews_state_orange | validation 2016-2017 | 0.070 | 0.023 | 2 | 1 |
| I_full_ews_state_orange | validation 2018-2019 | 0.262 | 0.203 | 5 | 4 |

## 15. Lead time and missed events

| system | split | n_events | warned | detected_late | missed | median_lead_days | median_gap_to_onset | median_delay_days |
|---|---|---|---|---|---|---|---|---|
| A_hmm_hard_stress | train | 9 | 4 | 0 | 5 | 8.0 | 1.0 |  |
| A_hmm_hard_stress | validation | 7 | 3 | 2 | 2 | 3.0 | 1.0 | 9.0 |
| B_p_stress | train | 9 | 3 | 1 | 5 | 8.0 | 1.0 | 0.0 |
| B_p_stress | validation | 7 | 1 | 3 | 3 | 63.0 | 38.0 | 0.0 |
| C_p_non_normal | train | 9 | 3 | 1 | 5 | 8.0 | 1.0 | 0.0 |
| C_p_non_normal | validation | 7 | 1 | 3 | 3 | 63.0 | 37.0 | 0.0 |
| D_volatility_level | train | 9 | 6 | 0 | 3 | 2.0 | 1.0 |  |
| D_volatility_level | validation | 7 | 5 | 1 | 1 | 2.0 | 1.0 | 0.0 |
| E_ar_innovation | train | 9 | 9 | 0 | 0 | 13.0 | 2.0 |  |
| E_ar_innovation | validation | 7 | 7 | 0 | 0 | 5.0 | 1.0 |  |
| F_bocpd | train | 9 | 8 | 1 | 0 | 12.0 | 7.0 | 8.0 |
| F_bocpd | validation | 7 | 6 | 0 | 1 | 1.5 | 1.0 |  |
| G_hmm_bocpd | train | 9 | 6 | 0 | 3 | 3.5 | 1.0 |  |
| G_hmm_bocpd | validation | 7 | 7 | 0 | 0 | 3.0 | 1.0 |  |
| H_hmm_innovation_bocpd | train | 9 | 8 | 1 | 0 | 4.5 | 1.0 | 0.0 |
| H_hmm_innovation_bocpd | validation | 7 | 7 | 0 | 0 | 2.0 | 1.0 |  |
| I_full_ews_score | train | 9 | 7 | 1 | 1 | 4.0 | 1.0 | 21.0 |
| I_full_ews_score | validation | 7 | 6 | 1 | 0 | 2.5 | 1.0 | 2.0 |
| I_full_ews_state_amber | train | 9 | 6 | 0 | 3 | 7.0 | 1.0 |  |
| I_full_ews_state_amber | validation | 7 | 5 | 1 | 1 | 3.0 | 1.0 | 0.0 |
| I_full_ews_state_orange | train | 9 | 5 | 0 | 4 | 5.0 | 1.0 |  |
| I_full_ews_state_orange | validation | 7 | 5 | 1 | 1 | 10.0 | 5.0 | 1.0 |

Event detail for the state machine (>= AMBER), Validation:

| onset_date | peak_date | trough_loss_pct | status | lead_days | gap_last_alert_to_onset | delay_days | alert_on_onset_day |
|---|---|---|---|---|---|---|---|
| 2016-01-13 | 2016-01-05 | -9.2 | warned | 3.0 | 1.0 |  | True |
| 2016-06-27 | 2016-06-08 | -5.5 | missed |  |  |  | False |
| 2018-02-05 | 2018-01-26 | -10.1 | detected_late |  |  | 0.0 | True |
| 2018-06-27 | 2018-01-26 | -5.2 | warned | 18.0 | 11.0 |  | False |
| 2018-10-11 | 2018-09-20 | -19.3 | warned | 2.0 | 1.0 |  | True |
| 2019-05-29 | 2019-05-03 | -6.6 | warned | 1.0 | 1.0 |  | True |
| 2019-08-05 | 2019-07-26 | -6.0 | warned | 48.0 | 40.0 |  | True |

## 16. False alarms and alert stability

| system | split | alert_episodes | false_alert_episodes | false_alerts_per_year | episode_precision | mean_alert_episode_days | alert_transitions_per_year | time_share_alerted |
|---|---|---|---|---|---|---|---|---|
| A_hmm_hard_stress | train | 8 | 2 | 0.24 | 0.75 | 64.00 | 1.84 | 0.25 |
| A_hmm_hard_stress | validation | 4 | 0 | 0.00 | 1.00 | 52.25 | 2.00 | 0.21 |
| B_p_stress | train | 17 | 3 | 0.37 | 0.82 | 24.24 | 4.04 | 0.20 |
| B_p_stress | validation | 6 | 0 | 0.00 | 1.00 | 24.50 | 3.01 | 0.15 |
| C_p_non_normal | train | 13 | 3 | 0.37 | 0.77 | 32.23 | 3.06 | 0.20 |
| C_p_non_normal | validation | 5 | 0 | 0.00 | 1.00 | 30.40 | 2.50 | 0.15 |
| D_volatility_level | train | 22 | 8 | 0.98 | 0.64 | 18.73 | 5.39 | 0.20 |
| D_volatility_level | validation | 13 | 2 | 0.50 | 0.85 | 13.92 | 6.51 | 0.18 |
| E_ar_innovation | train | 112 | 20 | 2.45 | 0.82 | 3.67 | 27.43 | 0.20 |
| E_ar_innovation | validation | 47 | 14 | 3.51 | 0.70 | 3.85 | 23.55 | 0.18 |
| F_bocpd | train | 104 | 25 | 3.06 | 0.76 | 3.96 | 25.35 | 0.20 |
| F_bocpd | validation | 53 | 27 | 6.76 | 0.49 | 3.74 | 26.55 | 0.20 |
| G_hmm_bocpd | train | 94 | 22 | 2.69 | 0.77 | 4.37 | 23.02 | 0.20 |
| G_hmm_bocpd | validation | 44 | 7 | 1.75 | 0.84 | 5.05 | 22.04 | 0.22 |
| H_hmm_innovation_bocpd | train | 95 | 20 | 2.45 | 0.79 | 4.33 | 23.27 | 0.20 |
| H_hmm_innovation_bocpd | validation | 41 | 10 | 2.50 | 0.76 | 5.39 | 20.54 | 0.22 |
| I_full_ews_score | train | 88 | 16 | 1.96 | 0.82 | 4.67 | 21.55 | 0.20 |
| I_full_ews_score | validation | 38 | 8 | 2.00 | 0.79 | 5.50 | 19.04 | 0.21 |
| I_full_ews_state_amber | train | 46 | 9 | 1.10 | 0.80 | 10.22 | 11.27 | 0.23 |
| I_full_ews_state_amber | validation | 24 | 4 | 1.00 | 0.83 | 9.00 | 12.02 | 0.21 |
| I_full_ews_state_orange | train | 25 | 6 | 0.73 | 0.76 | 10.72 | 6.12 | 0.13 |
| I_full_ews_state_orange | validation | 9 | 0 | 0.00 | 1.00 | 13.22 | 4.51 | 0.12 |

Validation alert episodes of the state machine:

| start | end | days | max_level |
|---|---|---|---|
| 2016-01-08 | 2016-01-27 | 13 | RED |
| 2016-02-04 | 2016-02-17 | 9 | ORANGE |
| 2017-04-19 | 2017-04-21 | 3 | AMBER |
| 2017-05-19 | 2017-05-24 | 4 | AMBER |
| 2017-08-14 | 2017-08-25 | 10 | AMBER |
| 2018-02-05 | 2018-03-09 | 24 | RED |
| 2018-03-16 | 2018-03-20 | 3 | AMBER |
| 2018-03-26 | 2018-04-13 | 14 | ORANGE |
| 2018-04-25 | 2018-04-30 | 4 | AMBER |
| 2018-06-01 | 2018-06-12 | 8 | AMBER |
| 2018-07-17 | 2018-07-20 | 4 | AMBER |
| 2018-08-16 | 2018-08-22 | 5 | AMBER |
| 2018-09-10 | 2018-09-14 | 5 | AMBER |
| 2018-10-09 | 2018-11-01 | 18 | ORANGE |
| 2018-11-13 | 2018-12-03 | 14 | RED |
| 2018-12-06 | 2019-01-02 | 18 | ORANGE |
| 2019-02-06 | 2019-02-12 | 5 | AMBER |
| 2019-03-12 | 2019-03-14 | 3 | AMBER |
| 2019-04-11 | 2019-04-16 | 4 | AMBER |
| 2019-05-09 | 2019-05-21 | 9 | ORANGE |
| 2019-05-28 | 2019-06-07 | 9 | AMBER |
| 2019-08-05 | 2019-08-29 | 19 | ORANGE |
| 2019-09-05 | 2019-09-10 | 4 | AMBER |
| 2019-10-04 | 2019-10-14 | 7 | AMBER |

## 17. Ablations

| specification | split | pr_auc | warned | missed | false_alerts_per_year | time_share_alerted | alert_transitions_per_year |
|---|---|---|---|---|---|---|---|
| full | train | 0.311 | 6 | 3 | 1.102 | 0.228 | 11.265 |
| full | validation | 0.205 | 5 | 1 | 1.002 | 0.215 | 12.024 |
| no_bocpd | train | 0.328 | 5 | 3 | 1.347 | 0.227 | 9.551 |
| no_bocpd | validation | 0.197 | 5 | 1 | 0.501 | 0.177 | 7.014 |
| no_innovation | train | 0.331 | 6 | 3 | 0.612 | 0.212 | 7.592 |
| no_innovation | validation | 0.204 | 5 | 1 | 0.751 | 0.213 | 9.018 |
| no_velocity | train | 0.315 | 6 | 3 | 1.469 | 0.221 | 10.041 |
| no_velocity | validation | 0.210 | 5 | 1 | 0.751 | 0.219 | 10.020 |
| no_macro | train | 0.314 | 6 | 3 | 1.224 | 0.226 | 10.776 |
| no_macro | validation | 0.207 | 5 | 1 | 1.252 | 0.214 | 12.525 |
| no_credit_confirmation | train | 0.312 | 6 | 3 | 1.102 | 0.215 | 10.531 |
| no_credit_confirmation | validation | 0.208 | 6 | 0 | 1.252 | 0.223 | 12.024 |
| with_entropy | train | 0.307 | 6 | 3 | 1.102 | 0.228 | 11.510 |
| with_entropy | validation | 0.195 | 5 | 0 | 1.252 | 0.213 | 13.026 |
| no_hysteresis | train | 0.311 | 7 | 1 | 1.959 | 0.200 | 21.551 |
| no_hysteresis | validation | 0.205 | 6 | 0 | 2.004 | 0.208 | 19.038 |

Validation PR-AUC change versus the full specification (positive = removing/adding the item helps): no_bocpd -0.007, no_innovation -0.001, no_velocity +0.005, no_macro +0.003, no_credit_confirmation +0.003, with_entropy -0.010, no_hysteresis +0.000.
No ablation result was used to change the primary weights.

## 18. Entropy and velocity: do they add information?

| signal | train | validation |
|---|---|---|
| ar_innovation_stress | 0.259 | 0.153 |
| ar_surprise_chi2 | 0.260 | 0.147 |
| arhmm_surprise | 0.254 | 0.151 |
| normalized_entropy | 0.193 | 0.123 |
| p_non_normal | 0.354 | 0.181 |
| p_non_normal_plus_entropy | 0.247 | 0.131 |
| p_non_normal_plus_velocity | 0.266 | 0.165 |
| velocity_positive | 0.202 | 0.133 |

Loss rate by entropy tercile within regime bucket (Validation):

| regime_bucket | tercile | n | event_rate | mean_value |
|---|---|---|---|---|
| p_non_normal < 0.5 | low | 145 | 0.034 | 0.000 |
| p_non_normal < 0.5 | mid | 144 | 0.056 | 0.001 |
| p_non_normal < 0.5 | high | 144 | 0.104 | 0.138 |
| p_non_normal >= 0.5 | low | 184 | 0.179 | 0.000 |
| p_non_normal >= 0.5 | mid | 184 | 0.152 | 0.003 |
| p_non_normal >= 0.5 | high | 184 | 0.109 | 0.176 |

Velocity before argmax switches S1 -> S2/S3 (threshold = Train 90th percentile of positive velocity, 10-session look-back):

| split | switches | share_preceded | median_lead |
|---|---|---|---|
| train | 12 | 0.92 | 7.00 |
| validation | 9 | 0.56 | 4.00 |

Mean entropy 10-6 sessions before a switch vs 2-0 sessions before: train 0.048 -> 0.423, validation 0.073 -> 0.367. Entropy rising only at the switch is a coincident, not a leading, signal.

## 19. AR-HMM surprise versus AR(1) innovation

The frozen AR-HMM K=3 predictive surprise (-log p(x_t | past)) is direction-free; the AR(1) market innovation is directional. Compare the rows `arhmm_surprise`, `ar_surprise_chi2` (direction-free AR(1)) and `ar_innovation_stress` (directional) in section 18, and B2 vs B3 in section 8. The AR-HMM is used only as a supplementary signal; it was not refitted.

## 20. Specification selection

| model | reference | point_ap_diff | mean_ap_diff | ci95_low | ci95_high | share_positive | valid_reps |
|---|---|---|---|---|---|---|---|
| B_p_stress | I_full_ews_score | -0.0247 | -0.0360 | -0.1586 | 0.0368 | 0.2410 | 1000 |
| C_p_non_normal | I_full_ews_score | -0.0237 | -0.0292 | -0.1478 | 0.0364 | 0.2960 | 1000 |
| D_volatility_level | I_full_ews_score | -0.0436 | -0.0550 | -0.2032 | 0.0213 | 0.1790 | 1000 |
| E_ar_innovation | I_full_ews_score | -0.0514 | -0.0591 | -0.1896 | 0.0095 | 0.0910 | 1000 |
| F_bocpd | I_full_ews_score | -0.1146 | -0.1228 | -0.3027 | -0.0065 | 0.0020 | 1000 |
| G_hmm_bocpd | I_full_ews_score | -0.0188 | -0.0213 | -0.0926 | 0.0160 | 0.1390 | 1000 |
| H_hmm_innovation_bocpd | I_full_ews_score | -0.0114 | -0.0156 | -0.0633 | 0.0057 | 0.1320 | 1000 |

- Best Validation PR-AUC: I full EWS (score, 20% budget).
- Not significantly worse than the best: B P(stress), C P(non-normal), D volatility level, E AR(1) innovation, G HMM + BOCPD, H HMM + innov. + BOCPD, I full EWS (score, 20% budget).
- Retained specification: **C P(non-normal)** - the full system is not demonstrably better than a simpler method, so the simpler specification is retained.

## 21. Causality, leakage controls and limitations

- Causality test on the real data: the second half of the Validation factors was replaced by a distorted copy and every signal, score and alert before the cut was recomputed and compared (section 3).
- Thresholds, ECDF scales, BOCPD priors and channel normalizations are estimated on Train rows only.
- Filtered (not smoothed) HMM probabilities; Validation filtering continues from the last Train filtered distribution.
- Limitations: few Validation events; Validation reused from Stages 5/5.5; Train signals are in-sample; the -5% event definition and the 63-session lead window are conventions; long alert episodes can warn mechanically, so lead time must be read with false alarms and time on alert; the outcome shares SPY momentum with the equity factor.

## 22. Conclusions and handoff

- The full EWS has the highest Validation PR-AUC; it is significantly better than 1 of 5 single-signal baselines (F_bocpd).
- Retained specification: **C P(non-normal)** - the full system is not demonstrably better than a simpler method, so the simpler specification is retained.
- Stage 6 output for later use: daily early_warning_{train,validation}.csv (score, level, reason, drivers). How alerts should affect allocation is a Stage 7 question and was not studied here.
- Test (2020 onward) remains untouched.

Figures: `figures/early_warning/bocpd_change_point_diagnostics.png`, `figures/early_warning/factor_level_vs_innovation_bocpd.png`, `figures/early_warning/hmm_regime_uncertainty.png`, `figures/early_warning/probability_velocity.png`, `figures/early_warning/early_warning_score_timeline.png`, `figures/early_warning/alert_state_timeline.png`, `figures/early_warning/warning_lead_time_analysis.png`, `figures/early_warning/false_alarm_analysis.png`, `figures/early_warning/early_warning_benchmark_comparison.png`, `figures/early_warning/early_warning_ablation.png`
