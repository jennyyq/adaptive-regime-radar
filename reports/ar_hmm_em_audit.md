# Stage 5.5 audit - AR-HMM EM monotonicity, M-step optimality and reproducibility

Train only (2058 rows, 2057 scored); 10 seeds x 5 starts per K; every start's trajectory is in `data/processed/ar_hmm/audit/em_trajectories.csv`. Test not read; Stage 0-5 untouched.

## Question

The first real Stage 5.5 run reported `all_monotone = False` for ar_hmm_k3 while all checks passed. EM (and constrained generalized EM) cannot decrease the likelihood if the M-step maximizes the Q-function, so a decrease is either rounding error or an M-step that is not a maximizer.

## Monotonicity of every start

| k | variance_rule | starts | converged_share | starts_with_any_decrease | starts_with_decrease_beyond_tol | total_decreases_beyond_tol | max_decrease | max_fp_error_estimate | genuine | numerical | best_loglik_per_obs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | additive_legacy | 10 | 1.000000 | 0 | 0 | 0 | 0.000000 | 0.000000 | 0 | 0 | 2.604174 |
| 1 | floor | 10 | 1.000000 | 0 | 0 | 0 | 0.000000 | 0.000000 | 0 | 0 | 2.604240 |
| 2 | additive_legacy | 50 | 1.000000 | 0 | 0 | 0 | 0.000000 | 0.000000 | 0 | 0 | 3.077065 |
| 2 | floor | 50 | 1.000000 | 0 | 0 | 0 | 0.000000 | 0.000000 | 0 | 0 | 3.077482 |
| 3 | additive_legacy | 50 | 1.000000 | 5 | 5 | 84 | 0.032456 | 0.000000 | 5 | 0 | 3.184233 |
| 3 | floor | 50 | 1.000000 | 0 | 0 | 0 | 0.000000 | 0.000000 | 0 | 0 | 3.184941 |

Classification: a decrease is 'numerical' if it is at most 100 x the floating-point error estimate of the objective (|naive sum - exact sum| of the T summands + T x eps x max|summand|), otherwise 'genuine'. Tolerance used by the pipeline: 1e-07 nats on the total objective.

Legacy starts with decreases beyond tolerance:

| k | seed | start_seed | n_iter | n_decreases_beyond_tol | max_decrease | iteration_of_max_decrease | fp_error_estimate | classification |
|---|---|---|---|---|---|---|---|---|
| 3 | 20240532 | 202405320 | 83 | 41 | 0.03245640 | 48.00000000 | 0.00000000 | genuine |
| 3 | 20240534 | 202405342 | 43 | 1 | 0.00174190 | 43.00000000 | 0.00000000 | genuine |
| 3 | 20240535 | 202405354 | 100 | 1 | 0.00083063 | 100.00000000 | 0.00000000 | genuine |
| 3 | 20240536 | 202405360 | 100 | 1 | 0.00083063 | 100.00000000 | 0.00000000 | genuine |
| 3 | 20240538 | 202405380 | 81 | 40 | 0.03191241 | 47.00000000 | 0.00000000 | genuine |

## M-step optimality and objective consistency

q_floor_minus_q_legacy_at_same_estep > 0 means the legacy variance update leaves Q-function value unused at every iteration; max_q_gain_feasible_move_floor <= 0 means no feasible single-coordinate change of (intercept, phi, var) improves Q at the exact M-step.

| k | variance_rule | best_seed_start | q_floor_minus_q_legacy_at_same_estep | max_q_gain_feasible_move_floor | mstep_optimal_floor | em_objective | filter_score_sum | objective_vs_filter_abs_diff | min_innovation_var | variance_floor_active_share |
|---|---|---|---|---|---|---|---|---|---|---|
| 3 | floor | 202405320 | 0.51838259 | -0.00000084 | True | 6551.42301711 | 6551.42301711 | 0.00000000 | 0.00306223 | 0.00000000 |
| 3 | additive_legacy | 202405400 | 0.42036893 |  |  | 6549.96671358 | 6549.96671358 | 0.00000000 | 0.00317281 | 0.00000000 |
| 2 | floor | 202405310 | 0.35356929 | -0.00000133 | True | 6330.38078607 | 6330.38078607 | 0.00000000 | 0.00456378 | 0.00000000 |
| 2 | additive_legacy | 202405310 | 0.34336879 |  |  | 6329.52371680 | 6329.52371680 | 0.00000000 | 0.00478183 | 0.00000000 |
| 1 | floor | 202405310 | 0.13593422 | -0.00000514 | True | 5356.92108237 | 5356.92108237 | 0.00000000 | 0.00783308 | 0.00000000 |
| 1 | additive_legacy | 202405310 | 0.13593422 |  |  | 5356.78514815 | 5356.78514815 | 0.00000000 | 0.00793308 | 0.00000000 |

## Reproducibility and effect of the fix

| k | refit_abs_loglik_diff | refit_max_param_diff | floor_minus_legacy_best_loglik_per_obs | train_ari_floor_vs_legacy | saved_model_version | saved_variance_rule | saved_loglik | recomputed_loglik_of_saved_params | audit_best_loglik_same_rule | abs_diff_saved_vs_audit_best |
|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 0.00000000 | 0.00000000 | 0.00070797 | 0.32817665 | stage5.5-2026-10-21 | additive_legacy | 6549.96671358 | 6549.96671358 | 6549.96671358 | 0.00000000 |
| 2 | 0.00000000 | 0.00000000 | 0.00041666 | 0.93351967 | stage5.5-2026-10-21 | additive_legacy | 6329.52371680 | 6329.52371680 | 6329.52371680 | 0.00000000 |
| 1 | 0.00000000 | 0.00000000 | 0.00006608 | 1.00000000 | stage5.5-2026-10-21 | additive_legacy | 5356.78514815 | 5356.78514815 | 5356.78514815 | 0.00000000 |

## Checks

| name | status | detail |
|---|---|---|
| Exact M-step: no EM start decreases beyond tolerance | PASS | 0 of 110 starts; max decrease 0.00e+00 |
| Legacy v1 M-step reproduces the reported non-monotonicity (informational) | INFO | 5 of 110 starts; max decrease 3.25e-02; genuine 5 |
| M-step is the exact constrained maximizer of Q (floor rule) | PASS | max Q gain of any feasible move -8.4e-07 |
| EM objective = sum of filter log scores (fit/score consistency) | PASS | max |difference| 1.8e-12 |
| Saved 07b models reproduced by re-running their rule (same data, seeds, starts) | PASS | max |log-likelihood difference| 0.0e+00 |
| Refit reproduces the best fit exactly | PASS | max |log-likelihood difference| 0.0e+00 |
| No test file read | PASS | split_manifest.json, train_features.csv, train_returns.csv, validation_features.csv, validation_returns.csv |

## Conclusion (computed)

- Legacy rule: 5 starts decreased beyond tolerance, 5 classified genuine (max 3.25e-02 nats vs floating-point error <= 1.6e-11). The decreases are not rounding error: the variance update s2 + 1e-4 is not a Q maximizer.
- Exact rule: 0 starts decreased beyond tolerance (max decrease 0.00e+00).
- Stage 5.5 results must be re-run with scripts/07b_compare_ar_hmm.py (exact M-step); the frozen Stage 5 HMM is not affected.
