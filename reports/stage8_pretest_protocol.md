# Stage 8 pre-Test protocol (Phase 8A)

**Lock fingerprint:** `73998035e0857aad7a9ccaa091f410166b9c9f369c05896f3f18aec684e012a6`  
**Created (UTC):** 2026-10-21T11:53:27+00:00  
**Test access authorized:** YES - all preflight checks PASS  
**Test data read before this lock:** none (the Test split files were not opened; only their recorded hashes in the Stage 3 split manifest exist).

This document was generated before the reserved Test split was opened. Everything below is frozen in `data/processed/backtesting/stage8_pretest_lock.json` together with the hashes of every project module, configuration file and frozen artifact. Changing any of them invalidates the lock; the gated Test loader then refuses to read Test.

## 1. Research question

Does the Full EWS-enhanced portfolio (P2) improve realized downside protection and risk-adjusted performance beyond the strategic benchmark (P0) and the simpler HMM-based allocation (P1), after the conservative t+1 execution delay and transaction costs?

Terminology: Sequential out-of-sample portfolio backtest with FROZEN regime / early-warning models (no refit, no retraining schedule) and causally re-estimated portfolio risk inputs (Ledoit-Wolf covariance re-estimated at every weekly decision from returns through the decision date).

## 2. Frozen strategies

| id | definition |
|---|---|
| P0_strategic | frozen strategic benchmark, weekly 1% no-trade band, turnover cap 20% (Stage 7 rule, posture = benchmark) |
| P1_regime_aware | role baseline_regime only; d1 = min(1, a * baseline_p_non_normal), a from the Stage 7 handoff (Train-calibrated) |
| P2_ews_enhanced | role challenger_ews only; d2 = level map GREEN 0, AMBER 1/3, ORANGE 2/3, RED 1, RECOVERY 1/3 |
| B1_60_40 | same weekly calendar and conservative_close_t_plus_1 execution; trade back to 60/40 when the one-way distance exceeds the 1% no-trade band; no turnover cap; same cost model; outside the nine-ETF mandate limits (contextual only) |
| B2_constant_defensiveness | identical optimizer, constraints, calendar, execution and costs as P1/P2 with a CONSTANT defensiveness d |
| B3_strategic_buy_and_hold | strategic weights held from the close of the last Validation session, never rebalanced, no costs (contextual only; not a substitute for P0) |

Frozen constants: P1_parity_scale_a = 0.28269502197950724, cost_stress_threshold_spy_21d = 0.30517114338438883, B2_constant_d = 0.1674765144152899, baseline_q80_threshold = 1.0, first_decision_date = '2008-10-31', last_validation_session = '2019-12-31'

## 3. Models and versions

| artifact | value |
|---|---|
| hmm | data/processed/hmm/models/hmm_std_diag_k3.json (stage5-2026-10-21c) |
| ar1 | data/processed/ar_hmm/models/ar1.json (stage5.5-2026-10-21b) |
| ar_hmm_k3 | data/processed/ar_hmm/models/ar_hmm_k3.json (stage5.5-2026-10-21b) |
| stage6_1_handoff | stage6.1-2026-10-21b |
| stage7 | stage7-2026-10-21 |
| stage7_policy | stage7-policy-2026-10-21 |
| stage8_code | stage8-2026-10-21 |

## 4. Data definitions and boundary

- **returns**: daily log return log(P_t/P_{t-1}) from adjusted close (dividends and splits); same-day
- **simple_returns**: R = exp(r) - 1
- **factors**: equity_trend, interest_rate, inflation, volatility, liquidity_credit
- **test_split**: Stage 3 test split (split_manifest.json); opened only by backtest_integrity.load_test_split after this lock
- **test_file_hashes_expected**: {'test_features.csv': '6aea72ba62b75a60730436a0cd0906f191a2123a61fd087ac4f7394d57d5be66', 'test_returns.csv': 'f5e67c3ccad2fb93bb416d946295b1bce062bc1c17dbcdd5b6a56f8b3581a2dd'}
- **train**: 2007-10-30..2015-12-31 (2058)
- **validation**: 2016-01-04..2019-12-31 (1006)
- **missing_data**: no filling; any missing Test value fails the Test integrity checks

## 5. Decision, execution and return accounting

- **decision_rule.optimizer**: Stage 7 primary policy (portfolio_optimizer.solve_policy, target_then_trade), reused without modification
- **decision_rule.pre_trade_weights_at_decision**: actual ledger holdings at the close of the decision date t (known at t). They equal the Stage 7 ex-ante proxy (previous target drifted through t) whenever the previous order was executed in full; they differ only after a hold inside the no-trade band or an execution-day projection (the Stage 7 proxy would then ignore realized drift).
- **decision_rule.risk_inputs**: Ledoit-Wolf 504 sessions, minimum 252, equilibrium expected returns (SR_b 0.30), all from simple returns dated <= t
- **decision_rule.signals**: date-t value of the portfolio's own role; intra-week signals are not acted upon
- **execution.convention**: conservative_close_t_plus_1: decision after the close of t; execution at the close of t+1; first return of the new holdings is that of t+2; holdings held before the execution earn the return of t+1
- **execution.hold_orders**: a decision inside the no-trade band is an explicit no-trade order (no execution, holdings keep drifting)
- **execution.trade_orders**: at the close of t+1 compute the actual pre-trade weights (drift through t+1). If 0.5*sum|target - pre| <= the decision's turnover cap, execute the decision-time target exactly; otherwise execute pre + (cap / TO) * (target - pre) (cap-preserving straight-line projection, the Stage 7 transition rule). No t+1 signal is used.
- **execution.price_assumption**: adjusted close of t+1 (adjusted total-return series; not proof of historical fills)
- **execution.costs**: deducted from wealth at the execution close, pro rata across holdings, so post-trade weights equal the executed target
- **return_accounting.inputs**: daily log returns of adjusted closes from the split files, converted exactly to simple returns R = exp(r) - 1
- **return_accounting.portfolio_return**: R_p,t = sum_i w_i,t-1 R_i,t with the weights actually held over the session (never a weighted sum of log returns)
- **return_accounting.gross**: returns of the actual holdings before trading costs (the trading path is identical with and without costs because costs are deducted pro rata)
- **return_accounting.net**: (1 + gross_t) * (1 - cost_fraction_t) - 1 on execution sessions, gross otherwise

## 6. Transaction costs

- **primary_bps**: 10
- **sensitivity_bps**: [5, 25]
- **basis**: bps of GROSS traded notional = sum_i |w_post,i - w_pre,i| = 2 x one-way turnover
- **stress_multiplier**: 2.0
- **stress_indicator**: trailing 21-session SPY realized volatility (simple returns through the decision date t, annualized) above the Train-calibrated threshold in the Stage 7 handoff; evaluated at t, applied to the execution at t+1; identical for every portfolio; uses no HMM / EWS role signal

## 7. Performance metrics

- **annualization_sessions**: 252
- **risk_free_for_evaluation**: realized daily simple return of BIL (evaluation benchmark only; never added to portfolio returns)
- **cagr**: wealth_T ^ (252 / T) - 1, T = number of Test sessions
- **volatility**: sample standard deviation of daily returns x sqrt(252)
- **sharpe**: mean(daily excess return over BIL) x 252 / (std(daily excess) x sqrt(252))
- **sortino**: mean daily excess x 252 / downside deviation; downside deviation = sqrt(mean(min(excess, 0)^2)) x sqrt(252)
- **calmar**: CAGR / |maximum drawdown|
- **max_drawdown**: minimum of wealth / running maximum - 1 on the daily net wealth path (initial 1.0 included)
- **var_cvar**: 95% historical daily VaR / CVaR of returns (Stage 7 Rockafellar-Uryasev convention)
- **worst_week_month**: compounded calendar-week (Monday-Sunday) and calendar-month returns
- **tracking_error_information_ratio**: std / mean of daily active returns vs the reference, annualized
- **rolling**: {'sharpe_window_sessions': 252, 'volatility_window_sessions': 63}

## 8. Stress windows and objective events (fixed before Test)

| window | start | end |
|---|---|---|
| 2020_pandemic_shock | 2020-02-01 | 2020-04-30 |
| 2020_2021_recovery | 2020-05-01 | 2021-12-31 |
| 2022_inflation_tightening | 2022-01-01 | 2022-12-31 |
| 2023_onward_normalization | 2023-01-01 | sample end |

- **window_origin**: calendar-month boundaries taken from the public macro chronology named in the Stage 8 brief (pandemic shock 2020, inflation / tightening 2022, later normalization); not chosen from any portfolio result
- **objective_events**: SPY drawdown events in Test by the frozen Stage 6 rule: onset when cumulative log return falls 5% below its running peak, re-armed above -2.5%; major events have a trough <= -10%
- **executable_protection**: a de-risking execution counts as protection for an event only if its EXECUTION date is before the event onset date; executions between onset and trough are reactions
- **de_risking_execution**: an execution that lowers the portfolio's equity weight AND leaves it at least 1 percentage point below the strategic equity weight (50%), i.e. a move into a defensive posture rather than a rebalance back to the benchmark; searched from 21 sessions before the event peak to the trough
- **recovery_participation**: net return from the close of the trough date over the next 63 sessions (or to the sample end), compared with P0

## 9. Statistical procedures

- **paired_series**: daily net returns at 10 bps
- **mean_active_return**: Newey-West (Bartlett) HAC standard error, lag = floor(4 (T/100)^(2/9)), normal 95% interval, two-sided p-value; Holm adjustment across the three primary comparisons
- **bootstrap**: {'method': 'stationary bootstrap (Politis-Romano), paired across all portfolios and BIL', 'mean_block_length_primary': 21, 'mean_block_length_secondary': 63, 'replications': 5000, 'seed': 20200102, 'interval': '95% percentile'}
- **bootstrapped_statistics**: ['annualized mean active return', 'Sharpe difference', 'Sortino difference', 'CAGR difference', 'volatility difference', 'maximum drawdown difference (approximate: resampled paths do not preserve the full crash sequence)']
- **caveat**: annualized Sharpe, CAGR and maximum drawdown are path statistics, not independent observations; the number of independent stress episodes is reported

## 10. Decision rules for the final conclusion (set before Test)

- **economically_meaningful_sharpe_difference**: 0.1
- **economically_meaningful_drawdown_reduction_pct_points**: 2.0
- **statistically_supported**: 95% stationary-bootstrap interval (block 21) of the net Sharpe difference excludes 0
- **robust_to_costs**: the sign of the net Sharpe difference is the same at 5, 10 and 25 bps
- **similar_to_constant**: |net Sharpe difference vs B2| < 0.10 and its 95% interval contains 0
- **explained_by_exposure**: the average-exposure part of the gross active return is larger in absolute value than the timing part (exposure-driven); otherwise timing-driven, called statistically supported only if the 95% interval of the regression alpha on the reference excludes 0
- **Outcome 1 - Full EWS adds credible incremental value**: P2 - P1 net Sharpe difference >= 0.10 with a 95% interval excluding 0, P2 - P0 net Sharpe interval excluding 0 on the positive side, and both robust to costs
- **Outcome 3 - Strategic allocation remains preferable**: both P1 - P0 and P2 - P0 net Sharpe differences <= 0 at 10 bps
- **Outcome 2 - Simple HMM allocation is sufficient**: not Outcome 1; P1 - P0 net Sharpe difference > 0 with a 95% interval excluding 0; and P2 - P1 difference < 0.10 or its interval contains 0
- **Outcome 4 - Evidence is inconclusive**: every other case

## 11. Allowed sensitivity analyses

- **costs_bps**: [5, 10, 25]
- **initialization**: ['continuation (primary)', 'restart_benchmark_at_test_start']
- **model_variants**: {'cov_lw_252': 'Ledoit-Wolf on 252 sessions (Stage 7 permitted sensitivity)', 'cov_ewma_hl63': 'EWMA covariance, half-life 63 (Stage 7 permitted sensitivity)', 'mapping_P1_full_range_P2_continuous': 'd1 = p, d2 = continuous EWS score map (Stage 7 permitted sensitivity)'}
- **model_variant_protocol**: each variant and its reference restart from the strategic benchmark at the Test start; robustness evidence only; the primary policy is never replaced

## 12. Model-selection restrictions

- No frozen HMM, AR(1), AR-HMM, BOCPD or EWS parameter, threshold, weight or mapping is re-estimated or changed.
- No Stage 7 policy parameter (risk budget, constraints, band, cap, penalties, solver, costs) is changed.
- No new comparator, window, metric, block length or sensitivity case is added after Test access.
- No grid search over risk aversion, thresholds, hazard, K, multipliers, rebalance frequency, caps or penalties.
- A sensitivity variant never replaces the primary policy.
- A genuine error found after Test access is corrected only as a labelled amendment that preserves the original result.

## 13. Implementation ambiguities resolved before Test

- Pre-trade weights at a decision are the ACTUAL ledger holdings at the close of t (known at t). The Stage 7 proxy (previous target drifted through t) equals them whenever the previous order was executed in full; after a hold inside the no-trade band it would ignore the realized drift of the holding period (consecutive holds would accumulate the error). Train/Validation difference vs the published Stage 7 targets: P0_strategic mean |dw| 1.1e-03; P1_regime_aware mean |dw| 7.6e-04; P2_ews_enhanced mean |dw| 6.3e-04.
- A hold decision is a no-trade order: nothing is executed at t+1 even if drift moves the holdings further; the next weekly decision re-evaluates them.
- Execution-day turnover above the decision's cap (drift between t and t+1) is handled by the Stage 7 straight-line transition rule (cap-preserving projection) and reported as an exception.
- The cost-stress flag is evaluated at the decision date t (Stage 7: 'known at the decision date') and applied to the t+1 execution; it is the same for every portfolio.
- The calendar week containing the last Validation session is completed with Test sessions; its decision (last session of that week) is the first Test-period decision. No other decision is created.
- Costs are deducted pro rata at the execution close; therefore the trading path is identical under 0 / 5 / 10 / 25 bps and every cost scenario is evaluated on the same holdings.

## 14. Preflight checks

| check | status | detail |
|---|---|---|
| Stage 7 handoff status READY | PASS | all blocking checks PASS; WARN: No library warnings: RuntimeWarning: Values in x were outside bounds during a minimize step, clipping to bounds |
| Frozen configuration hashes match (portfolio_policy.json, project.json) | PASS | configs/portfolio_policy.json, configs/project.json |
| Stage 6.1 handoff manifest hash matches | PASS | data/processed/early_warning/stage7_handoff/stage7_handoff_manifest.json |
| Stage 7 output hashes match the handoff (targets, risk estimates, sensitivity, ...) | PASS | 17 files; mismatched: none |
| Stage 7 code unchanged since the handoff | PASS | 10 files; changed: none |
| Frozen model files (HMM K=3, AR(1), AR-HMM) unchanged - no refit | PASS | hmm: data/processed/hmm/models/hmm_std_diag_k3.json, ar1: data/processed/ar_hmm/models/ar1.json, ar_hmm_k3: data/processed/ar_hmm/models/ar_hmm_k3.json |
| Train / Validation split files unchanged | PASS | 4 files; changed: none |
| Portfolio constraints, risk-budget mapping and benchmark match the frozen Stage 7 policy | PASS | constraints =; benchmark = configs/project.json |
| Signal roles separated: P1 -> baseline_regime, P2 -> challenger_ews | PASS | {"P1_regime_aware": "baseline_regime", "P2_ews_enhanced": "challenger_ews"} |
| Execution timing is the approved conservative convention | PASS | conservative_close_t_plus_1 |
| Stage 7 read no Test data | PASS | test_data_loaded = False |
| Train-calibrated constants available (P1 scale a, cost-stress threshold) | PASS | a = 0.28269502197950724; stress threshold = 0.30517114338438883 |
| Stage 7 release checks: no FAIL | PASS | 30 checks; FAIL: none; documented WARN: RuntimeWarning: Values in x were outside bounds during a minimize step, clipping to bounds |
| No Test file read during preflight | PASS | split_manifest.json, train_features.csv, train_returns.csv, validation_features.csv, validation_returns.csv |
| Frozen HMM is hmm_std_diag_k3 (K = 3) and the AR models use the corrected variance-floor M-step | PASS | data/processed/hmm/models/hmm_std_diag_k3.json; AR variance rules ['floor', 'floor'] |
| Rebuilt Train/Validation signals identical to the Stage 6.1 handoff (17 digits, bit-for-bit) | PASS | 14 columns compared; max |difference| 0.0e+00 |
| Train constants reproduce the frozen values (scaler, innovation normalization, BOCPD priors, alert thresholds) - no refit | PASS | alert_thresholds 0.0e+00, innovation_normalization 0.0e+00, bocpd_priors 0.0e+00, factor_scaler 0.0e+00 |
| Signal prefix invariance on Validation (run through t == full run truncated at t) | PASS | cut 2018-01-02; max difference 0.0e+00 |
| Engine configuration equals the frozen Stage 7 optimizer (penalties, band, execution mode) | PASS | lambda_turnover 0.008, lambda_tracking 2.0, band 0.01, target_then_trade |
| Train-calibrated constants reproduced exactly (P1 scale a, cost-stress threshold, B2 constant d) | PASS | a 0.28269502197950724; threshold 0.30517114338438883; mean Train d2 0.1674765144152899 |
| Frozen Stage 7 engine reproduces the published Stage 7 targets on this machine | PASS | P0_strategic: 583 decisions, max |dw| 5.0e-11; P1_regime_aware: 583 decisions, max |dw| 5.0e-11; P2_ews_enhanced: 583 decisions, max |dw| 5.0e-11 |
| Ledger reconciliation on Train/Validation (all identities, all portfolios) | PASS | 50 identity checks; failed: none |
| Train/Validation decisions satisfy every hard constraint | PASS | 0 decisions with a hard violation |
| No pending order at the close of the last Validation session (boundary handled by the calendar rule) | PASS | pending: none; last Validation session 2019-12-31; the week of 2019-12-30 is completed with Test sessions in Phase 8B |
| Ledger-holdings decisions close to the Stage 7 proxy decisions (differences only from realized drift) | PASS | P0_strategic: mean |dw| 1.1e-03, max 0.009, same status 77.9%; P1_regime_aware: mean |dw| 7.6e-04, max 0.009, same status 84.9%; P2_ews_enhanced: mean |dw| 6.3e-04, max 0.009, same status 88.2% |
| Future returns cannot alter past decisions or holdings (Validation perturbation test) | PASS | returns after 2018-01-05 multiplied by -3; decisions and ledger up to that date bit-identical |
| No excluded session inside the pre-Test ledger window | PASS | 0 excluded sessions between 2008-10-31 and 2019-12-31 |
| Split manifest lists no excluded session after the last Validation date (a missing Test return would be an omitted holding return) | PASS | 0 excluded dates (metadata of split_manifest.json only; no Test value read) |
| Analytic ledger examples (two-asset delayed rebalance, zero trades, drift, cap, t+1, round trip) | PASS | 12 quantities; max error 2.2e-16 |
| Stage 8 unit tests pass before Test access (synthetic data; leakage-injection cases included) | PASS | 59 tests, 0 failures, 0 errors, 0 skipped |

## 15. Analytic ledger examples

| example | quantity | expected | computed | abs_error | status |
|---|---|---|---|---|---|
| E1 two-asset delayed rebalance | pre-trade wealth at the execution close | [1.02] | [1.02] | 0.0 | PASS |
| E1 two-asset delayed rebalance | gross notional at execution | [0.029411764706] | [0.029411764706] | 0.0 | PASS |
| E1 two-asset delayed rebalance | net wealth at day 3 (10 bps) | [1.03526955] | [1.03526955] | 2.220446049250313e-16 | PASS |
| E1 two-asset delayed rebalance | day-2 gross return earned by the OLD holdings | [-0.028571428571] | [-0.028571428571] | 3.469446951953614e-18 | PASS |
| E1 two-asset delayed rebalance | zero-cost scenario equals gross wealth | [1.0353] | [1.0353] | 0.0 | PASS |
| E2 zero trades | terminal wealth = sum w0 x price relative | [1.03008] | [1.03008] | 0.0 | PASS |
| E4 drift without rebalancing | terminal weights = w0 x price relative / sum | [0.588187325256, 0.411812674744] | [0.588187325256, 0.411812674744] | 1.1102230246251565e-16 | PASS |
| E2 zero trades | no transaction cost | [0.0] | [0.0] | 0.0 | PASS |
| E5 turnover-capped execution | executed weights | [0.6, 0.4] | [0.6, 0.4] | 0.0 | PASS |
| E5 turnover-capped execution | one-way turnover = cap | [0.1] | [0.1] | 2.7755575615628914e-17 | PASS |
| E6 alert on t executed at t+1 | gross returns (t+1 still fully exposed, t+2 in BIL) | [-0.05, -0.05, 0.0] | [-0.05, -0.05, 0.0] | 0.0 | PASS |
| E7 round trip | identical wealth under repeated simulation | [[1.0, 1.0, 1.0], [1.05, 1.05, 1.05], [1.019694, 1.019388, 1.01847], [1.036368996, 1.036057992, 1.03512498]] | [[1.0, 1.0, 1.0], [1.05, 1.05, 1.05], [1.019694, 1.019388, 1.01847], [1.036368996, 1.036057992, 1.03512498]] | 0.0 | PASS |

## 16. Holdings carried into Test (pre-Test information only)

| portfolio | SPY | EFA | EEM | IEF | TIP | HYG | GLD | DBC | BIL | last_pre_test_decision |
|---|---|---|---|---|---|---|---|---|---|---|
| P0_strategic | 0.3558 | 0.0997 | 0.0512 | 0.1456 | 0.0979 | 0.0495 | 0.0808 | 0.0709 | 0.0486 | 2019-12-27 |
| P1_regime_aware | 0.3558 | 0.0997 | 0.0512 | 0.1456 | 0.0979 | 0.0495 | 0.0808 | 0.0709 | 0.0486 | 2019-12-27 |
| P2_ews_enhanced | 0.3528 | 0.1005 | 0.0522 | 0.1458 | 0.0979 | 0.0496 | 0.0813 | 0.0709 | 0.0489 | 2019-12-27 |
| B1_60_40 | 0.6035 | 0.0 | 0.0 | 0.3965 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 2019-12-27 |
| B2_constant_defensiveness | 0.3344 | 0.0945 | 0.0479 | 0.1395 | 0.0962 | 0.0512 | 0.0757 | 0.067 | 0.0939 | 2019-12-27 |
| B3_strategic_buy_and_hold | 0.35 | 0.1 | 0.05 | 0.15 | 0.1 | 0.05 | 0.08 | 0.07 | 0.05 | none (bought at the last Validation close) |

Initial-state hash (recorded in the lock): `6a334d65edfc37e5ba6667b171f5a272735ce78f41110297ed7fceab7320fe16`

## 17. Hash snapshot

54 code files and 20 configuration / artifact files are hashed in the lock.
