# Stage 8 - Final sequential out-of-sample portfolio backtest

## STAGE 8: WARN - FINAL RESEARCH STATUS: COMPLETE WITH LIMITATIONS

Checks: 41 PASS, 2 WARN, 0 FAIL. Evaluation mode: **first_evaluation**. Pre-Test lock `73998035e0857aad...`. Test 2020-01-02 to 2026-10-19 (1708 sessions, 6.78 years).

**Prespecified outcome classification: Outcome 4 - Evidence is inconclusive**

## 1. Executive summary

- Net of 10 bps (x2 in cost stress), Test CAGR / volatility / Sharpe / max drawdown: P0 strategic 9.8% / 11.1% / 0.65 / -20.1%; P1 HMM regime 9.4% / 10.3% / 0.66 / -18.7%; P2 Full EWS 8.8% / 9.6% / 0.64 / -17.3%.
- P1 vs P0: net Sharpe difference 0.01 (95% stationary-bootstrap interval -0.02 to 0.03).
- P2 vs P0: net Sharpe difference -0.01 (95% stationary-bootstrap interval -0.17 to 0.13).
- P2 vs P1: net Sharpe difference -0.02 (95% stationary-bootstrap interval -0.17 to 0.12).
- Implementation: annualized one-way turnover / cost drag at 10 bps: P0 strategic 0.16 / 0.04% per year; P1 HMM regime 0.39 / 0.09% per year; P2 Full EWS 1.42 / 0.32% per year.
- Maximum drawdown: YES vs P0 (threshold 2.0 pp); vs P1: NO.
- Stress periods: net return: P2 above P1 in 1 and above P0 in 1 of 4 fixed windows; shallower in-window drawdown than P1 in 1; in the 11 objective SPY drawdown events P2 was already defensive (executed) before the onset in 9, de-risked only after the onset in 1, and did not de-risk in 1.
- Exposure vs timing: P2 Full EWS vs P0 strategic: exposure-driven (average weight difference dominates) (gross active -0.81%/yr = exposure -0.44% + timing -0.37%; beta 0.85); P2 Full EWS vs P1 HMM regime: timing component larger but not statistically distinguishable (alpha interval contains 0) (gross active -0.36%/yr = exposure 0.02% + timing -0.39%; beta 0.93); P1 HMM regime vs P0 strategic: exposure-driven (average weight difference dominates) (gross active -0.45%/yr = exposure -0.46% + timing 0.02%; beta 0.92).
- Costs: P1 vs P0: sign of the net Sharpe difference changes with the cost level; P2 vs P0: sign of the net Sharpe difference changes with the cost level; P2 vs P1: not higher at any cost level.
- Independent evidence is limited: 11 SPY drawdown events (>= 5%) in Test, 4 of them >= 10%; a single crisis can dominate the comparison.

## 2. Research question and investment mandate

Does the Full EWS-enhanced portfolio (P2) improve realized downside protection and risk-adjusted performance beyond the strategic benchmark (P0) and the simpler HMM-based allocation (P1), after the conservative t+1 execution delay and transaction costs?

Moderate-risk multi-asset client; nine US-listed ETFs; long only, no leverage; at most one rebalance per week. Benchmark (frozen): SPY 35%, EFA 10%, EEM 5%, IEF 15%, TIP 10%, HYG 5%, GLD 8%, DBC 7%, BIL 5%. Stage 7 constraints are PROPOSED research assumptions (not an approved mandate).

## 3. Research pipeline (Stages 0-8)

| stage | content |
|---|---|
| 0-2 | data download, point-in-time panels (ALFRED first releases where available) |
| 3 | five causal factors; Train 2007-2015 / Validation 2016-2019 / Test 2020- split (protocol v1.2) |
| 4 | regime exploration (GMM / K-Means) |
| 5 | frozen Gaussian HMM K=3 (hmm_std_diag_k3), forward filter |
| 5.5 | AR(1) and AR-HMM robustness (variance-floor M-step), frozen |
| 6 / 6.1 | BOCPD, EWS score, 5-level alert machine; audit and role-separated handoff |
| 7 | robust MVO, dynamic risk budgets, weekly target_then_trade policy (Train/Validation only) |
| 8 | pre-Test lock, then ONE sequential Test evaluation with a daily ledger |

## 4. Data partitions and Test protocol

- Train 2008-10-31 is the first decision (warm-up 252 sessions); Train/Validation history is used only to carry holdings forward to 2019-12-31 (no pre-Test performance is reported).
- Test 2020-01-02 - 2026-10-19: last valid session of the frozen local split (not chosen). Opened only through the lock-gated evaluation loader; access log: #1 2026-10-21T11:53:27+00:00 (first_evaluation).
- Boundary: the week of 2019-12-30 was incomplete in Stage 7; the frozen weekly rule completes it with Test sessions, so the first Test decision is 2020-01-03 (executed 2020-01-06).
- Wealth of every strategy = 1.0 at the close of 2019-12-31; the first Test return is 2020-01-02.

## 5. Model freeze and pre-Test lock

`data/processed/backtesting/stage8_pretest_lock.json` (fingerprint `73998035e0857aad7a9ccaa091f410166b9c9f369c05896f3f18aec684e012a6`) was written before Test was opened and records the protocol, every code / configuration / frozen-artifact hash, the frozen policies and constants, the initial state and the preflight results. Before reading Test the loader re-verified every hash; after the evaluation the hashes were checked again (section 25 checks).
Preflight: 30 PASS, 0 WARN, 0 FAIL (details in `reports/stage8_pretest_protocol.md`).

| portfolio | decisions | max_abs_weight_difference | same_status |
|---|---|---|---|
| P0_strategic | 583 | 4.999663183458125e-11 | True |
| P1_regime_aware | 583 | 4.998182423499031e-11 | True |
| P2_ews_enhanced | 583 | 4.99736363401837e-11 | True |

Frozen constants: P1_parity_scale_a = 0.28269502197950724, cost_stress_threshold_spy_21d = 0.30517114338438883, B2_constant_d = 0.1674765144152899, baseline_q80_threshold = 1.0, first_decision_date = '2008-10-31', last_validation_session = '2019-12-31'

## 6. Point-in-time limitations

- Factors are point-in-time from Stage 2/3 (ALFRED first releases where available; rule-based lags otherwise); revisions not captured by first-release data remain a limitation.
- ETF returns are adjusted-close total returns. Executing at the adjusted close of t+1 is an accounting convention; it does not prove that those prices were obtainable fills (no bid-ask, no market impact beyond the cost assumption, dividends assumed reinvested).
- Train signals come from models fitted on all of Train (retrospective); Validation was reused in Stages 5-7. Only the Test period is an untouched evaluation.

## 7. Frozen portfolio strategies

| id | rule |
|---|---|
| P0_strategic | frozen strategic benchmark, weekly 1% no-trade band, turnover cap 20% (Stage 7 rule, posture = benchmark) |
| P1_regime_aware | role baseline_regime only; d1 = min(1, a * baseline_p_non_normal), a from the Stage 7 handoff (Train-calibrated) |
| P2_ews_enhanced | role challenger_ews only; d2 = level map GREEN 0, AMBER 1/3, ORANGE 2/3, RED 1, RECOVERY 1/3 |
| B1_60_40 | same weekly calendar and conservative_close_t_plus_1 execution; trade back to 60/40 when the one-way distance exceeds the 1% no-trade band; no turnover cap; same cost model; outside the nine-ETF mandate limits (contextual only) |
| B2_constant_defensiveness | identical optimizer, constraints, calendar, execution and costs as P1/P2 with a CONSTANT defensiveness d |
| B3_strategic_buy_and_hold | strategic weights held from the close of the last Validation session, never rebalanced, no costs (contextual only; not a substitute for P0) |

P1 scale a = 0.28269502197950724 (Train-calibrated in Stage 7); B2 constant d = 0.1674765144152899 (mean Train defensiveness of P1 and P2).

## 8. Sequential signal-generation framework

Sequential out-of-sample portfolio backtest with FROZEN regime / early-warning models (no refit, no retraining schedule) and causally re-estimated portfolio risk inputs (Ledoit-Wolf covariance re-estimated at every weekly decision from returns through the decision date). The Stage 6 generator `early_warning.build_signals` is reused unchanged and run once over Train + Validation + Test: the HMM forward filter, AR(1) innovations, AR-HMM surprise, BOCPD run-length posterior and the alert state machine all continue chronologically; nothing is reset at the Test boundary. This is NOT monthly HMM retraining: no parameter is re-estimated in Test. Verified: prefix invariance through a Test date, invariance of earlier signals to perturbed later factors, and bit-identity with the Stage 6.1 handoff on Train/Validation.

## 9. Portfolio optimization and execution conventions

- **optimizer**: Stage 7 primary policy (portfolio_optimizer.solve_policy, target_then_trade), reused without modification
- **pre_trade_weights_at_decision**: actual ledger holdings at the close of the decision date t (known at t). They equal the Stage 7 ex-ante proxy (previous target drifted through t) whenever the previous order was executed in full; they differ only after a hold inside the no-trade band or an execution-day projection (the Stage 7 proxy would then ignore realized drift).
- **risk_inputs**: Ledoit-Wolf 504 sessions, minimum 252, equilibrium expected returns (SR_b 0.30), all from simple returns dated <= t
- **signals**: date-t value of the portfolio's own role; intra-week signals are not acted upon
- **convention**: conservative_close_t_plus_1: decision after the close of t; execution at the close of t+1; first return of the new holdings is that of t+2; holdings held before the execution earn the return of t+1
- **hold_orders**: a decision inside the no-trade band is an explicit no-trade order (no execution, holdings keep drifting)
- **trade_orders**: at the close of t+1 compute the actual pre-trade weights (drift through t+1). If 0.5*sum|target - pre| <= the decision's turnover cap, execute the decision-time target exactly; otherwise execute pre + (cap / TO) * (target - pre) (cap-preserving straight-line projection, the Stage 7 transition rule). No t+1 signal is used.
- **price_assumption**: adjusted close of t+1 (adjusted total-return series; not proof of historical fills)
- **costs**: deducted from wealth at the execution close, pro rata across holdings, so post-trade weights equal the executed target

Implementation ambiguities resolved before Test:

- Pre-trade weights at a decision are the ACTUAL ledger holdings at the close of t (known at t). The Stage 7 proxy (previous target drifted through t) equals them whenever the previous order was executed in full; after a hold inside the no-trade band it would ignore the realized drift of the holding period (consecutive holds would accumulate the error). Train/Validation difference vs the published Stage 7 targets: P0_strategic mean |dw| 1.1e-03; P1_regime_aware mean |dw| 7.6e-04; P2_ews_enhanced mean |dw| 6.3e-04.
- A hold decision is a no-trade order: nothing is executed at t+1 even if drift moves the holdings further; the next weekly decision re-evaluates them.
- Execution-day turnover above the decision's cap (drift between t and t+1) is handled by the Stage 7 straight-line transition rule (cap-preserving projection) and reported as an exception.
- The cost-stress flag is evaluated at the decision date t (Stage 7: 'known at the decision date') and applied to the t+1 execution; it is the same for every portfolio.
- The calendar week containing the last Validation session is completed with Test sessions; its decision (last session of that week) is the first Test-period decision. No other decision is created.
- Costs are deducted pro rata at the execution close; therefore the trading path is identical under 0 / 5 / 10 / 25 bps and every cost scenario is evaluated on the same holdings.

## 10. Actual daily holdings and rebalancing

| portfolio | decisions | hold_decisions_share | executions_with_trades | rebalances_per_year | mean_defensiveness_d | share_decisions_defensive_d_gt_0 | mean_weight_equity | mean_weight_duration | mean_weight_credit | mean_weight_real_assets | mean_weight_BIL |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P0_strategic | 355 | 0.777 | 79 | 11.656 |  |  | 0.501 | 0.249 | 0.050 | 0.151 | 0.050 |
| P1_regime_aware | 355 | 0.710 | 103 | 15.197 | 0.166 | 1.000 | 0.469 | 0.243 | 0.052 | 0.142 | 0.093 |
| P2_ews_enhanced | 355 | 0.651 | 124 | 18.295 | 0.156 | 0.287 | 0.470 | 0.249 | 0.051 | 0.144 | 0.086 |
| B1_60_40 | 355 | 0.839 | 57 | 8.410 |  |  | 0.602 | 0.398 | 0.000 | 0.000 | 0.000 |
| B2_constant_defensiveness | 355 | 0.789 | 75 | 11.066 | 0.167 | 1.000 | 0.470 | 0.242 | 0.052 | 0.142 | 0.094 |
| B3_strategic_buy_and_hold |  |  | 0 | 0.000 |  |  | 0.548 | 0.201 | 0.041 | 0.168 | 0.041 |

Holdings at the start of Test (carried from the pre-Test continuation; recorded in the lock):

| portfolio | SPY | EFA | EEM | IEF | TIP | HYG | GLD | DBC | BIL | last_pre_test_decision |
|---|---|---|---|---|---|---|---|---|---|---|
| P0_strategic | 0.3558 | 0.0997 | 0.0512 | 0.1456 | 0.0979 | 0.0495 | 0.0808 | 0.0709 | 0.0486 | 2019-12-27 |
| P1_regime_aware | 0.3558 | 0.0997 | 0.0512 | 0.1456 | 0.0979 | 0.0495 | 0.0808 | 0.0709 | 0.0486 | 2019-12-27 |
| P2_ews_enhanced | 0.3528 | 0.1005 | 0.0522 | 0.1458 | 0.0979 | 0.0496 | 0.0813 | 0.0709 | 0.0489 | 2019-12-27 |
| B1_60_40 | 0.6035 | 0.0 | 0.0 | 0.3965 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 2019-12-27 |
| B2_constant_defensiveness | 0.3344 | 0.0945 | 0.0479 | 0.1395 | 0.0962 | 0.0512 | 0.0757 | 0.067 | 0.0939 | 2019-12-27 |
| B3_strategic_buy_and_hold | 0.35 | 0.1 | 0.05 | 0.15 | 0.1 | 0.05 | 0.08 | 0.07 | 0.05 | none (bought at the last Validation close) |

## 11. Transaction-cost methodology

- **primary_bps**: 10
- **sensitivity_bps**: [5, 25]
- **basis**: bps of GROSS traded notional = sum_i |w_post,i - w_pre,i| = 2 x one-way turnover
- **stress_multiplier**: 2.0
- **stress_indicator**: trailing 21-session SPY realized volatility (simple returns through the decision date t, annualized) above the Train-calibrated threshold in the Stage 7 handoff; evaluated at t, applied to the execution at t+1; identical for every portfolio; uses no HMM / EWS role signal
- Train-calibrated stress threshold (annualized 21-session SPY volatility): 0.3052.

## 12. Test-period performance

Net of 10 bps:

| portfolio | cumulative_return | cagr | arithmetic_annual_return | annual_volatility | sharpe | sortino | calmar | max_drawdown | max_drawdown_sessions_underwater | worst_day | worst_week | worst_month | var95_daily | cvar95_daily | downside_deviation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P0_strategic | 88.63% | 9.82% | 9.99% | 11.13% | 0.652 | 0.907 | 0.489 | -20.08% | 102 | -5.99% | -8.08% | -7.61% | 0.98% | 1.65% | 8.00% |
| P1_regime_aware | 83.55% | 9.37% | 9.49% | 10.26% | 0.659 | 0.916 | 0.502 | -18.69% | 105 | -5.40% | -7.41% | -6.73% | 0.93% | 1.52% | 7.38% |
| P2_ews_enhanced | 77.07% | 8.80% | 8.90% | 9.64% | 0.640 | 0.889 | 0.508 | -17.30% | 534 | -3.97% | -5.80% | -7.31% | 0.94% | 1.46% | 6.94% |
| B1_60_40 | 81.42% | 9.19% | 9.53% | 12.18% | 0.559 | 0.788 | 0.432 | -21.28% | 542 | -5.79% | -8.18% | -7.43% | 1.11% | 1.81% | 8.64% |
| B2_constant_defensiveness | 83.30% | 9.35% | 9.49% | 10.48% | 0.645 | 0.897 | 0.491 | -19.07% | 102 | -5.64% | -7.68% | -7.17% | 0.92% | 1.56% | 7.54% |
| B3_strategic_buy_and_hold | 94.31% | 10.30% | 10.46% | 11.46% | 0.674 | 0.939 | 0.546 | -18.86% | 106 | -5.37% | -7.76% | -7.40% | 1.07% | 1.72% | 8.24% |

Calendar-year net returns:

| year | P0_strategic | P1_regime_aware | P2_ews_enhanced | B1_60_40 | B2_constant_defensiveness | B3_strategic_buy_and_hold |
|---|---|---|---|---|---|---|
| 2020 | 13.43% | 12.14% | 14.15% | 16.58% | 12.68% | 12.29% |
| 2021 | 13.20% | 12.76% | 12.77% | 14.96% | 12.41% | 13.75% |
| 2022 | -11.21% | -10.21% | -11.60% | -16.68% | -10.64% | -11.95% |
| 2023 | 13.69% | 13.08% | 13.65% | 16.81% | 13.11% | 14.16% |
| 2024 | 12.28% | 12.00% | 9.64% | 14.35% | 11.85% | 14.26% |
| 2025 | 18.95% | 17.71% | 14.61% | 14.24% | 18.04% | 20.43% |
| 2026 | 8.96% | 8.44% | 8.96% | 6.46% | 8.44% | 9.98% |

## 13. P0 versus P1

|  | active_return_annualized | tracking_error | information_ratio | cagr_difference | sharpe_difference | sortino_difference | max_drawdown_difference | up_capture_vs_b | down_capture_vs_b |
|---|---|---|---|---|---|---|---|---|---|
| P1 HMM regime - P0 strategic | -0.50% | 0.98% | -0.505 | -0.44% | 0.007 | 0.009 | 1.39% | 0.934 | 0.931 |

| statistic | method | estimate | ci_low | ci_high | p_value | p_value_holm |
|---|---|---|---|---|---|---|
| mean_active_return | Newey-West HAC | -0.0050 | -0.0118 | 0.0018 | 0.1527 | 0.4581 |
| sharpe_difference | stationary bootstrap, mean block 21 (primary) | 0.0068 | -0.0220 | 0.0336 | 0.6940 |  |
| sortino_difference | stationary bootstrap, mean block 21 (primary) | 0.0090 | -0.0362 | 0.0490 | 0.7408 |  |
| cagr_difference | stationary bootstrap, mean block 21 (primary) | -0.0044 | -0.0119 | 0.0025 | 0.2188 |  |
| volatility_difference | stationary bootstrap, mean block 21 (primary) | -0.0087 | -0.0117 | -0.0058 | 0.0000 |  |
| max_drawdown_difference | stationary bootstrap, mean block 21 (primary) | 0.0139 | 0.0057 | 0.0264 | 0.0016 |  |
| mean_annualized_difference | stationary bootstrap, mean block 21 (primary) | -0.0050 | -0.0111 | 0.0013 | 0.1176 |  |
| sharpe_difference | stationary bootstrap, mean block 63 (secondary) | 0.0068 | -0.0192 | 0.0318 | 0.6832 |  |
| sortino_difference | stationary bootstrap, mean block 63 (secondary) | 0.0090 | -0.0295 | 0.0454 | 0.7248 |  |
| cagr_difference | stationary bootstrap, mean block 63 (secondary) | -0.0044 | -0.0110 | 0.0017 | 0.1568 |  |
| volatility_difference | stationary bootstrap, mean block 63 (secondary) | -0.0087 | -0.0121 | -0.0052 | 0.0000 |  |
| max_drawdown_difference | stationary bootstrap, mean block 63 (secondary) | 0.0139 | 0.0069 | 0.0243 | 0.0032 |  |
| mean_annualized_difference | stationary bootstrap, mean block 63 (secondary) | -0.0050 | -0.0104 | 0.0006 | 0.0744 |  |
| alpha_on_reference_excess | OLS, Newey-West HAC | 0.0008 | -0.0024 | 0.0039 | 0.6355 |  |
| beta_on_reference_excess | OLS, Newey-West HAC (p-value vs beta = 1) | 0.9212 | 0.9139 | 0.9286 | 0.0000 |  |

**Does P1 outperform P0 on risk-adjusted performance after costs?** NOT DEMONSTRATED - higher point estimate, but the 95% interval contains 0.

## 14. P1 versus P2

|  | active_return_annualized | tracking_error | information_ratio | cagr_difference | sharpe_difference | sortino_difference | max_drawdown_difference | up_capture_vs_b | down_capture_vs_b |
|---|---|---|---|---|---|---|---|---|---|
| P2 Full EWS - P1 HMM regime | -0.59% | 1.76% | -0.338 | -0.58% | -0.019 | -0.027 | 1.39% | 0.975 | 0.982 |

| statistic | method | estimate | ci_low | ci_high | p_value | p_value_holm |
|---|---|---|---|---|---|---|
| mean_active_return | Newey-West HAC | -0.0059 | -0.0178 | 0.0059 | 0.3264 | 0.4581 |
| sharpe_difference | stationary bootstrap, mean block 21 (primary) | -0.0190 | -0.1703 | 0.1154 | 0.7000 |  |
| sortino_difference | stationary bootstrap, mean block 21 (primary) | -0.0266 | -0.2768 | 0.1771 | 0.7076 |  |
| cagr_difference | stationary bootstrap, mean block 21 (primary) | -0.0058 | -0.0203 | 0.0084 | 0.4056 |  |
| volatility_difference | stationary bootstrap, mean block 21 (primary) | -0.0062 | -0.0152 | 0.0015 | 0.1680 |  |
| max_drawdown_difference | stationary bootstrap, mean block 21 (primary) | 0.0139 | -0.0237 | 0.0501 | 0.6412 |  |
| mean_annualized_difference | stationary bootstrap, mean block 21 (primary) | -0.0059 | -0.0191 | 0.0066 | 0.3500 |  |
| sharpe_difference | stationary bootstrap, mean block 63 (secondary) | -0.0190 | -0.1809 | 0.1160 | 0.7224 |  |
| sortino_difference | stationary bootstrap, mean block 63 (secondary) | -0.0266 | -0.2933 | 0.1729 | 0.7328 |  |
| cagr_difference | stationary bootstrap, mean block 63 (secondary) | -0.0058 | -0.0209 | 0.0080 | 0.4224 |  |
| volatility_difference | stationary bootstrap, mean block 63 (secondary) | -0.0062 | -0.0162 | 0.0022 | 0.2544 |  |
| max_drawdown_difference | stationary bootstrap, mean block 63 (secondary) | 0.0139 | -0.0257 | 0.0481 | 0.7180 |  |
| mean_annualized_difference | stationary bootstrap, mean block 63 (secondary) | -0.0059 | -0.0195 | 0.0062 | 0.3552 |  |
| alpha_on_reference_excess | OLS, Newey-West HAC | -0.0010 | -0.0138 | 0.0119 | 0.8836 |  |
| beta_on_reference_excess | OLS, Newey-West HAC (p-value vs beta = 1) | 0.9264 | 0.8648 | 0.9881 | 0.0193 |  |

**Does P2 outperform P1 on risk-adjusted performance after costs?** NO - net Sharpe ratio not higher.

## 15. Gross versus net performance

| portfolio | gross CAGR | net CAGR (10 bps) | cost drag (CAGR points) | gross Sharpe | net Sharpe |
|---|---|---|---|---|---|
| P0_strategic | 9.86% | 9.82% | 0.04% | 0.655 | 0.652 |
| P1_regime_aware | 9.47% | 9.37% | 0.09% | 0.667 | 0.659 |
| P2_ews_enhanced | 9.14% | 8.80% | 0.35% | 0.673 | 0.640 |
| B1_60_40 | 9.22% | 9.19% | 0.03% | 0.561 | 0.559 |
| B2_constant_defensiveness | 9.39% | 9.35% | 0.04% | 0.649 | 0.645 |
| B3_strategic_buy_and_hold | 10.30% | 10.30% | 0.00% | 0.674 | 0.674 |

## 16. Drawdown and downside-risk comparison

| portfolio | peak_date | trough_date | recovery_date | depth | sessions_peak_to_trough | sessions_underwater |
|---|---|---|---|---|---|---|
| P0_strategic | 2020-02-19 | 2020-03-23 | 2020-07-15 | -20.08% | 23 | 102 |
| P0_strategic | 2021-12-27 | 2022-09-30 | 2023-12-19 | -16.85% | 192 | 498 |
| P0_strategic | 2025-02-20 | 2025-04-08 | 2025-05-12 | -8.97% | 33 | 56 |
| P1_regime_aware | 2020-02-19 | 2020-03-23 | 2020-07-20 | -18.69% | 23 | 105 |
| P1_regime_aware | 2021-12-27 | 2022-09-30 | 2023-12-14 | -15.45% | 192 | 495 |
| P1_regime_aware | 2025-02-20 | 2025-04-08 | 2025-05-12 | -8.02% | 33 | 56 |
| P2_ews_enhanced | 2021-11-09 | 2022-09-30 | 2023-12-26 | -17.30% | 224 | 534 |
| P2_ews_enhanced | 2020-02-19 | 2020-03-18 | 2020-07-08 | -15.45% | 20 | 97 |
| P2_ews_enhanced | 2025-02-20 | 2025-04-08 | 2025-06-26 | -8.57% | 33 | 87 |

Does P2 reduce maximum drawdown by an economically meaningful amount? YES vs P0 (threshold 2.0 pp); vs P1: NO. Reduction vs P0 2.78 pp (bootstrap 95% interval of the drawdown difference -0.93% to 7.16%; approximate, resampled paths do not keep the crash sequence).

## 17. Risk-adjusted performance

| portfolio | sharpe | sortino | calmar | information_ratio_vs_P0 | information_ratio_vs_P1 |
|---|---|---|---|---|---|
| P0_strategic | 0.652 | 0.907 | 0.489 |  | 0.505 |
| P1_regime_aware | 0.659 | 0.916 | 0.502 | -0.505 |  |
| P2_ews_enhanced | 0.640 | 0.889 | 0.508 | -0.454 | -0.338 |
| B1_60_40 | 0.559 | 0.788 | 0.432 | -0.128 | 0.011 |
| B2_constant_defensiveness | 0.645 | 0.897 | 0.491 | -0.754 | 0.006 |
| B3_strategic_buy_and_hold | 0.674 | 0.939 | 0.546 | 0.351 | 0.552 |

Risk matching (regression of excess returns on the reference's excess returns, Newey-West):

| portfolio | reference | alpha_annualized | alpha_ci_low | alpha_ci_high | alpha_p_value | beta | beta_ci_low | beta_ci_high | r_squared |
|---|---|---|---|---|---|---|---|---|---|
| P1_regime_aware | P0_strategic | 0.08% | -0.24% | 0.39% | 0.635 | 0.921 | 0.914 | 0.929 | 0.998 |
| P2_ews_enhanced | P0_strategic | -0.01% | -1.40% | 1.37% | 0.985 | 0.852 | 0.789 | 0.915 | 0.967 |
| B2_constant_defensiveness | P0_strategic | -0.07% | -0.14% | -0.01% | 0.035 | 0.942 | 0.941 | 0.943 | 1.000 |
| P2_ews_enhanced | P1_regime_aware | -0.10% | -1.38% | 1.19% | 0.884 | 0.926 | 0.865 | 0.988 | 0.973 |

DESCRIPTIVE ex-post volatility normalization (scaled to P0's realized volatility; not a tradable strategy):

| portfolio_a | portfolio_b | descriptive_vol_matched_cagr_a | descriptive_vol_matched_max_drawdown_a |
|---|---|---|---|
| P1_regime_aware | P0_strategic | 9.90% | -20.17% |
| P2_ews_enhanced | P0_strategic | 9.67% | -19.84% |
| P2_ews_enhanced | P1_regime_aware | 9.67% | -19.84% |
| P1_regime_aware | B2_constant_defensiveness | 9.90% | -20.17% |
| P2_ews_enhanced | B2_constant_defensiveness | 9.67% | -19.84% |

## 18. Crisis-event analysis

| window | portfolio | gross_cumulative_return | net_cumulative_return | max_drawdown_in_window | equity_weight_at_start | equity_weight_min | bil_weight_max | trades | cost_primary |
|---|---|---|---|---|---|---|---|---|---|
| 2020_pandemic_shock | P0_strategic | -5.44% | -5.49% | -20.08% | 0.492 | 0.478 | 0.054 | 8 | 0.05% |
| 2020_pandemic_shock | P1_regime_aware | -4.98% | -5.05% | -18.69% | 0.492 | 0.430 | 0.135 | 9 | 0.08% |
| 2020_pandemic_shock | P2_ews_enhanced | -3.06% | -3.32% | -15.45% | 0.491 | 0.284 | 0.263 | 10 | 0.27% |
| 2020_pandemic_shock | B1_60_40 | -1.85% | -1.91% | -19.27% | 0.595 | 0.562 | 0.000 | 6 | 0.06% |
| 2020_pandemic_shock | B2_constant_defensiveness | -5.06% | -5.12% | -19.07% | 0.461 | 0.448 | 0.102 | 8 | 0.06% |
| 2020_pandemic_shock | B3_strategic_buy_and_hold | -5.49% | -5.49% | -18.86% | 0.495 | 0.415 | 0.060 | 0 | 0.00% |
| 2020_2021_recovery | P0_strategic | 36.09% | 36.04% | -4.85% | 0.514 | 0.482 | 0.052 | 15 | 0.04% |
| 2020_2021_recovery | P1_regime_aware | 33.47% | 33.35% | -4.55% | 0.461 | 0.431 | 0.116 | 21 | 0.09% |
| 2020_2021_recovery | P2_ews_enhanced | 33.91% | 33.34% | -4.61% | 0.302 | 0.293 | 0.252 | 30 | 0.43% |
| 2020_2021_recovery | B1_60_40 | 34.85% | 34.80% | -5.81% | 0.611 | 0.579 | 0.000 | 13 | 0.03% |
| 2020_2021_recovery | B2_constant_defensiveness | 33.68% | 33.62% | -4.59% | 0.483 | 0.451 | 0.090 | 14 | 0.04% |
| 2020_2021_recovery | B3_strategic_buy_and_hold | 35.33% | 35.33% | -4.87% | 0.466 | 0.459 | 0.054 | 0 | 0.00% |
| 2022_inflation_tightening | P0_strategic | -11.17% | -11.21% | -16.79% | 0.503 | 0.479 | 0.054 | 14 | 0.04% |
| 2022_inflation_tightening | P1_regime_aware | -10.18% | -10.21% | -15.37% | 0.451 | 0.429 | 0.145 | 14 | 0.04% |
| 2022_inflation_tightening | P2_ews_enhanced | -11.14% | -11.60% | -17.15% | 0.446 | 0.297 | 0.252 | 29 | 0.52% |
| 2022_inflation_tightening | B1_60_40 | -16.65% | -16.68% | -20.95% | 0.608 | 0.580 | 0.000 | 8 | 0.03% |
| 2022_inflation_tightening | B2_constant_defensiveness | -10.60% | -10.64% | -15.97% | 0.475 | 0.450 | 0.109 | 14 | 0.04% |
| 2022_inflation_tightening | B3_strategic_buy_and_hold | -11.95% | -11.95% | -17.43% | 0.555 | 0.493 | 0.048 | 0 | 0.00% |
| 2023_onward_normalization | P0_strategic | 65.62% | 65.43% | -8.97% | 0.494 | 0.479 | 0.053 | 41 | 0.11% |
| 2023_onward_normalization | P1_regime_aware | 62.26% | 61.67% | -8.02% | 0.449 | 0.422 | 0.144 | 58 | 0.37% |
| 2023_onward_normalization | P2_ews_enhanced | 57.04% | 55.60% | -8.57% | 0.502 | 0.255 | 0.252 | 54 | 0.92% |
| 2023_onward_normalization | B1_60_40 | 62.60% | 62.46% | -10.54% | 0.607 | 0.571 | 0.000 | 30 | 0.09% |
| 2023_onward_normalization | B2_constant_defensiveness | 62.12% | 61.94% | -8.39% | 0.471 | 0.443 | 0.107 | 38 | 0.11% |
| 2023_onward_normalization | B3_strategic_buy_and_hold | 72.76% | 72.76% | -10.72% | 0.519 | 0.519 | 0.045 | 0 | 0.00% |

Signals inside each window:

| window | portfolio | hmm_p_non_normal_mean | ews_share_ge_amber | ews_share_red | ews_level_at_start | ews_max_level |
|---|---|---|---|---|---|---|
| 2020_pandemic_shock | P1_regime_aware | 0.800 |  |  |  |  |
| 2020_pandemic_shock | P2_ews_enhanced |  | 0.742 | 0.500 | GREEN | RED |
| 2020_2021_recovery | P1_regime_aware | 0.592 |  |  |  |  |
| 2020_2021_recovery | P2_ews_enhanced |  | 0.197 | 0.031 | ORANGE | RED |
| 2022_inflation_tightening | P1_regime_aware | 1.000 |  |  |  |  |
| 2022_inflation_tightening | P2_ews_enhanced |  | 0.442 | 0.056 | GREEN | RED |
| 2023_onward_normalization | P1_regime_aware | 0.485 |  |  |  |  |
| 2023_onward_normalization | P2_ews_enhanced |  | 0.129 | 0.039 | GREEN | RED |

Objective SPY drawdown events (frozen Stage 6 rule), 11 in Test:

| event | portfolio | spy_peak_date | spy_onset_date (-5%) | spy_trough_date | spy_trough_loss_pct | loss_peak_to_trough | equity_at_peak | equity_before_onset | equity_at_trough | first_de_risking_signal_date | first_de_risking_execution_date | de_risking_timing | recovery_return_63_sessions | protection_vs_P0 | recovery_participation_vs_P0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | P0_strategic | 2020-02-19 | 2020-02-25 | 2020-03-23 | -33.717 | -20.08% | 0.507 | 0.493 | 0.500 |  |  | none | 22.19% | 0.00% | 0.00% |
| 1 | P1_regime_aware | 2020-02-19 | 2020-02-25 | 2020-03-23 | -33.717 | -18.69% | 0.500 | 0.486 | 0.445 | 2020-02-28 | 2020-03-02 | after onset (reaction) | 20.08% | 1.39% | -2.11% |
| 1 | P2_ews_enhanced | 2020-02-19 | 2020-02-25 | 2020-03-23 | -33.717 | -15.38% | 0.507 | 0.493 | 0.372 | 2020-02-28 | 2020-03-02 | after onset (reaction) | 16.36% | 4.70% | -5.82% |
| 2 | P0_strategic | 2020-09-02 | 2020-09-08 | 2020-09-23 | -9.440 | -4.85% | 0.506 | 0.498 | 0.492 |  |  | none | 8.21% | 0.00% | 0.00% |
| 2 | P1_regime_aware | 2020-09-02 | 2020-09-08 | 2020-09-23 | -9.440 | -4.43% | 0.459 | 0.451 | 0.442 | 2020-08-14 | 2020-08-17 | before onset (executable protection) | 7.39% | 0.42% | -0.83% |
| 2 | P2_ews_enhanced | 2020-09-02 | 2020-09-08 | 2020-09-23 | -9.440 | -4.61% | 0.445 | 0.437 | 0.492 | 2020-08-28 | 2020-08-31 | before onset (executable protection) | 8.21% | 0.24% | 0.00% |
| 3 | P0_strategic | 2020-09-02 | 2020-10-27 | 2020-10-30 | -8.346 | -4.76% | 0.506 | 0.502 | 0.495 |  |  | none | 11.05% | 0.00% | 0.00% |
| 3 | P1_regime_aware | 2020-09-02 | 2020-10-27 | 2020-10-30 | -8.346 | -4.55% | 0.459 | 0.461 | 0.454 | 2020-08-14 | 2020-08-17 | before onset (executable protection) | 10.43% | 0.21% | -0.62% |
| 3 | P2_ews_enhanced | 2020-09-02 | 2020-10-27 | 2020-10-30 | -8.346 | -4.52% | 0.445 | 0.502 | 0.495 | 2020-08-28 | 2020-08-31 | before onset (executable protection) | 11.05% | 0.24% | 0.00% |
| 4 | P0_strategic | 2021-09-02 | 2021-09-30 | 2021-10-04 | -5.114 | -2.61% | 0.507 | 0.499 | 0.493 |  |  | none | 4.75% | 0.00% | 0.00% |
| 4 | P1_regime_aware | 2021-09-02 | 2021-09-30 | 2021-10-04 | -5.114 | -2.38% | 0.454 | 0.456 | 0.447 | 2021-10-01 | 2021-10-04 | after onset (reaction) | 4.26% | 0.23% | -0.49% |
| 4 | P2_ews_enhanced | 2021-09-02 | 2021-09-30 | 2021-10-04 | -5.114 | -2.45% | 0.502 | 0.434 | 0.429 | 2021-08-20 | 2021-08-23 | before onset (executable protection) | 4.27% | 0.15% | -0.47% |
| 5 | P0_strategic | 2022-01-03 | 2022-01-19 | 2022-10-12 | -24.496 | -16.38% | 0.506 | 0.500 | 0.491 |  |  | none | 10.19% | 0.00% | 0.00% |
| 5 | P1_regime_aware | 2022-01-03 | 2022-01-19 | 2022-10-12 | -24.496 | -14.99% | 0.454 | 0.446 | 0.440 | 2022-03-18 | 2022-03-21 | after onset (reaction) | 9.25% | 1.39% | -0.95% |
| 5 | P2_ews_enhanced | 2022-01-03 | 2022-01-19 | 2022-10-12 | -24.496 | -16.75% | 0.500 | 0.492 | 0.491 | 2021-12-03 | 2021-12-06 | before onset (executable protection) | 10.20% | -0.36% | 0.01% |
| 6 | P0_strategic | 2022-01-03 | 2023-08-15 | 2023-10-27 | -11.667 | -7.85% | 0.506 | 0.499 | 0.487 |  |  | none | 10.52% | 0.00% | 0.00% |
| 6 | P1_regime_aware | 2022-01-03 | 2023-08-15 | 2023-10-27 | -11.667 | -7.29% | 0.454 | 0.500 | 0.487 | 2022-03-18 | 2022-03-21 | before onset (executable protection) | 10.52% | 0.56% | -0.00% |
| 6 | P2_ews_enhanced | 2022-01-03 | 2023-08-15 | 2023-10-27 | -11.667 | -8.25% | 0.500 | 0.499 | 0.487 | 2021-12-03 | 2021-12-06 | before onset (executable protection) | 10.52% | -0.41% | 0.00% |
| 7 | P0_strategic | 2024-03-27 | 2024-04-19 | 2024-04-19 | -5.354 | -2.25% | 0.502 | 0.490 | 0.488 |  |  | none | 6.43% | 0.00% | 0.00% |
| 7 | P1_regime_aware | 2024-03-27 | 2024-04-19 | 2024-04-19 | -5.354 | -2.25% | 0.502 | 0.490 | 0.488 |  |  | none | 6.39% | 0.00% | -0.05% |
| 7 | P2_ews_enhanced | 2024-03-27 | 2024-04-19 | 2024-04-19 | -5.354 | -2.25% | 0.502 | 0.490 | 0.488 |  |  | none | 6.43% | 0.00% | -0.00% |
| 8 | P0_strategic | 2024-07-16 | 2024-08-02 | 2024-08-05 | -8.406 | -4.15% | 0.505 | 0.495 | 0.500 |  |  | none | 5.66% | 0.00% | 0.00% |
| 8 | P1_regime_aware | 2024-07-16 | 2024-08-02 | 2024-08-05 | -8.406 | -3.68% | 0.451 | 0.441 | 0.448 | 2024-06-14 | 2024-06-17 | before onset (executable protection) | 5.21% | 0.47% | -0.45% |
| 8 | P2_ews_enhanced | 2024-07-16 | 2024-08-02 | 2024-08-05 | -8.406 | -3.94% | 0.505 | 0.435 | 0.424 | 2024-07-26 | 2024-07-29 | before onset (executable protection) | 2.93% | 0.21% | -2.73% |
| 9 | P0_strategic | 2025-02-19 | 2025-03-04 | 2025-04-08 | -18.755 | -8.95% | 0.499 | 0.500 | 0.498 |  |  | none | 15.04% | 0.00% | 0.00% |
| 9 | P1_regime_aware | 2025-02-19 | 2025-03-04 | 2025-04-08 | -18.755 | -8.01% | 0.449 | 0.440 | 0.444 | 2025-02-14 | 2025-02-18 | before onset (executable protection) | 13.53% | 0.94% | -1.51% |
| 9 | P2_ews_enhanced | 2025-02-19 | 2025-03-04 | 2025-04-08 | -18.755 | -8.56% | 0.502 | 0.438 | 0.279 | 2025-02-28 | 2025-03-03 | before onset (executable protection) | 10.67% | 0.40% | -4.36% |
| 10 | P0_strategic | 2025-10-29 | 2025-11-20 | 2025-11-20 | -5.071 | -2.21% | 0.504 | 0.494 | 0.490 |  |  | none | 8.04% | 0.00% | 0.00% |
| 10 | P1_regime_aware | 2025-10-29 | 2025-11-20 | 2025-11-20 | -5.071 | -1.95% | 0.444 | 0.434 | 0.431 | 2025-10-10 | 2025-10-13 | before onset (executable protection) | 7.64% | 0.26% | -0.40% |
| 10 | P2_ews_enhanced | 2025-10-29 | 2025-11-20 | 2025-11-20 | -5.071 | -1.91% | 0.434 | 0.424 | 0.420 | 2025-10-17 | 2025-10-20 | before onset (executable protection) | 7.65% | 0.31% | -0.39% |
| 11 | P0_strategic | 2026-01-27 | 2026-03-19 | 2026-03-30 | -8.885 | -4.29% | 0.501 | 0.495 | 0.489 |  |  | none | 7.35% | 0.00% | 0.00% |
| 11 | P1_regime_aware | 2026-01-27 | 2026-03-19 | 2026-03-30 | -8.885 | -3.89% | 0.501 | 0.437 | 0.432 | 2026-03-06 | 2026-03-09 | before onset (executable protection) | 6.63% | 0.40% | -0.73% |
| 11 | P2_ews_enhanced | 2026-01-27 | 2026-03-19 | 2026-03-30 | -8.885 | -3.81% | 0.501 | 0.427 | 0.422 | 2026-03-06 | 2026-03-09 | before onset (executable protection) | 6.74% | 0.48% | -0.62% |

Protection is credited only when the de-risking EXECUTION precedes the onset date; an alert executed after the -5% threshold is a reaction. Negative recovery participation shows a missed rebound.

Does P2 provide additional value in the predeclared stress periods? net return: P2 above P1 in 1 and above P0 in 1 of 4 fixed windows; shallower in-window drawdown than P1 in 1; in the 11 objective SPY drawdown events P2 was already defensive (executed) before the onset in 9, de-risked only after the onset in 1, and did not de-risk in 1.

## 19. Portfolio return attribution

**P2 Full EWS - P1 HMM regime** (annualized; reconciliation error 8.7e-18):

| component | group | annualized_mean | sum_daily | mean_weight_difference |
|---|---|---|---|---|
| asset allocation (total) | equity | -0.21% | -1.46% | 0.07% |
| average-exposure part | equity | 0.01% | 0.10% | 0.07% |
| timing part | equity | -0.23% | -1.55% | 0.07% |
| asset allocation (total) | duration | -0.09% | -0.64% | 0.53% |
| average-exposure part | duration | 0.01% | 0.07% | 0.53% |
| timing part | duration | -0.10% | -0.71% | 0.53% |
| asset allocation (total) | credit | 0.03% | 0.19% | -0.11% |
| average-exposure part | credit | -0.00% | -0.03% | -0.11% |
| timing part | credit | 0.03% | 0.22% | -0.11% |
| asset allocation (total) | gold | -0.04% | -0.30% | 0.02% |
| average-exposure part | gold | 0.00% | 0.02% | 0.02% |
| timing part | gold | -0.05% | -0.32% | 0.02% |
| asset allocation (total) | commodities | 0.00% | 0.02% | 0.12% |
| average-exposure part | commodities | 0.02% | 0.13% | 0.12% |
| timing part | commodities | -0.02% | -0.10% | 0.12% |
| asset allocation (total) | bil | -0.04% | -0.26% | -0.63% |
| average-exposure part | bil | -0.02% | -0.12% | -0.63% |
| timing part | bil | -0.02% | -0.15% | -0.63% |
| gross active return (sum of groups) | all | -0.36% | -2.45% | n/a |
| gross active return (ledger) | all | -0.36% | -2.45% | n/a |
| average-exposure part (all groups) | all | 0.02% | 0.17% | n/a |
| timing part (all groups) | all | -0.39% | -2.62% | n/a |
| transaction-cost effect | all | -0.23% | -1.57% | n/a |
| net active return | all | -0.59% | -4.02% | n/a |
| terminal wealth difference (compounded) | all | -0.58% | -6.49% | n/a |
| compounding residual (terminal difference - sum of daily net active) | all | n/a | -2.46% | n/a |

**P1 HMM regime - P0 strategic** (annualized; reconciliation error 1.1e-17):

| component | group | annualized_mean | sum_daily | mean_weight_difference |
|---|---|---|---|---|
| asset allocation (total) | equity | -0.44% | -2.99% | -3.17% |
| average-exposure part | equity | -0.46% | -3.09% | -3.17% |
| timing part | equity | 0.01% | 0.10% | -3.17% |
| asset allocation (total) | duration | -0.00% | -0.03% | -0.51% |
| average-exposure part | duration | -0.00% | -0.01% | -0.51% |
| timing part | duration | -0.00% | -0.02% | -0.51% |
| asset allocation (total) | credit | 0.01% | 0.04% | 0.23% |
| average-exposure part | credit | 0.01% | 0.06% | 0.23% |
| timing part | credit | -0.00% | -0.02% | 0.23% |
| asset allocation (total) | gold | -0.06% | -0.41% | -0.50% |
| average-exposure part | gold | -0.08% | -0.55% | -0.50% |
| timing part | gold | 0.02% | 0.13% | -0.50% |
| asset allocation (total) | commodities | -0.05% | -0.37% | -0.35% |
| average-exposure part | commodities | -0.05% | -0.35% | -0.35% |
| timing part | commodities | -0.00% | -0.01% | -0.35% |
| asset allocation (total) | bil | 0.11% | 0.72% | 4.30% |
| average-exposure part | bil | 0.12% | 0.79% | 4.30% |
| timing part | bil | -0.01% | -0.07% | 4.30% |
| gross active return (sum of groups) | all | -0.45% | -3.04% | n/a |
| gross active return (ledger) | all | -0.45% | -3.04% | n/a |
| average-exposure part (all groups) | all | -0.46% | -3.15% | n/a |
| timing part (all groups) | all | 0.02% | 0.11% | n/a |
| transaction-cost effect | all | -0.05% | -0.32% | n/a |
| net active return | all | -0.50% | -3.36% | n/a |
| terminal wealth difference (compounded) | all | -0.44% | -5.08% | n/a |
| compounding residual (terminal difference - sum of daily net active) | all | n/a | -1.72% | n/a |

**P2 Full EWS - P0 strategic** (annualized; reconciliation error 6.9e-18):

| component | group | annualized_mean | sum_daily | mean_weight_difference |
|---|---|---|---|---|
| asset allocation (total) | equity | -0.66% | -4.45% | -3.10% |
| average-exposure part | equity | -0.44% | -2.99% | -3.10% |
| timing part | equity | -0.22% | -1.46% | -3.10% |
| asset allocation (total) | duration | -0.10% | -0.67% | 0.02% |
| average-exposure part | duration | 0.01% | 0.06% | 0.02% |
| timing part | duration | -0.11% | -0.73% | 0.02% |
| asset allocation (total) | credit | 0.04% | 0.24% | 0.12% |
| average-exposure part | credit | 0.00% | 0.03% | 0.12% |
| timing part | credit | 0.03% | 0.21% | 0.12% |
| asset allocation (total) | gold | -0.11% | -0.72% | -0.48% |
| average-exposure part | gold | -0.08% | -0.53% | -0.48% |
| timing part | gold | -0.03% | -0.19% | -0.48% |
| asset allocation (total) | commodities | -0.05% | -0.35% | -0.22% |
| average-exposure part | commodities | -0.03% | -0.23% | -0.22% |
| timing part | commodities | -0.02% | -0.12% | -0.22% |
| asset allocation (total) | bil | 0.07% | 0.46% | 3.67% |
| average-exposure part | bil | 0.10% | 0.68% | 3.67% |
| timing part | bil | -0.03% | -0.22% | 3.67% |
| gross active return (sum of groups) | all | -0.81% | -5.48% | n/a |
| gross active return (ledger) | all | -0.81% | -5.48% | n/a |
| average-exposure part (all groups) | all | -0.44% | -2.98% | n/a |
| timing part (all groups) | all | -0.37% | -2.51% | n/a |
| transaction-cost effect | all | -0.28% | -1.90% | n/a |
| net active return | all | -1.09% | -7.38% | n/a |
| terminal wealth difference (compounded) | all | -1.02% | -11.56% | n/a |
| compounding residual (terminal difference - sum of daily net active) | all | n/a | -4.18% | n/a |

Is any advantage mainly explained by reduced equity exposure rather than superior timing? P2 Full EWS vs P0 strategic: exposure-driven (average weight difference dominates) (gross active -0.81%/yr = exposure -0.44% + timing -0.37%; beta 0.85); P2 Full EWS vs P1 HMM regime: timing component larger but not statistically distinguishable (alpha interval contains 0) (gross active -0.36%/yr = exposure 0.02% + timing -0.39%; beta 0.93); P1 HMM regime vs P0 strategic: exposure-driven (average weight difference dominates) (gross active -0.45%/yr = exposure -0.46% + timing 0.02%; beta 0.92).

## 20. Turnover and implementation costs

| portfolio | annualized_one_way_turnover | gross_traded_notional_total | rebalances_per_year | average_holding_period_years | total_cost_fraction_10bps | cost_primary_in_stress | cost_primary_in_normal | executions_in_stress |
|---|---|---|---|---|---|---|---|---|
| P0_strategic | 0.1592 | 2.1579 | 11.6557 | 6.2817 | 0.0025 | 0.0007 | 0.0018 | 10 |
| P1_regime_aware | 0.3944 | 5.3470 | 15.1967 | 2.5352 | 0.0058 | 0.0008 | 0.0049 | 10 |
| P2_ews_enhanced | 1.4155 | 19.1885 | 18.2951 | 0.7064 | 0.0215 | 0.0046 | 0.0169 | 16 |
| B1_60_40 | 0.1231 | 1.6688 | 8.4098 | 8.1230 | 0.0021 | 0.0008 | 0.0013 | 9 |
| B2_constant_defensiveness | 0.1522 | 2.0636 | 11.0656 | 6.5690 | 0.0025 | 0.0008 | 0.0017 | 10 |
| B3_strategic_buy_and_hold | 0.0000 | 0.0000 | 0.0000 |  | 0.0000 | 0.0000 | 0.0000 | 0 |

Does additional EWS complexity justify its operational cost? NO - the prespecified bar (net Sharpe +0.10 over P1 with an interval excluding 0, robust to costs) is not met (extra one-way turnover 1.02 per year; extra total cost 1.57% of wealth).

## 21. Constant-defensiveness comparison

- P1 HMM regime - B2: net Sharpe difference 0.01 (95% interval -0.02 to 0.04); similar to permanent defensiveness.
- P2 Full EWS - B2: net Sharpe difference -0.01 (95% interval -0.16 to 0.13); similar to permanent defensiveness.
B2 holds the same average defensiveness every week without timing; it isolates the value of dynamic risk timing from simply holding less risk.

## 22. Statistical significance and uncertainty

1708 daily observations; 11 independent SPY drawdown events (4 >= 10%). Annualized Sharpe, CAGR and maximum drawdown are path statistics; their intervals come from a paired stationary bootstrap (block 21 primary, 63 secondary, 5000 draws, fixed seed) and are approximate.

Primary settings (Newey-West for the mean; stationary bootstrap with mean block 21 for path statistics):

| portfolio_a | portfolio_b | statistic | estimate | ci_low | ci_high | p_value | p_value_holm |
|---|---|---|---|---|---|---|---|
| P1_regime_aware | P0_strategic | mean_active_return | -0.0050 | -0.0118 | 0.0018 | 0.1527 | 0.4581 |
| P2_ews_enhanced | P0_strategic | mean_active_return | -0.0109 | -0.0267 | 0.0049 | 0.1758 | 0.4581 |
| P2_ews_enhanced | P1_regime_aware | mean_active_return | -0.0059 | -0.0178 | 0.0059 | 0.3264 | 0.4581 |
| P1_regime_aware | B2_constant_defensiveness | mean_active_return | -0.0000 | -0.0035 | 0.0034 | 0.9866 |  |
| P2_ews_enhanced | B2_constant_defensiveness | mean_active_return | -0.0060 | -0.0194 | 0.0074 | 0.3827 |  |
| P1_regime_aware | P0_strategic | sharpe_difference | 0.0068 | -0.0220 | 0.0336 | 0.6940 |  |
| P1_regime_aware | P0_strategic | sortino_difference | 0.0090 | -0.0362 | 0.0490 | 0.7408 |  |
| P1_regime_aware | P0_strategic | cagr_difference | -0.0044 | -0.0119 | 0.0025 | 0.2188 |  |
| P1_regime_aware | P0_strategic | volatility_difference | -0.0087 | -0.0117 | -0.0058 | 0.0000 |  |
| P1_regime_aware | P0_strategic | max_drawdown_difference | 0.0139 | 0.0057 | 0.0264 | 0.0016 |  |
| P1_regime_aware | P0_strategic | mean_annualized_difference | -0.0050 | -0.0111 | 0.0013 | 0.1176 |  |
| P2_ews_enhanced | P0_strategic | sharpe_difference | -0.0121 | -0.1686 | 0.1261 | 0.7596 |  |
| P2_ews_enhanced | P0_strategic | sortino_difference | -0.0175 | -0.2778 | 0.1922 | 0.7588 |  |
| P2_ews_enhanced | P0_strategic | cagr_difference | -0.0102 | -0.0279 | 0.0078 | 0.2496 |  |
| P2_ews_enhanced | P0_strategic | volatility_difference | -0.0149 | -0.0265 | -0.0050 | 0.0000 |  |
| P2_ews_enhanced | P0_strategic | max_drawdown_difference | 0.0278 | -0.0093 | 0.0716 | 0.2772 |  |
| P2_ews_enhanced | P0_strategic | mean_annualized_difference | -0.0109 | -0.0264 | 0.0049 | 0.1704 |  |
| P2_ews_enhanced | P1_regime_aware | sharpe_difference | -0.0190 | -0.1703 | 0.1154 | 0.7000 |  |
| P2_ews_enhanced | P1_regime_aware | sortino_difference | -0.0266 | -0.2768 | 0.1771 | 0.7076 |  |
| P2_ews_enhanced | P1_regime_aware | cagr_difference | -0.0058 | -0.0203 | 0.0084 | 0.4056 |  |
| P2_ews_enhanced | P1_regime_aware | volatility_difference | -0.0062 | -0.0152 | 0.0015 | 0.1680 |  |
| P2_ews_enhanced | P1_regime_aware | max_drawdown_difference | 0.0139 | -0.0237 | 0.0501 | 0.6412 |  |
| P2_ews_enhanced | P1_regime_aware | mean_annualized_difference | -0.0059 | -0.0191 | 0.0066 | 0.3500 |  |
| P1_regime_aware | B2_constant_defensiveness | sharpe_difference | 0.0136 | -0.0156 | 0.0396 | 0.3840 |  |
| P1_regime_aware | B2_constant_defensiveness | sortino_difference | 0.0191 | -0.0255 | 0.0579 | 0.4256 |  |
| P1_regime_aware | B2_constant_defensiveness | cagr_difference | 0.0002 | -0.0034 | 0.0035 | 0.9132 |  |
| P1_regime_aware | B2_constant_defensiveness | volatility_difference | -0.0022 | -0.0038 | -0.0006 | 0.0068 |  |
| P1_regime_aware | B2_constant_defensiveness | max_drawdown_difference | 0.0038 | -0.0012 | 0.0111 | 0.0936 |  |
| P1_regime_aware | B2_constant_defensiveness | mean_annualized_difference | -0.0000 | -0.0032 | 0.0030 | 0.9808 |  |
| P2_ews_enhanced | B2_constant_defensiveness | sharpe_difference | -0.0054 | -0.1617 | 0.1324 | 0.8268 |  |
| P2_ews_enhanced | B2_constant_defensiveness | sortino_difference | -0.0074 | -0.2674 | 0.2004 | 0.8172 |  |
| P2_ews_enhanced | B2_constant_defensiveness | cagr_difference | -0.0056 | -0.0212 | 0.0097 | 0.4516 |  |
| P2_ews_enhanced | B2_constant_defensiveness | volatility_difference | -0.0084 | -0.0186 | 0.0003 | 0.0744 |  |
| P2_ews_enhanced | B2_constant_defensiveness | max_drawdown_difference | 0.0177 | -0.0185 | 0.0573 | 0.5420 |  |
| P2_ews_enhanced | B2_constant_defensiveness | mean_annualized_difference | -0.0060 | -0.0199 | 0.0075 | 0.3748 |  |

Secondary block length (63 sessions), net Sharpe difference:

| portfolio_a | portfolio_b | estimate | ci_low | ci_high | p_value |
|---|---|---|---|---|---|
| P1_regime_aware | P0_strategic | 0.0068 | -0.0192 | 0.0318 | 0.6832 |
| P2_ews_enhanced | P0_strategic | -0.0121 | -0.1749 | 0.1239 | 0.7764 |
| P2_ews_enhanced | P1_regime_aware | -0.0190 | -0.1809 | 0.1160 | 0.7224 |
| P1_regime_aware | B2_constant_defensiveness | 0.0136 | -0.0122 | 0.0378 | 0.3372 |
| P2_ews_enhanced | B2_constant_defensiveness | -0.0054 | -0.1697 | 0.1304 | 0.8448 |

Do statistical confidence intervals support the economic conclusions? no primary net Sharpe difference has a 95% interval excluding 0. A 95% interval that contains 0 is not evidence that the strategies are equivalent.

## 23. Prespecified sensitivity analyses

Net Sharpe ratio by cost assumption:

| portfolio | gross | net_5bps | net_10bps | net_25bps |
|---|---|---|---|---|
| P0_strategic | 0.655 | 0.654 | 0.652 | 0.647 |
| P1_regime_aware | 0.667 | 0.663 | 0.659 | 0.646 |
| P2_ews_enhanced | 0.673 | 0.656 | 0.640 | 0.590 |
| B1_60_40 | 0.561 | 0.560 | 0.559 | 0.555 |
| B2_constant_defensiveness | 0.649 | 0.647 | 0.645 | 0.640 |
| B3_strategic_buy_and_hold | 0.674 | 0.674 | 0.674 | 0.674 |

P1 vs P0: sign of the net Sharpe difference changes with the cost level; P2 vs P0: sign of the net Sharpe difference changes with the cost level; P2 vs P1: not higher at any cost level.

Model and initialization sensitivity (robustness only; the primary policy is never replaced):

| variant | initialization | portfolio | cagr | annual_volatility | sharpe | max_drawdown | annualized_one_way_turnover | max_abs_weight_difference_vs_restart_reference |
|---|---|---|---|---|---|---|---|---|
| restart_reference | restart_benchmark_at_test_start | P0_strategic | 9.82% | 11.13% | 0.652 | -20.08% | 0.158 | 0.000 |
| restart_reference | restart_benchmark_at_test_start | P1_regime_aware | 9.37% | 10.26% | 0.659 | -18.69% | 0.393 | 0.000 |
| restart_reference | restart_benchmark_at_test_start | P2_ews_enhanced | 8.80% | 9.64% | 0.640 | -17.30% | 1.414 | 0.000 |
| restart_reference | restart_benchmark_at_test_start | B1_60_40 | 9.19% | 12.17% | 0.559 | -21.28% | 0.124 |  |
| restart_reference | restart_benchmark_at_test_start | B2_constant_defensiveness | 9.36% | 10.48% | 0.646 | -19.07% | 0.158 |  |
| restart_reference | restart_benchmark_at_test_start | B3_strategic_buy_and_hold | 10.30% | 11.46% | 0.674 | -18.86% | 0.000 |  |
| cov_lw_252 | restart_benchmark_at_test_start | P1_regime_aware | 9.32% | 10.26% | 0.654 | -18.72% | 0.388 | 0.025 |
| cov_lw_252 | restart_benchmark_at_test_start | P2_ews_enhanced | 8.82% | 9.63% | 0.643 | -17.58% | 1.425 | 0.063 |
| cov_ewma_hl63 | restart_benchmark_at_test_start | P1_regime_aware | 9.37% | 10.23% | 0.660 | -18.52% | 0.417 | 0.039 |
| cov_ewma_hl63 | restart_benchmark_at_test_start | P2_ews_enhanced | 8.73% | 9.69% | 0.631 | -17.35% | 1.589 | 0.061 |
| mapping_P1_full_range_P2_continuous | restart_benchmark_at_test_start | P1_regime_aware | 8.03% | 8.16% | 0.653 | -15.37% | 1.155 | 0.157 |
| mapping_P1_full_range_P2_continuous | restart_benchmark_at_test_start | P2_ews_enhanced | 8.60% | 9.72% | 0.616 | -16.87% | 2.358 | 0.202 |
| primary | continuation (primary) | P0_strategic | 9.82% | 11.13% | 0.652 | -20.08% | 0.159 | 0.006 |
| primary | continuation (primary) | P1_regime_aware | 9.37% | 10.26% | 0.659 | -18.69% | 0.394 | 0.006 |
| primary | continuation (primary) | P2_ews_enhanced | 8.80% | 9.64% | 0.640 | -17.30% | 1.416 | 0.005 |
| primary | continuation (primary) | B1_60_40 | 9.19% | 12.18% | 0.559 | -21.28% | 0.123 |  |
| primary | continuation (primary) | B2_constant_defensiveness | 9.35% | 10.48% | 0.645 | -19.07% | 0.152 |  |
| primary | continuation (primary) | B3_strategic_buy_and_hold | 10.30% | 11.46% | 0.674 | -18.86% | 0.000 |  |

## 24. Model failures and unfavorable market conditions

- Event 1 (SPY peak 2020-02-19): P2 recovery over 63 sessions after the trough 16.4% vs P0 22.2%; de-risking: after onset (reaction).
- Event 9 (SPY peak 2025-02-19): P2 recovery over 63 sessions after the trough 10.7% vs P0 15.0%; de-risking: before onset (executable protection).
- Event 8 (SPY peak 2024-07-16): P2 recovery over 63 sessions after the trough 2.9% vs P0 5.7%; de-risking: before onset (executable protection).
- P2 reacted only after the -5% onset in 1 event(s).
- 472 reported policy exceptions (portfolio_constraint_audit.csv):

| portfolio | reason | count |
|---|---|---|
| B2_constant_defensiveness | hold inside the no-trade band | 172 |
| P1_regime_aware | hold inside the no-trade band | 153 |
| P2_ews_enhanced | execution-day turnover 0.3010 > cap 0.3000: cap-preserving projection, 0.997 of the move executed | 1 |
| P2_ews_enhanced | hold inside the no-trade band | 145 |
| P2_ews_enhanced | turnover-cap-constrained transition | 1 |

- Solver warnings in Test decisions: 5 (documented; final solutions verified).

## 25. Research limitations

- One Test period (about seven years) with few independent stress episodes; results are dominated by a handful of events.
- Validation was reused in Stages 5-7; Train signals are retrospective reconstructions.
- Adjusted-close execution and the 5 / 10 / 25 bps cost scenarios are assumptions, not observed fills.
- P(non-normal) is saturated (often exactly 0 or 1), so P1 behaves almost like a two-state switch.
- Stage 7 constraints and risk-budget slopes are proposed research assumptions; they were not optimized but they shape every dynamic result.
- Gross / net returns ignore taxes, fund fees beyond the adjusted-close series and market impact.

## 26. Economic interpretation

Forecasting quality, regime classification, early warning, portfolio implementation and realized investment performance are different claims. Stage 6 measured ranking quality of the signals; Stage 7 showed that the policies change ex-ante risk; only this stage measures realized money outcomes after delay and costs. A better early-warning score does not by itself imply better portfolio performance, and lower volatility is not alpha.

- Attribution: P2 Full EWS vs P0 strategic: exposure-driven (average weight difference dominates) (gross active -0.81%/yr = exposure -0.44% + timing -0.37%; beta 0.85); P2 Full EWS vs P1 HMM regime: timing component larger but not statistically distinguishable (alpha interval contains 0) (gross active -0.36%/yr = exposure 0.02% + timing -0.39%; beta 0.93); P1 HMM regime vs P0 strategic: exposure-driven (average weight difference dominates) (gross active -0.45%/yr = exposure -0.46% + timing 0.02%; beta 0.92).
- Implementation: NO - the prespecified bar (net Sharpe +0.10 over P1 with an interval excluding 0, robust to costs) is not met.

## 27. Final investment conclusion

- **A. Does P1 outperform P0 on risk-adjusted performance after costs?** NOT DEMONSTRATED - higher point estimate, but the 95% interval contains 0.
- **B. Does P2 outperform P1 on risk-adjusted performance after costs?** NO - net Sharpe ratio not higher.
- **C. Does P2 reduce maximum drawdown by an economically meaningful amount?** YES vs P0 (threshold 2.0 pp); vs P1: NO.
- **D. Does P2 provide additional value in the predeclared stress periods?** net return: P2 above P1 in 1 and above P0 in 1 of 4 fixed windows; shallower in-window drawdown than P1 in 1; in the 11 objective SPY drawdown events P2 was already defensive (executed) before the onset in 9, de-risked only after the onset in 1, and did not de-risk in 1.
- **E. Are the improvements robust across transaction-cost assumptions?** P1 vs P0: sign of the net Sharpe difference changes with the cost level; P2 vs P0: sign of the net Sharpe difference changes with the cost level; P2 vs P1: not higher at any cost level.
- **F. Is any advantage mainly explained by reduced equity exposure rather than superior timing?** P2 Full EWS vs P0 strategic: exposure-driven (average weight difference dominates) (gross active -0.81%/yr = exposure -0.44% + timing -0.37%; beta 0.85); P2 Full EWS vs P1 HMM regime: timing component larger but not statistically distinguishable (alpha interval contains 0) (gross active -0.36%/yr = exposure 0.02% + timing -0.39%; beta 0.93); P1 HMM regime vs P0 strategic: exposure-driven (average weight difference dominates) (gross active -0.45%/yr = exposure -0.46% + timing 0.02%; beta 0.92).
- **G. Does additional EWS complexity justify its operational cost?** NO - the prespecified bar (net Sharpe +0.10 over P1 with an interval excluding 0, robust to costs) is not met.
- **H. Do statistical confidence intervals support the economic conclusions?** no primary net Sharpe difference has a 95% interval excluding 0.

**Outcome 4 - Evidence is inconclusive** (rule-based classification frozen before Test).

**STAGE 8: WARN** - **FINAL RESEARCH STATUS: COMPLETE WITH LIMITATIONS**

## 28. Next steps for presentation and potential deployment

- Presentation: use investment_decision_summary, cumulative_wealth_gross_net, portfolio_drawdowns, ews_incremental_attribution and performance_uncertainty; state the prespecified outcome label and the number of independent events.
- Deployment would require a live, append-only signal log (true point-in-time), paper trading with real quotes, ETF-specific cost estimates, a governance rule for model refits, and an approved client mandate for the proposed constraints. None of these is claimed here.

## Checks

| name | status | detail |
|---|---|---|
| Pre-Test protocol frozen and Phase 8A passed before Test access | PASS | lock 73998035e0857aad...; preflight 30 PASS, 0 WARN |
| Test factor order and ETF universe identical to Train/Validation | PASS | factors ['equity_trend', 'interest_rate', 'inflation', 'volatility', 'liquidity_credit']; return columns ['SPY', 'EFA', 'EEM', 'IEF', 'TIP', 'HYG', 'GLD', 'DBC', 'BIL'] |
| Test dates strictly chronological, unique, identical in factor and return files | PASS | 1708 sessions 2020-01-02..2026-10-19 |
| No overlap with Validation; Test starts after the last Validation session | PASS | last Validation 2019-12-31, first Test 2020-01-02 |
| Test sessions match the frozen Stage 3 split manifest (membership, first / last date, rows) | PASS | manifest 2020-01-02..2026-10-19 (1708 rows) |
| No missing or non-finite Test factors / returns (no filling) | PASS | missing factors 0, missing returns 0 |
| No excluded session between the last Validation date and the Test end (no omitted holding returns) | PASS | 0 excluded sessions in the window |
| Return definition unchanged (daily log return of adjusted closes) | PASS | daily log return log(P_t/P_{t-1}) from adjusted close (dividends and splits); same-day |
| No implausible daily log return (|r| > 0.5) in the Test ETF returns | PASS | 0 values |
| Test opened only after a valid pre-Test lock (gated evaluation-only loader) | PASS | lock 73998035e0857aad...; access #1 (first_evaluation) |
| Appending Test leaves every Train/Validation signal unchanged (prefix invariance at the boundary) | PASS | max difference 0.0e+00 over 3064 sessions |
| Continued signals still identical to the frozen Stage 6.1 handoff on Train/Validation | PASS | 14 columns |
| No frozen signal constant re-estimated (thresholds, priors, scaler identical) | PASS | alert_thresholds 0.0e+00, innovation_normalization 0.0e+00, bocpd_priors 0.0e+00, factor_scaler 0.0e+00 |
| Test signal prefix invariance: running through t equals the full run truncated at t | PASS | cut 2023-05-24; max difference 0.0e+00 (HMM filter, BOCPD, EWS score, alert levels) |
| Changing Test factors after t leaves earlier HMM probabilities, BOCPD posteriors, EWS scores and alerts unchanged | PASS | cut 2023-05-24; difference before 0.0e+00, after inf (perturbation effective) |
| Alert state machine and HMM filter continue across the boundary (no reset to GREEN / prior) | PASS | EWS level 2019-12-31: GREEN; 2020-01-02: GREEN; P(non-normal) 2019-12-31: 0.0000 -> 0.0000 |
| Weekly calendar: pre-Test decisions unchanged; boundary week completed by the frozen rule | PASS | first Test decision 2020-01-03 executes 2020-01-06 (week of 2019-12-30); 355 Test decisions |
| Test initialization depends only on pre-Test information (continuation = locked preflight state) | PASS | holdings at 2019-12-31 identical to the lock (sha 6a334d65edfc...) |
| Changing Test returns after t leaves every decision, covariance input and holding up to t unchanged | PASS | cut 2023-05-26: P0_strategic identical, P1_regime_aware identical, P2_ews_enhanced identical |
| Ledger: Every session after the start is in the ledger (no omitted holding returns, no phantom dates) | PASS | 6 portfolios; failed: none |
| Ledger: Wealth finite and positive | PASS | 6 portfolios; failed: none |
| Ledger: Weights sum to one, no short positions, no leverage | PASS | 6 portfolios; failed: none; max error 1.8e-15 |
| Ledger: Gross return = sum of previous-close holdings x asset simple returns | PASS | 6 portfolios; failed: none; max error 1.4e-17 |
| Ledger: Holdings continuity: pre-trade weights = previous holdings drifted by the session return | PASS | 6 portfolios; failed: none; max error 1.1e-16 |
| Ledger: Trades recorded: gross notional = sum |post - pre| = 2 x one-way turnover; no unrecorded rebalancing | PASS | 6 portfolios; failed: none; max error 5.6e-17 |
| Ledger: Trades only on scheduled execution sessions (no phantom executions) | PASS | 6 portfolios; failed: none |
| Ledger: Transaction cost = bps / 1e4 x stress multiplier x gross notional (10 bps = 0.001) | PASS | 6 portfolios; failed: none; max error 0.0e+00 |
| Ledger: Net wealth = gross wealth path x (1 - cost) at each execution, compounded daily (costs counted once) | PASS | 6 portfolios; failed: none; max error 4.0e-15 |
| Ledger: Self-financing: pre-trade wealth = previous wealth x (1 + gross); post-trade wealth = pre-trade wealth - cost amount | PASS | 6 portfolios; failed: none; max error 2.2e-16 |
| First Test session return earned by the holdings carried from the last Validation close | PASS | 2020-01-02 included for every portfolio (no split-boundary gap) |
| Execution timing: every order executes at the close of the next session (t+1); new holdings first earn t+2 | PASS | 1775 Test executions audited |
| Previous holdings earn the t+1 return; new targets never earn it; traded notional uses execution-time pre-trade holdings | PASS | violations: none |
| Hard constraints: every Test decision-time target and executed portfolio compliant | PASS | targets violating 0; executions violating 0; passive-drift episodes reported separately: 6 |
| Execution-day turnover cap: exceptions handled by the frozen projection and reported | PASS | 1 cap-preserving projections in Test (listed in portfolio_constraint_audit.csv) |
| Optimizer: no hold fallback in Test (numerical failure) | PASS | 0 stage-3 fallbacks |
| Solver warnings documented; every affected final solution independently verified feasible | WARN | 5 warning records in Test decisions: RuntimeWarning: Values in x were outside bounds during a minimize step, clipping to bounds (x5); hard-constraint check of those decisions: all compliant |
| Role separation: P1 only baseline_regime, P2 only challenger_ews, P0 no signal | PASS | P0 posture = benchmark True; roles True; d1 = min(1, a p) True; d2 = level map True |
| P0, P1, P2 (and B1, B2) share universe, returns, calendar and execution convention | PASS | 355 Test decisions each |
| At most one execution per portfolio and calendar week | PASS | 1775 executions |
| Cost scenarios on one trading path: terminal wealth gross >= 5 >= 10 >= 25 bps for every portfolio (costs reduce wealth only, counted once) | PASS | total 10 bps cost fractions: P0 0.0025, P1 0.0058, P2 0.0215, B1 0.0021, B2 0.0025, B3 0.0000 |
| No Test result changed a frozen model, signal, policy or configuration (lock hashes after == before) | PASS | 54 code + 20 artifact hashes; frozen Stage 5-7 files changed: none |
| Automated tests on the real Stage 8 outputs | PASS | 9 tests, 0 failures, 0 errors, 0 skipped |
| No library warnings outside the documented solver warnings | WARN | RuntimeWarning: Values in x were outside bounds during a minimize step, clipping to bounds |

Figures: `reports/figures/backtesting/cumulative_wealth_gross_net.png`, `reports/figures/backtesting/portfolio_drawdowns.png`, `reports/figures/backtesting/rolling_sharpe.png`, `reports/figures/backtesting/rolling_volatility.png`, `reports/figures/backtesting/realized_asset_allocation.png`, `reports/figures/backtesting/regime_warning_portfolio_response.png`, `reports/figures/backtesting/crisis_performance_comparison.png`, `reports/figures/backtesting/ews_incremental_attribution.png`, `reports/figures/backtesting/turnover_and_costs.png`, `reports/figures/backtesting/risk_adjusted_performance_comparison.png`, `reports/figures/backtesting/annual_returns_heatmap.png`, `reports/figures/backtesting/transaction_cost_sensitivity.png`, `reports/figures/backtesting/performance_uncertainty.png`, `reports/figures/backtesting/investment_decision_summary.png`
