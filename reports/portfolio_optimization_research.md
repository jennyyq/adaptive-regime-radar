# Stage 7 - Robust portfolio optimization and dynamic risk budgeting

## STAGE 8 HANDOFF STATUS: READY

all blocking checks PASS; WARN: No library warnings: RuntimeWarning: Values in x were outside bounds during a minimize step, clipping to bounds

Checks: 29 PASS, 1 WARN, 0 FAIL. Train 2007-10-30..2015-12-31 (2058 sessions); Validation 2016-01-04..2019-12-31 (1006 sessions). Test not read. Everything below is EX ANTE or a retrospective diagnostic; no realized returns, Sharpe ratios, drawdowns or wealth curves are computed in Stage 7.

## 1. Executive summary

- A feasible, interpretable weekly allocation system was implemented for P0 (strategic), P1 (HMM P(non-normal)) and P2 (Full EWS) with one shared optimizer, risk model, expected-return model, constraint set, cost model, calendar and the frozen execution convention. 583 weekly decisions per portfolio.
- Optimizer outcomes for P1/P2: P1_regime_aware hold_within_no_trade_band 423, P1_regime_aware trade_to_target 160, P2_ews_enhanced hold_within_no_trade_band 386, P2_ews_enhanced trade_to_target 196, P2_ews_enhanced turnover_limited_transition 1.
- Validation mean ex-ante volatility: P0_strategic 7.11%, P1_regime_aware 6.74%, P2_ews_enhanced 6.79%; mean equity: P0_strategic 50.1%, P1_regime_aware 47.3%, P2_ews_enhanced 47.5%; mean BIL: P0_strategic 5.0%, P1_regime_aware 9.2%, P2_ews_enhanced 8.5%.
- P(non-normal) is saturated: on Train 20.4% of days equal 1.0 exactly and 29.8% are <= 0.001, so P1's continuous budget behaves almost like a two-state switch (mean Train d: P1 0.167 vs P2 0.167, equal by construction).
- Predicted risk differences are not realized investment improvements. Whether either signal adds value after costs must be decided by the Stage 8 chronological backtest.

## 2. Investment mandate

Moderate-risk multi-asset client with a long-term strategic allocation; nine US-listed ETFs; long only, no leverage or derivatives; at most one rebalance per week. Benchmark (frozen): SPY 35%, EFA 10%, EEM 5%, IEF 15%, TIP 10%, HYG 5%, GLD 8%, DBC 7%, BIL 5%.

Proposed constraints (research assumptions, not an approved mandate):

| ETF | benchmark | max weight |
|---|---|---|
| SPY | 35% | 45% |
| EFA | 10% | 20% |
| EEM | 5% | 10% |
| IEF | 15% | 35% |
| TIP | 10% | 25% |
| HYG | 5% | 10% |
| GLD | 8% | 15% |
| DBC | 7% | 10% |
| BIL | 5% | 40% |

Maximum active weight per ETF: 20%. Group limits: equity 20%-55%, duration 0%-45%, credit 0%-10%, real_assets 0%-20%, liquidity 2%-40%. Turnover cap 20% one-way per rebalance (rising with defensiveness, see section 8).

## 3. Data and temporal protocol

- Returns: the split files store daily LOG returns of adjusted closes (daily log return log(P_t/P_{t-1}) from adjusted close (dividends and splits); same-day); no missing values in the nine-ETF common sample. They are converted exactly to SIMPLE returns, R = exp(r) - 1; portfolio returns, covariances, CVaR scenarios and expected returns are all in simple-return units, annualized with 252 sessions.
- Decisions: last trading session of every calendar week; execution at the close of t+1; first holding return t+2 (frozen conservative_close_t_plus_1). Exactly one execution per calendar week. The last Train decision (2015-12-31) executes on the first Validation session and is flagged split-crossing. The final Validation week is incomplete without the Test calendar and has no decision.
- Warm-up: a decision needs >= 252 past sessions; the first decision is 2008-10-31.
- Train decisions use frozen HMM/AR/EWS models fitted on all of Train: they are retrospective research reconstructions, not live or deployable historical decisions. Covariances and expected returns are always as-of t.

## 4. Expected-return model

Primary (low alpha): benchmark-implied equilibrium returns mu_t = rf_t + (SR_b / sigma_b,t) Sigma_t w_b with a PROPOSED benchmark Sharpe ratio SR_b = 0.3. With gamma_t = SR_b / sigma_b,t the unconstrained optimum is the benchmark, so every deviation is driven by the risk budget and the constraints, not by noisy means.

| asset | mean mu_eq train | mean mu_eq validation |
|---|---|---|
| BIL | -0.0001 | 0.0110 |
| DBC | 0.0397 | 0.0354 |
| EEM | 0.0910 | 0.0593 |
| EFA | 0.0735 | 0.0481 |
| GLD | 0.0188 | 0.0161 |
| HYG | 0.0264 | 0.0248 |
| IEF | -0.0061 | 0.0089 |
| SPY | 0.0610 | 0.0466 |
| TIP | -0.0006 | 0.0123 |

Historical trailing means and their standard errors (sensitivity only):

| asset | hist_mean_mean | hist_se_mean |
|---|---|---|
| BIL | 0.0062 | 0.0019 |
| DBC | -0.0367 | 0.1080 |
| EEM | 0.0619 | 0.1340 |
| EFA | 0.0474 | 0.1005 |
| GLD | 0.0283 | 0.0904 |
| HYG | 0.0424 | 0.0430 |
| IEF | 0.0213 | 0.0356 |
| SPY | 0.1154 | 0.0921 |
| TIP | 0.0178 | 0.0293 |

A one-standard-error band of a 2-year mean is several percentage points per year for equities - larger than the equilibrium premia. This is why raw historical means are not used (section 12 reports the weight sensitivity).

## 5. Covariance and risk estimation

Primary: Ledoit-Wolf shrinkage (to a scaled identity, analytic intensity; identical to sklearn's estimator) on the trailing 504 daily simple returns through t, annualized x252; PSD floor at 1e-08 x mean variance (never needed for Ledoit-Wolf). Alternatives (robustness only): sample 504, EWMA half-life 63 (effective sample size reported), Ledoit-Wolf 252 and expanding.

|  | benchmark_vol_ex_ante | evaluation_only_subsequent_realized_benchmark_vol_21d | condition_number | min_eigenvalue | vol_BIL | mean_pairwise_correlation | mean_abs_weekly_change_asset_vol | lw_shrinkage | effective_n |
|---|---|---|---|---|---|---|---|---|---|
| ('train', 'ewma_hl63') | 0.1156 | 0.0979 | 16496.3790 | 0.0000 | 0.0045 | 0.1711 | 0.0027 |  | 179.5249 |
| ('train', 'ledoit_wolf_252') | 0.1156 | 0.0979 | 194.3733 | 0.0014 | 0.0332 | 0.1719 | 0.0018 | 0.0337 | 252.0000 |
| ('train', 'ledoit_wolf_504 (primary)') | 0.1262 | 0.0979 | 299.2951 | 0.0012 | 0.0304 | 0.1713 | 0.0011 | 0.0221 | 486.3893 |
| ('train', 'ledoit_wolf_expanding') | 0.1640 | 0.0979 | 365.2974 | 0.0015 | 0.0362 | 0.1520 | 0.0005 | 0.0187 | 1156.4827 |
| ('train', 'sample_504') | 0.1272 | 0.0979 | 15142.4270 | 0.0000 | 0.0055 | 0.1654 | 0.0010 |  | 486.3893 |
| ('validation', 'ewma_hl63') | 0.0696 | 0.0626 | 13836.0082 | 0.0000 | 0.0025 | 0.1798 | 0.0017 |  | 180.3668 |
| ('validation', 'ledoit_wolf_252') | 0.0705 | 0.0626 | 174.7756 | 0.0005 | 0.0212 | 0.1666 | 0.0010 | 0.0369 | 252.0000 |
| ('validation', 'ledoit_wolf_504 (primary)') | 0.0709 | 0.0626 | 288.1802 | 0.0003 | 0.0163 | 0.1716 | 0.0005 | 0.0200 | 504.0000 |
| ('validation', 'ledoit_wolf_expanding') | 0.1200 | 0.0626 | 546.1406 | 0.0004 | 0.0200 | 0.1585 | 0.0001 | 0.0105 | 2562.6010 |
| ('validation', 'sample_504') | 0.0713 | 0.0626 | 11187.3756 | 0.0000 | 0.0026 | 0.1804 | 0.0005 |  | 504.0000 |

Predicted-versus-subsequent risk (DEVELOPMENT DIAGNOSTIC ONLY - uses returns after t, never used for a decision): benchmark volatility predicted at t versus realized over the 21 sessions after the first holding date:

|  | mean |log(predicted / subsequent)| |
|---|---|
| ('train', 'ewma_hl63') | 0.323 |
| ('train', 'ledoit_wolf_252') | 0.368 |
| ('train', 'ledoit_wolf_504 (primary)') | 0.430 |
| ('train', 'ledoit_wolf_expanding') | 0.659 |
| ('train', 'sample_504') | 0.434 |
| ('validation', 'ewma_hl63') | 0.354 |
| ('validation', 'ledoit_wolf_252') | 0.375 |
| ('validation', 'ledoit_wolf_504 (primary)') | 0.423 |
| ('validation', 'ledoit_wolf_expanding') | 0.750 |
| ('validation', 'sample_504') | 0.426 |

BIL's estimated volatility is a fraction of a percent; without the BIL cap and the active-weight limit a risk-averse optimizer would load into it. BIL is treated as the liquidity sleeve with an explicit cap.

## 6. Optimization mathematics

Step 1 - posture target (path independent, depends only on information at t and the role's signal):

maximize  w'mu - (gamma/2) w'Sigma w - (lambda_TE/2) (w - w_b)'Sigma(w - w_b)

subject to sum w = 1, 0 <= w <= cap, |w - w_b| <= active limit, static group limits, risk-budget group limits and w'Sigma w <= (vol_multiplier(d) sigma_b)^2.

Step 2 - trade rule (identical for P0, P1, P2; P0's posture is the benchmark): hold if the one-way distance to the posture is <= the no-trade band (1%); else trade to the posture if within the turnover cap; else move along the straight line toward it as far as the cap allows (a convex combination of feasible points, so every linear hard constraint holds) and report the unmet budget.

- Units: annualized simple returns and covariance. gamma_t = SR_b / sigma_b,t; lambda_TE = 2.0. With equilibrium returns and gamma = delta, the d = 0 posture is exactly the benchmark.
- Why two steps: a single problem with a linear turnover penalty was tested and is kept as a sensitivity case. It is path dependent - de-risking is forced by the binding budget, but re-risking is only rewarded by the very flat objective near the benchmark, so the penalty's no-trade region traps the portfolio in defensive positions after an alert (a ratchet). The two-step rule removes this while the hard turnover cap and the no-trade band still control trading.
- Solver: SLSQP (scipy 1.15.3) with analytic gradients from two starting points (benchmark and an LP-feasible point); the strictly concave objective has a unique optimum (max start disagreement 1.6e-05).
- Deterministic fallback: if the rule's result violated a hard constraint (only possible if drift pushed the pre-trade weights outside the policy), a slack formulation keeps every hard constraint and penalizes the unmet budget; if that fails, hold pre-trade / previous target / benchmark if it satisfies every hard constraint. Never equal weights.

## 7. CVaR model

Ex-ante 95% CVaR from the trailing 504 daily simple-return scenarios through t (empirical, equally weighted); VaR = 95% loss quantile, CVaR = Rockafellar-Uryasev value (exact for the empirical distribution). Secondary optimizer: an LP that maximizes expected return net of L1 turnover and tracking penalties subject to CVaR95 <= vol_multiplier(d) x benchmark CVaR95 and the same linear constraints.

| portfolio | mvo_vol | cvar_lp_vol | mvo_cvar95 | cvar_lp_cvar95 | max_abs_weight_diff_vs_mvo |
|---|---|---|---|---|---|
| P1_regime_aware | 0.0674 | 0.0675 | 0.0101 | 0.0101 | 0.0205 |
| P2_ews_enhanced | 0.0679 | 0.0679 | 0.0102 | 0.0102 | 0.0166 |

LP status: optimal 416. Tail estimates rest on ~25 days per 504-session window; they are unstable and are not drawdowns.

## 8. Dynamic risk budgeting

| d | vol multiplier | equity max | credit max | BIL min | turnover cap |
|---|---|---|---|---|---|
| 0.00 | 1.000 | 0.550 | 0.100 | 0.020 | 0.200 |
| 0.33 | 0.883 | 0.467 | 0.083 | 0.080 | 0.233 |
| 0.67 | 0.767 | 0.383 | 0.067 | 0.140 | 0.267 |
| 1.00 | 0.650 | 0.300 | 0.050 | 0.200 | 0.300 |

### Regime-aware allocation (P1)

d1 = min(1, a x P(non-normal)) with a = 0.2827, calibrated on Train so that mean Train d1 = mean Train d2. The map is Lipschitz (no jump near p = 1). Because P(non-normal) is saturated (Train: 20.4% exactly 1.0, 48.3% >= 0.999, 29.8% <= 0.001), P1 alternates mostly between d = 0 and d = a: a regime switch rather than a graded signal. P(non-normal) also pools S2 (neutral) with S3 (stress), so P1 is moderately defensive in neutral regimes too.

### EWS-enhanced allocation (P2)

d2 = GREEN 0, AMBER 1/3, ORANGE 2/3, RED 1, RECOVERY 1/3 (AMBER posture held until the frozen state machine returns to GREEN). The level at the decision date is used; intra-week escalations are recorded (signal_week_max) and acted upon only at the next decision.

Do higher warning levels produce more defensive targets? Mean ex-ante characteristics by P2 level:

|  | budget_d | budget_vol_multiplier | vol_ex_ante | exposure_equity | exposure_duration | exposure_credit | w_GLD | w_DBC | w_BIL |
|---|---|---|---|---|---|---|---|---|---|
| ('train', 'AMBER') | 0.333 | 0.883 | 0.094 | 0.439 | 0.237 | 0.050 | 0.070 | 0.062 | 0.141 |
| ('train', 'GREEN') | 0.000 | 1.000 | 0.129 | 0.501 | 0.249 | 0.050 | 0.080 | 0.070 | 0.050 |
| ('train', 'ORANGE') | 0.667 | 0.767 | 0.092 | 0.380 | 0.229 | 0.050 | 0.061 | 0.055 | 0.223 |
| ('train', 'RECOVERY') | 0.333 | 0.883 | 0.111 | 0.440 | 0.239 | 0.050 | 0.071 | 0.063 | 0.137 |
| ('train', 'RED') | 1.000 | 0.650 | 0.086 | 0.300 | 0.282 | 0.050 | 0.047 | 0.073 | 0.249 |
| ('validation', 'AMBER') | 0.333 | 0.883 | 0.060 | 0.440 | 0.236 | 0.052 | 0.070 | 0.062 | 0.140 |
| ('validation', 'GREEN') | 0.000 | 1.000 | 0.073 | 0.501 | 0.249 | 0.050 | 0.079 | 0.070 | 0.050 |
| ('validation', 'ORANGE') | 0.667 | 0.767 | 0.049 | 0.377 | 0.225 | 0.057 | 0.059 | 0.054 | 0.229 |
| ('validation', 'RECOVERY') | 0.333 | 0.883 | 0.061 | 0.440 | 0.234 | 0.052 | 0.070 | 0.062 | 0.142 |
| ('validation', 'RED') | 1.000 | 0.650 | 0.042 | 0.301 | 0.290 | 0.050 | 0.044 | 0.066 | 0.249 |

P1 by defensiveness bucket:

|  | budget_d | vol_ex_ante | exposure_equity | exposure_duration | exposure_credit | w_GLD | w_DBC | w_BIL |
|---|---|---|---|---|---|---|---|---|
| ('train', 'd~0') | 0.000 | 0.137 | 0.501 | 0.249 | 0.050 | 0.080 | 0.070 | 0.050 |
| ('train', 'low d') | 0.037 | 0.121 | 0.494 | 0.249 | 0.050 | 0.079 | 0.069 | 0.060 |
| ('train', 'high d') | 0.279 | 0.108 | 0.451 | 0.240 | 0.050 | 0.072 | 0.064 | 0.123 |
| ('validation', 'd~0') | 0.000 | 0.076 | 0.502 | 0.248 | 0.050 | 0.079 | 0.070 | 0.050 |
| ('validation', 'low d') | 0.024 | 0.076 | 0.496 | 0.249 | 0.050 | 0.079 | 0.069 | 0.058 |
| ('validation', 'high d') | 0.277 | 0.061 | 0.450 | 0.238 | 0.052 | 0.071 | 0.063 | 0.125 |

## 9. Investment constraints

Binding frequency of each constraint (share of P1/P2 decisions):

| constraint | P1_regime_aware | P2_ews_enhanced |
|---|---|---|
| credit_max_budget | 0.000 | 0.124 |
| equity_max_budget | 0.000 | 0.154 |
| max_BIL | 0.000 | 0.033 |
| sum_weights | 1.000 | 1.000 |
| turnover_cap | 0.000 | 0.002 |
| volatility_target | 0.274 | 0.336 |

Hard-constraint violations in published targets: 0. Turnover-limited transitions (risk budget not fully reached within one week): 1.

## 10. Portfolio results

Representative Validation conditions (dates chosen by fixed rules, not by performance):

|  | signal_date | signal | budget_d | SPY | EFA | EEM | IEF | TIP | HYG | GLD | DBC | BIL | equity | duration | credit | real_assets | liquidity | vol_ex_ante | cvar95_daily | effective_n | solver_status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ('P1 lowest d', 'P0_strategic') | 2017-02-03 |  |  | 0.352 | 0.103 | 0.051 | 0.146 | 0.098 | 0.050 | 0.078 | 0.072 | 0.049 | 0.507 | 0.244 | 0.050 | 0.150 | 0.049 | 0.081 | 0.012 | 5.418 | hold_within_no_trade_band |
| ('P1 lowest d', 'P1_regime_aware') | 2017-02-03 | 8.35623198414126e-08 | 0.000 | 0.352 | 0.103 | 0.051 | 0.146 | 0.098 | 0.050 | 0.078 | 0.072 | 0.049 | 0.507 | 0.244 | 0.050 | 0.150 | 0.049 | 0.081 | 0.012 | 5.418 | hold_within_no_trade_band |
| ('P1 lowest d', 'P2_ews_enhanced') | 2017-02-03 | GREEN | 0.000 | 0.352 | 0.103 | 0.051 | 0.146 | 0.098 | 0.050 | 0.078 | 0.072 | 0.049 | 0.507 | 0.244 | 0.050 | 0.150 | 0.049 | 0.081 | 0.012 | 5.418 | hold_within_no_trade_band |
| ('P2 RED', 'P0_strategic') | 2018-11-23 |  |  | 0.352 | 0.101 | 0.052 | 0.152 | 0.100 | 0.050 | 0.082 | 0.061 | 0.050 | 0.505 | 0.252 | 0.050 | 0.143 | 0.050 | 0.061 | 0.010 | 5.390 | hold_within_no_trade_band |
| ('P2 RED', 'P1_regime_aware') | 2018-11-23 | 1.0 | 0.283 | 0.316 | 0.091 | 0.043 | 0.140 | 0.100 | 0.051 | 0.072 | 0.058 | 0.129 | 0.451 | 0.239 | 0.051 | 0.130 | 0.129 | 0.055 | 0.009 | 5.971 | hold_within_no_trade_band |
| ('P2 RED', 'P2_ews_enhanced') | 2018-11-23 | RED | 1.000 | 0.231 | 0.035 | 0.034 | 0.107 | 0.182 | 0.050 | 0.049 | 0.061 | 0.250 | 0.300 | 0.289 | 0.050 | 0.111 | 0.250 | 0.040 | 0.006 | 5.827 | trade_to_target |
| ('P1 highest d', 'P0_strategic') | 2019-02-08 |  |  | 0.353 | 0.100 | 0.051 | 0.149 | 0.099 | 0.050 | 0.080 | 0.070 | 0.049 | 0.504 | 0.248 | 0.050 | 0.150 | 0.049 | 0.069 | 0.011 | 5.399 | hold_within_no_trade_band |
| ('P1 highest d', 'P1_regime_aware') | 2019-02-08 | 1.0 | 0.283 | 0.317 | 0.091 | 0.045 | 0.138 | 0.098 | 0.052 | 0.072 | 0.063 | 0.125 | 0.453 | 0.236 | 0.052 | 0.134 | 0.125 | 0.062 | 0.010 | 5.992 | hold_within_no_trade_band |
| ('P1 highest d', 'P2_ews_enhanced') | 2019-02-08 | AMBER | 0.333 | 0.308 | 0.089 | 0.043 | 0.138 | 0.097 | 0.053 | 0.070 | 0.062 | 0.141 | 0.440 | 0.234 | 0.053 | 0.132 | 0.141 | 0.060 | 0.009 | 6.089 | trade_to_target |
| ('P2 ORANGE', 'P0_strategic') | 2019-08-23 |  |  | 0.350 | 0.100 | 0.050 | 0.150 | 0.100 | 0.050 | 0.080 | 0.070 | 0.050 | 0.500 | 0.250 | 0.050 | 0.150 | 0.050 | 0.073 | 0.012 | 5.441 | trade_to_target |
| ('P2 ORANGE', 'P1_regime_aware') | 2019-08-23 | 0.9999999999999998 | 0.283 | 0.314 | 0.091 | 0.044 | 0.139 | 0.098 | 0.053 | 0.071 | 0.063 | 0.127 | 0.449 | 0.237 | 0.053 | 0.134 | 0.127 | 0.066 | 0.010 | 6.041 | trade_to_target |
| ('P2 ORANGE', 'P2_ews_enhanced') | 2019-08-23 | ORANGE | 0.667 | 0.259 | 0.078 | 0.036 | 0.127 | 0.096 | 0.057 | 0.061 | 0.054 | 0.233 | 0.373 | 0.223 | 0.057 | 0.115 | 0.233 | 0.056 | 0.009 | 6.103 | hold_within_no_trade_band |
| ('P2 RECOVERY', 'P0_strategic') | 2019-08-30 |  |  | 0.352 | 0.100 | 0.051 | 0.149 | 0.099 | 0.050 | 0.079 | 0.070 | 0.050 | 0.503 | 0.249 | 0.050 | 0.149 | 0.050 | 0.074 | 0.012 | 5.406 | hold_within_no_trade_band |
| ('P2 RECOVERY', 'P1_regime_aware') | 2019-08-30 | 0.9999999999995335 | 0.283 | 0.316 | 0.091 | 0.045 | 0.138 | 0.097 | 0.052 | 0.070 | 0.063 | 0.126 | 0.452 | 0.236 | 0.052 | 0.133 | 0.126 | 0.067 | 0.010 | 6.007 | hold_within_no_trade_band |
| ('P2 RECOVERY', 'P2_ews_enhanced') | 2019-08-30 | RECOVERY | 0.333 | 0.307 | 0.089 | 0.043 | 0.137 | 0.097 | 0.053 | 0.070 | 0.062 | 0.141 | 0.440 | 0.234 | 0.053 | 0.132 | 0.141 | 0.065 | 0.010 | 6.101 | trade_to_target |
| ('P2 AMBER', 'P0_strategic') | 2019-10-11 |  |  | 0.351 | 0.102 | 0.051 | 0.149 | 0.099 | 0.050 | 0.080 | 0.068 | 0.050 | 0.504 | 0.248 | 0.050 | 0.148 | 0.050 | 0.075 | 0.012 | 5.429 | hold_within_no_trade_band |
| ('P2 AMBER', 'P1_regime_aware') | 2019-10-11 | 0.999994761797165 | 0.283 | 0.314 | 0.093 | 0.045 | 0.138 | 0.096 | 0.052 | 0.071 | 0.061 | 0.128 | 0.453 | 0.235 | 0.052 | 0.133 | 0.128 | 0.068 | 0.010 | 6.029 | hold_within_no_trade_band |
| ('P2 AMBER', 'P2_ews_enhanced') | 2019-10-11 | AMBER | 0.333 | 0.309 | 0.091 | 0.044 | 0.134 | 0.095 | 0.053 | 0.069 | 0.063 | 0.141 | 0.444 | 0.230 | 0.053 | 0.132 | 0.141 | 0.067 | 0.010 | 6.090 | hold_within_no_trade_band |
| ('P2 GREEN', 'P0_strategic') | 2019-12-27 |  |  | 0.355 | 0.099 | 0.051 | 0.147 | 0.098 | 0.049 | 0.080 | 0.072 | 0.049 | 0.505 | 0.245 | 0.049 | 0.152 | 0.049 | 0.076 | 0.012 | 5.374 | hold_within_no_trade_band |
| ('P2 GREEN', 'P1_regime_aware') | 2019-12-27 | 4.998929043376066e-06 | 0.000 | 0.355 | 0.099 | 0.051 | 0.147 | 0.098 | 0.049 | 0.080 | 0.072 | 0.049 | 0.505 | 0.245 | 0.049 | 0.152 | 0.049 | 0.076 | 0.012 | 5.374 | hold_within_no_trade_band |
| ('P2 GREEN', 'P2_ews_enhanced') | 2019-12-27 | GREEN | 0.000 | 0.350 | 0.100 | 0.050 | 0.150 | 0.100 | 0.050 | 0.081 | 0.071 | 0.050 | 0.500 | 0.249 | 0.050 | 0.151 | 0.050 | 0.075 | 0.012 | 5.447 | hold_within_no_trade_band |

Mean ex-ante characteristics:

|  | vol_ex_ante | cvar95_daily_ex_ante | expected_return_ex_ante | tracking_error_ex_ante | effective_n | herfindahl | active_share |
|---|---|---|---|---|---|---|---|
| ('P0_strategic', 'train') | 0.1264 | 0.0190 | 0.0379 | 0.0008 | 5.4297 | 0.1842 | 0.0044 |
| ('P0_strategic', 'validation') | 0.0711 | 0.0107 | 0.0323 | 0.0006 | 5.4256 | 0.1843 | 0.0049 |
| ('P1_regime_aware', 'train') | 0.1197 | 0.0180 | 0.0359 | 0.0071 | 5.7205 | 0.1752 | 0.0440 |
| ('P1_regime_aware', 'validation') | 0.0674 | 0.0101 | 0.0311 | 0.0040 | 5.7405 | 0.1746 | 0.0453 |
| ('P2_ews_enhanced', 'train') | 0.1197 | 0.0180 | 0.0359 | 0.0077 | 5.6023 | 0.1789 | 0.0443 |
| ('P2_ews_enhanced', 'validation') | 0.0679 | 0.0102 | 0.0313 | 0.0037 | 5.5884 | 0.1793 | 0.0411 |

Turnover (ex-ante proxy against drifted pre-trade weights) and prospective costs (fraction of portfolio per year):

|  | decisions | mean_turnover | median_turnover | max_turnover | annualized_turnover | share_at_turnover_cap | stress_friction_share | annual_cost_5bps | annual_cost_10bps | annual_cost_25bps |
|---|---|---|---|---|---|---|---|---|---|---|
| ('P0_strategic', 'train') | 375 | 0.0030 | 0.0000 | 0.0257 | 0.1583 |  | 0.0907 | 0.0002 | 0.0004 | 0.0010 |
| ('P0_strategic', 'validation') | 208 | 0.0018 | 0.0000 | 0.0151 | 0.0924 |  | 0.0048 | 0.0001 | 0.0002 | 0.0005 |
| ('P1_regime_aware', 'train') | 375 | 0.0068 | 0.0000 | 0.0798 | 0.3518 | 0.0000 | 0.0907 | 0.0004 | 0.0008 | 0.0020 |
| ('P1_regime_aware', 'validation') | 208 | 0.0068 | 0.0000 | 0.0774 | 0.3536 | 0.0000 | 0.0048 | 0.0004 | 0.0007 | 0.0018 |
| ('P2_ews_enhanced', 'train') | 375 | 0.0259 | 0.0000 | 0.2005 | 1.3446 | 0.0027 | 0.0907 | 0.0016 | 0.0033 | 0.0082 |
| ('P2_ews_enhanced', 'validation') | 208 | 0.0257 | 0.0000 | 0.2481 | 1.3339 | 0.0000 | 0.0048 | 0.0014 | 0.0027 | 0.0068 |

Mean share of ex-ante volatility by ETF:

|  | SPY | EFA | EEM | IEF | TIP | HYG | GLD | DBC | BIL |
|---|---|---|---|---|---|---|---|---|---|
| ('P0_strategic', 'train') | 0.556 | 0.194 | 0.116 | -0.022 | 0.000 | 0.034 | 0.048 | 0.074 | -0.000 |
| ('P0_strategic', 'validation') | 0.587 | 0.174 | 0.114 | -0.013 | 0.006 | 0.032 | 0.020 | 0.080 | 0.000 |
| ('P1_regime_aware', 'train') | 0.558 | 0.191 | 0.114 | -0.023 | 0.000 | 0.036 | 0.048 | 0.075 | 0.000 |
| ('P1_regime_aware', 'validation') | 0.586 | 0.174 | 0.111 | -0.013 | 0.007 | 0.035 | 0.020 | 0.080 | 0.001 |
| ('P2_ews_enhanced', 'train') | 0.554 | 0.189 | 0.115 | -0.022 | 0.001 | 0.036 | 0.048 | 0.077 | 0.001 |
| ('P2_ews_enhanced', 'validation') | 0.585 | 0.173 | 0.110 | -0.013 | 0.008 | 0.035 | 0.020 | 0.082 | 0.001 |

## 11. Train versus Validation diagnostics

Train rows are retrospective reconstructions (signals from models fitted on all of Train).

|  | decisions | mean_abs_weekly_weight_change | mean_vol | mean_equity | mean_bil | mean_effective_n |
|---|---|---|---|---|---|---|
| ('P0_strategic', 'train') | 375 | 0.0010 | 0.1264 | 0.5012 | 0.0498 | 5.4297 |
| ('P0_strategic', 'validation') | 208 | 0.0008 | 0.0711 | 0.5012 | 0.0498 | 5.4256 |
| ('P1_regime_aware', 'train') | 375 | 0.0018 | 0.1197 | 0.4727 | 0.0914 | 5.7205 |
| ('P1_regime_aware', 'validation') | 208 | 0.0020 | 0.0674 | 0.4728 | 0.0925 | 5.7405 |
| ('P2_ews_enhanced', 'train') | 375 | 0.0061 | 0.1197 | 0.4719 | 0.0878 | 5.6023 |
| ('P2_ews_enhanced', 'validation') | 208 | 0.0061 | 0.0679 | 0.4754 | 0.0849 | 5.5884 |
| ('P1_vs_P2', 'train') | 375 |  | 0.0063 | 0.0278 | 0.0352 |  |
| ('P1_vs_P2', 'validation') | 208 |  | 0.0034 | 0.0275 | 0.0368 |  |

P1 versus P2 target differences:

| split | decisions | share_dates_identical | mean_active_share_between | mean_vol | mean_equity | mean_bil |
|---|---|---|---|---|---|---|
| train | 375 | 0.1307 | 0.0397 | 0.0063 | 0.0278 | 0.0352 |
| validation | 208 | 0.0865 | 0.0413 | 0.0034 | 0.0275 | 0.0368 |

## 12. Sensitivity analysis

Validation decisions, every variant restarted from the benchmark at the first Validation decision; changes are measured against the identically restarted reference. None of these variants was used to choose the primary policy.

|  | decisions | mean_abs_weight_change_vs_reference | max_abs_weight_change_vs_reference | mean_vol_ex_ante | mean_cvar95_daily | mean_equity | mean_duration | mean_bil | mean_turnover | mean_budget_d | fallbacks | hard_violations |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ('reference', 'P1_regime_aware') | 208 | 0.0000 | 0.0000 | 0.0674 | 0.0101 | 0.4728 | 0.2427 | 0.0925 | 0.0071 | 0.1559 | 0 | 0 |
| ('reference', 'P2_ews_enhanced') | 208 | 0.0000 | 0.0000 | 0.0679 | 0.0102 | 0.4754 | 0.2461 | 0.0849 | 0.0257 | 0.1378 | 0 | 0 |
| ('cov_sample_504', 'P1_regime_aware') | 208 | 0.0025 | 0.0236 | 0.0678 | 0.0101 | 0.4738 | 0.2359 | 0.1017 | 0.0083 | 0.1559 | 0 | 0 |
| ('cov_sample_504', 'P2_ews_enhanced') | 208 | 0.0016 | 0.0501 | 0.0683 | 0.0102 | 0.4762 | 0.2424 | 0.0897 | 0.0290 | 0.1378 | 0 | 0 |
| ('cov_ewma_hl63', 'P1_regime_aware') | 208 | 0.0026 | 0.0244 | 0.0657 | 0.0101 | 0.4740 | 0.2356 | 0.1018 | 0.0084 | 0.1559 | 0 | 0 |
| ('cov_ewma_hl63', 'P2_ews_enhanced') | 208 | 0.0021 | 0.1151 | 0.0661 | 0.0102 | 0.4762 | 0.2419 | 0.0897 | 0.0303 | 0.1378 | 1 | 0 |
| ('cov_lw_252', 'P1_regime_aware') | 208 | 0.0015 | 0.0206 | 0.0667 | 0.0101 | 0.4726 | 0.2455 | 0.0874 | 0.0068 | 0.1559 | 0 | 0 |
| ('cov_lw_252', 'P2_ews_enhanced') | 208 | 0.0010 | 0.0296 | 0.0671 | 0.0102 | 0.4751 | 0.2479 | 0.0817 | 0.0248 | 0.1378 | 0 | 0 |
| ('cov_lw_expanding', 'P1_regime_aware') | 208 | 0.0015 | 0.0128 | 0.1138 | 0.0101 | 0.4739 | 0.2404 | 0.0955 | 0.0073 | 0.1559 | 0 | 0 |
| ('cov_lw_expanding', 'P2_ews_enhanced') | 208 | 0.0015 | 0.0576 | 0.1145 | 0.0102 | 0.4762 | 0.2439 | 0.0873 | 0.0262 | 0.1378 | 0 | 0 |
| ('er_shrunk_historical', 'P1_regime_aware') | 208 | 0.0699 | 0.2127 | 0.0635 | 0.0095 | 0.4670 | 0.2719 | 0.0771 | 0.0460 | 0.1559 | 1 | 0 |
| ('er_shrunk_historical', 'P2_ews_enhanced') | 208 | 0.0678 | 0.2262 | 0.0635 | 0.0095 | 0.4643 | 0.2729 | 0.0722 | 0.0582 | 0.1378 | 2 | 0 |
| ('er_plus_1se', 'P1_regime_aware') | 208 | 0.0673 | 0.1973 | 0.0673 | 0.0098 | 0.4560 | 0.2964 | 0.0480 | 0.0101 | 0.1559 | 2 | 0 |
| ('er_plus_1se', 'P2_ews_enhanced') | 208 | 0.0666 | 0.2299 | 0.0678 | 0.0099 | 0.4599 | 0.2957 | 0.0448 | 0.0239 | 0.1378 | 2 | 0 |
| ('er_minus_1se', 'P1_regime_aware') | 208 | 0.0896 | 0.2054 | 0.0328 | 0.0046 | 0.2104 | 0.4483 | 0.2489 | 0.0052 | 0.1559 | 1 | 0 |
| ('er_minus_1se', 'P2_ews_enhanced') | 208 | 0.0914 | 0.2069 | 0.0326 | 0.0046 | 0.2080 | 0.4492 | 0.2494 | 0.0086 | 0.1378 | 1 | 0 |
| ('gamma_x0.5', 'P1_regime_aware') | 208 | 0.0000 | 0.0000 | 0.0674 | 0.0101 | 0.4728 | 0.2427 | 0.0925 | 0.0071 | 0.1559 | 0 | 0 |
| ('gamma_x0.5', 'P2_ews_enhanced') | 208 | 0.0000 | 0.0000 | 0.0679 | 0.0102 | 0.4754 | 0.2461 | 0.0849 | 0.0257 | 0.1378 | 0 | 0 |
| ('gamma_x2', 'P1_regime_aware') | 208 | 0.0486 | 0.2011 | 0.0457 | 0.0067 | 0.3073 | 0.2704 | 0.2487 | 0.0039 | 0.1559 | 1 | 0 |
| ('gamma_x2', 'P2_ews_enhanced') | 208 | 0.0502 | 0.2016 | 0.0457 | 0.0067 | 0.3066 | 0.2711 | 0.2490 | 0.0072 | 0.1378 | 1 | 0 |
| ('single_step_turnover_penalized', 'P1_regime_aware') | 208 | 0.0160 | 0.0606 | 0.0654 | 0.0098 | 0.4578 | 0.2567 | 0.0902 | 0.0022 | 0.1559 | 0 | 0 |
| ('single_step_turnover_penalized', 'P2_ews_enhanced') | 208 | 0.0308 | 0.1466 | 0.0629 | 0.0094 | 0.4223 | 0.2090 | 0.1836 | 0.0070 | 0.1378 | 0 | 0 |
| ('no_trade_band_0', 'P1_regime_aware') | 208 | 0.0010 | 0.0089 | 0.0672 | 0.0101 | 0.4717 | 0.2435 | 0.0923 | 0.0094 | 0.1559 | 0 | 0 |
| ('no_trade_band_0', 'P2_ews_enhanced') | 208 | 0.0008 | 0.0090 | 0.0677 | 0.0101 | 0.4743 | 0.2468 | 0.0852 | 0.0277 | 0.1378 | 0 | 0 |
| ('no_trade_band_0.02', 'P1_regime_aware') | 208 | 0.0013 | 0.0159 | 0.0675 | 0.0101 | 0.4742 | 0.2416 | 0.0920 | 0.0066 | 0.1559 | 0 | 0 |
| ('no_trade_band_0.02', 'P2_ews_enhanced') | 208 | 0.0008 | 0.0097 | 0.0681 | 0.0102 | 0.4764 | 0.2447 | 0.0845 | 0.0253 | 0.1378 | 0 | 0 |
| ('turnover_cap_halved', 'P1_regime_aware') | 208 | 0.0000 | 0.0000 | 0.0674 | 0.0101 | 0.4728 | 0.2427 | 0.0925 | 0.0071 | 0.1559 | 0 | 0 |
| ('turnover_cap_halved', 'P2_ews_enhanced') | 208 | 0.0004 | 0.0784 | 0.0680 | 0.0102 | 0.4762 | 0.2461 | 0.0838 | 0.0257 | 0.1378 | 7 | 0 |
| ('tracking_penalty_0', 'P1_regime_aware') | 208 | 0.0000 | 0.0000 | 0.0674 | 0.0101 | 0.4728 | 0.2427 | 0.0925 | 0.0071 | 0.1559 | 0 | 0 |
| ('tracking_penalty_0', 'P2_ews_enhanced') | 208 | 0.0000 | 0.0000 | 0.0679 | 0.0102 | 0.4754 | 0.2461 | 0.0849 | 0.0257 | 0.1378 | 0 | 0 |
| ('mapping_P1_full_range_P2_continuous', 'P1_regime_aware') | 208 | 0.0250 | 0.1421 | 0.0580 | 0.0086 | 0.3908 | 0.2685 | 0.1628 | 0.0240 | 0.5516 | 3 | 0 |
| ('mapping_P1_full_range_P2_continuous', 'P2_ews_enhanced') | 208 | 0.0065 | 0.2004 | 0.0673 | 0.0101 | 0.4706 | 0.2464 | 0.0906 | 0.0421 | 0.1617 | 3 | 0 |
| ('mapping_steeper_volatility', 'P1_regime_aware') | 208 | 0.0041 | 0.0411 | 0.0658 | 0.0099 | 0.4609 | 0.2398 | 0.1105 | 0.0100 | 0.1559 | 0 | 0 |
| ('mapping_steeper_volatility', 'P2_ews_enhanced') | 208 | 0.0036 | 0.0918 | 0.0666 | 0.0100 | 0.4650 | 0.2521 | 0.0926 | 0.0354 | 0.1378 | 2 | 0 |

## 13. Failure cases

- Turnover-limited transitions: 1 decisions (P2_ews_enhanced 1); the unmet budget is recorded per decision.
- BIL at its cap: P0_strategic 0.0%, P1_regime_aware 0.0%, P2_ews_enhanced 0.0% of decisions.
- Saturation: P1's budget is effectively binary (Train distinct d values: 718; most mass at the two extremes).
- Excessive defensiveness is limited by design: the equity floor (20%), the active-weight limit and the BIL cap bound how far any posture can move from the benchmark.
- Hold fallbacks or hard-constraint violations: see the checks in section 16 (any occurrence fails the stage).

## 14. Research limitations

- Validation (2016-2019) was reused in Stages 5-6.1; all Validation conclusions are development-sample evidence.
- HMM/AR/EWS parameters, Train-quantile thresholds and the P1 parity scale use all of Train; Train allocations are retrospective.
- Point-in-time macro handling is inherited from Stage 2 (ALFRED first releases where available).
- Gaussian HMM probabilities are saturated near 0 and 1; P(non-normal) is a regime score, not a calibrated loss probability, and pools S2 with S3.
- Few independent stress events (7 Validation SPY -5% drawdowns); risk-budget responses cannot be validated statistically on so few events.
- Expected returns are equilibrium assumptions; historical means are far too noisy to be forecasts.
- Transaction costs are illustrative scenarios (5/10/25 bps of traded notional, x2 in stress), not observed costs; adjusted closes are not guaranteed executable prices.
- Turnover is an ex-ante proxy (drift through t); exact execution-day turnover, after-cost returns and all realized performance belong to the Stage 8 walk-forward backtest.

## 15. Stage 8 handoff

`data/processed/portfolio_optimization/stage8_handoff_manifest.json` freezes: universe, benchmark, constraints, expected-return rule, covariance method, optimizer objective and penalties, risk-budget mappings (incl. the Train-calibrated P1 scale), CVaR parameters, weekly calendar, cost scenarios, execution timing, role bindings, solver and fallback policy, and every relevant hash. Primary policies to evaluate: P0_strategic, P1_regime_aware, P2_ews_enhanced (primary specification); listed sensitivity cases only.

## 16. Checks

| name | status | detail |
|---|---|---|
| Stage 6.1 handoff status READY | PASS | all blocking checks PASS. Documented limitations (WARN, not blocking): Every requested exposure budget is attained (else flagged NOT exposure-matched): A_p_non_normal@orange_exposure: target 13.02%, a |
| Split-file hashes match the handoff manifest | PASS | 4 files; mismatched: none |
| Stage 6 output hashes match the handoff manifest | PASS | 24 files; mismatched: none |
| Frozen model files (HMM, AR(1), AR-HMM) match the handoff hashes | PASS | hmm: data/processed/hmm/models/hmm_std_diag_k3.json, ar1: data/processed/ar_hmm/models/ar1.json, ar_hmm_k3: data/processed/ar_hmm/models/ar_hmm_k3.json |
| Stage 6 / 6.1 signal code unchanged since the handoff | PASS | 7 files; changed: none |
| Frozen model ID is hmm_std_diag_k3 for both roles | PASS | {'baseline_regime': 'hmm_std_diag_k3', 'challenger_ews': 'hmm_std_diag_k3'} |
| Signal dates strictly chronological and identical to the split calendar | PASS | 3064 sessions 2007-10-30..2019-12-31 |
| Decision timestamps valid (available after 16:00 ET on t; execution on t+1) | PASS | both roles |
| No Test file loaded | PASS | split_manifest.json, train_features.csv, train_returns.csv, validation_features.csv, validation_returns.csv |
| Return files: nine ETFs, daily LOG returns of adjusted closes, no missing values | PASS | definition from split manifest: daily log return log(P_t/P_{t-1}) from adjusted close (dividends and splits); same-day; missing 0 |
| Strategic benchmark loaded from configs/project.json, sums to 1, feasible under the policy | PASS | SPY 0.35, EFA 0.10, EEM 0.05, IEF 0.15, TIP 0.10, HYG 0.05, GLD 0.08, DBC 0.07, BIL 0.05 |
| Maximum-defensive (d = 1) linear constraints feasible | PASS | equity <= 0.30, credit <= 0.05, BIL >= 0.20 |
| Every covariance estimate positive definite after stabilization | PASS | min eigenvalue 3.01e-06; floored eigenvalues 0 (summed over estimators and dates) |
| Every published target satisfies all hard constraints | PASS | 0 of 1749 targets violate a hard constraint |
| Weights sum to one, no shorts, no leverage | PASS | max |sum-1| 2.7e-15; min weight 1.50e-02 |
| Risk-budget postures attainable under the static policy constraints | PASS | 0 decisions where even an unconstrained-turnover posture cannot meet the risk budget (closest feasible posture used and reported) |
| Optimizer: no hold fallback; transitions reported | PASS | 1166 decisions: target reached or held in band 1165, turnover-limited transition 1, hold fallback 0 |
| Solutions independent of the starting point (convex problem) | PASS | max weight difference between starts 1.6e-05 |
| BIL concentration below its cap on most decisions | PASS | BIL at its cap on 0.0% of decisions |
| P2 risk budget weakly decreasing GREEN -> AMBER -> ORANGE -> RED | PASS | AMBER 0.883, GREEN 1.000, ORANGE 0.767, RECOVERY 0.883, RED 0.650 |
| P0 posture equals the strategic benchmark; executed P0 weights within the no-trade band | PASS | 583 decisions; P0 traded back to the benchmark on 117 |
| Normal budget (d = 0) posture equals the benchmark for P1 and P2 (no built-in defensiveness) | PASS | 414 decisions with d = 0; max distance 2.7e-08 |
| P0, P1, P2 share identical decision, execution and holding dates | PASS | 583 decisions each |
| At most one execution per calendar week; execution after the signal; holding after execution | PASS | 1 decision(s) cross the Train/Validation boundary |
| P1 uses only role baseline_regime, P2 only role challenger_ews | PASS | defensiveness series built from one guarded role frame each |
| Changing every return after a decision date leaves all targets up to that date unchanged | PASS | cut 2018-01-05; P1 and P2 recomputed bit-for-bit |
| Every unmet risk budget is reported (only transitions or holds inside the no-trade band) | PASS | 519 decisions with an unmet budget component (all turnover-limited transitions) |
| Stage 5, 5.5, 6 and 6.1 artifacts unchanged (hashes before == after) | PASS | 109 files; changed: none |
| Automated tests on the real Train/Validation outputs | PASS | 13 tests, 0 failures, 0 errors, 0 skipped |
| No library warnings | WARN | RuntimeWarning: Values in x were outside bounds during a minimize step, clipping to bounds |

**Final: 29 PASS, 1 WARN, 0 FAIL - Stage 8 handoff READY.**

Automated real-data tests: 13 run, 0 failures, 0 errors, 0 skipped.

Figures: `figures/portfolio_optimization/strategic_asset_allocation.png`, `figures/portfolio_optimization/portfolio_weight_heatmap.png`, `figures/portfolio_optimization/portfolio_asset_class_allocation.png`, `figures/portfolio_optimization/regime_vs_ews_allocation.png`, `figures/portfolio_optimization/risk_budget_timeline.png`, `figures/portfolio_optimization/portfolio_risk_contribution.png`, `figures/portfolio_optimization/efficient_frontier_comparison.png`, `figures/portfolio_optimization/portfolio_turnover_comparison.png`, `figures/portfolio_optimization/cvar_risk_comparison.png`, `figures/portfolio_optimization/stress_scenario_allocation.png`, `figures/portfolio_optimization/portfolio_constraint_diagnostics.png`
