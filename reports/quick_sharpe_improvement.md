# Quick Sharpe-improvement experiment (POST-TEST EXPLORATORY)

**Status: exploratory, after the Test period was examined.** None of these results is an independent out-of-sample evaluation. The official result remains Stage 8 V1 (`data/processed/backtesting/`, lock `73998035e0857aad...`).

Test 2020-01-02 to 2026-10-19 (1708 sessions, 6.78 years). Primary cost 10 bps of gross traded notional (x2 in cost stress).

## 1. Declared experiments (no other values tried)

| experiment | no-trade band | GREEN | AMBER | ORANGE | RED | RECOVERY |
|---|---|---|---|---|---|---|
| Exp A: band 2% | 2% | 0.000 | 0.333 | 0.667 | 1.000 | 0.333 |
| Exp B: RECOVERY = 0 | 1% | 0.000 | 0.333 | 0.667 | 1.000 | 0.000 |
| Exp C: moderate map | 1% | 0.000 | 0.250 | 0.500 | 0.750 | 0.000 |
| Exp D: moderate map + band 2% | 2% | 0.000 | 0.250 | 0.500 | 0.750 | 0.000 |

Original P2: band 1%, GREEN 0, AMBER 1/3, ORANGE 2/3, RED 1, RECOVERY 1/3. Alert states, signals, optimizer, constraints, calendar, execution lag and costs are unchanged.

## 2. Comparison table (net of 10 bps unless stated)

|  | gross Sharpe | net Sharpe | net CAGR | volatility | max drawdown | turnover / yr | cost drag / yr | 2020 | 2022 | 2024-2025 | mean equity | mean BIL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P0 original | 0.655 | 0.652 | 9.82% | 11.13% | -20.08% | 0.16 | 0.04% | 13.43% | -11.21% | 33.55% | 50.12% | 4.98% |
| P1 original | 0.667 | 0.659 | 9.37% | 10.26% | -18.69% | 0.39 | 0.09% | 12.14% | -10.21% | 31.83% | 46.95% | 9.28% |
| P2 original | 0.673 | 0.640 | 8.80% | 9.64% | -17.30% | 1.42 | 0.35% | 14.15% | -11.60% | 25.65% | 47.02% | 8.65% |
| B2 constant d (V1) | 0.649 | 0.645 | 9.35% | 10.48% | -19.07% | 0.15 | 0.04% | 12.68% | -10.64% | 32.02% | 46.97% | 9.45% |
| Exp A: band 2% | 0.673 | 0.641 | 8.81% | 9.65% | -17.29% | 1.38 | 0.34% | 14.24% | -11.60% | 25.72% | 47.21% | 8.63% |
| Exp B: RECOVERY = 0 | 0.663 | 0.626 | 8.72% | 9.77% | -17.91% | 1.61 | 0.39% | 14.61% | -12.33% | 25.27% | 47.46% | 7.98% |
| Exp C: moderate map | 0.669 | 0.641 | 9.08% | 10.09% | -17.52% | 1.25 | 0.31% | 14.42% | -11.92% | 27.65% | 48.23% | 7.41% |
| Exp D: moderate map + band 2% | 0.669 | 0.642 | 9.09% | 10.10% | -17.56% | 1.22 | 0.30% | 14.45% | -11.96% | 27.75% | 48.44% | 7.40% |

|  | net Sharpe 25 bps | net CAGR 25 bps | mean d (Test) | beta vs P0 | alpha vs P0 | recovery participation vs P0 (mean over events) | peak-to-trough protection vs P0 |
|---|---|---|---|---|---|---|---|
| P0 original | 0.647 | 9.75% | 0.000 |  |  | 0.00% | 0.00% |
| P1 original | 0.646 | 9.24% | 0.166 | 0.921 | 0.08% | -0.74% | 0.57% |
| P2 original | 0.590 | 8.28% | 0.156 | 0.852 | -0.01% | -1.31% | 0.54% |
| B2 constant d (V1) | 0.640 | 9.29% | 0.167 | 0.942 | -0.07% | -0.57% | 0.41% |
| Exp A: band 2% | 0.593 | 8.31% | 0.156 | 0.852 | -0.00% | -1.29% | 0.55% |
| Exp B: RECOVERY = 0 | 0.570 | 8.14% | 0.128 | 0.863 | -0.15% | -1.18% | 0.32% |
| Exp C: moderate map | 0.600 | 8.62% | 0.096 | 0.899 | -0.05% | -0.87% | 0.28% |
| Exp D: moderate map + band 2% | 0.602 | 8.65% | 0.096 | 0.899 | -0.04% | -0.85% | 0.28% |

## 3. Candidates against P2, P1 and the constant-defensiveness benchmark

| candidate | reference | net_sharpe_difference | sharpe_ci_low | sharpe_ci_high | max_drawdown_difference | net_sharpe_difference_25bps | mean_equity_difference | gross_active_return | average_exposure_part | timing_part | cost_effect | driver |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Exp A: band 2% | P2 original | 0.001 | -0.008 | 0.008 | 0.01% | 0.002 | 0.19% | 0.01% | 0.02% | -0.02% | 0.01% | average-exposure component dominates; Sharpe interval contains 0 |
| Exp A: band 2% | P1 original | -0.018 | -0.168 | 0.118 | 1.40% | -0.053 | 0.26% | -0.36% | 0.05% | -0.40% | -0.22% | timing component dominates; Sharpe interval contains 0 |
| Exp A: band 2% | B2 constant d (V1) | -0.004 | -0.161 | 0.134 | 1.77% | -0.047 | 0.24% | -0.31% | 0.05% | -0.36% | -0.27% | timing component dominates; Sharpe interval contains 0 |
| Exp B: RECOVERY = 0 | P2 original | -0.014 | -0.050 | 0.018 | -0.61% | -0.020 | 0.44% | -0.01% | 0.06% | -0.07% | -0.04% | timing component dominates; Sharpe interval contains 0 |
| Exp B: RECOVERY = 0 | P1 original | -0.033 | -0.193 | 0.110 | 0.78% | -0.076 | 0.51% | -0.37% | 0.09% | -0.46% | -0.28% | timing component dominates; Sharpe interval contains 0 |
| Exp B: RECOVERY = 0 | B2 constant d (V1) | -0.019 | -0.183 | 0.127 | 1.16% | -0.070 | 0.49% | -0.33% | 0.09% | -0.41% | -0.33% | timing component dominates; Sharpe interval contains 0 |
| Exp C: moderate map | P2 original | 0.002 | -0.041 | 0.052 | -0.22% | 0.009 | 1.21% | 0.27% | 0.16% | 0.11% | 0.04% | average-exposure component dominates; Sharpe interval contains 0 |
| Exp C: moderate map | P1 original | -0.017 | -0.133 | 0.086 | 1.17% | -0.047 | 1.28% | -0.10% | 0.18% | -0.28% | -0.19% | timing component dominates; Sharpe interval contains 0 |
| Exp C: moderate map | B2 constant d (V1) | -0.004 | -0.123 | 0.101 | 1.55% | -0.040 | 1.26% | -0.05% | 0.18% | -0.23% | -0.24% | timing component dominates; Sharpe interval contains 0 |
| Exp D: moderate map + band 2% | P2 original | 0.002 | -0.040 | 0.052 | -0.26% | 0.011 | 1.43% | 0.27% | 0.18% | 0.09% | 0.05% | average-exposure component dominates; Sharpe interval contains 0 |
| Exp D: moderate map + band 2% | P1 original | -0.017 | -0.133 | 0.086 | 1.13% | -0.045 | 1.49% | -0.09% | 0.21% | -0.30% | -0.19% | timing component dominates; Sharpe interval contains 0 |
| Exp D: moderate map + band 2% | B2 constant d (V1) | -0.003 | -0.123 | 0.103 | 1.51% | -0.038 | 1.48% | -0.04% | 0.21% | -0.25% | -0.24% | timing component dominates; Sharpe interval contains 0 |

Sharpe intervals: paired stationary bootstrap (mean block 21, 5000 draws, Stage 8 seed). Attribution: holdings-based daily identity; 'average-exposure part' = Test-average weight difference x returns (descriptive). A difference driven mainly by the average-exposure part comes from holding more (or less) risk on average, not from better timing.

## 4. Selection (rules stated in the brief; no second round)

- Ranking by net Sharpe (10 bps): Exp D: moderate map + band 2% (0.642) > Exp C: moderate map (0.641) > Exp A: band 2% (0.641) > Exp B: RECOVERY = 0 (0.626).
- Highest net Sharpe: Exp D: moderate map + band 2%.
- Most defensible risk-return candidate: Exp D: moderate map + band 2% (among candidates whose max drawdown is not worse than P0, lowest sum of ranks of net Sharpe, max drawdown and turnover (ties: higher net Sharpe)).
- Improves on both P2 and P1 (net Sharpe, 10 bps): none.
- Of those, still above both at 25 bps: none.
- Maximum drawdown worse than P0 (-20.08%): none.
- Net Sharpe above 0.70 / 0.75 / 0.80: 0.7: none; 0.75: none; 0.8: none.

## 5. Integrity checks

| name | status | detail |
|---|---|---|
| V1 pre-Test lock still valid (no frozen code, config or artifact changed) | PASS | lock 73998035e0857aad...; Test read through the gated loader |
| Saved V1 Test signals (baseline_regime) unchanged | PASS | signals_baseline_regime_test.csv |
| Saved V1 Test signals (challenger_ews) unchanged | PASS | signals_challenger_ews_test.csv |
| Signal calendar = split calendar (Train + Validation + Test) | PASS | 4772 sessions |
| Engine configuration identical to the locked V1 configuration | PASS | band 0.01, lambda_tracking 2.0 |
| Reused engine reproduces the official V1 net returns of P0, P1, P2, B2 (10 significant digits) | PASS | P0 5.0e-12, P1 5.0e-12, P2 5.0e-12, B2 5.0e-12 |
| Daily ledger reconciliation for every strategy (returns, drift, notional, costs, wealth) | PASS | 80 identity checks; failed 0 |
| Hard portfolio constraints: every decision-time target and every executed trade compliant (long only, no leverage, caps, groups, active limit, turnover cap) | PASS | target violations 0 (all history); Test trade executions violating 0; reported separately - passive drift during no-trade holds: none; cap projections: P2 1, A 1, B 4 |
| Execution lag: every order executed at the close of the next session (t+1) | PASS | all strategies |
| Modified level maps applied exactly; alert states unchanged; defensiveness monotone in severity | PASS | A: GREEN 0.00, AMBER 0.33, ORANGE 0.67, RED 1.00, RECOVERY 0.33; B: GREEN 0.00, AMBER 0.33, ORANGE 0.67, RED 1.00, RECOVERY 0.00; C: GREEN 0.00, AMBER 0.25, ORANGE 0.50, RED 0.75, RECOVERY 0.00; D: GREEN 0.00, AMBER 0.25, ORANGE 0.50, RED 0.75, RECOVERY 0.00 |
| No-trade band applied as declared (every hold is within the variant's band) | PASS | A max held distance 0.0200 (band 0.02), B max held distance 0.0100 (band 0.01), C max held distance 0.0100 (band 0.01), D max held distance 0.0200 (band 0.02) |
| Causality: changing returns after a Test decision date leaves earlier decisions and holdings unchanged (experiments C and D) | PASS | cut 2023-05-26 |
| Original Stage 8 outputs, pre-Test lock and frozen Stage 5-7 artifacts unchanged | PASS | 31 Stage 8 files, 130 frozen files compared; lock valid True |
| Solver warnings (documented, solutions verified by the hard-constraint check) | WARN | 5 records in Test decisions |

## 6. Interpretation limits

- Four variants were evaluated on the same, already-seen Test period; the best of four is selected with hindsight and its Sharpe is biased upward.
- Differences of a few hundredths of Sharpe over ~7 years are inside the bootstrap intervals; they are not evidence of a better strategy.
- Any candidate would need a new, untouched evaluation period (live paper trading) before it could replace the frozen P2 policy.

Figure: `reports/figures/quick_sharpe_comparison.png`
