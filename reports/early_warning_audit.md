# Stage 6.1 - Final release audit: early-warning evaluation and Stage 7 handoff

## HANDOFF STATUS: READY

all blocking checks PASS. Documented limitations (WARN, not blocking): Every requested exposure budget is attained (else flagged NOT exposure-matched): A_p_non_normal@orange_exposure: target 13.02%, achieved 20.36%, 419 Train days tied at 1.0 - comparisons at these budgets are reported as NOT exposure-matched

Checks: 30 PASS, 1 WARN, 0 FAIL. Train 2007-10-30..2015-12-31; Validation 2016-01-04..2019-12-31. Test not read. Events (SPY -5% drawdown onsets): Train 9, Validation 7.

READY requires every check to PASS or WARN; any FAIL makes the handoff BLOCKED and the Stage 7 loader refuses it. WARN items are documented limitations, not blockers.

## 1. Blockers and warnings

| name | status | detail |
|---|---|---|
| Every requested exposure budget is attained (else flagged NOT exposure-matched) | WARN | A_p_non_normal@orange_exposure: target 13.02%, achieved 20.36%, 419 Train days tied at 1.0 - comparisons at these budgets are reported as NOT exposure-matched |

## 2. Automated tests on the real Train/Validation outputs

15 tests run, 0 failures, 0 errors, 0 skipped. Full runner output: `data/processed/early_warning/audit/real_data_test_results.txt`.

## 3. Amendment: earlier real-data test failure (root cause and fix)

- Failing test in the previous run: `test_baseline_rule_threshold_is_train_q80` (identified by elimination: it is the only real-data test whose result depends on the numerical precision of P(non-normal) at the frozen threshold; the runner output is now saved with full tracebacks).
- Root cause 1 (implementation error): the Stage 7 handoff CSVs stored floats with 10 significant digits. On Train, 419 of 2058 days have P(non-normal) exactly 1.0 in float64 and 87 more lie within 1e-10 of 1.0; the frozen 80th-percentile threshold is 1.0. Values such as 1 - 1e-11 were written as '1', so the frozen rule `p >= threshold` could not be reproduced from the handoff files.
- Root cause 2 (test defect): the test compared with `threshold - 1e-12`; under saturation that tolerance itself flips days in [1 - 1e-12, 1). The assertion is now exact (stricter, not relaxed).
- Fix: floats are written with 17 significant digits and read with round-trip precision; an audit check verifies bit-for-bit reproduction of every handoff value and rule. No threshold, score or rule changed.

## 4. Scope

- Non-tuning audit: no model was refitted, no BOCPD prior, EWS weight, threshold or alert rule was changed, no tie-breaking was used, and no specification is selected from these diagnostics.
- Stage 5, 5.5 and 6 artifacts are preserved (hashes verified before and after); all outputs are new files.
- Validation was used in Stages 5, 5.5 and 6: everything below is development-sample evidence. Train signals are in-sample reconstructions.
- Ranking performance (PR-AUC, ROC-AUC of a continuous score against a daily loss label) and operational event-warning performance (binary alerts against drawdown events) are reported separately.

## 5. Interpretation

- Ranking (statistical): Validation PR-AUC is 0.205 for the full EWS score and 0.181 for P(non-normal) (base rate 0.111). Block-bootstrap difference (EWS minus P(non-normal)): the 95% interval (-0.036 to +0.148) contains zero - the difference is not resolved by these data. This is NOT evidence that the two scores are equivalent; the interval is also compatible with differences as large as 0.148.
- Operational (strict matching) at the C Train exposure 22.84% (amber exposure; C = state>=AMBER): W=5: A 2/7 signal-warned (1 executable), 0.00 false/yr, B 5/7 signal-warned (4 executable), 2.00 false/yr, C 3/7 signal-warned (2 executable), 1.00 false/yr; W=21: A 2/7 signal-warned (1 executable), 0.00 false/yr, B 6/7 signal-warned (5 executable), 2.00 false/yr, C 4/7 signal-warned (3 executable), 1.00 false/yr; W=63: A 2/7 signal-warned (1 executable), 0.00 false/yr, B 6/7 signal-warned (5 executable), 2.00 false/yr, C 5/7 signal-warned (4 executable), 1.00 false/yr.
- Operational (strict matching) at the C Train exposure 13.02% (orange exposure; C = state>=ORANGE) - NOT A FULLY EXPOSURE-MATCHED COMPARISON: A_p_non_normal@orange_exposure is NOT matched (Train 20.36% vs target 13.02%): W=5: A 0/7 signal-warned (0 executable), 0.00 false/yr, B 3/7 signal-warned (1 executable), 1.00 false/yr, C 2/7 signal-warned (0 executable), 0.00 false/yr; W=21: A 0/7 signal-warned (0 executable), 0.00 false/yr, B 5/7 signal-warned (3 executable), 1.00 false/yr, C 3/7 signal-warned (1 executable), 0.00 false/yr; W=63: A 0/7 signal-warned (0 executable), 0.00 false/yr, B 6/7 signal-warned (4 executable), 1.00 false/yr, C 4/7 signal-warned (2 executable), 0.00 false/yr.
- Matching convention: across all Validation specification-window cells the permissive convention credits 17 more early warnings than the strict one-to-one convention.
- Signal versus executable lead (strict, W=21, Validation): 26 event x specification cells are signal-warned, of which 8 have a signal lead of one session. Under the conservative convention those positions change at the close of the onset day itself, so they offer no protection before the -5% threshold is crossed.
- Onset-day alerts: 9 Validation event x specification cells (W=21) have an alert first raised on the onset day; none of them is counted as an early warning.
- Sample size: 7 Validation events. A difference of one or two warned events between specifications is within what chance alone can produce; no event-level comparison here is statistically resolved.
- No new model is selected by this audit. The Stage 6 decision stands: P(non-normal) is the simple regime baseline and the full EWS is the challenger for Stage 7.

## 6. Exposure budgets under ties and saturation

Rule: alert when score >= c, c = Train quantile at 1 - target (numpy linear interpolation). With ties, the achievable Train exposures are the step values share(score >= v); no chronological or random tie-breaking is used. A budget counts as exposure-matched when the achieved Train share is within 0.5 percentage point of the target. For B the 'Train days == 1.0' column is not meaningful.

| spec | target Train share | threshold (full precision) | Train days tied at threshold | Train days == 1.0 | achieved Train share | share strictly above threshold | Validation share | status |
|---|---|---|---|---|---|---|---|---|
| A_p_non_normal@budget20 | 20.0000% | 1.0 | 419 | 419 | 20.3596% | 0.0000% | 15.1093% | EXPOSURE-MATCHED |
| A_p_non_normal@amber_exposure | 22.8377% | 0.9999999999990792 | 0 | 419 | 22.8377% | 22.8377% | 19.1849% | EXPOSURE-MATCHED |
| A_p_non_normal@orange_exposure | 13.0224% | 1.0 | 419 | 419 | 20.3596% | 0.0000% | 15.1093% | NOT EXPOSURE-MATCHED |
| B_ews_score@budget20 | 20.0000% | 0.6145357660525279 | 0 | 0 | 20.0195% | 20.0195% | 20.7753% | EXPOSURE-MATCHED |
| B_ews_score@amber_exposure | 22.8377% | 0.5813777141840852 | 0 | 0 | 22.8446% | 22.8446% | 22.6640% | EXPOSURE-MATCHED |
| B_ews_score@orange_exposure | 13.0224% | 0.6908392818661123 | 0 | 0 | 13.0541% | 13.0541% | 12.0278% | EXPOSURE-MATCHED |

- **A_p_non_normal@orange_exposure: NOT EXPOSURE-MATCHED.** NOT EXPOSURE-MATCHED: target 13.02% not attained; the attainable Train shares around it are 0.00% and 20.36% because 419 Train days are tied at the threshold 1.0. Every comparison that uses this budget is labelled accordingly; it is not an equal-exposure comparison.

State-machine budgets (they define the AMBER / ORANGE exposure targets):

| spec | rule | train_time_share | validation_time_share |
|---|---|---|---|
| C_ews_state>=AMBER | alert level >= AMBER (RECOVERY counts as no alert) | 22.8377% | 21.4712% |
| C_ews_state>=ORANGE | alert level >= ORANGE (RECOVERY counts as no alert) | 13.0224% | 11.8290% |

## 7. Ranking performance (statistical)

Daily scores against daily labels on identical dates. The state machine is a discretization of the EWS score and has no separate ranking.

|  | PR-AUC A_p_non_normal | PR-AUC B_ews_score |
|---|---|---|
| ('SPY 21d loss >= 5%', 'train') | 0.348 | 0.311 |
| ('SPY 21d loss >= 5%', 'validation') | 0.181 | 0.205 |
| ('SPY 21d loss >= 5%', 'validation 2016-2017') | 0.060 | 0.070 |
| ('SPY 21d loss >= 5%', 'validation 2018-2019') | 0.217 | 0.262 |
| ('portfolio 21d loss >= 3%', 'train') | 0.320 | 0.280 |
| ('portfolio 21d loss >= 3%', 'validation') | 0.104 | 0.137 |
| ('portfolio 21d loss >= 3%', 'validation 2016-2017') | 0.049 | 0.043 |
| ('portfolio 21d loss >= 3%', 'validation 2018-2019') | 0.127 | 0.176 |

|  | ROC-AUC A_p_non_normal | ROC-AUC B_ews_score |
|---|---|---|
| ('SPY 21d loss >= 5%', 'train') | 0.669 | 0.637 |
| ('SPY 21d loss >= 5%', 'validation') | 0.657 | 0.643 |
| ('SPY 21d loss >= 5%', 'validation 2016-2017') | 0.549 | 0.601 |
| ('SPY 21d loss >= 5%', 'validation 2018-2019') | 0.455 | 0.528 |
| ('portfolio 21d loss >= 3%', 'train') | 0.679 | 0.638 |
| ('portfolio 21d loss >= 3%', 'validation') | 0.644 | 0.682 |
| ('portfolio 21d loss >= 3%', 'validation 2016-2017') | 0.919 | 0.910 |
| ('portfolio 21d loss >= 3%', 'validation 2018-2019') | 0.426 | 0.559 |

Validation block bootstrap (63-session blocks) of the PR-AUC difference, EWS minus P(non-normal):

| label | model | reference | point_ap_diff | ci95_low | ci95_high | share_positive | valid_reps |
|---|---|---|---|---|---|---|---|
| SPY 21d loss >= 5% | B_ews_score | A_p_non_normal | 0.0237 | -0.0364 | 0.1478 | 0.7040 | 1000 |
| portfolio 21d loss >= 3% | B_ews_score | A_p_non_normal | 0.0335 | -0.0340 | 0.1318 | 0.9242 | 989 |

- SPY 21d loss >= 5%: the 95% interval (-0.036 to +0.148) contains zero - the difference is not resolved by these data. This is NOT evidence that the two scores are equivalent; the interval is also compatible with differences as large as 0.148.
- portfolio 21d loss >= 3%: the 95% interval (-0.034 to +0.132) contains zero - the difference is not resolved by these data. This is NOT evidence that the two scores are equivalent; the interval is also compatible with differences as large as 0.132.

## 8. Operational event-warning performance - Validation

W/L/M = warned / detected late (first alert within 21 sessions after onset) / missed. Strict executable W = strict warnings with signal lead >= 2 (protection begins before onset under the conservative convention). Unmatched episodes: redundant = started inside an event's window but another episode was matched; premature = started 64..W+1 sessions before an onset; in-event = started while a drawdown was under way; only the remainder counts as strict false alerts.

### Warning window W = 5

| spec | exposure budget | time_on_alert | episodes/yr | permissive W/L/M | permissive false/yr | strict W/L/M | strict executable W | strict median lead | strict false/yr | strict precision | redundant / premature / in-event |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A_p_non_normal@budget20 | matched | 15.1% | 1.3 | 0/3/4 | 0.00 | 0/3/4 | 0 |  | 0.00 | 0.60 | 0 / 0 / 2 |
| A_p_non_normal@amber_exposure | matched | 19.2% | 1.3 | 2/2/3 | 0.00 | 2/2/3 | 1 | 2 | 0.00 | 0.80 | 0 / 0 / 1 |
| A_p_non_normal@orange_exposure | NOT matched (Train 20.36% vs target 13.02%) | 15.1% | 1.3 | 0/3/4 | 0.00 | 0/3/4 | 0 |  | 0.00 | 0.60 | 0 / 0 / 2 |
| B_ews_score@budget20 | matched | 20.8% | 9.5 | 6/1/0 | 3.01 | 6/1/0 | 5 | 2 | 2.00 | 0.18 | 1 / 11 / 11 |
| B_ews_score@amber_exposure | matched | 22.7% | 9.3 | 6/1/0 | 3.51 | 5/2/0 | 4 | 2 | 2.00 | 0.19 | 1 / 13 / 8 |
| B_ews_score@orange_exposure | matched | 12.0% | 7.8 | 3/1/3 | 1.00 | 3/1/3 | 1 | 1 | 1.00 | 0.13 | 1 / 6 / 16 |
| C_ews_state>=AMBER | matched | 21.5% | 6.0 | 3/3/1 | 2.00 | 3/3/1 | 2 | 2 | 1.00 | 0.25 | 0 / 7 / 7 |
| C_ews_state>=ORANGE | matched | 11.8% | 2.3 | 3/2/2 | 0.00 | 2/2/3 | 0 | 1 | 0.00 | 0.44 | 0 / 2 / 3 |

### Warning window W = 10

| spec | exposure budget | time_on_alert | episodes/yr | permissive W/L/M | permissive false/yr | strict W/L/M | strict executable W | strict median lead | strict false/yr | strict precision | redundant / premature / in-event |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A_p_non_normal@budget20 | matched | 15.1% | 1.3 | 0/3/4 | 0.00 | 0/3/4 | 0 |  | 0.00 | 0.60 | 0 / 0 / 2 |
| A_p_non_normal@amber_exposure | matched | 19.2% | 1.3 | 2/2/3 | 0.00 | 2/2/3 | 1 | 2 | 0.00 | 0.80 | 0 / 0 / 1 |
| A_p_non_normal@orange_exposure | NOT matched (Train 20.36% vs target 13.02%) | 15.1% | 1.3 | 0/3/4 | 0.00 | 0/3/4 | 0 |  | 0.00 | 0.60 | 0 / 0 / 2 |
| B_ews_score@budget20 | matched | 20.8% | 9.5 | 6/1/0 | 3.01 | 6/1/0 | 5 | 2 | 2.00 | 0.18 | 1 / 11 / 11 |
| B_ews_score@amber_exposure | matched | 22.7% | 9.3 | 6/1/0 | 3.51 | 6/1/0 | 5 | 2 | 2.00 | 0.19 | 1 / 12 / 9 |
| B_ews_score@orange_exposure | matched | 12.0% | 7.8 | 4/1/2 | 1.00 | 4/1/2 | 2 | 2 | 1.00 | 0.16 | 1 / 5 / 16 |
| C_ews_state>=AMBER | matched | 21.5% | 6.0 | 3/3/1 | 2.00 | 3/3/1 | 2 | 2 | 1.00 | 0.25 | 0 / 7 / 7 |
| C_ews_state>=ORANGE | matched | 11.8% | 2.3 | 3/2/2 | 0.00 | 3/2/2 | 1 | 1 | 0.00 | 0.56 | 0 / 1 / 3 |

### Warning window W = 21

| spec | exposure budget | time_on_alert | episodes/yr | permissive W/L/M | permissive false/yr | strict W/L/M | strict executable W | strict median lead | strict false/yr | strict precision | redundant / premature / in-event |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A_p_non_normal@budget20 | matched | 15.1% | 1.3 | 0/3/4 | 0.00 | 0/3/4 | 0 |  | 0.00 | 0.60 | 0 / 0 / 2 |
| A_p_non_normal@amber_exposure | matched | 19.2% | 1.3 | 2/2/3 | 0.00 | 2/2/3 | 1 | 2 | 0.00 | 0.80 | 0 / 0 / 1 |
| A_p_non_normal@orange_exposure | NOT matched (Train 20.36% vs target 13.02%) | 15.1% | 1.3 | 0/3/4 | 0.00 | 0/3/4 | 0 |  | 0.00 | 0.60 | 0 / 0 / 2 |
| B_ews_score@budget20 | matched | 20.8% | 9.5 | 6/1/0 | 3.01 | 6/1/0 | 5 | 2 | 2.00 | 0.18 | 5 / 7 / 11 |
| B_ews_score@amber_exposure | matched | 22.7% | 9.3 | 6/1/0 | 3.26 | 6/1/0 | 5 | 2 | 2.00 | 0.19 | 6 / 7 / 9 |
| B_ews_score@orange_exposure | matched | 12.0% | 7.8 | 5/1/1 | 1.00 | 5/1/1 | 3 | 3 | 1.00 | 0.19 | 3 / 2 / 16 |
| C_ews_state>=AMBER | matched | 21.5% | 6.0 | 4/2/1 | 2.00 | 4/2/1 | 3 | 2 | 1.00 | 0.25 | 1 / 6 / 7 |
| C_ews_state>=ORANGE | matched | 11.8% | 2.3 | 3/2/2 | 0.00 | 3/2/2 | 1 | 1 | 0.00 | 0.56 | 0 / 1 / 3 |

### Warning window W = 63

| spec | exposure budget | time_on_alert | episodes/yr | permissive W/L/M | permissive false/yr | strict W/L/M | strict executable W | strict median lead | strict false/yr | strict precision | redundant / premature / in-event |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A_p_non_normal@budget20 | matched | 15.1% | 1.3 | 1/3/3 | 0.00 | 0/3/4 | 0 |  | 0.00 | 0.60 | 0 / 0 / 2 |
| A_p_non_normal@amber_exposure | matched | 19.2% | 1.3 | 3/2/2 | 0.00 | 2/2/3 | 1 | 2 | 0.00 | 0.80 | 0 / 0 / 1 |
| A_p_non_normal@orange_exposure | NOT matched (Train 20.36% vs target 13.02%) | 15.1% | 1.3 | 1/3/3 | 0.00 | 0/3/4 | 0 |  | 0.00 | 0.60 | 0 / 0 / 2 |
| B_ews_score@budget20 | matched | 20.8% | 9.5 | 6/1/0 | 2.00 | 6/1/0 | 5 | 2 | 2.00 | 0.18 | 12 / 0 / 11 |
| B_ews_score@amber_exposure | matched | 22.7% | 9.3 | 6/1/0 | 2.00 | 6/1/0 | 5 | 2 | 2.00 | 0.19 | 13 / 0 / 9 |
| B_ews_score@orange_exposure | matched | 12.0% | 7.8 | 6/0/1 | 1.00 | 6/0/1 | 4 | 5 | 1.00 | 0.19 | 4 / 0 / 17 |
| C_ews_state>=AMBER | matched | 21.5% | 6.0 | 5/1/1 | 1.00 | 5/1/1 | 4 | 3 | 1.00 | 0.25 | 6 / 0 / 8 |
| C_ews_state>=ORANGE | matched | 11.8% | 2.3 | 5/1/1 | 0.00 | 4/2/1 | 2 | 6 | 0.00 | 0.67 | 0 / 0 / 3 |

Descriptive natural threshold, P(non-normal) >= 0.5:

| spec | exposure budget | time_on_alert | episodes/yr | permissive W/L/M | permissive false/yr | strict W/L/M | strict executable W | strict median lead | strict false/yr | strict precision | redundant / premature / in-event | window |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_p_non_normal@0.5 | descriptive | 54.9% | 2.5 | 6/0/1 | 1.50 | 2/0/5 | 2 | 4 | 1.25 | 0.20 | 0 / 3 / 0 | 5 |
| A_p_non_normal@0.5 | descriptive | 54.9% | 2.5 | 6/0/1 | 1.50 | 3/0/4 | 3 | 4 | 1.25 | 0.30 | 0 / 2 / 0 | 10 |
| A_p_non_normal@0.5 | descriptive | 54.9% | 2.5 | 6/0/1 | 1.50 | 4/0/3 | 4 | 6 | 1.25 | 0.40 | 0 / 1 / 0 | 21 |
| A_p_non_normal@0.5 | descriptive | 54.9% | 2.5 | 7/0/0 | 1.25 | 5/0/2 | 5 | 7 | 1.25 | 0.50 | 0 / 0 / 0 | 63 |

## 9. Signal lead versus executable protection lead (strict matching)

Signal lead = onset - first session of the matched alert episode (the signal exists after that close). Under the frozen convention the position changes at the close of the next session, so the executable protection lead = signal lead - 1 sessions of pre-onset returns. A signal lead of 1 is a warning on paper but gives no executable protection before onset.

| spec | W=5: signal / executable | W=10: signal / executable | W=21: signal / executable | W=63: signal / executable |
|---|---|---|---|---|
| A_p_non_normal@budget20 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| A_p_non_normal@amber_exposure | 2 / 1 | 2 / 1 | 2 / 1 | 2 / 1 |
| A_p_non_normal@orange_exposure | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| B_ews_score@budget20 | 6 / 5 | 6 / 5 | 6 / 5 | 6 / 5 |
| B_ews_score@amber_exposure | 5 / 4 | 6 / 5 | 6 / 5 | 6 / 5 |
| B_ews_score@orange_exposure | 3 / 1 | 4 / 2 | 5 / 3 | 6 / 4 |
| C_ews_state>=AMBER | 3 / 2 | 3 / 2 | 4 / 3 | 5 / 4 |
| C_ews_state>=ORANGE | 2 / 0 | 3 / 1 | 3 / 1 | 4 / 2 |

Event detail (Validation, strict, W=21): signal lead, executable lead, SPY drawdown from its peak at the execution close, and SPY return from the execution close to the onset close.

|  | signal_lead_sessions | executable_lead_sessions | drawdown_at_signal_pct | drawdown_at_execution_pct | spy_return_execution_to_onset_pct |
|---|---|---|---|---|---|
| (datetime.date(2016, 1, 13), 'A_p_non_normal@amber_exposure') | 4.00 | 3.00 | -3.63 | -4.69 | -1.61 |
| (datetime.date(2018, 2, 5), 'A_p_non_normal@amber_exposure') | 1.00 | 0.00 | -3.88 | -7.90 | 0.00 |
| (datetime.date(2016, 1, 13), 'B_ews_score@budget20') | 5.00 | 4.00 | -1.26 | -3.63 | -2.69 |
| (datetime.date(2018, 2, 5), 'B_ews_score@budget20') | 1.00 | 0.00 | -3.88 | -7.90 | 0.00 |
| (datetime.date(2018, 6, 27), 'B_ews_score@budget20') | 2.00 | 1.00 | -4.63 | -4.42 | -0.83 |
| (datetime.date(2018, 10, 11), 'B_ews_score@budget20') | 4.00 | 3.00 | -1.52 | -1.52 | -5.44 |
| (datetime.date(2019, 5, 29), 'B_ews_score@budget20') | 3.00 | 2.00 | -4.04 | -3.83 | -1.59 |
| (datetime.date(2019, 8, 5), 'B_ews_score@budget20') | 2.00 | 1.00 | -2.37 | -3.11 | -3.01 |
| (datetime.date(2016, 1, 13), 'B_ews_score@amber_exposure') | 6.00 | 5.00 | 0.00 | -1.26 | -5.02 |
| (datetime.date(2018, 2, 5), 'B_ews_score@amber_exposure') | 1.00 | 0.00 | -3.88 | -7.90 | 0.00 |
| (datetime.date(2018, 6, 27), 'B_ews_score@amber_exposure') | 2.00 | 1.00 | -4.63 | -4.42 | -0.83 |
| (datetime.date(2018, 10, 11), 'B_ews_score@amber_exposure') | 4.00 | 3.00 | -1.52 | -1.52 | -5.44 |
| (datetime.date(2019, 5, 29), 'B_ews_score@amber_exposure') | 3.00 | 2.00 | -4.04 | -3.83 | -1.59 |
| (datetime.date(2019, 8, 5), 'B_ews_score@amber_exposure') | 2.00 | 1.00 | -2.37 | -3.11 | -3.01 |
| (datetime.date(2016, 1, 13), 'B_ews_score@orange_exposure') | 3.00 | 2.00 | -4.69 | -4.59 | -1.71 |
| (datetime.date(2018, 2, 5), 'B_ews_score@orange_exposure') | 1.00 | 0.00 | -3.88 | -7.90 | 0.00 |
| (datetime.date(2018, 6, 27), 'B_ews_score@orange_exposure') | 18.00 | 17.00 | -4.15 | -3.69 | -1.58 |
| (datetime.date(2018, 10, 11), 'B_ews_score@orange_exposure') | 1.00 | 0.00 | -4.78 | -6.87 | 0.00 |
| (datetime.date(2019, 5, 29), 'B_ews_score@orange_exposure') | 7.00 | 6.00 | -2.79 | -3.43 | -2.00 |
| (datetime.date(2016, 1, 13), 'C_ews_state>=AMBER') | 3.00 | 2.00 | -4.69 | -4.59 | -1.71 |
| (datetime.date(2018, 6, 27), 'C_ews_state>=AMBER') | 18.00 | 17.00 | -4.15 | -3.69 | -1.58 |
| (datetime.date(2018, 10, 11), 'C_ews_state>=AMBER') | 2.00 | 1.00 | -1.66 | -4.78 | -2.20 |
| (datetime.date(2019, 5, 29), 'C_ews_state>=AMBER') | 1.00 | 0.00 | -4.72 | -5.36 | 0.00 |
| (datetime.date(2016, 1, 13), 'C_ews_state>=ORANGE') | 1.00 | 0.00 | -3.82 | -6.22 | 0.00 |
| (datetime.date(2018, 10, 11), 'C_ews_state>=ORANGE') | 1.00 | 0.00 | -4.78 | -6.87 | 0.00 |
| (datetime.date(2019, 5, 29), 'C_ews_state>=ORANGE') | 10.00 | 9.00 | -3.62 | -3.05 | -2.38 |

## 10. Operational event-warning performance - Train (in-sample)

### W = 21

| spec | exposure budget | time_on_alert | episodes/yr | permissive W/L/M | permissive false/yr | strict W/L/M | strict executable W | strict median lead | strict false/yr | strict precision | redundant / premature / in-event |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A_p_non_normal@budget20 | matched | 20.4% | 1.6 | 3/1/5 | 0.37 | 2/2/5 | 2 | 6 | 0.37 | 0.31 | 0 / 0 / 6 |
| A_p_non_normal@amber_exposure | matched | 22.8% | 1.1 | 4/0/5 | 0.24 | 3/0/6 | 2 | 6 | 0.24 | 0.33 | 0 / 0 / 4 |
| A_p_non_normal@orange_exposure | NOT matched (Train 20.36% vs target 13.02%) | 20.4% | 1.6 | 3/1/5 | 0.37 | 2/2/5 | 2 | 6 | 0.37 | 0.31 | 0 / 0 / 6 |
| B_ews_score@budget20 | matched | 20.0% | 10.8 | 6/1/2 | 3.43 | 6/1/2 | 6 | 4 | 1.96 | 0.08 | 8 / 13 / 44 |
| B_ews_score@amber_exposure | matched | 22.8% | 11.9 | 6/1/2 | 3.55 | 6/1/2 | 6 | 4 | 2.33 | 0.07 | 8 / 13 / 50 |
| B_ews_score@orange_exposure | matched | 13.0% | 7.6 | 6/0/3 | 2.20 | 6/0/3 | 6 | 4 | 1.71 | 0.10 | 6 / 4 / 32 |
| C_ews_state>=AMBER | matched | 22.8% | 5.6 | 6/0/3 | 1.71 | 5/1/3 | 4 | 6 | 1.10 | 0.13 | 1 / 6 / 24 |
| C_ews_state>=ORANGE | matched | 13.0% | 3.1 | 5/0/4 | 0.86 | 4/1/4 | 3 | 4 | 0.73 | 0.20 | 0 / 1 / 13 |

### W = 63

| spec | exposure budget | time_on_alert | episodes/yr | permissive W/L/M | permissive false/yr | strict W/L/M | strict executable W | strict median lead | strict false/yr | strict precision | redundant / premature / in-event |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A_p_non_normal@budget20 | matched | 20.4% | 1.6 | 3/1/5 | 0.37 | 2/2/5 | 2 | 6 | 0.37 | 0.31 | 0 / 0 / 6 |
| A_p_non_normal@amber_exposure | matched | 22.8% | 1.1 | 4/0/5 | 0.24 | 3/0/6 | 2 | 6 | 0.24 | 0.33 | 0 / 0 / 4 |
| A_p_non_normal@orange_exposure | NOT matched (Train 20.36% vs target 13.02%) | 20.4% | 1.6 | 3/1/5 | 0.37 | 2/2/5 | 2 | 6 | 0.37 | 0.31 | 0 / 0 / 6 |
| B_ews_score@budget20 | matched | 20.0% | 10.8 | 7/1/1 | 1.96 | 6/1/2 | 6 | 4 | 1.96 | 0.08 | 21 / 0 / 44 |
| B_ews_score@amber_exposure | matched | 22.8% | 11.9 | 7/1/1 | 2.33 | 7/1/1 | 7 | 4 | 2.33 | 0.08 | 20 / 0 / 50 |
| B_ews_score@orange_exposure | matched | 13.0% | 7.6 | 6/0/3 | 1.71 | 6/0/3 | 6 | 4 | 1.71 | 0.10 | 10 / 0 / 32 |
| C_ews_state>=AMBER | matched | 22.8% | 5.6 | 6/0/3 | 1.10 | 6/0/3 | 5 | 7 | 1.10 | 0.13 | 6 / 0 / 25 |
| C_ews_state>=ORANGE | matched | 13.0% | 3.1 | 5/0/4 | 0.73 | 4/1/4 | 3 | 4 | 0.73 | 0.20 | 1 / 0 / 13 |

## 11. Event-by-event detail (Validation, W = 21)

**Permissive** (W lead = signal lead; L +d = delay; M = missed)

| onset | A_p_non_normal@budget20 | A_p_non_normal@amber_exposure | A_p_non_normal@orange_exposure | B_ews_score@budget20 | B_ews_score@amber_exposure | B_ews_score@orange_exposure | C_ews_state>=AMBER | C_ews_state>=ORANGE |
|---|---|---|---|---|---|---|---|---|
| 2016-01-13 | L +0 (onset-day alert) | W 4 | L +0 (onset-day alert) | W 5 | W 6 | W 3 | W 3 | W 1 |
| 2016-06-27 | M | M | M | L +2 | L +0 (onset-day alert) | M | M | M |
| 2018-02-05 | L +0 (onset-day alert) | W 1 | L +0 (onset-day alert) | W 1 | W 1 | W 1 | L +0 (onset-day alert) | L +1 |
| 2018-06-27 | M | M | M | W 2 (onset-day alert) | W 2 | W 18 | W 18 | M |
| 2018-10-11 | L +11 | L +9 | L +11 | W 4 | W 4 | W 1 | W 2 | W 1 |
| 2019-05-29 | M | M | M | W 3 | W 3 | W 7 | W 1 | W 10 |
| 2019-08-05 | M | L +11 | M | W 2 | W 2 | L +0 (onset-day alert) | L +0 (onset-day alert) | L +2 |

**Strict** (W lead = signal lead; L +d = delay; M = missed)

| onset | A_p_non_normal@budget20 | A_p_non_normal@amber_exposure | A_p_non_normal@orange_exposure | B_ews_score@budget20 | B_ews_score@amber_exposure | B_ews_score@orange_exposure | C_ews_state>=AMBER | C_ews_state>=ORANGE |
|---|---|---|---|---|---|---|---|---|
| 2016-01-13 | L +0 (onset-day alert) | W 4 | L +0 (onset-day alert) | W 5 | W 6 | W 3 | W 3 | W 1 |
| 2016-06-27 | M | M | M | L +2 | L +0 (onset-day alert) | M | M | M |
| 2018-02-05 | L +0 (onset-day alert) | W 1 | L +0 (onset-day alert) | W 1 | W 1 | W 1 | L +0 (onset-day alert) | L +1 |
| 2018-06-27 | M | M | M | W 2 (onset-day alert) | W 2 | W 18 | W 18 | M |
| 2018-10-11 | L +11 | L +9 | L +11 | W 4 | W 4 | W 1 | W 2 | W 1 |
| 2019-05-29 | M | M | M | W 3 | W 3 | W 7 | W 1 | W 10 |
| 2019-08-05 | M | L +11 | M | W 2 | W 2 | L +0 (onset-day alert) | L +0 (onset-day alert) | L +2 |

## 12. Matching-convention audit

- Permissive (Stage 6): any alert session in [onset - W, onset - 1] counts; one long episode can warn several events and an episode that started long before the window still counts.
- Strict: an episode must START in [onset - W, onset - 1]; each episode warns at most one event (chronological, most recent start); an older episode still active at onset is a stale alert, not a warning.
- In both conventions the early window ends at onset - 1, so an alert first raised on the onset date is detected late (delay 0), never an early warning.
- The strict tables are re-verified from the raw alert series by an independent invariant check.
- Caveat: strict matching counts episode starts, so a flickering alert can be credited where one continuous alert would not; compare episodes per year.

| spec | (5, 'permissive') | (5, 'strict') | (10, 'permissive') | (10, 'strict') | (21, 'permissive') | (21, 'strict') | (63, 'permissive') | (63, 'strict') |
|---|---|---|---|---|---|---|---|---|
| A_p_non_normal@budget20 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| A_p_non_normal@amber_exposure | 2 | 2 | 2 | 2 | 2 | 2 | 3 | 2 |
| A_p_non_normal@orange_exposure | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| B_ews_score@budget20 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 |
| B_ews_score@amber_exposure | 6 | 5 | 6 | 6 | 6 | 6 | 6 | 6 |
| B_ews_score@orange_exposure | 3 | 3 | 4 | 4 | 5 | 5 | 6 | 6 |
| C_ews_state>=AMBER | 3 | 3 | 3 | 3 | 4 | 4 | 5 | 5 |
| C_ews_state>=ORANGE | 3 | 2 | 3 | 3 | 3 | 3 | 5 | 4 |

## 13. Stage 7 handoff

Manifest: `data/processed/early_warning/stage7_handoff/stage7_handoff_manifest.json` (handoff_status = **READY**).

Execution convention (frozen):

- **name**: conservative_close_t_plus_1
- **signal**: computed after the US close of t from information through the close of t
- **execution**: at the close of session t+1 (column execution_date / execution_time)
- **first_holding_return**: return_(t+2) = log(P_(t+2) / P_(t+1)) (column first_holding_return_date)
- **forbidden**: crediting a date-t signal with return_t or return_(t+1)
- **stage3_protocol_text**: ARR-stage3-protocol-v1.2 credits return_(t+1); that assumes execution at the close of t with information computed at that close and is optimistic by one session. It is not the Stage 7 convention; if ever shown, it must be labelled as an optimistic sensitivity.

### baseline_regime: Simple regime baseline

- Status: Stage 6 retained specification (pre-specified rule: least complex score not significantly worse than the best Validation PR-AUC).
- Signal columns: baseline_p_non_normal, baseline_alert_ge_q80; timing columns: signal_available_after, execution_date, execution_time, first_holding_return_date.

| file | split | rows | first_signal_date | last_signal_date | first_signal_available_after | first_execution_date | last_execution_date | rows_without_execution_date | sha256 |
|---|---|---|---|---|---|---|---|---|---|
| stage7_baseline_regime_train.csv | train | 2058 | 2007-10-30 | 2015-12-31 | 2007-10-30T16:00:00-04:00 | 2007-10-31 | 2016-01-04 | 0 | f0b664c55d3f4c893502f3291a2cb2bafcd46f898b0ff9054374e3ed7a76d53e |
| stage7_baseline_regime_validation.csv | validation | 1006 | 2016-01-04 | 2019-12-31 | 2016-01-04T16:00:00-05:00 | 2016-01-05 | 2019-12-31 | 1 | 9b7a4d266df9000cf5e0f7e26763a9a48512766a5bf6a1f97e2333806d18dde1 |

- Frozen rule: baseline_p_non_normal >= threshold with threshold 1.0 (numpy.nanquantile(Train P(non-normal), 0.8), linear interpolation (Stage 6 equal 20% budget)); Train days tied at the threshold 419, exactly 1.0 419; achieved Train share 20.3596%, Validation 15.1093%.
- Frozen model: hmm_std_diag_k3 (data/processed/hmm/models/hmm_std_diag_k3.json, sha256 90b93d15c91906f9412f68855fb606da28afef2a3303094f81d5c357d6aafd0b).

### challenger_ews: Full EWS challenger

- Status: challenger only; NOT adopted by the Stage 6 selection rule.
- Signal columns: challenger_ews_score, challenger_alert_level, challenger_alert_ge_amber, challenger_alert_ge_orange; timing columns: signal_available_after, execution_date, execution_time, first_holding_return_date.

| file | split | rows | first_signal_date | last_signal_date | first_signal_available_after | first_execution_date | last_execution_date | rows_without_execution_date | sha256 |
|---|---|---|---|---|---|---|---|---|---|
| stage7_challenger_ews_train.csv | train | 2058 | 2007-10-30 | 2015-12-31 | 2007-10-30T16:00:00-04:00 | 2007-10-31 | 2016-01-04 | 0 | 2f667d61e45bde70570022102d27dd2a0b82e7c4796e9d9653ef2842b010f26a |
| stage7_challenger_ews_validation.csv | validation | 1006 | 2016-01-04 | 2019-12-31 | 2016-01-04T16:00:00-05:00 | 2016-01-05 | 2019-12-31 | 1 | 023f084c53fbd6c48d3e5a9707cf2bac48678ac3133f02556197045427e2cf15 |

- Frozen thresholds: amber_in 0.6145357660525279, amber_out 0.5096056427959338, orange_in 0.7102663989842651, orange_out 0.6729408353664328, red_in 0.7944761486632409, red_out 0.7376187078020813.
- EWS weights: regime 0.25, confirmation 0.25, innovation 0.2, bocpd 0.15, velocity 0.1, macro 0.05, entropy 0.0.
- Rules: confirmation: 3 of 5 sessions above entry; emergency_red: {'p_stress': 0.9, 'confirmation_with_p_stress': 0.95, 'innovation_pct': 0.995, 'confirmation_with_innovation': 0.9}; recovery_clear_days: 10; alert_flag: RECOVERY counts as no alert.
- Frozen models: hmm data/processed/hmm/models/hmm_std_diag_k3.json sha256 90b93d15c91906f9412f68855fb606da28afef2a3303094f81d5c357d6aafd0b; ar1 data/processed/ar_hmm/models/ar1.json sha256 104d52e74c14b837ce4ffebabcac6259cd0cd43640888f0cdfc5573926f1fe60; ar_hmm_k3 data/processed/ar_hmm/models/ar_hmm_k3.json sha256 ed32ea8a8d63df0ebf5b675874b223f3fc04e0a8a729ef2a5d21aa42489087ff.

Permitted portfolio inputs: baseline_regime: baseline_p_non_normal, baseline_alert_ge_q80; challenger_ews: challenger_ews_score, challenger_alert_level, challenger_alert_ge_amber, challenger_alert_ge_orange; timing (both roles): signal_available_after, execution_date, execution_time, first_holding_return_date.

Descriptive only (never portfolio inputs): alert_reason, dominant_risk_drivers, contribution_*, component_*, p_s1, p_s2, p_s3, normalized_entropy, *_velocity, hmm_state (argmax), bocpd_*, innovation diagnostics, all Stage 6 / 6.1 tables and figures.

Forbidden: forward_outcomes.csv (fwd_ret_*, fwd_mdd_*), daily loss labels, risk_events.csv, any smoothed or Viterbi HMM output, any test-period file, thresholds or weights re-estimated on Validation, return_t or return_(t+1) for a date-t signal.

Controls against confusing P(non-normal) with full-EWS alerts:

- baseline and challenger live in different files with role prefixes 'baseline_' / 'challenger_'
- load_stage7_inputs(role, split) verifies READY status, the file hash and the exact column set
- requesting a column of the other role raises HandoffError; there is no combined role
- assert_single_role(frame) rejects frames that mix both prefixes
- results must be labelled with the role name; P(non-normal) is never called an EWS alert

Input hashes: split files train_features.csv d5f93bb6d133e966809f4cc26ba8fb4ef7b0fe110fa37955648715c81c0b5602, train_returns.csv a6d07d372b8adb6f820c623b23e4fe96922512693f722fee38ed23d2d2af25f3, validation_features.csv 9cb05de766b7949f7b13920b95f82c264224221d974bf0ea489da995acecfb97, validation_returns.csv 30ca6a5ca9530643e4db0270e485dcd7725ce6dffb238609bd7bb5b2b35e530e; 24 Stage 6 output files and 7 code files hashed in the manifest.

## 14. Remaining limitations

**Point in time**

- Execution convention frozen as conservative (close t+1, first holding return t+2). The Stage 3 protocol text credits return_(t+1) and is superseded for Stage 7; Stage 3 files were not modified.
- Train-period HMM, AR(1), AR-HMM and EWS values use parameters estimated on all of Train (in-sample); only Validation values are out-of-sample with respect to parameter estimation.
- Train-quantile thresholds and ECDF scales use the full Train period; a live system in, e.g., 2010 could not have known them.
- Macro factor point-in-time handling is inherited from Stage 2 (ALFRED first releases where available); any revision-sensitive series not covered there remains a residual risk.
- Signals are computed from end-of-day data that become final only after the close; intraday execution on the signal day is not supported.
- The last Validation rows have no execution / holding date because the Test calendar is not read.

**Statistical**

- Validation (2016-2019) was used in Stages 5, 5.5 and 6: all Validation results are development-sample evidence, not independent out-of-sample tests.
- Only 7 Validation and 9 Train SPY -5% drawdown events; event-level differences of one or two events are within noise.
- Block-bootstrap PR-AUC intervals containing zero do not establish equivalence.
- P(non-normal) is saturated at 1.0 on a large share of days; threshold rules on it are step functions and some exposure budgets are unattainable (see the exposure audit).
- The equity factor contains SPY momentum, so signals and SPY outcomes are not fully independent.
- Warning windows, event threshold (-5%) and late window (21) are conventions.

## 15. All checks

| name | status | detail |
|---|---|---|
| Stage 6 outputs were produced by the matching Stage 6 code | PASS | manifest model_version stage6-2026-10-21 |
| Stage 6 run had no FAILed checks | PASS | none |
| Frozen hmm file unchanged since Stage 6 | PASS | data/processed/hmm/models/hmm_std_diag_k3.json sha256 90b93d15c919... |
| Frozen ar1 file unchanged since Stage 6 | PASS | data/processed/ar_hmm/models/ar1.json sha256 104d52e74c14... |
| Frozen ar_hmm_k3 file unchanged since Stage 6 | PASS | data/processed/ar_hmm/models/ar_hmm_k3.json sha256 ed32ea8a8d63... |
| Split file train_features.csv unchanged since Stage 6 | PASS | sha256 d5f93bb6d133... |
| Split file train_returns.csv unchanged since Stage 6 | PASS | sha256 a6d07d372b8a... |
| Split file validation_features.csv unchanged since Stage 6 | PASS | sha256 9cb05de766b7... |
| Split file validation_returns.csv unchanged since Stage 6 | PASS | sha256 30ca6a5ca953... |
| Rebuilt signals reproduce the saved Stage 6 outputs (6 significant digits) | PASS | max relative difference: p_non_normal 4.3e-06, early_warning_score 4.8e-06, p_s2 4.7e-06, p_s3 4.3e-06 |
| Rebuilt alert levels identical to the saved Stage 6 alert levels | PASS | 0 differing days |
| Frozen Stage 6 thresholds reproduced from Train | PASS | max |difference| 0.0e+00 |
| P(non-normal) = P(S2) + P(S3) (filtered probabilities) | PASS | max |difference| 1.8e-15 |
| Event definition reproduces the Stage 6 risk events | PASS | Train 9, Validation 7 SPY -5% drawdown onsets |
| Permissive convention at W=63 reproduces the Stage 6 event results | PASS | 4 specifications x 2 splits x 5 counts |
| No alert first triggered on the onset date is counted as an early warning | PASS | 112 event-window rows with an alert first triggered on onset day; all classified late/missed unless an earlier in-window alert exists |
| Strict matching is one-to-one and episode accounting balances | PASS | duplicate matches 0; unbalanced none |
| Warned counts are non-decreasing in the window length | PASS | all specifications |
| Strict convention never credits more early warnings than the permissive one | PASS | 38 of 72 spec/split/window cells are stricter |
| Exposure audit is internally consistent (achieved shares recomputed, no tie-breaking) | PASS | 6 budget rules; tolerance 0.5% of Train sessions |
| Every requested exposure budget is attained (else flagged NOT exposure-matched) | WARN | A_p_non_normal@orange_exposure: target 13.02%, achieved 20.36%, 419 Train days tied at 1.0 - comparisons at these budgets are reported as NOT exposure-matched |
| Strict one-to-one matching verified independently from the raw alert series | PASS | 9 specifications x 2 splits x 4 windows, 0 violations |
| Executable protection lead = signal lead - 1 (execution at the close of t+1) | PASS | 88 warned event rows have signal lead 1 and therefore no executable protection before onset |
| No test file read | PASS | split_manifest.json, train_features.csv, train_returns.csv, validation_features.csv, validation_returns.csv |
| Stage 7 handoff files load through the guarded loader; cross-role and non-READY loads refused | PASS | 4 files verified; cross-role refused True; PENDING refused True |
| Handoff files reproduce the frozen signals and rules bit-for-bit (float64 round trip) | PASS | baseline_p_non_normal, baseline_alert_ge_q80, challenger_ews_score, challenger_alert_level; Train and Validation |
| Execution convention in the files: signal after close t, execute at close t+1, first holding return t+2 | PASS | execution_date = next session, first_holding_return_date = session after that, signal_available_after = t 16:00 America/New_York; last rows blank (Test calendar not read) |
| No handoff row on or after the test start | PASS | last signal 2019-12-31, test start 2020-01-01 |
| Automated tests on the real Train/Validation outputs | PASS | 15 tests, 0 failures, 0 errors, 0 skipped |
| Stage 5, 5.5 and 6 artifacts unchanged (hashes before == after) | PASS | 106 files; changed: none |
| No library warnings | PASS | none |

**Final: 30 PASS, 1 WARN, 0 FAIL - handoff READY.**

Figures: `figures/early_warning_audit/audit_window_sensitivity.png`, `figures/early_warning_audit/audit_event_matrix.png`
