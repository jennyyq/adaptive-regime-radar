# Adaptive Regime Radar

## A Multi-Asset Regime Detection, Early-Warning, and Dynamic Portfolio Allocation Framework

Adaptive Regime Radar is an end-to-end quantitative research framework designed for **moderate-risk multi-asset investors**.

The project combines:

- point-in-time macro and market data engineering,
- economic risk-factor construction,
- unsupervised market regime detection,
- Hidden Markov Models,
- autoregressive regime-model robustness analysis,
- Bayesian Online Change Point Detection,
- a multi-layer early-warning system,
- dynamic risk budgeting,
- constrained portfolio optimization,
- realistic execution and transaction-cost modeling,
- and a final sequential out-of-sample portfolio evaluation.

The central investment question is:

> **Can market-regime information and early-warning signals improve downside protection and risk-adjusted portfolio performance relative to a strategic multi-asset allocation?**

---

# Research Status

**Status: COMPLETE — Stages 0–8**

The full research pipeline has been implemented and evaluated.

| Stage | Description | Status |
|---|---|---|
| 0–2 | Data acquisition and point-in-time alignment | Complete |
| 3 | Five economic risk factors and chronological dataset split | Complete |
| 4 | Regime exploration using K-Means and Gaussian Mixture Models | Complete |
| 5 | Gaussian HMM regime detection | Complete |
| 5.5 | AR-HMM robustness analysis | Complete |
| 6 | Bayesian change-point detection and Early Warning System | Complete |
| 6.1 | Early-warning audit and role-separated portfolio handoff | Complete |
| 7 | Robust portfolio optimization and dynamic risk budgeting | Complete |
| 8 | Final sequential out-of-sample portfolio backtest | Complete |

The official final evaluation uses an untouched Test period from **2020-01-02 to 2026-10-19**.

The Test set was accessed only after the Stage 8 protocol, code, model artifacts, portfolio policies, execution rules, and evaluation criteria had been frozen in a pre-Test lock.

---

# Final Research Findings

Three primary portfolios were evaluated:

- **P0 — Strategic Allocation:** fixed strategic multi-asset benchmark.
- **P1 — HMM Regime-Aware:** portfolio defensiveness driven by the frozen HMM regime probability.
- **P2 — Full Early-Warning System:** portfolio defensiveness driven by the complete EWS alert state.

All strategies use the same asset universe, optimization framework, weekly decision calendar, conservative `t+1` execution convention, and transaction-cost model.

### Final Test Performance

Test period: **2020-01-02 to 2026-10-19**

Primary transaction cost: **10 bps of gross traded notional**, doubled during predefined cost-stress periods.

| Portfolio | CAGR | Volatility | Sharpe | Sortino | Max Drawdown |
|---|---:|---:|---:|---:|---:|
| P0 Strategic | **9.82%** | 11.13% | 0.652 | 0.907 | -20.08% |
| P1 HMM Regime-Aware | 9.37% | 10.26% | **0.659** | **0.916** | -18.69% |
| P2 Full EWS | 8.80% | **9.64%** | 0.640 | 0.889 | **-17.30%** |

The HMM-based portfolio achieved a modestly higher point-estimate Sharpe ratio than the strategic benchmark, but the improvement was not statistically distinguishable from zero.

The full EWS reduced realized volatility and maximum drawdown more aggressively, but the additional downside protection did **not** translate into superior net risk-adjusted returns.

Its incremental performance was constrained by:

- lower average risky-asset exposure,
- negative timing contribution,
- missed participation in several market recoveries,
- and substantially higher portfolio turnover and transaction costs.

The official Stage 8 prespecified conclusion is:

> **Outcome 4 — Evidence is inconclusive.**

The results support the value of regime and early-warning information as a **risk-management overlay**, but do not establish statistically robust investment alpha.

---

# Research Integrity

A major objective of this project is to separate genuine out-of-sample evidence from retrospective model improvement.

The chronological research split is:

| Partition | Period | Purpose |
|---|---|---|
| Train | 2007-10-30 – 2015-12-31 | Model estimation |
| Validation | 2016-01-04 – 2019-12-31 | Model selection and development |
| Test | 2020-01-02 – 2026-10-19 | Final out-of-sample evaluation |

The Test set was not used during Stages 4–7.

Before Test access, Stage 8 generated a cryptographic pre-Test lock containing hashes of the frozen:

- source code,
- configuration,
- model artifacts,
- portfolio policies,
- execution assumptions,
- evaluation protocol,
- and initial portfolio states.

The final backtest then ran sequentially through the Test period without refitting the frozen regime or early-warning models.

The final Stage 8 audit produced:

```text
41 PASS
2 WARN
0 FAIL
```

The two warnings concern documented numerical solver/library warnings; all resulting portfolios passed independent hard-constraint checks.

---

# Post-Test Exploratory Analysis

After the official Test evaluation had been completed and examined, four small policy variants were evaluated to understand whether the Full EWS result was primarily caused by excessive defensiveness or trading frequency.

These experiments modified only:

- the no-trade band,
- and the mapping from EWS alert states to portfolio defensiveness.

They did **not** modify the underlying signals, HMM, BOCPD model, optimizer, portfolio constraints, execution lag, or cost methodology.

The best exploratory variant increased net Sharpe only from approximately:

```text
0.640 → 0.642
```

while the original HMM portfolio remained stronger at approximately:

```text
0.659
```

These experiments are explicitly classified as:

> **POST-TEST EXPLORATORY ANALYSIS**

They are **not independent out-of-sample evidence** and do not replace the official Stage 8 result.

No further parameter search is used to revise the reported Test performance.

---

# Asset Universe

The strategic portfolio contains nine US-listed ETFs:

| Asset | Strategic Weight | Role |
|---|---:|---|
| SPY | 35% | US equities |
| EFA | 10% | Developed ex-US equities |
| EEM | 5% | Emerging-market equities |
| IEF | 15% | US Treasury duration |
| TIP | 10% | Inflation-linked bonds |
| HYG | 5% | High-yield credit |
| GLD | 8% | Gold |
| DBC | 7% | Broad commodities |
| BIL | 5% | Treasury bills / defensive cash proxy |

Additional ETFs, volatility indices, and macroeconomic series are used for signal construction and diagnostics.

---

# Data Sources

## Yahoo Finance

Investment universe:

```text
SPY EFA EEM IEF TIP HYG GLD DBC BIL
```

Equity-trend and breadth indicators:

```text
QQQ IWM RSP
XLB XLE XLF XLI XLK XLP XLU XLV XLY
```

Credit:

```text
LQD
```

Volatility:

```text
^VIX ^VIX3M ^MOVE
```

## FRED / ALFRED

Interest rates:

```text
DGS3MO
DGS2
DGS10
T10Y2Y
T10Y3M
```

Inflation:

```text
DFII10
T10YIE
T5YIE
CPIAUCSL
CPILFESL
```

Liquidity and credit:

```text
BAA10Y
BAMLH0A0HYM2
NFCI
```

---

# Point-in-Time Data Framework

The project explicitly models when information would have been available to an investor.

Core rules include:

- The trading calendar follows actual SPY sessions.
- Signals are formed after the close of date `t`.
- Portfolio decisions cannot use information released after the decision timestamp.
- Macro observations are aligned using their estimated or observed release dates rather than observation dates.
- ALFRED first-release vintages are used where available.
- Conservative release lags are used where historical vintages are unavailable.
- Missing prices are not forward-filled.
- Missing returns remain `NaN`.
- Incomplete same-day bars are removed.
- Adjusted-close total-return data are used for portfolio accounting.

For CPI and NFCI, ALFRED first-release information is used when available. Earlier periods without usable historical vintages fall back to documented conservative release-date rules.

---

# Five Economic Risk Factors

The project compresses a broad market and macro dataset into five interpretable economic factors.

| Factor | Interpretation | Main Components |
|---|---|---|
| **Equity Trend** | Higher = stronger / broader risk-on trend | SPY momentum, 12-1 momentum, sector breadth, RSP/SPY, IWM/SPY |
| **Interest Rate Conditions** | Higher = tighter rates / larger rate shock | 2Y change, 10Y change, yield-curve inversion, Treasury volatility |
| **Inflation Pressure** | Higher = greater inflation pressure | CPI, core CPI, breakevens, TIP/IEF, commodities |
| **Volatility Stress** | Higher = greater market stress | VIX, realized volatility, downside volatility, VIX term structure, cross-asset correlation |
| **Liquidity / Credit Stress** | Higher = tighter liquidity / credit | BAA spread, HYG/LQD, NFCI, HYG illiquidity |

Each component is transformed using a trailing robust standardization procedure:

```text
Trailing median / MAD
→ economic direction adjustment
→ winsorization at ±5
→ equal-weight aggregation
```

The standardization uses only information available through each date.

Outputs include:

```text
data/processed/risk_factors.csv
data/processed/factor_components.csv
reports/factor_diagnostics.md
```

---

# Stage 4 — Market Regime Exploration

Run:

```bash
python3 scripts/06_explore_regimes.py
```

Stage 4 compares:

- K-Means,
- diagonal-covariance GMM,
- full-covariance GMM,

for:

```text
K = 2, 3, 4, 5, 6
```

Models are fitted exclusively on Train.

Validation is used only for scoring and model-complexity selection.

The original Stage 4 reference model was:

```text
gmm_diag_k2
```

with an economically interpretable two-state structure:

```text
S1 = Normal / Low Stress
S2 = Stress / Risk-Off
```

K=3 and K=4 remained relevant HMM candidates because their Validation performance was not decisively inferior and a dynamic transition model could support richer state structure.

---

# Stage 5 — Hidden Markov Model

Run:

```bash
python3 scripts/07_fit_hmm.py
```

The final selected regime model is:

```text
hmm_std_diag_k3
```

It is a three-state Gaussian HMM with diagonal covariance.

Candidate models include:

- Standard Gaussian HMM,
- Sticky HMM,
- K = 2, 3, 4,
- diagonal and full covariance variants.

Each candidate uses multiple fixed random seeds and multiple EM initializations.

Only causal forward-filtered probabilities are used for investment signals:

```text
P(Z_t | x_1, ..., x_t)
```

Smoothed probabilities and Viterbi paths are used only for diagnostics.

### Important HMM Finding

The HMM produces economically meaningful regime segmentation, but diagnostics reveal substantial serial dependence in factor residuals.

Therefore:

> High filtered state probabilities should not be interpreted as perfectly calibrated probabilities of an objectively observed economic state.

This limitation motivated the Stage 5.5 autoregressive robustness analysis.

---

# Stage 5.5 — AR-HMM Robustness

Run:

```bash
python3 scripts/07b_compare_ar_hmm.py
```

Additional models include:

- frozen Gaussian HMM,
- AR-HMM K=3,
- AR-HMM K=2,
- single-state AR(1),
- i.i.d. Gaussian reference model.

The purpose is not to replace the Stage 5 result automatically, but to investigate whether serial dependence materially affects predictive calibration and state inference.

An additional EM audit is available through:

```bash
python3 scripts/07c_audit_ar_hmm_em.py
```

The audit checks:

- EM objective trajectories,
- M-step optimality,
- variance-floor behavior,
- reproducibility,
- and consistency with previously saved models.

---

# Stage 6 — Early Warning System

Run:

```bash
python3 scripts/08_build_early_warning.py
```

The EWS combines slow-moving regime information with faster market-change indicators.

### Slow Layer

- HMM filtered probabilities
- normalized state entropy
- five-day regime-probability velocity

### Fast Layer

- AR(1) one-step innovations
- Bayesian Online Change Point Detection

### Confirmation Layer

- factor-level stress
- rates and inflation pressure

The composite EWS uses fixed component weights:

```text
Regime         0.25
Confirmation   0.25
Innovation     0.20
BOCPD          0.15
Velocity       0.10
Macro          0.05
```

Component scaling and thresholds are estimated from Train only.

The alert state machine produces:

```text
GREEN
AMBER
ORANGE
RED
RECOVERY
```

and incorporates confirmation rules, hysteresis, emergency escalation, and recovery logic.

---

# Stage 6.1 — Early-Warning Audit

Run:

```bash
python3 scripts/08b_audit_early_warning.py
```

Stage 6.1 audits the frozen Stage 6 signals without refitting or retuning the model.

It evaluates warning performance over:

```text
5 / 10 / 21 / 63 sessions
```

using both:

- the original permissive event-matching definition,
- and a stricter one-alert-episode-to-one-event framework.

Signal-ranking metrics and event-warning metrics are reported separately.

The Stage 7 handoff is role-separated:

```text
baseline_regime
challenger_ews
```

to prevent accidental mixing of the HMM baseline and EWS challenger.

---

# Stage 7 — Robust Portfolio Optimization

Run:

```bash
python3 scripts/09_optimize_portfolio.py
```

Three portfolios share the same optimization and implementation framework.

### P0 — Strategic Benchmark

Strategic multi-asset allocation with no regime signal.

### P1 — HMM Regime-Aware

Defensiveness is determined by:

```text
baseline_p_non_normal
```

### P2 — Full EWS

Defensiveness is determined by:

```text
challenger_alert_level
```

The portfolio engine uses:

- Ledoit-Wolf covariance estimation,
- 504-session primary risk window,
- equilibrium expected returns,
- long-only constraints,
- no leverage,
- group exposure constraints,
- dynamic risk budgets,
- turnover penalties,
- tracking penalties,
- no-trade bands,
- turnover caps,
- and a multi-stage optimization fallback.

The primary optimization problem is conceptually:

```text
maximize

w'μ
- γ/2 · w'Σw
- turnover penalty
- tracking penalty
```

subject to the portfolio and dynamic risk-budget constraints.

Stage 7 performs ex-ante portfolio research only and does not report realized Test performance.

---

# Stage 8 — Final Sequential Out-of-Sample Backtest

Run:

```bash
python3 scripts/10_backtest_portfolios.py
```

Stage 8 consists of two phases.

## Phase 8A — Pre-Test Lock

Before Test is opened, the system:

1. validates all Stage 7 handoff files,
2. verifies hashes of frozen artifacts,
3. reconstructs Train/Validation signals,
4. reproduces Stage 7 portfolio decisions,
5. runs accounting and execution tests,
6. freezes the evaluation protocol,
7. records initial portfolio states,
8. writes the cryptographic pre-Test lock.

To run only the preflight:

```bash
python3 scripts/10_backtest_portfolios.py --preflight-only
```

## Phase 8B — Final Evaluation

Test data can only be loaded through the lock-gated evaluation loader.

The frozen models are continued chronologically through Test without parameter refitting.

The primary execution convention is:

```text
Signal observed after close(t)
        ↓
Portfolio decision at t
        ↓
Execution at close(t+1)
        ↓
New holdings first earn return(t+2)
```

The previous holdings continue earning the `t+1` return.

Transaction costs are charged on gross traded notional.

Primary cost:

```text
10 bps
```

Sensitivity:

```text
5 bps
25 bps
```

Costs are doubled during predefined market-stress periods.

---

# Final Economic Interpretation

The project produces an important distinction between four separate questions:

```text
Regime classification
        ↓
Risk detection
        ↓
Portfolio response
        ↓
Realized investment performance
```

Success at one layer does not automatically imply success at the next.

The final results suggest that the regime and early-warning framework can meaningfully alter portfolio risk and reduce downside exposure.

However, the Full EWS does not generate robust incremental alpha relative to the simpler HMM strategy or the strategic benchmark.

In particular:

- P1 reduced volatility and drawdown modestly.
- P2 reduced downside exposure more aggressively.
- P2 incurred materially higher turnover.
- EWS timing contribution was not consistently positive.
- Recovery participation was sometimes sacrificed.
- Net Sharpe improvements were not statistically significant.

The main research conclusion is therefore:

> **Early-warning information appears more useful as a dynamic risk-management overlay than as a standalone source of return timing alpha.**

---

# Quick Start

Open the project directory in VS Code:

```text
adaptive-regime-radar
```

Confirm that the terminal and VS Code use the same Python interpreter:

```bash
which python3
python3 --version
```

Python **3.10 or later** is required.

No virtual environment is required by the project.

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Then run the complete pipeline sequentially:

```bash
python3 scripts/00_check_env.py
python3 scripts/01_download_data.py
python3 scripts/02_build_panels.py
python3 scripts/03_build_features.py
python3 scripts/04_build_factors.py
python3 scripts/05_prepare_datasets.py
python3 scripts/06_explore_regimes.py
python3 scripts/07_fit_hmm.py
python3 scripts/07c_audit_ar_hmm_em.py
python3 scripts/07b_compare_ar_hmm.py
python3 scripts/08_build_early_warning.py
python3 scripts/08b_audit_early_warning.py
python3 scripts/09_optimize_portfolio.py
python3 scripts/10_backtest_portfolios.py
```

Run the offline test suite with:

```bash
python3 -m unittest discover -s tests -v
```

---

# FRED API Key

A FRED API key is strongly recommended.

Create one through the Federal Reserve Bank of St. Louis and copy:

```bash
cp .env.example .env
```

Then add the API key to `.env`.

The file is excluded through `.gitignore`.

With an API key, supported macro series can use ALFRED vintage information to improve historical point-in-time alignment.

Without a key, the pipeline remains operational but uses documented conservative release-date approximations where required.

---

# Project Structure

```text
adaptive-regime-radar/
│
├── configs/
│   ├── project.json
│   ├── portfolio_policy.json
│   └── stage8_protocol.json
│
├── scripts/
│   ├── 00_check_env.py
│   ├── 01_download_data.py
│   ├── 02_build_panels.py
│   ├── 03_build_features.py
│   ├── 04_build_factors.py
│   ├── 05_prepare_datasets.py
│   ├── 06_explore_regimes.py
│   ├── 07_fit_hmm.py
│   ├── 07b_compare_ar_hmm.py
│   ├── 07c_audit_ar_hmm_em.py
│   ├── 08_build_early_warning.py
│   ├── 08b_audit_early_warning.py
│   ├── 09_optimize_portfolio.py
│   └── 10_backtest_portfolios.py
│
├── src/regime_radar/
│   ├── sources/
│   ├── pit.py
│   ├── panels.py
│   ├── features.py
│   ├── factors.py
│   ├── dataset.py
│   ├── regime_*.py
│   ├── hmm_*.py
│   ├── ar_hmm*.py
│   ├── bocpd.py
│   ├── innovation_signals.py
│   ├── early_warning*.py
│   ├── alert_state_machine.py
│   ├── portfolio_*.py
│   └── backtest_*.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
│
├── reports/
│   └── figures/
│
├── docs/
├── notebooks/
├── tests/
└── requirements.txt
```

---

# Main Research Outputs

Important reports include:

```text
reports/data_quality.md
reports/factor_diagnostics.md
reports/dataset_split_audit.md
reports/research_protocol.md
reports/regime_exploration.md
reports/hmm_model_selection.md
reports/ar_hmm_robustness.md
reports/ar_hmm_em_audit.md
reports/early_warning_research.md
reports/early_warning_audit.md
reports/portfolio_optimization_research.md
reports/stage8_pretest_protocol.md
reports/final_backtesting_research.md
```

Final Stage 8 figures include:

```text
cumulative_wealth_gross_net.png
portfolio_drawdowns.png
rolling_sharpe.png
rolling_volatility.png
realized_asset_allocation.png
regime_warning_portfolio_response.png
crisis_performance_comparison.png
ews_incremental_attribution.png
turnover_and_costs.png
risk_adjusted_performance_comparison.png
annual_returns_heatmap.png
transaction_cost_sensitivity.png
performance_uncertainty.png
investment_decision_summary.png
```

---

# Known Limitations

This is a research framework rather than a production trading system.

Important limitations include:

1. The final Test period contains only a limited number of independent market-stress episodes.
2. Validation data were reused during several development stages.
3. Some historical macro release dates require conservative approximations.
4. Adjusted-close execution is an accounting convention rather than evidence of obtainable historical fills.
5. Transaction costs are modeled rather than reconstructed from historical ETF quotes.
6. The Gaussian HMM has meaningful residual serial dependence and imperfect predictive calibration.
7. Stage 7 portfolio constraints and risk-budget functions are research assumptions rather than an approved investment mandate.
8. Dynamic risk reduction does not necessarily generate positive market-timing alpha.
9. Post-Test experiments are exploratory and cannot be interpreted as independent out-of-sample validation.

---

# Reproducibility and Research Philosophy

The project follows several principles throughout the pipeline:

- chronological Train / Validation / Test separation,
- point-in-time information alignment,
- causal signal construction,
- forward filtering rather than hindsight smoothing for decisions,
- explicit execution delays,
- realistic transaction-cost deductions,
- reproducible fixed random seeds,
- artifact hashing,
- frozen pre-Test protocols,
- sequential portfolio accounting,
- uncertainty estimation,
- and explicit reporting of negative results.

The objective is not to maximize a historical backtest metric through repeated parameter search.

Instead, the project aims to answer a more useful investment-research question:

> **Which parts of a regime-aware investment process survive realistic implementation, costs, timing constraints, and genuinely out-of-sample evaluation?**

## Reproducing the results

- The official numbers above come from a data snapshot downloaded in October 2026 (Test end 2026-10-19).
  Market data are not distributed with this repository.
- A fresh run downloads current data: the Test period will extend to the latest session and vendor-adjusted
  histories may be revised, so results will differ from the reported figures.
- On a fresh clone, `scripts/10_backtest_portfolios.py` performs a new pre-Test lock and a new first Test
  evaluation for that data snapshot. The committed `data/processed/backtesting/stage8_pretest_lock.json`
  documents the lock of the official run.
- Optional, post-test exploratory only (not part of the official evaluation):
  `python3 scripts/11_quick_sharpe_improvement.py`

Recommended setup for other users:

    python3 -m venv .venv && source .venv/bin/activate
    python3 -m pip install -r requirements.txt
    cp .env.example .env   # optional FRED API key

`hmmlearn` is optional (used only for a cross-check test).

---

# Disclaimer

This repository is a quantitative research project for educational and investment-research purposes.

It does not constitute investment advice, an investment recommendation, or a production-ready trading system.

Past simulated performance does not guarantee future results.
