"""Stage 5.5: predictive calibration, residual dependence, regime quality and the adoption decision for the
AR-HMM robustness experiment.

Every model is reduced to the same object: for each scored day t a Gaussian mixture predictive distribution
    p(x_t | x_1..x_t-1) = sum_k w_t,k N(x_t; m_t,k, diag(v_k)),   w_t = P(Z_t | x_1..x_t-1)
(Gaussian HMM: m_t,k = mu_k; AR-HMM: m_t,k = c_k + Phi_k x_t-1; AR(1): K = 1; i.i.d. Gaussian: K = 1, m = mean).
All diagnostics are computed from these one-step-ahead mixtures, i.e. before x_t enters the filter.
Scored dates are identical for every model: Train rows 2..T (row 1 is the first lag) and all Validation rows.
"""

from __future__ import annotations

STAGE55_CODE_VERSION = "stage5.5-2026-10-21b"  # must match scripts/07b_compare_ar_hmm.py

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2, norm
from sklearn.metrics import adjusted_rand_score

from . import regime_diagnostics as rd
from .ar_hmm import (ARFit, ARParams, conditional_means, filter_from_log_emission, filter_train_validation_ar,
                     fit_ar_hmm_best, lagged_pairs, log_emission_ar)
from .hmm_filtering import FilterResult, forward_filter
from .hmm_models import HMMParams, expected_durations, stress_scores

TRADING_DAYS = 252
COVERAGE_LEVELS = (0.5, 0.8, 0.9, 0.95)
ACF_LAGS = (1, 5, 21)
LJUNG_BOX_LAGS = 21
NW_LAGS = 21

# Adoption criteria, fixed on 2026-10-21 BEFORE any real-data AR-HMM result was produced.
ADOPTION_RULES = {
    "min_validation_gain_vs_frozen": 0.10,     # nats/day, B - A, and Newey-West t >= 2
    "min_validation_gain_vs_ar1": 0.0,         # B - C > 0 with Newey-West t >= 2 (latent states add value beyond AR)
    "nw_t": 2.0,
    "max_residual_acf1_ratio": 0.5,            # mean |lag-1 ACF| of Validation quantile residuals: B <= 0.5 x A
    "coverage90_band": (0.85, 0.95),           # B's Validation 90% interval coverage inside this band ...
    "coverage90_must_improve": True,           # ... and closer to 0.90 than A's
    "min_converged_seed_share": 0.8,
    "min_mean_pairwise_ari_train": 0.80,
    "min_train_state_share": 0.03,
    "max_switch_ratio_vs_frozen": 2.0,         # Validation switches/year of B <= 2 x those of A (usable regimes)
    "stress_state_rule": "highest-stress state satisfies the Stage 4 'Liquidity Crisis' rule on its Train profile "
                         "(volatility >= 1.0, liquidity_credit >= 1.0, equity_trend <= -0.5)",
}
ADOPTION_PROCEDURE = [
    "Replace the frozen HMM for Stage 6 only if ALL hold: (1) Validation log score B - A >= 0.10 nats/day with NW t "
    ">= 2; (2) B - C > 0 with NW t >= 2, i.e. the latent states add predictive value beyond a single-regime AR(1); "
    "(3) mean |lag-1 autocorrelation| of Validation quantile residuals at most half of A's; (4) Validation 90% "
    "interval coverage within [0.85, 0.95] and closer to 0.90 than A's; (5) >= 80% of seeds converge, EM monotone, "
    "seed ARI >= 0.80, every Train state >= 3% of days, Validation switches per year at most twice A's; (6) a "
    "stress state that meets the Stage 4 Liquidity Crisis rule.",
    "If (1)-(4) hold but (5) or (6) fails: keep the frozen HMM for regimes; the AR-HMM is a predictive-density "
    "challenger only.",
    "If (2) fails: the gain comes from autoregression, not from latent states; keep the frozen HMM for regime "
    "identification and report AR(1) as the density benchmark.",
    "Otherwise (mixed evidence): retain the frozen HMM; the AR-HMM remains a robustness challenger.",
    "Validation (2016-2019) was already used in Stage 5. This is a robustness experiment on the development sample, "
    "not a new holdout; the specification (K=3 primary, K=2 robustness, diagonal Phi and Sigma, |phi| <= 0.99) was "
    "fixed in advance and is not tuned on these results.",
]


ESTIMATION_AMENDMENTS = [
    {"version": "stage5.5 v1 -> v1.1", "date": "2026-10-21",
     "finding": "The v1 M-step set the innovation variance to s2 + 1e-4 (weighted residual variance plus a constant) and "
                "floored pi at 1e-8. s2 + 1e-4 is not the maximizer of the EM Q-function, so EM was not guaranteed to "
                "increase the likelihood. In the first real run some AR-HMM K=3 starts decreased their objective "
                "(all_monotone = False), and no check failed because only convergence of the reference fit was tested.",
     "fix": "Variance update replaced by the exact constrained maximizer max(s2, 1e-4); pi = gamma_1 (exact). Every "
            "start's trajectory is kept; any decrease above MONOTONE_ABS_TOL = 1e-7 nats now FAILS the run, and a "
            "Q-function optimality check verifies the M-step on Train. Audit: scripts/07c_audit_ar_hmm_em.py.",
     "first_run_outcome": "v1 (real data): ar_hmm_k3 Train 3.184 / Validation 3.354 nats per day, verdict "
                          "challenger_density; criteria 5 and 6 failed. All Stage 5.5 results are re-estimated with "
                          "the exact M-step; the frozen Stage 5 HMM is unaffected (it is not refitted)."},
]


# --------------------------------------------------------------------------------------
# Unified predictive representation
# --------------------------------------------------------------------------------------
@dataclass
class Predictive:
    """One-step-ahead Gaussian mixture predictions for one split."""
    weights: np.ndarray      # (T, K) P(Z_t | x_1..x_t-1)
    means: np.ndarray        # (T, K, D)
    var: np.ndarray          # (K, D)
    x: np.ndarray            # (T, D) realized
    log_score: np.ndarray    # (T,)
    filtered: np.ndarray     # (T, K)

    @property
    def mixture_mean(self) -> np.ndarray:
        return (self.weights[:, :, None] * self.means).sum(axis=1)

    @property
    def mixture_var(self) -> np.ndarray:
        second = (self.weights[:, :, None] * (self.var[None] + self.means ** 2)).sum(axis=1)
        return second - self.mixture_mean ** 2


@dataclass
class ModelResult:
    model_id: str
    description: str
    k: int
    n_parameters: int
    train: Predictive
    validation: Predictive
    params: object
    fits: list = field(default_factory=list)       # seed-level fits (new models only)
    stress_states: list[int] = field(default_factory=list)
    frozen: bool = False
    info: dict = field(default_factory=dict)

    @property
    def train_labels(self) -> np.ndarray:
        return self.train.filtered.argmax(axis=1)

    @property
    def validation_labels(self) -> np.ndarray:
        return self.validation.filtered.argmax(axis=1)

    def p_stress(self, split: str) -> np.ndarray:
        p = self.train if split == "train" else self.validation
        return p.filtered[:, self.stress_states].sum(axis=1) if self.stress_states else np.zeros(len(p.x))


def _predictive(res: FilterResult, means: np.ndarray, var: np.ndarray, x: np.ndarray) -> Predictive:
    return Predictive(res.predicted, means, var, np.asarray(x, float), res.log_score, res.filtered)


def frozen_hmm_result(params: HMMParams, x_train: np.ndarray, x_validation: np.ndarray, stress_states: list[int],
                      model_id: str) -> ModelResult:
    """Model A: the frozen Stage 5 HMM, filtered with its saved parameters (no refit). Train day 1 is filtered
    (from pi) but not scored, so that every model is scored on the same dates."""
    tr = forward_filter(params, x_train)
    va = forward_filter(params, x_validation, previous_filtered=tr.last_filtered)
    var = np.stack([np.diag(c) for c in params.full_covars()])
    tr_cut = FilterResult(tr.log_predicted[1:], tr.log_filtered[1:], tr.log_score[1:])
    m_tr = np.broadcast_to(params.means[None], (len(x_train) - 1,) + params.means.shape)
    m_va = np.broadcast_to(params.means[None], (len(x_validation),) + params.means.shape)
    k, d = params.means.shape
    n_par = (k - 1) + k * (k - 1) + 2 * k * d
    return ModelResult(model_id, "A - frozen Stage 5 Gaussian HMM (no refit)", k, n_par,
                       _predictive(tr_cut, m_tr, var, x_train[1:]), _predictive(va, m_va, var, x_validation),
                       params, stress_states=list(stress_states), frozen=True,
                       info={"train_day1_log_score_excluded": float(tr.log_score[0]),
                             "validation_avg_loglik_all_days": float(va.log_score.mean())})


def iid_gaussian_result(x_train: np.ndarray, x_validation: np.ndarray) -> ModelResult:
    """Reference only: one Gaussian with Train mean and variance (no dynamics, no states)."""
    mu, var = x_train.mean(axis=0), x_train.var(axis=0) + 1e-4
    res = {}
    for split, x in (("train", x_train[1:]), ("validation", x_validation)):
        ls = -0.5 * (np.log(2 * math.pi * var) + (x - mu) ** 2 / var).sum(axis=1)
        one = np.ones((len(x), 1))
        res[split] = Predictive(one, np.broadcast_to(mu[None, None], (len(x), 1, len(mu))), var[None], x, ls, one)
    return ModelResult("iid_gaussian", "reference - single i.i.d. Gaussian (no dynamics, no states)", 1,
                       2 * len(mu), res["train"], res["validation"], {"mean": mu, "var": var})


def _order_by_profile(params: ARParams, x_train, train_features: pd.DataFrame, columns) -> np.ndarray:
    y, lag = lagged_pairs(x_train)
    labels = filter_from_log_emission(log_emission_ar(params, y, lag), params.transmat,
                                      initial_predicted=params.startprob).filtered.argmax(axis=1)
    prof = rd.factor_profile(train_features.iloc[1:][list(columns)], labels, params.k)[list(columns)]
    score = stress_scores(prof.to_numpy(), list(columns))
    score = np.where(np.isnan(score), np.inf, score)
    return np.argsort(score, kind="stable")


def ar_model_result(model_id: str, description: str, k: int, x_train, x_validation, train_features: pd.DataFrame,
                    seeds, n_init: int, columns) -> ModelResult:
    """Fit AR-HMM (k >= 2) or AR(1) (k = 1) on Train only, order states by Train profile stress score, filter."""
    fits = [fit_ar_hmm_best(x_train, k, s, n_init) for s in (seeds if k > 1 else seeds[:1])]
    pool = [f for f in fits if f.converged] or fits
    ref = max(pool, key=lambda f: f.train_loglik)
    if k > 1:
        ref.params = ref.params.permuted(_order_by_profile(ref.params, x_train, train_features, columns))
    tr, va = filter_train_validation_ar(ref.params, x_train, x_validation)
    y_tr, lag_tr = lagged_pairs(x_train)
    y_va, lag_va = lagged_pairs(x_validation, previous=x_train[-1])
    result = ModelResult(model_id, description, k, ref.params.n_parameters(),
                         _predictive(tr, conditional_means(ref.params, lag_tr), ref.params.var, y_tr),
                         _predictive(va, conditional_means(ref.params, lag_va), ref.params.var, y_va),
                         ref.params, fits=fits)
    result.info.update({"reference_fit": ref, "x_train": x_train, "x_validation": x_validation})
    if k > 1:
        prof = rd.factor_profile(train_features.iloc[1:][list(columns)], result.train_labels, k)[list(columns)]
        scores = stress_scores(prof.to_numpy(), list(columns))
        overall = float(stress_scores(train_features.iloc[1:][list(columns)].to_numpy(), list(columns)).mean())
        chosen = [int(j) for j in np.flatnonzero(scores > overall)]
        result.stress_states = chosen if 0 < len(chosen) < k else [int(np.nanargmax(scores))]
    return result


# --------------------------------------------------------------------------------------
# Mixture predictive distribution: PIT, intervals, CRPS
# --------------------------------------------------------------------------------------
def mixture_cdf(p: Predictive, values: np.ndarray) -> np.ndarray:
    """(T, D) P(X_t,d <= values_t,d) under the marginal mixture of factor d."""
    s = np.sqrt(p.var)[None]
    return (p.weights[:, :, None] * norm.cdf((np.asarray(values)[:, None, :] - p.means) / s)).sum(axis=1)


def pit(p: Predictive) -> np.ndarray:
    return mixture_cdf(p, p.x)


def mixture_quantile(p: Predictive, q: float, iterations: int = 80) -> np.ndarray:
    """(T, D) q-quantile of each marginal mixture by bisection (exact up to 1e-12 x range)."""
    s = np.sqrt(p.var)[None]
    lo = (p.means - 10 * s).min(axis=1)
    hi = (p.means + 10 * s).max(axis=1)
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        below = mixture_cdf(p, mid) < q
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return 0.5 * (lo + hi)


def _crps_a(mu, var):
    s = np.sqrt(var)
    z = mu / s
    return 2 * s * norm.pdf(z) + mu * (2 * norm.cdf(z) - 1)


def mixture_crps(p: Predictive) -> np.ndarray:
    """(T, D) CRPS of the marginal Gaussian mixture (closed form, Grimit et al. 2006)."""
    w, m, v = p.weights[:, :, None], p.means, p.var[None]
    first = (w * _crps_a(p.x[:, None, :] - m, v)).sum(axis=1)
    pair = w[:, :, None] * w[:, None, :] * _crps_a(m[:, :, None] - m[:, None, :], v[:, :, None] + v[:, None, :])
    return first - 0.5 * pair.sum(axis=(1, 2))


def quantile_residuals(p: Predictive) -> np.ndarray:
    """Phi^-1(PIT): N(0,1) and serially independent if the predictive distribution is correct."""
    return norm.ppf(np.clip(pit(p), 1e-10, 1 - 1e-10))


def calibration_table(m: ModelResult, columns) -> pd.DataFrame:
    rows = []
    for split, p in (("train", m.train), ("validation", m.validation)):
        u, crps = pit(p), mixture_crps(p)
        widths = {lvl: mixture_quantile(p, 0.5 + lvl / 2) - mixture_quantile(p, 0.5 - lvl / 2)
                  for lvl in COVERAGE_LEVELS}
        for j, f in enumerate(list(columns) + ["all factors"]):
            uu = u[:, j] if j < len(columns) else u.ravel()
            row = {"model_id": m.model_id, "split": split, "factor": f, "n": len(uu),
                   "pit_mean": float(uu.mean()), "pit_std": float(uu.std()),
                   "pit_ks": float(np.abs(np.sort(uu) - (np.arange(1, len(uu) + 1) - 0.5) / len(uu)).max()),
                   "crps": float(crps[:, j].mean() if j < len(columns) else crps.mean())}
            for lvl in COVERAGE_LEVELS:
                inside = (uu > 0.5 - lvl / 2) & (uu < 0.5 + lvl / 2)
                row[f"coverage_{int(lvl * 100)}"] = float(inside.mean())
                wd = widths[lvl][:, j] if j < len(columns) else widths[lvl]
                row[f"mean_width_{int(lvl * 100)}"] = float(np.mean(wd))
            rows.append(row)
    return pd.DataFrame(rows)


def pit_histogram(m: ModelResult, split: str, bins: int = 10) -> np.ndarray:
    p = m.train if split == "train" else m.validation
    h, _ = np.histogram(pit(p).ravel(), bins=bins, range=(0, 1))
    return h / h.sum() * bins


# --------------------------------------------------------------------------------------
# Residual dependence
# --------------------------------------------------------------------------------------
def acf(x: np.ndarray, lag: int) -> float:
    x = np.asarray(x, float)
    x = x - x.mean()
    den = (x * x).sum()
    return float((x[lag:] * x[:-lag]).sum() / den) if den > 0 and len(x) > lag else np.nan


def ljung_box(x: np.ndarray, lags: int = LJUNG_BOX_LAGS) -> tuple[float, float]:
    n = len(x)
    q = n * (n + 2) * sum(acf(x, h) ** 2 / (n - h) for h in range(1, lags + 1))
    return float(q), float(chi2.sf(q, lags))


def residual_table(m: ModelResult, columns) -> pd.DataFrame:
    rows = []
    for split, p in (("train", m.train), ("validation", m.validation)):
        qr = quantile_residuals(p)
        pearson = (p.x - p.mixture_mean) / np.sqrt(p.mixture_var)
        labels = p.filtered.argmax(axis=1)
        own = (p.x - p.means[np.arange(len(labels)), labels]) / np.sqrt(p.var[labels])
        same = labels[1:] == labels[:-1]
        for j, f in enumerate(columns):
            q, pv = ljung_box(qr[:, j])
            a, b = own[1:, j][same], own[:-1, j][same]
            rows.append({"model_id": m.model_id, "split": split, "factor": f,
                         **{f"qres_acf_lag{L}": acf(qr[:, j], L) for L in ACF_LAGS},
                         "qres_sq_acf_lag1": acf(qr[:, j] ** 2, 1),
                         "ljung_box_q21": q, "ljung_box_p21": pv,
                         "qres_mean": float(qr[:, j].mean()), "qres_std": float(qr[:, j].std()),
                         "pearson_innovation_std": float(pearson[:, j].std()),
                         "within_state_resid_acf1": float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else np.nan})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Scores, paired comparisons
# --------------------------------------------------------------------------------------
def paired_test(a: np.ndarray, b: np.ndarray) -> dict:
    d = np.asarray(a) - np.asarray(b)
    se = rd.newey_west_se(d, NW_LAGS)
    t = float(d.mean() / se) if se and se > 0 else np.nan
    return {"mean_diff": float(d.mean()), "nw_se": se, "nw_t": t,
            "ci95_low": float(d.mean() - 1.96 * se) if se else np.nan,
            "ci95_high": float(d.mean() + 1.96 * se) if se else np.nan, "share_days_better": float((d > 0).mean())}


def comparison_table(models: dict[str, ModelResult], frozen_id: str, ar1_id: str) -> pd.DataFrame:
    rows = []
    for mid, m in models.items():
        row = {"model_id": mid, "description": m.description, "k": m.k, "n_parameters": m.n_parameters,
               "train_avg_log_score": float(m.train.log_score.mean()),
               "validation_avg_log_score": float(m.validation.log_score.mean()),
               "train_validation_gap": float(m.train.log_score.mean() - m.validation.log_score.mean()),
               "train_scored_days": len(m.train.log_score), "validation_scored_days": len(m.validation.log_score)}
        for ref, tag in ((frozen_id, "vs_frozen_hmm"), (ar1_id, "vs_ar1")):
            if ref in models and ref != mid:
                for split in ("train", "validation"):
                    t = paired_test(getattr(m, split).log_score, getattr(models[ref], split).log_score)
                    row.update({f"{split}_{k}_{tag}": v for k, v in t.items()})
        fits = m.fits
        if fits:
            ref = m.info["reference_fit"]
            row.update({"converged": ref.converged, "n_iter": ref.n_iter, "monotone": ref.monotone,
                        "phi_bound_share": ref.bound_share, "reference_seed": ref.seed})
        else:
            row.update({"converged": True, "n_iter": np.nan, "monotone": True, "phi_bound_share": np.nan,
                        "reference_seed": np.nan})
        rows.append(row)
    return pd.DataFrame(rows)


def daily_scores(models: dict[str, ModelResult], split: str, index) -> pd.DataFrame:
    return pd.DataFrame({m: getattr(r, split).log_score for m, r in models.items()}, index=index)


# --------------------------------------------------------------------------------------
# Regime quality
# --------------------------------------------------------------------------------------
def initialization_table(m: ModelResult) -> dict:
    fits: list[ARFit] = m.fits
    labels = [filter_train_validation_ar(f.params, m.info["x_train"], m.info["x_validation"]) for f in fits]
    tr_labels = [a.filtered.argmax(axis=1) for a, _ in labels]
    va_labels = [b.filtered.argmax(axis=1) for _, b in labels]
    obj = np.array([f.train_loglik for f in fits]) / fits[0].n_obs
    starts = np.array([o for f in fits for o in f.start_objectives]) / fits[0].n_obs
    pair_tr = rd.pairwise_agreement(tr_labels, m.k)
    pair_va = rd.pairwise_agreement(va_labels, m.k)
    return {"model_id": m.model_id, "n_seeds": len(fits), "n_init_per_seed": len(fits[0].start_objectives),
            "converged_share": float(np.mean([f.converged for f in fits])),
            "all_monotone": bool(all(f.monotone and f.all_starts_monotone for f in fits)),
            "n_starts": int(sum(len(f.start_histories) for f in fits)),
            "n_starts_with_decrease": int(sum(sum(_n_dec(h) > 0 for h in f.start_histories) for f in fits)),
            "max_objective_decrease": float(max(f.max_decrease for f in fits)),
            "empty_state_events": int(sum(f.empty_state_events for f in fits)),
            "loglik_per_obs_best": float(obj.max()), "loglik_per_obs_range": float(obj.max() - obj.min()),
            "best_basin_share": float(np.mean(obj >= obj.max() - 1e-3)),
            "single_start_best_basin_share": float(np.mean(starts >= starts.max() - 1e-3)),
            "train_mean_pairwise_ari": pair_tr["mean_ari"], "train_min_pairwise_ari": pair_tr["min_ari"],
            "validation_mean_pairwise_ari": pair_va["mean_ari"],
            "mean_phi_bound_share": float(np.mean([f.bound_share for f in fits]))}


def _n_dec(history, tol=None) -> int:
    from .ar_hmm import MONOTONE_ABS_TOL
    d = np.diff(np.asarray(history, float))
    return int((d < -(MONOTONE_ABS_TOL if tol is None else tol)).sum())


def monotonicity_check(models: dict, ids) -> tuple[str, str]:
    """FAIL if ANY EM start of any AR model (selected or not) decreased its objective by more than
    MONOTONE_ABS_TOL. A correct EM/GEM step can never decrease the objective, so this is a bug signal."""
    from .ar_hmm import MONOTONE_ABS_TOL
    bad, parts = [], []
    for m in ids:
        fits = models[m].fits
        n = sum(len(f.start_histories) for f in fits)
        nd = sum(sum(_n_dec(h) > 0 for h in f.start_histories) for f in fits)
        mx = max(f.max_decrease for f in fits)
        parts.append(f"{m}: {nd}/{n} starts with decrease > {MONOTONE_ABS_TOL:g} (max {mx:.1e})")
        if nd:
            bad.append(m)
    return ("FAIL" if bad else "PASS"), "; ".join(parts)


def mstep_optimality(params: ARParams, x_train: np.ndarray, step: float = 1e-4) -> dict:
    """At the E-step of the fitted parameters, run one M-step and verify that no feasible single-coordinate move
    (intercept, phi, var; +/- step) increases the EM Q-function. A positive gain means the M-step is not the
    maximizer it claims to be."""
    from .ar_hmm import PHI_BOUND, REG_VAR, _m_step, expected_complete_loglik, log_emission_ar
    from .hmm_models import _forward_backward

    y, lag = lagged_pairs(x_train)
    gamma, xi, _ = _forward_backward(params, log_emission_ar(params, y, lag))
    new = _m_step(params, y, lag, gamma, xi)[0]
    q0 = expected_complete_loglik(new, y, lag, gamma, xi)
    best_gain, where = -np.inf, ""
    for blk in ("intercept", "phi", "var"):
        arr = getattr(new, blk)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                for eps in (step, -step):
                    alt = new.copy()
                    a = getattr(alt, blk)
                    a[i, j] += eps * (a[i, j] if blk == "var" else 1.0)
                    if (blk == "phi" and abs(a[i, j]) > PHI_BOUND) or (blk == "var" and a[i, j] < REG_VAR):
                        continue
                    g = expected_complete_loglik(alt, y, lag, gamma, xi) - q0
                    if g > best_gain:
                        best_gain, where = g, f"{blk}[S{i + 1},{j}] {eps:+g}"
    return {"max_q_gain_from_feasible_move": float(best_gain), "argmax_move": where,
            "q_value": float(q0), "optimal": bool(best_gain <= 1e-9)}


def regime_summary(m: ModelResult) -> dict:
    out = {"model_id": m.model_id}
    for split in ("train", "validation"):
        p = getattr(m, split)
        lab = p.filtered.argmax(axis=1)
        sw = rd.switching_stats(lab)
        out.update({f"{split}_shares": " / ".join(f"S{i + 1} {s:.0%}" for i, s in enumerate(rd.population(lab, m.k))),
                    f"{split}_min_state_share": float(rd.population(lab, m.k).min()),
                    f"{split}_switches_per_year": sw["switches_per_year"],
                    f"{split}_median_duration_days": sw["median_duration_days"],
                    f"{split}_mean_max_filtered_prob": float(p.filtered.max(axis=1).mean()),
                    f"{split}_mean_normalized_entropy": float(rd.normalized_entropy(p.filtered).mean()),
                    f"{split}_share_uncertain_days": float(((m.p_stress(split) > 0.2) & (m.p_stress(split) < 0.8)).mean())
                    if m.stress_states else np.nan})
    if m.k > 1:
        a = m.params.transmat
        imp = float(TRADING_DAYS * (m.validation.filtered[:-1] * (1 - np.diag(a))[None]).sum(axis=1).mean())
        out.update({"self_transitions": " / ".join(f"{v:.4f}" for v in np.diag(a)),
                    "expected_durations_days": " / ".join(f"{v:.1f}" for v in expected_durations(a)),
                    "validation_implied_switches_per_year": imp,
                    "validation_realized_to_implied_switch_ratio":
                        out["validation_switches_per_year"] / imp if imp > 0 else np.nan,
                    "stress_states": " ".join(f"S{j + 1}" for j in m.stress_states)})
    return out


def transition_rows(m: ModelResult) -> list[dict]:
    if m.k < 2:
        return []
    a = m.params.transmat
    dur = expected_durations(a)
    return [{"model_id": m.model_id, "from_state": f"S{i + 1}", "to_state": f"S{j + 1}", "probability": a[i, j],
             "expected_duration_days_from_state": dur[i] if i == j else np.nan}
            for i in range(m.k) for j in range(m.k)]


def state_profiles(m: ModelResult, train_feat: pd.DataFrame, val_feat: pd.DataFrame, columns, scale, mean
                   ) -> pd.DataFrame:
    """Factor profiles (original units, filtered argmax), plus AR coefficients and innovation s.d. per state."""
    tr_feat = train_feat.iloc[len(train_feat) - len(m.train.x):][list(columns)]
    p_tr = rd.factor_profile(tr_feat, m.train_labels, m.k)
    p_va = rd.factor_profile(val_feat[list(columns)], m.validation_labels, m.k)
    rows = []
    for s in range(m.k):
        st = f"S{s + 1}"
        label, rule = rd.hypothesis_label(p_tr.loc[st])
        shift = (float(np.sqrt(((p_tr.loc[st, list(columns)] - p_va.loc[st, list(columns)]) ** 2).sum()))
                 if p_va.loc[st, "n_obs"] > 0 else np.nan)
        for split, prof in (("train", p_tr), ("validation", p_va)):
            row = {"model_id": m.model_id, "state": st, "split": split, "n_obs": int(prof.loc[st, "n_obs"]),
                   "share": prof.loc[st, "share"], **{f: prof.loc[st, f] for f in columns},
                   "is_stress_state": s in m.stress_states, "hypothesis_label": label, "label_rule": rule,
                   "train_validation_profile_distance": shift}
            if isinstance(m.params, ARParams):
                row.update({f"phi_{f}": m.params.phi[s, j] for j, f in enumerate(columns)})
                row.update({f"innov_sd_{f}": float(np.sqrt(m.params.var[s, j]) * scale[j])
                            for j, f in enumerate(columns)})
                row.update({f"long_run_mean_{f}": float(m.params.long_run_mean[s, j] * scale[j] + mean[j])
                            for j, f in enumerate(columns)})
            rows.append(row)
    return pd.DataFrame(rows)


def match_states(frozen: ModelResult, other: ModelResult, profiles: pd.DataFrame, columns) -> dict:
    """Label-invariant agreement (ARI) and an economic matching of states by Train factor-profile distance."""
    pa = profiles[(profiles["model_id"] == frozen.model_id) & (profiles["split"] == "train")].set_index("state")
    pb = profiles[(profiles["model_id"] == other.model_id) & (profiles["split"] == "train")].set_index("state")
    cost = np.array([[np.sqrt(((pa.loc[a, list(columns)] - pb.loc[b, list(columns)]) ** 2).sum()) for b in pb.index]
                     for a in pa.index])
    cost = np.nan_to_num(cost, nan=1e6)
    r, c = linear_sum_assignment(cost)
    table = pd.crosstab(pd.Series(frozen.train_labels, name="frozen"), pd.Series(other.train_labels, name="other"))
    stress_a = np.isin(frozen.train_labels, frozen.stress_states)
    stress_b = np.isin(other.train_labels, other.stress_states)
    return {"model_id": other.model_id, "reference": frozen.model_id,
            "train_ari": float(adjusted_rand_score(frozen.train_labels, other.train_labels)),
            "validation_ari": float(adjusted_rand_score(frozen.validation_labels, other.validation_labels)),
            "profile_matching": "; ".join(f"{pa.index[i]}<->{pb.index[j]} (distance {cost[i, j]:.2f})"
                                          for i, j in zip(r, c)),
            "train_stress_overlap_recall": float(stress_b[stress_a].mean()) if stress_a.any() else np.nan,
            "train_stress_overlap_precision": float(stress_a[stress_b].mean()) if stress_b.any() else np.nan,
            "validation_p_stress_corr": float(np.corrcoef(frozen.p_stress("validation"),
                                                          other.p_stress("validation"))[0, 1])
            if other.p_stress("validation").std() > 0 and frozen.p_stress("validation").std() > 0 else np.nan,
            "crosstab_train": table.to_dict()}


# --------------------------------------------------------------------------------------
# Adoption decision
# --------------------------------------------------------------------------------------
def adoption_decision(comp: pd.DataFrame, calib: pd.DataFrame, resid: pd.DataFrame, init: pd.DataFrame,
                      regimes: pd.DataFrame, profiles: pd.DataFrame, cand: str, frozen: str, ar1: str) -> dict:
    r = ADOPTION_RULES
    row = comp.set_index("model_id").loc[cand]
    checks = {}
    checks["1_gain_vs_frozen"] = bool(row["validation_mean_diff_vs_frozen_hmm"] >= r["min_validation_gain_vs_frozen"]
                                      and row["validation_nw_t_vs_frozen_hmm"] >= r["nw_t"])
    checks["2_gain_vs_ar1"] = bool(row["validation_mean_diff_vs_ar1"] > r["min_validation_gain_vs_ar1"]
                                   and row["validation_nw_t_vs_ar1"] >= r["nw_t"])
    v = resid[resid["split"] == "validation"].groupby("model_id")["qres_acf_lag1"].apply(lambda s: s.abs().mean())
    checks["3_residual_dependence"] = bool(v[cand] <= r["max_residual_acf1_ratio"] * v[frozen])
    c = calib[(calib["split"] == "validation") & (calib["factor"] == "all factors")].set_index("model_id")["coverage_90"]
    lo, hi = r["coverage90_band"]
    checks["4_coverage90"] = bool(lo <= c[cand] <= hi and abs(c[cand] - 0.9) < abs(c[frozen] - 0.9))
    i = init.set_index("model_id").loc[cand]
    g = regimes.set_index("model_id").loc[cand]
    sw = regimes.set_index("model_id")["validation_switches_per_year"]
    checks["5_stability_and_persistence"] = bool(
        i["converged_share"] >= r["min_converged_seed_share"] and i["all_monotone"]
        and i["train_mean_pairwise_ari"] >= r["min_mean_pairwise_ari_train"]
        and g["train_min_state_share"] >= r["min_train_state_share"]
        and (frozen not in sw or sw[cand] <= r["max_switch_ratio_vs_frozen"] * max(sw[frozen], 1.0)))
    p = profiles[(profiles["model_id"] == cand) & (profiles["split"] == "train") & profiles["is_stress_state"]]
    top = p.assign(score=p["volatility"] + p["liquidity_credit"] - p["equity_trend"]).sort_values("score").tail(1)
    checks["6_stress_state"] = bool(len(top) and top["volatility"].iloc[0] >= 1.0 and top["liquidity_credit"].iloc[0] >= 1.0
                                    and top["equity_trend"].iloc[0] <= -0.5)
    if all(checks.values()):
        verdict, use = "replace", f"{cand} (AR-HMM) for Stage 6"
    elif checks["1_gain_vs_frozen"] and checks["2_gain_vs_ar1"] and checks["3_residual_dependence"] \
            and checks["4_coverage90"]:
        verdict, use = "challenger_density", f"{frozen} for regimes; {cand} as predictive-density challenger"
    elif not checks["2_gain_vs_ar1"]:
        verdict, use = "retain_ar_explains_gain", f"{frozen} for regimes; {ar1} as density benchmark"
    else:
        verdict, use = "retain_mixed", f"{frozen}; {cand} as robustness challenger"
    return {"candidate": cand, "checks": checks, "verdict": verdict, "stage6_model": use,
            "validation_mean_abs_qres_acf1": {k: float(x) for k, x in v.items()},
            "validation_coverage90": {k: float(x) for k, x in c.items()}}


def file_sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def directory_hashes(directory: Path) -> dict[str, str]:
    return {str(p.relative_to(directory)): file_sha256(p) for p in sorted(directory.rglob("*")) if p.is_file()} \
        if directory.exists() else {}


def save_json(obj, path: Path) -> None:
    from .hmm_diagnostics import save_json as _save
    _save(obj, path)
