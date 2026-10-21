"""Stage 5.5: Gaussian autoregressive hidden Markov model (Markov-switching AR(1) with diagonal dynamics).

Model (state k, factor d, all factors standardized with the Train scaler)
    X_t,d = mu_k,d + phi_k,d (X_t-1,d - mu_k,d) + eps_t,d,      eps_t,d | Z_t = k ~ N(0, s2_k,d)
          = c_k,d + phi_k,d X_t-1,d + eps_t,d,                  c_k,d = mu_k,d (1 - phi_k,d)
    Z_t first-order Markov: P(Z_t = j | Z_t-1 = i) = A_ij.
    Phi_k and Sigma_k are diagonal, so given the state the five factors are independent AR(1) processes.

Likelihood (conditional on the first observation)
    The first Train observation X_1 is used only as the lag of X_2; the hidden chain starts at t = 2 with
    P(Z_2 = k) = pi_k. For t >= 2:
        p(x_t | x_1..x_t-1) = sum_k P(Z_t = k | x_1..x_t-1) N(x_t; c_k + Phi_k x_t-1, Sigma_k)
    The emission of day t depends on x_t and x_t-1 only. This is a genuine AR-HMM: the state selects the
    autoregression; lagged values are NOT appended as extra independent features.

Estimation (Train only): EM / Baum-Welch. E-step = scaled forward-backward on the conditional emission
densities (same algorithm as Stage 5, reused). M-step per state and factor = weighted least squares of x_t on
(1, x_t-1) with weights P(Z_t = k | x_1..x_T); |phi| is constrained to PHI_BOUND (the constrained optimum
of the convex weighted SSE is the clipped phi with the optimal intercept, so EM stays monotone); innovation
variances get a floor REG_VAR. Several K-Means-based starts per seed; the best Train log-likelihood is kept.

Filtering for decisions: log-space forward recursion on the same conditional emissions. Validation continues
from the last Train filtered distribution and uses the last Train observation as the first lag.
K = 1 gives the single-regime Gaussian AR(1) baseline (closed-form least squares).
"""

from __future__ import annotations

STAGE55_CODE_VERSION = "stage5.5-2026-10-21b"  # must match scripts/07b_compare_ar_hmm.py

import math
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.special import logsumexp

from .hmm_filtering import FilterResult
from .hmm_models import _forward_backward, start_seeds

PHI_BOUND = 0.99           # |phi| <= 0.99: stationary, and the state mean still moves the forecast >= 1%/day
REG_VAR = 1e-4             # innovation-variance FLOOR: var = max(s2, REG_VAR) (exact constrained M-step)
VARIANCE_RULES = ("floor", "additive_legacy")   # "additive_legacy" = v1 rule var = s2 + REG_VAR (audit only)
AR_MAX_ITER = 1000
AR_TOL = 1e-6              # objective change per observation
AR_N_INIT = 5              # EM starts per seed (as Stage 5 v1.2)
STARTPROB_FLOOR = 1e-8
MIN_STATE_WEIGHT = 1e-6
# Monotonicity tolerance on the TOTAL Train objective (nats). Double-precision error of the scaled forward
# recursion is bounded by about T * eps * max|log c_t| ~ 2057 * 2.2e-16 * 50 ~ 2e-11; 1e-7 is >1000x that bound
# and far below the decreases a wrong M-step produces (1e-4 to 1e-3 in the Stage 5.5 audit).
MONOTONE_ABS_TOL = 1e-7


# --------------------------------------------------------------------------------------
# Parameters and emissions
# --------------------------------------------------------------------------------------
@dataclass
class ARParams:
    startprob: np.ndarray     # (K,)
    transmat: np.ndarray      # (K, K)
    intercept: np.ndarray     # (K, D)  c_k
    phi: np.ndarray           # (K, D)  diagonal AR coefficients
    var: np.ndarray           # (K, D)  diagonal innovation variances

    @property
    def k(self) -> int:
        return len(self.startprob)

    @property
    def long_run_mean(self) -> np.ndarray:
        """mu_k = c_k / (1 - phi_k) (finite because |phi| <= PHI_BOUND < 1)."""
        return self.intercept / (1.0 - self.phi)

    def copy(self) -> "ARParams":
        return ARParams(self.startprob.copy(), self.transmat.copy(), self.intercept.copy(), self.phi.copy(),
                        self.var.copy())

    def permuted(self, order) -> "ARParams":
        o = np.asarray(order, dtype=int)
        return ARParams(self.startprob[o], self.transmat[np.ix_(o, o)], self.intercept[o], self.phi[o], self.var[o])

    def n_parameters(self) -> int:
        k, d = self.intercept.shape
        return (k - 1) + k * (k - 1) + 3 * k * d

    def to_dict(self) -> dict:
        return {"startprob": self.startprob.tolist(), "transmat": self.transmat.tolist(),
                "intercept": self.intercept.tolist(), "phi": self.phi.tolist(), "var": self.var.tolist(),
                "long_run_mean": self.long_run_mean.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "ARParams":
        return cls(*(np.asarray(d[key], float) for key in ("startprob", "transmat", "intercept", "phi", "var")))


def lagged_pairs(x: np.ndarray, previous: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(y, x_lag). Without `previous` the first row is used only as a lag (y = x[1:]). With `previous` (the
    last observation before x, e.g. the last Train row for Validation) every row of x is scored."""
    x = np.asarray(x, dtype=float)
    if previous is None:
        return x[1:], x[:-1]
    return x, np.vstack([np.asarray(previous, float)[None, :], x[:-1]])


def conditional_means(params: ARParams, x_lag: np.ndarray) -> np.ndarray:
    """(T, K, D) E[x_t | x_t-1, Z_t = k] = c_k + phi_k * x_t-1."""
    return params.intercept[None] + params.phi[None] * np.asarray(x_lag, float)[:, None, :]


def log_emission_ar(params: ARParams, y: np.ndarray, x_lag: np.ndarray) -> np.ndarray:
    """(T, K) log N(y_t; c_k + phi_k y_t-1, diag(var_k)). Row t uses y[t] and x_lag[t] only."""
    m = conditional_means(params, x_lag)
    v = params.var[None]
    return -0.5 * (np.log(2 * math.pi * v) + (np.asarray(y, float)[:, None, :] - m) ** 2 / v).sum(axis=2)


# --------------------------------------------------------------------------------------
# Causal filtering (log space)
# --------------------------------------------------------------------------------------
def _log(p) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return np.log(np.asarray(p, dtype=float))


def filter_from_log_emission(log_b: np.ndarray, transmat: np.ndarray, initial_predicted: np.ndarray | None = None,
                             previous_filtered: np.ndarray | None = None) -> FilterResult:
    """predicted_t = filtered_t-1 @ A; score_t = logsumexp(log predicted_t + log b_t); filtered_t = update.
    The predicted distribution and the score of day t are computed before day t's emission enters."""
    if (initial_predicted is None) == (previous_filtered is None):
        raise ValueError("give exactly one of initial_predicted or previous_filtered")
    log_a = _log(transmat)
    t_len, k = log_b.shape
    log_pred, log_filt, score = np.empty((t_len, k)), np.empty((t_len, k)), np.empty(t_len)
    current = (_log(initial_predicted) if previous_filtered is None
               else logsumexp(_log(previous_filtered)[:, None] + log_a, axis=0))
    for t in range(t_len):
        if t > 0:
            current = logsumexp(log_filt[t - 1][:, None] + log_a, axis=0)
        log_pred[t] = current
        joint = current + log_b[t]
        score[t] = logsumexp(joint)
        log_filt[t] = joint - score[t]
    return FilterResult(log_pred, log_filt, score)


def filter_train_validation_ar(params: ARParams, x_train: np.ndarray, x_validation: np.ndarray
                               ) -> tuple[FilterResult, FilterResult]:
    """Train scored from its 2nd row (chain starts with pi); Validation continues from the last Train filtered
    distribution with the last Train row as the first lag (no reset at the boundary)."""
    y_tr, lag_tr = lagged_pairs(x_train)
    tr = filter_from_log_emission(log_emission_ar(params, y_tr, lag_tr), params.transmat,
                                  initial_predicted=params.startprob)
    y_va, lag_va = lagged_pairs(x_validation, previous=np.asarray(x_train)[-1])
    va = filter_from_log_emission(log_emission_ar(params, y_va, lag_va), params.transmat,
                                  previous_filtered=tr.last_filtered)
    return tr, va


def brute_force_loglik_ar(params: ARParams, x: np.ndarray) -> float:
    """Exact conditional log-likelihood by enumerating all K^(T-1) paths (tests only)."""
    import itertools

    y, lag = lagged_pairs(x)
    log_b = log_emission_ar(params, y, lag)
    log_a, log_pi = _log(params.transmat), _log(params.startprob)
    totals = []
    for path in itertools.product(range(params.k), repeat=len(y)):
        s = log_pi[path[0]] + log_b[0, path[0]]
        for t in range(1, len(y)):
            s += log_a[path[t - 1], path[t]] + log_b[t, path[t]]
        totals.append(s)
    return float(logsumexp(totals))


# --------------------------------------------------------------------------------------
# M-step and initialization
# --------------------------------------------------------------------------------------
def weighted_ar_regression(y: np.ndarray, x_lag: np.ndarray, w: np.ndarray, reg: float = REG_VAR,
                           bound: float = PHI_BOUND, variance_rule: str = "floor"
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per factor, the exact maximizer of the weighted Gaussian log-likelihood
        sum_t w_t log N(y_t; c + phi x_t, var)   subject to |phi| <= bound and var >= reg.
    (c, phi): the weighted SSE is convex; profiling out c leaves a convex quadratic in phi, so the constrained
    optimum is the clipped phi with its optimal c. var: the objective is unimodal in var with maximum at the
    weighted residual variance s2, so the constrained optimum is max(s2, reg). Returns (c, phi, var, bound_active).
    variance_rule="additive_legacy" reproduces the v1 rule var = s2 + reg, which is NOT the maximizer."""
    w = np.asarray(w, float)
    sw = w.sum()
    sx, sy = w @ x_lag, w @ y
    sxx, sxy = w @ (x_lag * x_lag), w @ (x_lag * y)
    den = sw * sxx - sx ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        phi_hat = np.where(den > 1e-12 * max(sw, 1.0) ** 2, (sw * sxy - sx * sy) / den, 0.0)
    phi = np.clip(phi_hat, -bound, bound)
    c = (sy - phi * sx) / sw
    resid = y - c[None] - phi[None] * x_lag
    s2 = (w @ resid ** 2) / sw
    if variance_rule == "floor":
        var = np.maximum(s2, reg)
    elif variance_rule == "additive_legacy":
        var = s2 + reg
    else:
        raise ValueError(f"unknown variance_rule {variance_rule}")
    return c, phi, var, np.abs(phi_hat) > bound


def _m_step(params: ARParams, y, lag, gamma, xi_sum, variance_rule: str = "floor"
            ) -> tuple[ARParams, list[int], np.ndarray]:
    """Exact EM M-step: pi = gamma_1; A_ij proportional to expected transitions; per state the constrained
    weighted AR regression. (The legacy rule also floored pi at 1e-8, as in v1.)"""
    new = params.copy()
    k = params.k
    if variance_rule == "additive_legacy":
        start = np.maximum(gamma[0], STARTPROB_FLOOR)
        new.startprob = start / start.sum()
    else:
        new.startprob = gamma[0] / gamma[0].sum()
    if k > 1:
        rows = xi_sum.sum(axis=1, keepdims=True)
        new.transmat = np.where(rows > 0, xi_sum / np.where(rows > 0, rows, 1.0), 1.0 / k)
        new.transmat /= new.transmat.sum(axis=1, keepdims=True)
    weights = gamma.sum(axis=0)
    empty = [int(j) for j in np.flatnonzero(weights < MIN_STATE_WEIGHT)]
    bound_active = np.zeros_like(params.phi, dtype=bool)
    for j in range(k):
        if j in empty:
            continue
        c, phi, var, active = weighted_ar_regression(y, lag, gamma[:, j], variance_rule=variance_rule)
        new.intercept[j], new.phi[j], new.var[j] = c, phi, var
        bound_active[j] = active
    return new, empty, bound_active


def initial_params_ar(y: np.ndarray, lag: np.ndarray, k: int, seed: int, variance_rule: str = "floor") -> ARParams:
    """One K-Means run on the levels (n_init=1, seed-specific) -> hard-label AR regressions per state and a
    smoothed label transition matrix. K = 1 starts from the pooled regression."""
    d = y.shape[1]
    if k == 1:
        labels = np.zeros(len(y), dtype=int)
    else:
        from sklearn.cluster import KMeans
        labels = KMeans(n_clusters=k, n_init=1, random_state=seed).fit_predict(y)
    c, phi, var = np.empty((k, d)), np.empty((k, d)), np.empty((k, d))
    for j in range(k):
        w = (labels == j).astype(float)
        if w.sum() < 10:
            w = np.ones(len(y))
        c[j], phi[j], var[j], _ = weighted_ar_regression(y, lag, w, variance_rule=variance_rule)
    counts = np.ones((k, k))
    np.add.at(counts, (labels[:-1], labels[1:]), 1)
    return ARParams(np.full(k, 1.0 / k), counts / counts.sum(axis=1, keepdims=True), c, phi, var)


# --------------------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------------------
@dataclass
class ARFit:
    k: int
    seed: int
    params: ARParams
    train_loglik: float                 # conditional on the first Train row
    n_iter: int
    converged: bool
    monotone: bool
    history: list[float] = field(default_factory=list)
    empty_state_events: int = 0
    bound_share: float = 0.0            # share of phi coefficients at the |phi| bound in the final M-step
    seconds: float = 0.0
    n_obs: int = 0
    start_objectives: list[float] = field(default_factory=list)
    start_converged: list[bool] = field(default_factory=list)
    all_starts_monotone: bool = True
    warnings: list[str] = field(default_factory=list)
    n_decreases: int = 0                 # objective decreases larger than MONOTONE_ABS_TOL
    max_decrease: float = 0.0            # largest decrease of the total objective (nats), any size
    variance_rule: str = "floor"
    start_histories: list[list[float]] = field(default_factory=list)   # every start's trajectory (best-of-n)
    start_seeds: list[int] = field(default_factory=list)


def fit_ar_hmm(x_train: np.ndarray, k: int, seed: int, max_iter: int = AR_MAX_ITER, tol: float = AR_TOL,
               init: ARParams | None = None, variance_rule: str = "floor") -> ARFit:
    """EM for one start on the Train sequence (rows 2..T scored, row 1 = first lag). The monitored objective is
    exactly the conditional log-likelihood used for scoring (same emissions, same pi convention)."""
    started = time.perf_counter()
    x = np.asarray(x_train, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("training data contain NaN/inf")
    y, lag = lagged_pairs(x)
    n = len(y)
    params = init.copy() if init is not None else initial_params_ar(y, lag, k, seed, variance_rule)
    history, converged, empties = [], False, 0
    previous, it = -np.inf, 0
    bound_active = np.zeros_like(params.phi, dtype=bool)
    for it in range(1, max_iter + 1):
        log_b = log_emission_ar(params, y, lag)
        gamma, xi_sum, loglik = _forward_backward(params, log_b)
        history.append(loglik)
        if abs(loglik - previous) < tol * n:
            converged = True
            break
        previous = loglik
        params, empty, bound_active = _m_step(params, y, lag, gamma, xi_sum, variance_rule)
        empties += len(empty)
    loglik = _forward_backward(params, log_emission_ar(params, y, lag))[2]
    if not converged:
        history.append(loglik)
    steps = np.diff(np.asarray(history))
    n_dec = int((steps < -MONOTONE_ABS_TOL).sum())
    max_dec = float(max(0.0, -steps.min())) if len(steps) else 0.0
    notes = []
    if not converged:
        notes.append(f"no convergence after {max_iter} iterations")
    if n_dec:
        notes.append(f"EM objective decreased {n_dec} times (max {max_dec:.2e} nats)")
    if empties:
        notes.append(f"{empties} empty-state M-steps")
    return ARFit(k, int(seed), params, loglik, it, converged, n_dec == 0, history, empties,
                 float(bound_active.mean()), time.perf_counter() - started, n, warnings=notes, n_decreases=n_dec,
                 max_decrease=max_dec, variance_rule=variance_rule)


def fit_ar_hmm_best(x_train: np.ndarray, k: int, seed: int, n_init: int = AR_N_INIT, **kwargs) -> ARFit:
    """Best Train log-likelihood among n_init starts (same start seeds as the Stage 5 HMM). Every start's
    trajectory is kept so that monotonicity is audited for all starts, not only the selected one."""
    seeds = start_seeds(seed, n_init if k > 1 else 1)
    fits = [fit_ar_hmm(x_train, k, s, **kwargs) for s in seeds]
    pool = [f for f in fits if f.converged] or fits
    best = max(pool, key=lambda f: f.train_loglik)
    best.seed = int(seed)
    best.start_objectives = [f.train_loglik for f in fits]
    best.start_converged = [f.converged for f in fits]
    best.all_starts_monotone = all(f.monotone for f in fits)
    best.start_histories = [list(f.history) for f in fits]
    best.start_seeds = [int(s) for s in seeds]
    best.n_decreases = int(sum(f.n_decreases for f in fits))
    best.max_decrease = float(max(f.max_decrease for f in fits))
    best.seconds = float(sum(f.seconds for f in fits))
    return best


def expected_complete_loglik(params: ARParams, y: np.ndarray, lag: np.ndarray, gamma: np.ndarray,
                             xi_sum: np.ndarray) -> float:
    """EM Q-function: E[log p(x_2..T, Z | x_1, theta)] under the E-step posteriors (gamma, xi). The M-step must
    maximize it over the feasible set (|phi| <= PHI_BOUND, var >= REG_VAR); used by the audit and the tests."""
    with np.errstate(divide="ignore"):
        log_pi, log_a = np.log(params.startprob), np.log(params.transmat)
    q = float(np.sum(np.where(gamma[0] > 0, gamma[0] * log_pi, 0.0)))
    q += float(np.sum(np.where(xi_sum > 0, xi_sum * log_a, 0.0)))
    q += float(np.sum(gamma * log_emission_ar(params, y, lag)))
    return q
