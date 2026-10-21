"""Stage 5: Gaussian hidden Markov models (Standard and Sticky) estimated on Train only.

Model
  Z_t in {1..K} is a first-order Markov chain:  P(Z_1 = k) = pi_k,  P(Z_t = j | Z_(t-1) = i) = A_ij.
  Emissions:  X_t | Z_t = k ~ N(mu_k, Sigma_k), Sigma_k diagonal or full, X_t the five standardized factors.
  Log-likelihood of one sequence x_1..x_T:
      log p(x_1:T) = sum_t log p(x_t | x_1:t-1),
      p(x_t | x_1:t-1) = sum_k P(Z_t = k | x_1:t-1) N(x_t; mu_k, Sigma_k)          (forward recursion)

Estimation (Baum-Welch / EM, one continuous sequence, Train only)
  E-step: scaled forward-backward (Rabiner scaling; emission densities are rescaled by their per-day maximum
          so nothing underflows). Smoothed quantities are used ONLY inside EM on Train; they never become an
          output for decisions.
  M-step: closed-form updates. Covariances get `reg_covar` added to the diagonal (as in Stage 4), so the
          emission update is a regularized ML step.

Sticky HMM (finite-K MAP analogue of the sticky HDP-HMM of Fox et al., 2011)
  Each row of A has a Dirichlet prior  A_i. ~ Dir(alpha_i1..alpha_iK),  alpha_ij = 1 + kappa * 1[i = j].
  The M-step maximizes  E[log p(x, z)] + log p(A), which gives
      A_ij = (n_ij + kappa * 1[i = j]) / (sum_j n_ij + kappa),     n_ij = expected transition counts.
  The prior therefore enters every EM iteration (it changes the E-step of the next iteration and so the
  states themselves); it is not a post-hoc relabelling. kappa = w * (T_train - 1) / K, i.e. w is the prior
  pseudo-count as a fraction of the average number of observed transitions per row. The monitored
  objective is the log-posterior  log p(x | theta) + kappa * sum_i log A_ii  (+ const), which EM increases
  monotonically. w = 0 is the Standard HMM (maximum likelihood).

Nothing in this module reads files; callers pass Train arrays only.
"""

from __future__ import annotations

STAGE5_CODE_VERSION = "stage5-2026-10-21c"  # must match scripts/07_fit_hmm.py

import math
import time
import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.special import logsumexp

HMM_K = (2, 3, 4)
STANDARD_COVARIANCE_TYPES = ("diag", "full")    # diag = primary baseline; full = controlled challenger
STICKY_COVARIANCE_TYPES = ("diag",)
STICKY_PRIOR_WEIGHTS = (0.10, 0.30, 1.00)       # predetermined, roughly log-spaced; w = kappa / ((T_train - 1) / K)
# w = 1 means the prior adds as many pseudo self-transitions per row as the average number of observed transitions
# per row; even then A_ii is not forced: off-diagonal mass still comes from the data.
HMM_REG_COVAR = 1e-4                            # same as Stage 4 GMM (data are standardized)
HMM_N_INIT = 5                                  # starts per seed; the seed's estimate = best Train objective
                                                # (same structure as the Stage 4 GMM, n_init = 5 per seed)
HMM_MAX_ITER = 1000
HMM_TOL = 1e-6                                  # convergence: objective change per observation
STARTPROB_FLOOR = 1e-8                          # keeps log pi finite after a one-sequence M-step
MIN_STATE_WEIGHT = 1e-6                         # expected occupancy below this = empty state (kept, flagged)
MONOTONE_TOLERANCE = 1e-8                       # allowed objective decrease per observation (rounding)


# --------------------------------------------------------------------------------------
# Specification and parameters
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class HMMSpec:
    family: str            # "standard" | "sticky"
    k: int
    covariance_type: str   # "diag" | "full"
    prior_weight: float = 0.0

    @property
    def model_id(self) -> str:
        base = f"hmm_{'std' if self.family == 'standard' else 'sticky'}_{self.covariance_type}_k{self.k}"
        return base if self.family == "standard" else f"{base}_w{self.prior_weight:.2f}"

    def kappa(self, n_train: int) -> float:
        return 0.0 if self.family == "standard" else float(self.prior_weight) * (n_train - 1) / self.k

    def n_parameters(self, n_features: int) -> int:
        d, k = n_features, self.k
        cov = k * d if self.covariance_type == "diag" else k * d * (d + 1) // 2
        return (k - 1) + k * (k - 1) + k * d + cov


def candidate_specs(ks=HMM_K) -> list[HMMSpec]:
    specs = [HMMSpec("standard", k, c) for k in ks for c in STANDARD_COVARIANCE_TYPES]
    specs += [HMMSpec("sticky", k, c, w) for k in ks for c in STICKY_COVARIANCE_TYPES for w in STICKY_PRIOR_WEIGHTS]
    return specs


@dataclass
class HMMParams:
    startprob: np.ndarray      # (K,)
    transmat: np.ndarray       # (K, K), rows sum to one
    means: np.ndarray          # (K, D)
    covars: np.ndarray         # (K, D) diag or (K, D, D) full
    covariance_type: str

    @property
    def k(self) -> int:
        return len(self.startprob)

    def copy(self) -> "HMMParams":
        return HMMParams(self.startprob.copy(), self.transmat.copy(), self.means.copy(), self.covars.copy(),
                         self.covariance_type)

    def permuted(self, order) -> "HMMParams":
        """Relabel states: new state i is old state order[i]."""
        o = np.asarray(order, dtype=int)
        return HMMParams(self.startprob[o], self.transmat[np.ix_(o, o)], self.means[o], self.covars[o],
                         self.covariance_type)

    def full_covars(self) -> np.ndarray:
        if self.covariance_type == "full":
            return self.covars
        return np.stack([np.diag(c) for c in self.covars])

    def to_dict(self) -> dict:
        return {"startprob": self.startprob.tolist(), "transmat": self.transmat.tolist(),
                "means": self.means.tolist(), "covars": self.covars.tolist(),
                "covariance_type": self.covariance_type}

    @classmethod
    def from_dict(cls, d: dict) -> "HMMParams":
        return cls(np.asarray(d["startprob"], float), np.asarray(d["transmat"], float),
                   np.asarray(d["means"], float), np.asarray(d["covars"], float), d["covariance_type"])


def log_emission(params: HMMParams, x: np.ndarray) -> np.ndarray:
    """log N(x_t; mu_k, Sigma_k) for every day t and state k -> (T, K)."""
    x = np.asarray(x, dtype=float)
    t, d = x.shape
    out = np.empty((t, params.k))
    for k in range(params.k):
        diff = x - params.means[k]
        if params.covariance_type == "diag":
            var = params.covars[k]
            out[:, k] = -0.5 * (d * math.log(2 * math.pi) + np.log(var).sum() + (diff ** 2 / var).sum(axis=1))
        else:
            chol = np.linalg.cholesky(params.covars[k])
            sol = np.linalg.solve(chol, diff.T)
            out[:, k] = -0.5 * (d * math.log(2 * math.pi) + 2 * np.log(np.diag(chol)).sum() + (sol ** 2).sum(axis=0))
    return out


def log_prior(transmat: np.ndarray, kappa: float) -> float:
    """Dirichlet log-density of A up to a constant: kappa * sum_i log A_ii (alpha_ij - 1 = kappa * 1[i=j])."""
    if kappa == 0:
        return 0.0
    return float(kappa * np.log(np.clip(np.diag(transmat), 1e-300, None)).sum())


def map_transmat(expected_counts: np.ndarray, kappa: float) -> np.ndarray:
    """M-step for A under the sticky Dirichlet prior (kappa = 0 gives the ML update)."""
    k = expected_counts.shape[0]
    num = expected_counts + kappa * np.eye(k)
    rows = num.sum(axis=1, keepdims=True)
    out = np.where(rows > 0, num / np.where(rows > 0, rows, 1.0), np.full((k, k), 1.0 / k))
    return out / out.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------------------
# E-step: scaled forward-backward (Train only, inside EM)
# --------------------------------------------------------------------------------------
def _forward_backward(params: HMMParams, log_b: np.ndarray):
    t_len, k = log_b.shape
    shift = log_b.max(axis=1)
    b = np.exp(log_b - shift[:, None])
    a = params.transmat
    alpha = np.empty((t_len, k))
    scale = np.empty(t_len)
    cur = params.startprob * b[0]
    scale[0] = cur.sum()
    alpha[0] = cur / scale[0]
    for t in range(1, t_len):
        cur = (alpha[t - 1] @ a) * b[t]
        s = cur.sum()
        scale[t] = s
        alpha[t] = cur / s
    beta = np.empty((t_len, k))
    beta[-1] = 1.0
    for t in range(t_len - 2, -1, -1):
        beta[t] = a @ (b[t + 1] * beta[t + 1]) / scale[t + 1]
    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)
    weighted = b[1:] * beta[1:] / scale[1:, None]
    xi_sum = (alpha[:-1].T @ weighted) * a
    loglik = float(np.log(scale).sum() + shift.sum())
    return gamma, xi_sum, loglik


def _m_step(params: HMMParams, x: np.ndarray, gamma: np.ndarray, xi_sum: np.ndarray, kappa: float,
            reg_covar: float) -> tuple[HMMParams, list[int]]:
    new = params.copy()
    k, d = params.means.shape
    start = np.maximum(gamma[0], STARTPROB_FLOOR)
    new.startprob = start / start.sum()
    new.transmat = map_transmat(xi_sum, kappa)
    weights = gamma.sum(axis=0)
    empty = [int(j) for j in np.flatnonzero(weights < MIN_STATE_WEIGHT)]
    for j in range(k):
        if j in empty:          # keep the previous emission parameters for an (almost) empty state
            continue
        g = gamma[:, j]
        mu = g @ x / weights[j]
        diff = x - mu
        new.means[j] = mu
        if params.covariance_type == "diag":
            new.covars[j] = (g @ diff ** 2) / weights[j] + reg_covar
        else:
            cov = (diff * g[:, None]).T @ diff / weights[j]
            new.covars[j] = 0.5 * (cov + cov.T) + reg_covar * np.eye(d)
    return new, empty


# --------------------------------------------------------------------------------------
# Initialization and fitting
# --------------------------------------------------------------------------------------
def initial_params(x: np.ndarray, k: int, covariance_type: str, seed: int, reg_covar: float = HMM_REG_COVAR
                   ) -> HMMParams:
    """Seed-dependent start: one K-Means run (n_init=1) -> means, within-cluster covariances, and a transition
    matrix from the K-Means label sequence with add-one smoothing. Standard and Sticky models with the same
    (K, covariance, seed) start from the identical point, so differences come from the prior alone."""
    from sklearn.cluster import KMeans

    labels = KMeans(n_clusters=k, n_init=1, random_state=seed).fit_predict(x)
    d = x.shape[1]
    means = np.empty((k, d))
    covars = np.empty((k, d)) if covariance_type == "diag" else np.empty((k, d, d))
    for j in range(k):
        block = x[labels == j]
        if len(block) < 2:
            block = x
        means[j] = block.mean(axis=0)
        if covariance_type == "diag":
            covars[j] = block.var(axis=0) + reg_covar
        else:
            covars[j] = np.cov(block, rowvar=False, bias=True) + reg_covar * np.eye(d)
    counts = np.ones((k, k))
    np.add.at(counts, (labels[:-1], labels[1:]), 1)
    trans = counts / counts.sum(axis=1, keepdims=True)
    return HMMParams(np.full(k, 1.0 / k), trans, means, covars, covariance_type)


@dataclass
class HMMFit:
    spec: HMMSpec
    seed: int
    params: HMMParams
    kappa: float
    train_loglik: float              # log p(x_train | theta) at the estimate
    objective: float                 # train_loglik + log prior (= train_loglik for Standard)
    n_iter: int
    converged: bool
    monotone: bool
    history: list[float] = field(default_factory=list)
    empty_state_events: int = 0
    seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)
    n_train: int = 0
    start_objectives: list[float] = field(default_factory=list)   # all starts of this seed (best-of-n_init)
    start_converged: list[bool] = field(default_factory=list)
    all_starts_monotone: bool = True


def fit_hmm(x_train: np.ndarray, spec: HMMSpec, seed: int, max_iter: int = HMM_MAX_ITER, tol: float = HMM_TOL,
            reg_covar: float = HMM_REG_COVAR, init: HMMParams | None = None) -> HMMFit:
    """EM / MAP-EM on ONE Train sequence. Raises nothing on non-convergence; it is recorded instead."""
    started = time.perf_counter()
    x = np.asarray(x_train, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("training data contain NaN/inf")
    n = len(x)
    kappa = spec.kappa(n)
    params = init.copy() if init is not None else initial_params(x, spec.k, spec.covariance_type, seed, reg_covar)
    history, monotone, converged, empties, notes = [], True, False, 0, []
    previous = -np.inf
    it = 0
    for it in range(1, max_iter + 1):
        log_b = log_emission(params, x)
        gamma, xi_sum, loglik = _forward_backward(params, log_b)
        obj = loglik + log_prior(params.transmat, kappa)
        history.append(obj)
        if obj < previous - MONOTONE_TOLERANCE * n:
            monotone = False
        if abs(obj - previous) < tol * n:
            converged = True
            break
        previous = obj
        params, empty = _m_step(params, x, gamma, xi_sum, kappa, reg_covar)
        empties += len(empty)
    # final quantities at the returned parameters
    loglik = _forward_backward(params, log_emission(params, x))[2]
    obj = loglik + log_prior(params.transmat, kappa)
    if not converged:
        notes.append(f"no convergence after {max_iter} iterations")
    if not monotone:
        notes.append("EM objective decreased")
    if empties:
        notes.append(f"{empties} empty-state M-steps")
    return HMMFit(spec, seed, params, kappa, loglik, obj, it, converged, monotone, history, empties,
                  time.perf_counter() - started, notes, n)


def start_seeds(seed: int, n_init: int = HMM_N_INIT) -> list[int]:
    """Deterministic, non-overlapping initialization seeds for one protocol seed."""
    return [int(seed) * 10 + j for j in range(n_init)]


def fit_hmm_best(x_train: np.ndarray, spec: HMMSpec, seed: int, n_init: int = HMM_N_INIT, **kwargs) -> HMMFit:
    """The estimator used for every protocol seed: n_init EM starts, keep the converged start with the highest
    Train objective (log-likelihood, or log-posterior for Sticky). Standard and Sticky models with the same
    seed use identical starting points. Single-start outcomes are kept for the local-optimum diagnostics."""
    fits = [fit_hmm(x_train, spec, s, **kwargs) for s in start_seeds(seed, n_init)]
    pool = [f for f in fits if f.converged] or fits
    best = max(pool, key=lambda f: f.objective)
    best.seed = int(seed)
    best.start_objectives = [f.objective for f in fits]
    best.start_converged = [f.converged for f in fits]
    best.all_starts_monotone = all(f.monotone for f in fits)
    best.seconds = sum(f.seconds for f in fits)
    return best


def inflate_covariance(params: HMMParams, factor: float) -> HMMParams:
    """Emission covariances multiplied by `factor` (>= 1), used only inside the filter for predictive
    recalibration. N(x; mu, c Sigma) is a proper density, so log scores stay comparable."""
    out = params.copy()
    out.covars = params.covars * float(factor)
    return out


# --------------------------------------------------------------------------------------
# Canonical state order and alignment
# --------------------------------------------------------------------------------------
STRESS_WEIGHTS = {"volatility": 1.0, "liquidity_credit": 1.0, "equity_trend": -1.0}   # same as Stage 4


def stress_scores(means_original: np.ndarray, columns: list[str]) -> np.ndarray:
    idx = {c: i for i, c in enumerate(columns)}
    return sum(w * means_original[:, idx[c]] for c, w in STRESS_WEIGHTS.items())


def canonical_order(params: HMMParams, columns: list[str], mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Order states by ascending stress score of the emission means in original factor units (S1 = lowest
    stress). This is a naming convention only; economic labels require the profiles."""
    original = params.means * scale + mean
    return np.argsort(stress_scores(original, columns), kind="stable")


def align_by_means(reference: HMMParams, other: HMMParams) -> np.ndarray:
    """Permutation p with other.permuted(p) matched to reference (Hungarian on emission-mean distances)."""
    from scipy.optimize import linear_sum_assignment

    cost = ((reference.means[:, None, :] - other.means[None, :, :]) ** 2).sum(axis=2)
    rows, cols = linear_sum_assignment(cost)
    perm = np.empty(reference.k, dtype=int)
    perm[rows] = cols
    return perm


def parameter_distance(a: HMMParams, b: HMMParams) -> dict:
    """Distances between two aligned parameter sets (standardized units)."""
    return {"max_mean_distance": float(np.sqrt(((a.means - b.means) ** 2).sum(axis=1)).max()),
            "max_transition_abs_diff": float(np.abs(a.transmat - b.transmat).max())}


def expected_durations(transmat: np.ndarray) -> np.ndarray:
    """Geometric expectation 1 / (1 - A_ii); infinite when A_ii = 1 (absorbing state)."""
    diag = np.diag(np.asarray(transmat, dtype=float))
    with np.errstate(divide="ignore"):
        return np.where(diag < 1.0, 1.0 / (1.0 - diag), np.inf)


def information_criteria(fit: HMMFit, n_features: int) -> dict:
    p = fit.spec.n_parameters(n_features)
    return {"n_parameters": p, "train_aic": -2 * fit.train_loglik + 2 * p,
            "train_bic": -2 * fit.train_loglik + p * math.log(fit.n_train)}


def logsumexp_rows(a: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return logsumexp(a, axis=1)
