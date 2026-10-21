"""Stage 5: causal (forward) filtering, one-step predictive densities, and retrospective comparators.

Decision quantities (causal, computed in log space):
    predicted  P(Z_t = k | x_1..x_(t-1))  = sum_i P(Z_(t-1) = i | x_1..x_(t-1)) A_ik
    filtered   P(Z_t = k | x_1..x_t)      = predicted_k * N(x_t; mu_k, Sigma_k) / p(x_t | x_1..x_(t-1))
    log score  log p(x_t | x_1..x_(t-1)) = logsumexp_k( log predicted_k + log N(x_t; mu_k, Sigma_k) )
The value for date t depends only on x_1..x_t and on parameters estimated from Train.

Sequence initialization
    Train:      predicted_1 = pi (estimated start distribution).
    Validation: the recursion continues from the last Train filtered distribution:
                predicted_(first validation day) = filtered_(last train day) @ A.
    This equals filtering the concatenated Train+Validation sequence with fixed Train parameters, and uses no
    Validation observation before its own date.

Retrospective comparators (NOT decision outputs): smoothed P(Z_t | x_1..x_T) and the Viterbi path use the
whole sequence, so earlier values change when later data change. They exist only to show the difference.
"""

from __future__ import annotations

STAGE5_CODE_VERSION = "stage5-2026-10-21c"  # must match scripts/07_fit_hmm.py

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from .hmm_models import HMMParams, log_emission


@dataclass
class FilterResult:
    log_predicted: np.ndarray     # (T, K)
    log_filtered: np.ndarray      # (T, K)
    log_score: np.ndarray         # (T,)  log p(x_t | x_1..x_(t-1))

    @property
    def predicted(self) -> np.ndarray:
        return np.exp(self.log_predicted)

    @property
    def filtered(self) -> np.ndarray:
        return np.exp(self.log_filtered)

    @property
    def last_filtered(self) -> np.ndarray:
        return np.exp(self.log_filtered[-1])


def _log(p: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return np.log(np.asarray(p, dtype=float))


def forward_filter(params: HMMParams, x: np.ndarray, initial_predicted: np.ndarray | None = None,
                   previous_filtered: np.ndarray | None = None) -> FilterResult:
    """Log-space forward filter. Give at most one of
        initial_predicted   -> P(Z_1) for the first day (default: params.startprob), or
        previous_filtered   -> filtered distribution of the day before x[0]; it is propagated by A first.
    """
    if initial_predicted is not None and previous_filtered is not None:
        raise ValueError("give initial_predicted or previous_filtered, not both")
    log_b = log_emission(params, x)
    log_a = _log(params.transmat)
    t_len, k = log_b.shape
    log_pred = np.empty((t_len, k))
    log_filt = np.empty((t_len, k))
    score = np.empty(t_len)
    if previous_filtered is not None:
        current = logsumexp(_log(previous_filtered)[:, None] + log_a, axis=0)
    else:
        current = _log(params.startprob if initial_predicted is None else initial_predicted)
    for t in range(t_len):
        if t > 0:
            current = logsumexp(log_filt[t - 1][:, None] + log_a, axis=0)
        log_pred[t] = current
        joint = current + log_b[t]
        score[t] = logsumexp(joint)
        log_filt[t] = joint - score[t]
    return FilterResult(log_pred, log_filt, score)


def filter_train_validation(params: HMMParams, x_train: np.ndarray, x_validation: np.ndarray
                            ) -> tuple[FilterResult, FilterResult]:
    """Train filtered from pi; Validation continues from the final Train filtered distribution."""
    train = forward_filter(params, x_train)
    validation = forward_filter(params, x_validation, previous_filtered=train.last_filtered)
    return train, validation


# --------------------------------------------------------------------------------------
# Retrospective comparators (use the whole sequence; never decision outputs)
# --------------------------------------------------------------------------------------
def smoothed_probabilities(params: HMMParams, x: np.ndarray, initial_predicted: np.ndarray | None = None
                           ) -> np.ndarray:
    """P(Z_t | x_1..x_T) by forward-backward in log space. RETROSPECTIVE: depends on future observations."""
    fwd = forward_filter(params, x, initial_predicted=initial_predicted)
    log_b = log_emission(params, x)
    log_a = _log(params.transmat)
    t_len, k = log_b.shape
    log_beta = np.zeros((t_len, k))
    for t in range(t_len - 2, -1, -1):
        log_beta[t] = logsumexp(log_a + (log_b[t + 1] + log_beta[t + 1])[None, :], axis=1)
    log_gamma = fwd.log_filtered + log_beta
    log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
    return np.exp(log_gamma)


def viterbi(params: HMMParams, x: np.ndarray, initial_predicted: np.ndarray | None = None) -> np.ndarray:
    """Most likely state path given the whole sequence. RETROSPECTIVE."""
    log_b = log_emission(params, x)
    log_a = _log(params.transmat)
    t_len, k = log_b.shape
    delta = _log(params.startprob if initial_predicted is None else initial_predicted) + log_b[0]
    back = np.zeros((t_len, k), dtype=int)
    for t in range(1, t_len):
        cand = delta[:, None] + log_a
        back[t] = cand.argmax(axis=0)
        delta = cand.max(axis=0) + log_b[t]
    path = np.empty(t_len, dtype=int)
    path[-1] = int(delta.argmax())
    for t in range(t_len - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def brute_force_loglik(params: HMMParams, x: np.ndarray) -> float:
    """Exact log p(x_1..x_T) by enumerating all K^T paths (tests only; tiny T)."""
    import itertools

    log_b = log_emission(params, x)
    log_a = _log(params.transmat)
    log_pi = _log(params.startprob)
    totals = []
    for path in itertools.product(range(params.k), repeat=len(x)):
        s = log_pi[path[0]] + log_b[0, path[0]]
        for t in range(1, len(x)):
            s += log_a[path[t - 1], path[t]] + log_b[t, path[t]]
        totals.append(s)
    return float(logsumexp(totals))
