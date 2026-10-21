"""Stage 6: Bayesian Online Change-Point Detection (Adams & MacKay, 2007) for a scalar input.

Model
  The input y_t is split into segments. Within a segment, y_t ~ N(mu, sigma^2) i.i.d. with the conjugate
  Normal-Inverse-Gamma prior (mu, sigma^2) ~ NIG(mu0, kappa0, alpha0, beta0); a new segment starts with
  constant hazard h = 1 / expected_run_length. Both the mean and the variance may change at a change point.
  Posterior predictive of y_t given a run of length r:
      Student-t(df = 2 alpha_r, loc = mu_r, scale^2 = beta_r (kappa_r + 1) / (alpha_r kappa_r)).

Recursion (log space), r_t = run length after observing y_t:
  pi_r      = log P(r_t-1 = r | y_1..t-1) + log StudentT(y_t | stats_r)
  growth    : log P(r_t = r + 1, y_1..t) = pi_r + log(1 - h)
  change    : log P(r_t = 0,     y_1..t) = logsumexp_r(pi_r) + log h
  evidence  : log p(y_t | y_1..t-1)     = logsumexp_r(pi_r)
  posterior : normalize; the statistics of run 0 restart from the prior after absorbing y_t.
  Everything at date t uses y_1..y_t only.

What it detects: shifts in the level and/or dispersion of y_t relative to the current segment. What it does not:
the direction or economic meaning of a change (a calm-down is a change too), slow drifts that look like noise,
and dependence inside segments (persistent inputs violate the i.i.d. assumption and create spurious changes).

Reported evidence: P(r_t < w | y_1..t) = probability that a change occurred within the last w sessions. With
constant hazard the PRIOR probability of that event is 1 - (1 - h)^w; the posterior is reported next to it.
Optional truncation at max_run_length drops the oldest run lengths and renormalizes (documented approximation).
"""

from __future__ import annotations

STAGE6_CODE_VERSION = "stage6-2026-10-21"  # must match scripts/08_build_early_warning.py

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp

HAZARD_RUN_LENGTHS = (63, 126, 252)       # predetermined grid (trading days)
PRIMARY_RUN_LENGTH = 126
RECENT_WINDOWS = (5, 10, 21)
PRIMARY_WINDOW = 10
PRIOR_KAPPA0 = 1.0
PRIOR_ALPHA0 = 2.0                         # finite predictive variance (df = 4)


@dataclass(frozen=True)
class NIGPrior:
    mu0: float
    kappa0: float
    alpha0: float
    beta0: float

    @classmethod
    def from_train(cls, y_train: np.ndarray, kappa0: float = PRIOR_KAPPA0, alpha0: float = PRIOR_ALPHA0) -> "NIGPrior":
        """mu0 = Train mean; beta0 chosen so the prior predictive variance equals the Train variance:
        var = scale^2 * df / (df - 2), scale^2 = beta0 (kappa0 + 1) / (alpha0 kappa0)."""
        y = np.asarray(y_train, float)
        y = y[np.isfinite(y)]
        v = float(y.var())
        df = 2 * alpha0
        scale2 = v * (df - 2) / df
        return cls(float(y.mean()), kappa0, alpha0, scale2 * alpha0 * kappa0 / (kappa0 + 1))

    def to_dict(self) -> dict:
        return {"mu0": self.mu0, "kappa0": self.kappa0, "alpha0": self.alpha0, "beta0": self.beta0}


def student_t_logpdf(y: float, mu: np.ndarray, kappa: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    df = 2 * alpha
    scale2 = beta * (kappa + 1) / (alpha * kappa)
    z2 = (y - mu) ** 2 / scale2
    return (gammaln((df + 1) / 2) - gammaln(df / 2) - 0.5 * np.log(df * math.pi * scale2)
            - (df + 1) / 2 * np.log1p(z2 / df))


@dataclass
class BOCPDResult:
    index: pd.Index
    p_change_now: np.ndarray            # P(r_t = 0 | y_1..t)
    p_recent: dict                      # w -> P(r_t < w | y_1..t)
    expected_run_length: np.ndarray
    map_run_length: np.ndarray
    log_evidence: np.ndarray            # log p(y_t | y_1..t-1)
    hazard: float
    run_length_matrix: np.ndarray | None = None   # optional (T, R_store) posterior for plots

    def frame(self, prefix: str) -> pd.DataFrame:
        out = {f"{prefix}_p_change_now": self.p_change_now, f"{prefix}_expected_run_length": self.expected_run_length,
               f"{prefix}_map_run_length": self.map_run_length, f"{prefix}_log_predictive": self.log_evidence}
        for w, v in self.p_recent.items():
            out[f"{prefix}_p_change_within_{w}d"] = v
        return pd.DataFrame(out, index=self.index)


def bocpd(y, prior: NIGPrior, expected_run_length: float, max_run_length: int | None = None,
          windows=RECENT_WINDOWS, store_matrix: int | None = None, index=None) -> BOCPDResult:
    """Run BOCPD over y (NaN values are skipped: the posterior is carried forward unchanged)."""
    y = np.asarray(y, float)
    t_len = len(y)
    h = 1.0 / float(expected_run_length)
    log_h, log_1mh = math.log(h), math.log1p(-h)
    log_r = np.array([0.0])                                   # P(r_0 = 0) = 1
    mu = np.array([prior.mu0])
    kappa = np.array([prior.kappa0])
    alpha = np.array([prior.alpha0])
    beta = np.array([prior.beta0])
    p_now, exp_rl, map_rl, evid = (np.full(t_len, np.nan) for _ in range(4))
    recent = {w: np.full(t_len, np.nan) for w in windows}
    mat = np.zeros((t_len, store_matrix)) if store_matrix else None
    for t in range(t_len):
        if np.isfinite(y[t]):
            pi = log_r + student_t_logpdf(y[t], mu, kappa, alpha, beta)
            ev = logsumexp(pi)
            growth = pi + log_1mh
            change = ev + log_h
            log_r = np.concatenate([[change], growth])
            log_r -= logsumexp(log_r)
            mu_new = (kappa * mu + y[t]) / (kappa + 1)
            beta_new = beta + kappa * (y[t] - mu) ** 2 / (2 * (kappa + 1))
            mu = np.concatenate([[prior.mu0], mu_new])
            kappa = np.concatenate([[prior.kappa0], kappa + 1])
            alpha = np.concatenate([[prior.alpha0], alpha + 0.5])
            beta = np.concatenate([[prior.beta0], beta_new])
            # the run-0 hypothesis has the PRIOR statistics: y_t closed the old segment, the new one is empty
            if max_run_length is not None and len(log_r) > max_run_length:
                log_r, mu, kappa, alpha, beta = (a[:max_run_length] for a in (log_r, mu, kappa, alpha, beta))
                log_r -= logsumexp(log_r)
            evid[t] = ev
        p = np.exp(log_r)
        rl = np.arange(len(p))
        p_now[t] = p[0]
        exp_rl[t] = float(p @ rl)
        map_rl[t] = int(np.argmax(p))
        for w in windows:
            recent[w][t] = float(p[:w].sum())
        if mat is not None:
            k = min(store_matrix, len(p))
            mat[t, :k] = p[:k]
    return BOCPDResult(pd.Index(range(t_len)) if index is None else index, p_now, recent, exp_rl, map_rl, evid, h, mat)


def prior_recent_probability(expected_run_length: float, w: int) -> float:
    """Prior probability that at least one change occurred in the last w sessions (constant hazard)."""
    return 1.0 - (1.0 - 1.0 / expected_run_length) ** w


def brute_force_run_length(y, prior: NIGPrior, expected_run_length: float) -> np.ndarray:
    """Exact P(r_T = r | y_1..T) by enumerating every change-point configuration (tests only; tiny T)."""
    import itertools

    y = np.asarray(y, float)
    t_len = len(y)
    h = 1.0 / expected_run_length

    def seg_loglik(seg):
        mu, kappa, alpha, beta, ll = prior.mu0, prior.kappa0, prior.alpha0, prior.beta0, 0.0
        for v in seg:
            ll += float(student_t_logpdf(v, np.array(mu), np.array(kappa), np.array(alpha), np.array(beta)))
            mu, beta = (kappa * mu + v) / (kappa + 1), beta + kappa * (v - mu) ** 2 / (2 * (kappa + 1))
            kappa, alpha = kappa + 1, alpha + 0.5
        return ll

    post = np.zeros(t_len + 1)
    # indicator c_t = 1 if a change happens at step t (t = 1..T-1; step 0 always starts a run, as in bocpd())
    for cps in itertools.product([0, 1], repeat=t_len):
        # in bocpd(), a "change" at step t means y_t closes a segment and r_t = 0 (y_t belongs to the OLD run)
        lp, start = 0.0, 0
        segs = []
        for t in range(t_len):
            lp += math.log(h) if cps[t] else math.log1p(-h)
            if cps[t]:
                segs.append(y[start:t + 1])
                start = t + 1
        segs.append(y[start:])
        # a change at step t means the next segment begins after y_t; the last run length is T - start
        lp += sum(seg_loglik(s) for s in segs if len(s))
        post[t_len - start] += math.exp(lp)
    return post / post.sum()


# --------------------------------------------------------------------------------------
# Synthetic scenario study (documents what BOCPD detects; no real data)
# --------------------------------------------------------------------------------------
SCENARIO_LENGTH, SCENARIO_CHANGE, SCENARIO_PRIOR_ROWS, SCENARIO_ALARM = 600, 300, 250, 0.5


def _episodes(flag: np.ndarray) -> list[tuple[int, int]]:
    out, start = [], None
    for i, v in enumerate(flag):
        if v and start is None:
            start = i
        if not v and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(flag)))
    return out


def scenario_series(seed: int = 7) -> dict:
    """Synthetic inputs; the change (if any) is at SCENARIO_CHANGE. Returns name -> (y, has_change)."""
    rng = np.random.default_rng(seed)
    n, c = SCENARIO_LENGTH, SCENARIO_CHANGE
    e = rng.normal(size=n)
    out = {"mean_shift": (np.r_[e[:c], e[c:] + 1.5], True),
           "variance_increase": (np.r_[e[:c], 3 * e[c:]], True),
           "single_outlier": (np.r_[e[:c], 6.0, e[c + 1:]], False),
           "gradual_drift": (e + np.r_[np.zeros(c), np.linspace(0, 2, n - c)], True),
           "consecutive_shocks": (e + np.r_[np.zeros(c), np.full(10, 2.0), np.zeros(20), np.full(10, 2.0),
                                            np.zeros(n - c - 40)], True)}
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.97 * x[t - 1] + 0.3 * e[t]
    phi = float(np.dot(x[1:SCENARIO_PRIOR_ROWS], x[:SCENARIO_PRIOR_ROWS - 1]) /
                np.dot(x[:SCENARIO_PRIOR_ROWS - 1], x[:SCENARIO_PRIOR_ROWS - 1]))
    out["ar1_no_change_on_levels"] = (x, False)
    out["ar1_no_change_on_innovations"] = (np.r_[np.nan, x[1:] - phi * x[:-1]], False)
    return out


def synthetic_scenarios(expected_run_length: float = PRIMARY_RUN_LENGTH, window: int = PRIMARY_WINDOW,
                        seeds=range(20)) -> pd.DataFrame:
    """Alarm = P(change within `window` sessions) > 0.5. Prior fitted on the first SCENARIO_PRIOR_ROWS points
    (the 'Train' part, before any change). Reports false-alarm episodes per 252 sessions on the pre-change part
    (or the whole series if there is no change) and the detection delay after the change."""
    rows = []
    for seed in seeds:
        for name, (y, has_change) in scenario_series(seed).items():
            prior = NIGPrior.from_train(y[:SCENARIO_PRIOR_ROWS])
            p = bocpd(y, prior, expected_run_length, windows=(window,)).p_recent[window]
            flag = np.nan_to_num(p) > SCENARIO_ALARM
            flag[:window] = False                      # the first run is trivially 'recent'
            pre = flag[:SCENARIO_CHANGE] if has_change else flag
            after = np.flatnonzero(flag[SCENARIO_CHANGE:]) if has_change else np.array([])
            rows.append({"scenario": name, "seed": seed, "has_change": has_change,
                         "false_alarm_episodes_per_252": len(_episodes(pre)) * 252 / len(pre),
                         "detected": bool(len(after)), "delay_sessions": float(after[0]) if len(after) else np.nan,
                         "alarm_share_after_change": float(flag[SCENARIO_CHANGE:].mean()) if has_change else np.nan})
    d = pd.DataFrame(rows)
    return d.groupby("scenario", sort=False).agg(
        has_change=("has_change", "first"), runs=("seed", "size"),
        false_alarm_episodes_per_252=("false_alarm_episodes_per_252", "mean"),
        detection_rate=("detected", "mean"), median_delay_sessions=("delay_sessions", "median"),
        alarm_share_after_change=("alarm_share_after_change", "mean")).reset_index()
