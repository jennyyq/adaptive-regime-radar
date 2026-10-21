"""Stage 5: running the HMM candidates, diagnostics, detection delay, selection and protocol checks.

Data rules
  * Only data/splits/{train,validation}_{features,returns}.csv are read (via the Stage 4 loader, which refuses
    the test split). The optional late-Train composition check reads data/processed/factor_components.csv
    line by line and stops at the first row after the last Train date, so no later row is parsed.
  * Every parameter (scaler, HMM parameters, prior strength choice, thresholds) comes from Train, or from the
    Train/Validation comparison for model selection. Validation is passed only to the causal filter.
  * The Stage 4 GMM reference is rebuilt from Train with the Stage 4 specification (hmm_robustness) and
    used only for predictions; its pickle is compared only when written by the same scikit-learn.

Selection rules are fixed in SELECTION_RULES / SELECTION_PROCEDURE before any real-data result is seen.
"""

from __future__ import annotations

STAGE5_CODE_VERSION = "stage5-2026-10-21c"  # must match scripts/07_fit_hmm.py

import csv
import json
import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from . import regime_diagnostics as rd
from .dataset import FEATURE_COLUMNS
from .factors import Check
from .hmm_filtering import FilterResult, filter_train_validation, smoothed_probabilities
from .hmm_robustness import PHASE_RULES, phase_consistency
from .hmm_models import (HMM_MAX_ITER, HMM_N_INIT, HMM_REG_COVAR, HMM_TOL, HMMFit, HMMParams, HMMSpec,
                         align_by_means, canonical_order, candidate_specs, expected_durations, fit_hmm_best,
                         information_criteria, log_emission, parameter_distance, stress_scores)

PASS, WARN, FAIL, NOT_RUN = "PASS", "WARN", "FAIL", "NOT RUN"
TRADING_DAYS = 252

SELECTION_RULES = {
    "min_converged_seed_share": 0.8,        # >= 80% of seeds converge
    "min_train_state_share": 0.03,          # every state >= 3% of Train days (filtered argmax)
    "min_mean_pairwise_ari_train": 0.80,    # seed stability of Train filtered assignments
    "min_validation_state_share": 0.01,     # below = 'collapsed in validation' (FLAG only)
    "near_absorbing_self_transition": 0.999,  # A_ii above = expected duration > 1000 days (FLAG only)
    "flag_median_run_days": 2,              # median filtered run <= 2 days (FLAG only)
    "nw_lags": 21,
    "nw_t_threshold": 2.0,                  # 'significantly worse' than the best validation log score
    "max_median_delay_increase_days": 5,    # Sticky vs Standard (same K, covariance): detection-delay budget
    "max_undetected_share_increase": 0.10,
    "min_switch_reduction": 0.20,           # Sticky must cut Validation switches/year by >= 20% to be preferred
    "best_basin_tolerance_per_obs": 1e-3,   # seeds within this objective/obs of the best = 'same optimum'
}
SELECTION_PROCEDURE = [
    "1. Eligibility: >= 80% of the 10 seeds converge and EM is monotone; every Train state >= 3% of days "
    "(filtered argmax); mean pairwise ARI of Train filtered assignments across seeds >= 0.80. A state below 1% of "
    "Validation days, a self-transition above 0.999 or a median run of <= 2 days is FLAGGED, not excluded.",
    "2. Validation evidence: the primary score is the average one-step-ahead predictive log density "
    "log p(x_t | x_1..x_t-1) on Validation, with parameters fixed at their Train estimates and the recursion "
    "continued from the last Train filtered distribution. The best eligible model defines the benchmark; a model "
    "is 'competitive' if its mean daily shortfall has a Newey-West t-statistic (21 lags) below 2.",
    "3. Parsimony: among competitive models choose the smallest K, then diagonal over full covariance. If that K is "
    "larger than the K of the best eligible smaller model (i.e. the smaller K is significantly worse), the larger K "
    "must also beat it in at least 3 of the 4 Validation calendar years (phase consistency); otherwise the smaller "
    "K is kept.",
    "4. Persistence vs delay (same K and covariance): a Sticky model is preferred to the Standard model only if it "
    "is competitive, reduces Validation switches per year by >= 20%, and respects the delay budget on Train+"
    "Validation: (a) versus factor-based stress onsets, the median days until P(stress) >= 0.5 rise by <= 5 and "
    "at most 10 percentage points more onsets go undetected within 63 days; (b) on the same sequence, the Sticky "
    "model enters the Standard model's stress episodes with a median lag of <= 5 days and misses <= 10% of them. "
    "Among qualifying Sticky models, the highest Validation log score wins. Otherwise the Standard model is kept.",
    "5. Train AIC/BIC, economic profiles, asset behaviour and timelines are reported for every model but do not "
    "rank models, so that no model is chosen for an attractive timeline. Crisis dates are never used.",
    "6. The selected HMM is compared with the frozen Stage 4 GMM (diag, K=2) by daily predictive log scores "
    "(Newey-West t-test). Classification agreement is reported separately and is not evidence of better forecasts.",
]

# Protocol amendments made after the first real-data run (recorded, not hidden)
PROTOCOL_AMENDMENTS = [
    {"version": "stage5 v1 -> v1.1", "date": "2026-10-21",
     "change": "Seed-stability eligibility is measured on the per-seed estimator 'best of 5 EM starts' instead of "
               "single EM starts.",
     "reason": "In v1 each seed was ONE EM start, so the pairwise ARI across seeds measured how often EM falls into "
               "a local optimum, not whether the estimator is reproducible. Stage 4 (whose 0.80 ARI threshold this "
               "rule copies) used n_init = 5 starts per seed for the GMM. In the first real run 7/10 single starts "
               "for Standard diag K=3 and 2/10 for K=4 reached the identical best optimum (parameter distance 0), "
               "yet K=3/K=4 were excluded with single-start ARI 0.725 / 0.643 although their Validation log "
               "scores were higher than K=2.",
     "first_run_outcome": "v1 rules selected hmm_std_diag_k2. The frequency of EM local optima is still reported "
                          "(single_start_best_basin_share in hmm_initialization_stability.csv)."},
    {"version": "stage5 v1 -> v1.1", "date": "2026-10-21",
     "change": "Detection lag is signed: if the model is already in its stress states when a reference onset "
               "occurs, the lag is minus the number of days the current stress run had lasted (capped at -63).",
     "reason": "In the first real run the K=2 models were already in stress at 100% of factor-score onsets, so the "
               "v1 lag was 0 by construction and could not distinguish models."},
    {"version": "stage5 v1 -> v1.1", "date": "2026-10-21",
     "change": "Report fixes only: '|' inside table cells replaced by '/', best eligible and best overall "
               "Validation model reported separately, emission-vs-transition evidence diagnostic added.",
     "reason": "Markdown tables were misaligned; the summary named the best eligible model as the best model."},
    {"version": "stage5 v1.1 -> v1.2", "date": "2026-10-21",
     "change": "Phase-consistency gate for a larger K (>= 3 of 4 Validation years), probability-calibration "
               "diagnostics with an out-of-sample covariance-inflation test (chosen on 2016-2017, judged on "
               "2018-2019), training-window sensitivity (4 predefined windows), Stage 4 GMM reference rebuilt from "
               "Train instead of unpickled, and a log of all library warnings.",
     "reason": "Requested review after the first real run: (1) K=3/K=4 scored higher on Validation than K=2, so a "
               "larger K needs evidence that the gain is not one episode; the yearly split was fixed before the "
               "yearly results were seen; (2) mean max filtered probability 0.990 on Validation needs a "
               "calibration test rather than ad hoc smoothing; (3) a scikit-learn version-mismatch warning came from "
               "unpickling the Stage 4 GMM.",
     "first_run_outcome": "Under v1 the selected model was hmm_std_diag_k2; best overall Validation score was "
                          "hmm_std_diag_k4 (-5.803 vs -6.159 per day)."},
]

# Detection-delay reference events (fixed before modelling; descriptive, not ground truth)
EVENT_RULES = {
    "stress_score_quantile": 0.80,   # factor stress score above its Train 80th percentile = exceedance day
    "close_gap_days": 5,             # gaps shorter than this inside an exceedance run are closed
    "min_episode_days": 10,          # episodes shorter than this are ignored
    "merge_gap_days": 21,            # episodes closer than this are merged (onset = first day)
    "max_lag_days": 63,              # not reached within 63 sessions = 'not detected'
    "thresholds": (0.5, 0.8),
    "event_window": (-10, 30),
}
LATE_TRAIN_MIN_ROWS = 500
GMM_REFERENCE_ID = "gmm_diag_k2"


# --------------------------------------------------------------------------------------
# Running all candidates
# --------------------------------------------------------------------------------------
@dataclass
class Candidate:
    spec: HMMSpec
    reference: HMMFit                       # canonical order applied
    seed_fits: list[HMMFit]                 # each aligned to the reference by emission means
    train: FilterResult
    validation: FilterResult
    train_labels: np.ndarray
    validation_labels: np.ndarray
    seed_train_labels: list[np.ndarray]
    seed_validation_labels: list[np.ndarray]
    stress_states: list[int]
    validation_smoothed_labels: np.ndarray
    flags: list[str] = field(default_factory=list)
    validation_emission_log_lr: float = float("nan")    # median |log N(x|top state) - log N(x|2nd state)|

    @property
    def model_id(self) -> str:
        return self.spec.model_id

    def stress_probability(self, split: str) -> np.ndarray:
        res = self.train if split == "train" else self.validation
        return res.filtered[:, self.stress_states].sum(axis=1)


def _apply_order(fit: HMMFit, order) -> HMMFit:
    fit.params = fit.params.permuted(order)
    return fit


def stress_state_set(params: HMMParams, columns, mean: np.ndarray, scale: np.ndarray, train_original: np.ndarray
                     ) -> list[int]:
    """States whose emission-mean stress score exceeds the Train average stress score (fixed rule)."""
    scores = stress_scores(params.means * scale + mean, list(columns))
    overall = float(stress_scores(train_original, list(columns)).mean())
    chosen = [int(j) for j in np.flatnonzero(scores > overall)]
    if not chosen or len(chosen) == params.k:
        chosen = [int(np.argmax(scores))]
    return chosen


def run_candidate(spec: HMMSpec, x_train: np.ndarray, x_validation: np.ndarray, seeds, mean: np.ndarray,
                  scale: np.ndarray, train_original: np.ndarray, columns=FEATURE_COLUMNS, n_init: int = HMM_N_INIT
                  ) -> Candidate:
    fits = [fit_hmm_best(x_train, spec, s, n_init) for s in seeds]
    pool = [f for f in fits if f.converged] or fits
    ref = max(pool, key=lambda f: f.objective)
    _apply_order(ref, canonical_order(ref.params, list(columns), mean, scale))
    for f in fits:
        if f is not ref:
            _apply_order(f, align_by_means(ref.params, f.params))
    tr, va = filter_train_validation(ref.params, x_train, x_validation)
    seed_tr, seed_va = [], []
    for f in fits:
        a, b = (tr, va) if f is ref else filter_train_validation(f.params, x_train, x_validation)
        seed_tr.append(a.filtered.argmax(axis=1))
        seed_va.append(b.filtered.argmax(axis=1))
    init_va = tr.last_filtered @ ref.params.transmat
    smooth_va = smoothed_probabilities(ref.params, x_validation, initial_predicted=init_va).argmax(axis=1)
    stress = stress_state_set(ref.params, columns, mean, scale, train_original)
    lb = np.sort(log_emission(ref.params, x_validation), axis=1)
    gap = float(np.median(lb[:, -1] - lb[:, -2]))
    return Candidate(spec, ref, fits, tr, va, tr.filtered.argmax(axis=1), va.filtered.argmax(axis=1),
                     seed_tr, seed_va, stress, smooth_va, validation_emission_log_lr=gap)


def run_all(x_train, x_validation, mean, scale, train_original, seeds, specs=None, progress=None,
            n_init: int = HMM_N_INIT) -> dict[str, Candidate]:
    out = {}
    for spec in specs or candidate_specs():
        out[spec.model_id] = run_candidate(spec, x_train, x_validation, seeds, mean, scale, train_original,
                                           n_init=n_init)
        if progress:
            progress(out[spec.model_id])
    return out


# --------------------------------------------------------------------------------------
# Per-model tables
# --------------------------------------------------------------------------------------
def _shares(labels, k) -> np.ndarray:
    return rd.population(labels, k)


def _share_text(shares) -> str:
    return " / ".join(f"S{i + 1} {s:.0%}" for i, s in enumerate(shares))


def probability_metrics(c: Candidate, split: str) -> dict:
    res = c.train if split == "train" else c.validation
    p = res.filtered
    ps = c.stress_probability(split)
    above = ps >= 0.5
    crossings = int((above[1:] != above[:-1]).sum())
    n = len(ps)
    uncertain = (ps > 0.2) & (ps < 0.8)
    runs = rd.run_lengths(uncertain.astype(int))
    unc_runs = runs[runs["state"] == 1]["length"]
    return {
        f"{split}_mean_max_filtered_prob": float(p.max(axis=1).mean()),
        f"{split}_share_max_prob_below_0.6": float((p.max(axis=1) < 0.6).mean()),
        f"{split}_mean_normalized_entropy": float(rd.normalized_entropy(p).mean()),
        f"{split}_mean_abs_daily_change_p_stress": float(np.abs(np.diff(ps)).mean()) if n > 1 else np.nan,
        f"{split}_p_stress_crossings_per_year": crossings / n * TRADING_DAYS if n > 1 else np.nan,
        f"{split}_share_uncertain_days": float(uncertain.mean()),
        f"{split}_mean_uncertain_spell_days": float(unc_runs.mean()) if len(unc_runs) else 0.0,
    }


def stability_row(c: Candidate) -> dict:
    fits = c.seed_fits
    n = c.reference.n_train
    objs = np.array([f.objective for f in fits]) / n
    best = objs.max()
    in_basin = objs >= best - SELECTION_RULES["best_basin_tolerance_per_obs"]
    pair_tr = rd.pairwise_agreement(c.seed_train_labels, c.spec.k)
    pair_va = rd.pairwise_agreement(c.seed_validation_labels, c.spec.k)
    dists = [parameter_distance(c.reference.params, f.params) for f, b in zip(fits, in_basin) if b]
    starts = np.array([o for f in fits for o in f.start_objectives]) / n
    single_basin = float(np.mean(starts >= starts.max() - SELECTION_RULES["best_basin_tolerance_per_obs"])) \
        if len(starts) else np.nan
    return {
        "model_id": c.model_id, "n_seeds": len(fits), "n_init_per_seed": len(fits[0].start_objectives),
        "converged_share": float(np.mean([f.converged for f in fits])),
        "all_monotone": bool(all(f.monotone and f.all_starts_monotone for f in fits)),
        "single_start_best_basin_share": single_basin,
        "single_start_converged_share": float(np.mean([c_ for f in fits for c_ in f.start_converged])),
        "empty_state_events": int(sum(f.empty_state_events for f in fits)),
        "reference_seed": c.reference.seed,
        "objective_per_obs_best": float(best), "objective_per_obs_worst": float(objs.min()),
        "objective_per_obs_range": float(best - objs.min()),
        "best_basin_share": float(in_basin.mean()),
        "train_mean_pairwise_ari": pair_tr["mean_ari"], "train_min_pairwise_ari": pair_tr["min_ari"],
        "validation_mean_pairwise_ari": pair_va["mean_ari"], "validation_min_pairwise_ari": pair_va["min_ari"],
        "train_mean_ari_vs_reference": float(np.mean([adjusted_rand_score(c.train_labels, l)
                                                       for l in c.seed_train_labels])),
        "best_basin_max_mean_distance": float(max(d["max_mean_distance"] for d in dists)),
        "best_basin_max_transition_diff": float(max(d["max_transition_abs_diff"] for d in dists)),
        "mean_iterations": float(np.mean([f.n_iter for f in fits])),
        "seconds_total": float(sum(f.seconds for f in fits)),
    }


def seed_table(cands: dict[str, Candidate]) -> pd.DataFrame:
    rows = []
    for c in cands.values():
        for f, ltr in zip(c.seed_fits, c.seed_train_labels):
            rows.append({"model_id": c.model_id, "seed": f.seed, "is_reference": f is c.reference,
                         "converged": f.converged, "monotone": f.monotone, "n_iter": f.n_iter,
                         "train_loglik": f.train_loglik, "objective": f.objective,
                         "objective_per_obs": f.objective / f.n_train,
                         "ari_vs_reference_train": adjusted_rand_score(c.train_labels, ltr),
                         "self_transitions": " / ".join(f"{a:.4f}" for a in np.diag(f.params.transmat)),
                         "notes": "; ".join(f.warnings) or "-"})
    return pd.DataFrame(rows)


def comparison_row(c: Candidate, n_features: int) -> dict:
    k = c.spec.k
    ref = c.reference
    ic = information_criteria(ref, n_features)
    sh_tr, sh_va = _shares(c.train_labels, k), _shares(c.validation_labels, k)
    sw_tr, sw_va = rd.switching_stats(c.train_labels), rd.switching_stats(c.validation_labels)
    durations = expected_durations(ref.params.transmat)
    row = {
        "model_id": c.model_id, "family": c.spec.family, "k": k, "covariance_type": c.spec.covariance_type,
        "prior_weight": c.spec.prior_weight, "kappa": ref.kappa, **ic,
        "converged": ref.converged, "n_iter": ref.n_iter, "monotone": ref.monotone,
        "train_loglik_total": ref.train_loglik,
        "train_avg_loglik": float(c.train.log_score.mean()),
        "validation_avg_loglik": float(c.validation.log_score.mean()),
        "train_validation_gap": float(c.train.log_score.mean() - c.validation.log_score.mean()),
        "train_map_objective_per_obs": ref.objective / ref.n_train,
        "self_transitions": " / ".join(f"{a:.4f}" for a in np.diag(ref.params.transmat)),
        "min_self_transition": float(np.diag(ref.params.transmat).min()),
        "max_self_transition": float(np.diag(ref.params.transmat).max()),
        "expected_durations_days": " / ".join("inf" if not np.isfinite(d) else f"{d:.1f}" for d in durations),
        "stress_states": " ".join(f"S{j + 1}" for j in c.stress_states),
        "train_shares": _share_text(sh_tr), "validation_shares": _share_text(sh_va),
        "train_min_state_share": float(sh_tr.min()), "validation_min_state_share": float(sh_va.min()),
        "train_switches_per_year": sw_tr["switches_per_year"],
        "validation_switches_per_year": sw_va["switches_per_year"],
        "train_median_duration_days": sw_tr["median_duration_days"],
        "validation_median_duration_days": sw_va["median_duration_days"],
        "validation_hindsight_revision_share": float(np.mean(c.validation_smoothed_labels != c.validation_labels)),
        "validation_median_emission_log_lr": c.validation_emission_log_lr,
        "mean_transition_log_odds": float(np.mean(np.log(np.diag(ref.params.transmat)
                                                         / np.clip(1 - np.diag(ref.params.transmat), 1e-300, None)))),
        "validation_filtered_vs_smoothed_ari": float(adjusted_rand_score(c.validation_smoothed_labels,
                                                                         c.validation_labels)),
    }
    row.update(probability_metrics(c, "train"))
    row.update(probability_metrics(c, "validation"))
    return row


def train_validation_table(comp: pd.DataFrame, stab: pd.DataFrame) -> pd.DataFrame:
    m = comp.merge(stab[["model_id", "train_mean_pairwise_ari", "converged_share", "best_basin_share"]],
                   on="model_id")
    cols = ["model_id", "family", "k", "covariance_type", "prior_weight", "train_avg_loglik", "validation_avg_loglik",
            "train_validation_gap", "train_shares", "validation_shares", "train_switches_per_year",
            "validation_switches_per_year", "train_median_duration_days", "validation_median_duration_days",
            "train_mean_normalized_entropy", "validation_mean_normalized_entropy", "train_mean_max_filtered_prob",
            "validation_mean_max_filtered_prob", "converged", "converged_share", "train_mean_pairwise_ari",
            "best_basin_share"]
    return m[cols].rename(columns={"train_mean_pairwise_ari": "initialization_stability_mean_ari"})


def transition_table(cands: dict[str, Candidate]) -> pd.DataFrame:
    rows = []
    for c in cands.values():
        a = c.reference.params.transmat
        dur = expected_durations(a)
        emp = rd.transition_counts(c.train_labels, c.spec.k)
        emp_va = rd.transition_counts(c.validation_labels, c.spec.k)
        for i in range(c.spec.k):
            for j in range(c.spec.k):
                rows.append({"model_id": c.model_id, "from_state": f"S{i + 1}", "to_state": f"S{j + 1}",
                             "probability": a[i, j],
                             "expected_duration_days_from_state": dur[i] if i == j else np.nan,
                             "train_filtered_argmax_transitions": int(emp[i, j]),
                             "validation_filtered_argmax_transitions": int(emp_va[i, j])})
    return pd.DataFrame(rows)


def duration_table(cands: dict[str, Candidate], index: dict[str, pd.DatetimeIndex]) -> pd.DataFrame:
    rows = []
    for c in cands.values():
        dur = expected_durations(c.reference.params.transmat)
        for split, labels in (("train", c.train_labels), ("validation", c.validation_labels)):
            runs = rd.run_lengths(labels)
            years = len(labels) / TRADING_DAYS
            for s in range(c.spec.k):
                r = runs[runs["state"] == s]["length"]
                rows.append({"model_id": c.model_id, "split": split, "state": f"S{s + 1}",
                             "share": float(np.mean(labels == s)), "n_runs": len(r),
                             "runs_per_year": len(r) / years,
                             "median_run_days": float(r.median()) if len(r) else np.nan,
                             "mean_run_days": float(r.mean()) if len(r) else np.nan,
                             "max_run_days": int(r.max()) if len(r) else 0,
                             "expected_duration_days_model": dur[s],
                             "rare_state_warning": bool(np.mean(labels == s) < 0.05)})
    return pd.DataFrame(rows)


def profile_table(cands: dict[str, Candidate], train_feat: pd.DataFrame, val_feat: pd.DataFrame,
                  mean: np.ndarray, scale: np.ndarray) -> pd.DataFrame:
    rows = []
    for c in cands.values():
        k = c.spec.k
        p_tr = rd.factor_profile(train_feat[FEATURE_COLUMNS], c.train_labels, k)
        p_va = rd.factor_profile(val_feat[FEATURE_COLUMNS], c.validation_labels, k)
        emis = c.reference.params.means * scale + mean
        for s in range(k):
            st = f"S{s + 1}"
            label, rule = rd.hypothesis_label(p_tr.loc[st])
            shift = (float(np.sqrt(((p_tr.loc[st, FEATURE_COLUMNS] - p_va.loc[st, FEATURE_COLUMNS]) ** 2).sum()))
                     if p_va.loc[st, "n_obs"] > 0 else np.nan)
            for split, prof in (("train", p_tr), ("validation", p_va)):
                rows.append({"model_id": c.model_id, "family": c.spec.family, "k": k, "state": st, "split": split,
                             "basis": "filtered argmax", "n_obs": int(prof.loc[st, "n_obs"]),
                             "share": prof.loc[st, "share"], **{f: prof.loc[st, f] for f in FEATURE_COLUMNS},
                             "is_stress_state": s in c.stress_states, "hypothesis_label": label, "label_rule": rule,
                             "train_validation_profile_distance": shift})
            rows.append({"model_id": c.model_id, "family": c.spec.family, "k": k, "state": st, "split": "train",
                         "basis": "emission mean", "n_obs": np.nan, "share": np.nan,
                         **{f: emis[s, i] for i, f in enumerate(FEATURE_COLUMNS)},
                         "is_stress_state": s in c.stress_states, "hypothesis_label": "-", "label_rule": "-",
                         "train_validation_profile_distance": np.nan})
    return pd.DataFrame(rows)


def asset_table(cands: dict[str, Candidate], train_returns: pd.DataFrame, val_returns: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for c in cands.values():
        for split, ret, labels in (("train", train_returns, c.train_labels),
                                   ("validation", val_returns, c.validation_labels)):
            st = rd.asset_statistics(ret, labels, c.spec.k)
            st.insert(0, "split", split)
            st.insert(0, "model_id", c.model_id)
            frames.append(st)
    out = pd.concat(frames, ignore_index=True)
    out["state_basis"] = "filtered argmax at close of t"
    return out


def probability_frames(cands: dict[str, Candidate], tr_index, va_index):
    tr, va, pv, lab_tr, lab_va = {}, {}, {}, {}, {}
    for c in cands.values():
        for j in range(c.spec.k):
            tr[f"{c.model_id}__S{j + 1}"] = c.train.filtered[:, j]
            va[f"{c.model_id}__S{j + 1}"] = c.validation.filtered[:, j]
            pv[f"{c.model_id}__S{j + 1}"] = c.validation.predicted[:, j]
        tr[f"{c.model_id}__p_stress"] = c.stress_probability("train")
        va[f"{c.model_id}__p_stress"] = c.stress_probability("validation")
        lab_tr[c.model_id] = [f"S{i + 1}" for i in c.train_labels]
        lab_va[c.model_id] = [f"S{i + 1}" for i in c.validation_labels]
    return (pd.DataFrame(tr, index=tr_index), pd.DataFrame(va, index=va_index), pd.DataFrame(pv, index=va_index),
            pd.DataFrame(lab_tr, index=tr_index), pd.DataFrame(lab_va, index=va_index))


# --------------------------------------------------------------------------------------
# Detection delay versus stability
# --------------------------------------------------------------------------------------
def stress_score_series(features: pd.DataFrame) -> pd.Series:
    return sum(w * features[c] for c, w in rd.STRESS_SCORE.items())


def episodes_from_indicator(indicator: np.ndarray, rules: dict = EVENT_RULES) -> list[tuple[int, int]]:
    """Close short gaps, drop short runs, merge nearby episodes. Returns (start, end) index pairs (end exclusive)."""
    ind = np.asarray(indicator, dtype=bool).copy()
    runs = rd.run_lengths(ind.astype(int))
    for r in runs.itertuples():
        if r.state == 0 and r.length < rules["close_gap_days"] and 0 < r.start and r.start + r.length < len(ind):
            ind[r.start:r.start + r.length] = True
    runs = rd.run_lengths(ind.astype(int))
    eps = [(int(r.start), int(r.start + r.length)) for r in runs.itertuples()
           if r.state == 1 and r.length >= rules["min_episode_days"]]
    merged = []
    for s, e in eps:
        if merged and s - merged[-1][1] < rules["merge_gap_days"]:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged


def onset_lags(p_stress: np.ndarray, onsets: list[int], threshold: float, max_lag: int) -> list[dict]:
    """Signed lag: days from the reference onset until P(stress) >= threshold. If the model is already above the
    threshold on the day before the onset, the lag is minus the length of that ongoing stress run (capped at
    -max_lag): the model led the reference."""
    out = []
    for o in onsets:
        already = o > 0 and p_stress[o - 1] >= threshold
        window = p_stress[o:o + max_lag + 1]
        hit = np.flatnonzero(window >= threshold)
        if already:
            back = p_stress[max(o - max_lag, 0):o][::-1] >= threshold
            lead = int(np.argmin(back)) if not back.all() else len(back)
            lag = -lead
        else:
            lag = int(hit[0]) if len(hit) else np.nan
        out.append({"onset_index": o, "lag_days": lag, "already_in_stress": bool(already),
                    "detected": bool(already or len(hit) > 0)})
    return out


def reference_events(train_feat: pd.DataFrame, val_feat: pd.DataFrame, gmm_labels: dict | None
                     ) -> dict[str, dict[str, list[tuple[int, int]]]]:
    """Reference onsets per split. R1: factor stress score above its Train 80th percentile (model-free).
    R2: Stage 4 GMM (diag, K=2) stress-state runs (a static detector). Neither is ground truth."""
    s_tr, s_va = stress_score_series(train_feat), stress_score_series(val_feat)
    q = float(s_tr.quantile(EVENT_RULES["stress_score_quantile"]))
    ev = {"factor_stress_score": {"train": episodes_from_indicator(s_tr.to_numpy() > q),
                                  "validation": episodes_from_indicator(s_va.to_numpy() > q)},
          "_threshold": q}
    if gmm_labels is not None:
        ev["gmm_static_stress"] = {split: episodes_from_indicator(gmm_labels[split] == 1)
                                   for split in ("train", "validation")}
    return ev


def detection_tables(cands: dict[str, Candidate], events: dict, tr_index, va_index
                     ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    idx = {"train": tr_index, "validation": va_index}
    lag_rows, ep_rows = [], []
    for ref_name, by_split in events.items():
        if ref_name.startswith("_"):
            continue
        for split, eps in by_split.items():
            for s, e in eps:
                ep_rows.append({"reference": ref_name, "split": split, "onset": idx[split][s].date().isoformat(),
                                "end": idx[split][e - 1].date().isoformat(), "length_days": e - s})
            onsets = [s for s, _ in eps]
            for c in cands.values():
                ps = c.stress_probability(split)
                for h in EVENT_RULES["thresholds"]:
                    for r in onset_lags(ps, onsets, h, EVENT_RULES["max_lag_days"]):
                        lag_rows.append({"model_id": c.model_id, "reference": ref_name, "split": split,
                                         "threshold": h, "onset": idx[split][r["onset_index"]].date().isoformat(),
                                         **{k: v for k, v in r.items() if k != "onset_index"}})
                # missed / unconditional shares
                inside = np.zeros(len(ps), dtype=bool)
                for s, e in eps:
                    inside[s:e] = True
                lag_rows.append({"model_id": c.model_id, "reference": ref_name, "split": split, "threshold": "share",
                                 "onset": "-", "lag_days": np.nan, "already_in_stress": False, "detected": False,
                                 "missed_share_in_episodes": float((ps[inside] < 0.5).mean()) if inside.any() else np.nan,
                                 "stress_share_outside_episodes": float((ps[~inside] >= 0.5).mean())
                                 if (~inside).any() else np.nan})
    lags = pd.DataFrame(lag_rows)
    onset_frame = pd.DataFrame(ep_rows)
    summary = detection_summary(lags)
    return lags, summary, onset_frame


def detection_summary(lags: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ev = lags[lags["threshold"] != "share"]
    shares = lags[lags["threshold"] == "share"]
    for (model_id, ref), g in ev.groupby(["model_id", "reference"]):
        for split_name, sub in (("train", g[g["split"] == "train"]), ("validation", g[g["split"] == "validation"]),
                                ("pooled", g)):
            row = {"model_id": model_id, "reference": ref, "split": split_name}
            for h in EVENT_RULES["thresholds"]:
                s = sub[sub["threshold"] == h]
                tag = f"p{int(h * 100)}"
                row["n_onsets"] = len(s)
                row[f"median_lag_{tag}"] = float(s["lag_days"].median()) if s["lag_days"].notna().any() else np.nan
                row[f"mean_lag_{tag}"] = float(s["lag_days"].mean()) if s["lag_days"].notna().any() else np.nan
                row[f"undetected_share_{tag}"] = float((~s["detected"]).mean()) if len(s) else np.nan
                row[f"detected_within_21d_{tag}"] = float((s["lag_days"] <= 21).mean()) if len(s) else np.nan
                row[f"already_in_stress_share_{tag}"] = float(s["already_in_stress"].mean()) if len(s) else np.nan
            sh = shares[(shares["model_id"] == model_id) & (shares["reference"] == ref)]
            if split_name != "pooled":
                sh = sh[sh["split"] == split_name]
            row["missed_share_in_episodes"] = float(sh["missed_share_in_episodes"].mean()) if len(sh) else np.nan
            row["stress_share_outside_episodes"] = float(sh["stress_share_outside_episodes"].mean()) if len(sh) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def paired_delay(std_ps: np.ndarray, sticky_ps: np.ndarray, rules: dict = EVENT_RULES, lead: int = 10) -> dict:
    """Same-sequence comparison. Episodes are runs of the Standard model's P(stress) >= 0.5 (event rules applied).
    entry lag = first day from (onset - lead) on which the Sticky P(stress) >= 0.5, minus the onset (negative =
    Sticky earlier); exit lag = first day from (end - lead) on which the Sticky P(stress) < 0.5, minus the
    Standard end. Not reached within max_lag_days = missed."""
    eps = episodes_from_indicator(std_ps >= 0.5, rules)
    entry, exit_, missed = [], [], 0
    n = len(std_ps)
    for s, e in eps:
        lo = max(s - lead, 0)
        hit = np.flatnonzero(sticky_ps[lo:min(s + rules["max_lag_days"] + 1, n)] >= 0.5)
        if len(hit):
            entry.append(lo + int(hit[0]) - s)
        else:
            missed += 1
        if e < n:
            lo = max(e - lead, s)
            out = np.flatnonzero(sticky_ps[lo:min(e + rules["max_lag_days"] + 1, n)] < 0.5)
            if len(out):
                exit_.append(lo + int(out[0]) - e)
    return {"n_standard_episodes": len(eps), "missed_share": missed / len(eps) if eps else np.nan,
            "median_entry_lag": float(np.median(entry)) if entry else np.nan,
            "mean_entry_lag": float(np.mean(entry)) if entry else np.nan,
            "max_entry_lag": float(np.max(entry)) if entry else np.nan,
            "median_exit_lag": float(np.median(exit_)) if exit_ else np.nan,
            "mean_exit_lag": float(np.mean(exit_)) if exit_ else np.nan}


def paired_delay_table(cands: dict[str, Candidate]) -> pd.DataFrame:
    rows = []
    for c in cands.values():
        if c.spec.family != "sticky":
            continue
        std_id = HMMSpec("standard", c.spec.k, c.spec.covariance_type).model_id
        if std_id not in cands:
            continue
        s = cands[std_id]
        for split in ("train", "validation", "pooled"):
            if split == "pooled":
                a = np.r_[s.stress_probability("train"), s.stress_probability("validation")]
                b = np.r_[c.stress_probability("train"), c.stress_probability("validation")]
            else:
                a, b = s.stress_probability(split), c.stress_probability(split)
            rows.append({"model_id": c.model_id, "standard_model": std_id, "split": split, **paired_delay(a, b)})
    return pd.DataFrame(rows)


def event_study(cands: dict[str, Candidate], events: dict, reference: str = "factor_stress_score") -> pd.DataFrame:
    """Average filtered stress probability around reference onsets (offset in trading days)."""
    lo, hi = EVENT_RULES["event_window"]
    rows = []
    for c in cands.values():
        for split in ("train", "validation"):
            ps = c.stress_probability(split)
            for off in range(lo, hi + 1):
                vals = [ps[s + off] for s, _ in events.get(reference, {}).get(split, []) if 0 <= s + off < len(ps)]
                rows.append({"model_id": c.model_id, "split": split, "offset": off,
                             "mean_p_stress": float(np.mean(vals)) if vals else np.nan, "n": len(vals)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------------------
def daily_score_frame(cands: dict[str, Candidate], va_index, gmm_scores: np.ndarray | None) -> pd.DataFrame:
    frame = pd.DataFrame({m: c.validation.log_score for m, c in cands.items()}, index=va_index)
    if gmm_scores is not None:
        frame[GMM_REFERENCE_ID] = gmm_scores
    return frame


def eligibility(comp: pd.DataFrame, stab: pd.DataFrame) -> pd.DataFrame:
    r = SELECTION_RULES
    m = comp.merge(stab, on="model_id")
    rows = []
    for row in m.itertuples():
        reasons, flags = [], []
        if row.converged_share < r["min_converged_seed_share"]:
            reasons.append(f"converged seeds {row.converged_share:.0%}")
        if not row.all_monotone:
            reasons.append("EM objective decreased")
        if row.train_min_state_share < r["min_train_state_share"]:
            reasons.append(f"smallest train state {row.train_min_state_share:.1%}")
        if not (row.train_mean_pairwise_ari >= r["min_mean_pairwise_ari_train"]):
            reasons.append(f"seed ARI {row.train_mean_pairwise_ari:.2f}")
        if row.validation_min_state_share < r["min_validation_state_share"]:
            flags.append("state collapsed in validation")
        if row.max_self_transition > r["near_absorbing_self_transition"]:
            flags.append("near-absorbing state")
        if row.validation_median_duration_days <= r["flag_median_run_days"]:
            flags.append("short validation runs")
        if not row.converged:
            flags.append("reference fit not converged")
        rows.append({"model_id": row.model_id, "eligible": not reasons, "ineligible_reasons": "; ".join(reasons) or "-",
                     "flags": "; ".join(flags) or "-"})
    return pd.DataFrame(rows)


def select_model(comp: pd.DataFrame, elig: pd.DataFrame, scores: pd.DataFrame, detection: pd.DataFrame,
                 paired: pd.DataFrame | None = None) -> dict:
    r = SELECTION_RULES
    table = comp.merge(elig, on="model_id")
    ok = table[table["eligible"]]
    out = {"rules": r, "procedure": SELECTION_PROCEDURE}
    if ok.empty:
        out.update({"selected": None, "reason": "no eligible model"})
        return out
    best = ok.loc[ok["validation_avg_loglik"].idxmax(), "model_id"]
    best_overall = table.loc[table["validation_avg_loglik"].idxmax(), "model_id"]
    tests = []
    for m in table["model_id"]:
        d = scores[best] - scores[m]
        se = rd.newey_west_se(d.to_numpy(), r["nw_lags"])
        t = float(d.mean() / se) if se and se > 0 else (0.0 if m == best else np.inf)
        tests.append({"model_id": m, "mean_daily_shortfall_vs_best": float(d.mean()), "nw_se": se, "nw_t_vs_best": t})
    tests = pd.DataFrame(tests)
    table = table.merge(tests, on="model_id")
    table["competitive"] = table["eligible"] & (table["nw_t_vs_best"] < r["nw_t_threshold"])
    comp_set = table[table["competitive"]]
    k_sel = int(comp_set["k"].min())
    table["allowed"] = table["competitive"]
    phase, fallback = None, False
    smaller = ok[ok["k"] < k_sel]
    if not smaller.empty:          # a larger K wins only if its advantage holds across Validation years
        at_k0 = comp_set[comp_set["k"] == k_sel]
        larger_id = at_k0.loc[at_k0["validation_avg_loglik"].idxmax(), "model_id"]
        smaller_id = smaller.loc[smaller["validation_avg_loglik"].idxmax(), "model_id"]
        phase = phase_consistency(scores, larger_id, smaller_id)
        if not phase["consistent"]:
            fallback = True
            k_sel = int(table.loc[table["model_id"] == smaller_id, "k"].iloc[0])
            table["allowed"] = table["eligible"] & (table["k"] == k_sel)
    allowed = table[table["allowed"]]
    at_k = allowed[allowed["k"] == k_sel]
    cov_sel = "diag" if (at_k["covariance_type"] == "diag").any() else "full"
    group = table[(table["k"] == k_sel) & (table["covariance_type"] == cov_sel)]
    std = group[group["family"] == "standard"]
    det = detection[(detection["reference"] == "factor_stress_score") & (detection["split"] == "pooled")]
    det = det.set_index("model_id")
    sticky_rows = []
    std_ok = not std.empty and bool(std["allowed"].iloc[0])
    std_id = std["model_id"].iloc[0] if not std.empty else None
    for row in group[group["family"] == "sticky"].itertuples():
        info = {"model_id": row.model_id, "competitive": bool(row.allowed)}
        if std_id is not None:
            base_sw = float(std["validation_switches_per_year"].iloc[0])
            info["switch_reduction"] = (1 - row.validation_switches_per_year / base_sw) if base_sw > 0 else np.nan
            info["median_delay_increase"] = float(det.loc[row.model_id, "median_lag_p50"]
                                                  - det.loc[std_id, "median_lag_p50"]) \
                if row.model_id in det.index and std_id in det.index else np.nan
            info["undetected_increase"] = float(det.loc[row.model_id, "undetected_share_p50"]
                                                - det.loc[std_id, "undetected_share_p50"]) \
                if row.model_id in det.index and std_id in det.index else np.nan
        else:
            info.update({"switch_reduction": np.nan, "median_delay_increase": np.nan, "undetected_increase": np.nan})
        pr = (paired[(paired["model_id"] == row.model_id) & (paired["split"] == "pooled")]
              if paired is not None and not paired.empty else pd.DataFrame())
        info["paired_median_entry_lag"] = float(pr["median_entry_lag"].iloc[0]) if len(pr) else np.nan
        info["paired_missed_share"] = float(pr["missed_share"].iloc[0]) if len(pr) else np.nan
        info["paired_median_exit_lag"] = float(pr["median_exit_lag"].iloc[0]) if len(pr) else np.nan
        delay_ok = (pd.notna(info["median_delay_increase"])
                    and info["median_delay_increase"] <= r["max_median_delay_increase_days"]
                    and (pd.isna(info["undetected_increase"])
                         or info["undetected_increase"] <= r["max_undetected_share_increase"])
                    and (pd.isna(info["paired_median_entry_lag"])
                         or info["paired_median_entry_lag"] <= r["max_median_delay_increase_days"])
                    and (pd.isna(info["paired_missed_share"])
                         or info["paired_missed_share"] <= r["max_undetected_share_increase"]))
        info["delay_ok"] = bool(delay_ok)
        info["qualifies"] = bool(row.allowed and delay_ok and pd.notna(info["switch_reduction"])
                                 and info["switch_reduction"] >= r["min_switch_reduction"])
        info["validation_avg_loglik"] = row.validation_avg_loglik
        sticky_rows.append(info)
    sticky = pd.DataFrame(sticky_rows)
    qual = sticky[sticky["qualifies"]] if not sticky.empty else sticky
    if not qual.empty:
        selected = qual.loc[qual["validation_avg_loglik"].idxmax(), "model_id"]
        why = ("Sticky model qualifies: competitive on Validation, >= 20% fewer Validation switches, detection-delay "
               "budget respected; highest Validation log score among qualifying Sticky models.")
    elif std_ok:
        selected = std_id
        detail = "; ".join(
            f"{x.model_id}: switch reduction " + ("n/a (Standard has no Validation switches)" if pd.isna(x.switch_reduction) else f"{x.switch_reduction:+.0%}") + f", competitive {x.competitive}, delay ok {x.delay_ok}"
            for x in sticky.itertuples()) if not sticky.empty else "no Sticky candidate at this K and covariance"
        why = ("Standard model kept: no Sticky model met all of competitiveness, a >= 20% switch reduction and the "
               f"detection-delay budget at this K and covariance ({detail}).")
    else:
        cand = group[group["allowed"]]
        selected = cand.loc[cand["validation_avg_loglik"].idxmax(), "model_id"]
        why = "Standard model not competitive at this K; best competitive model at this K and covariance."
    if fallback:
        why = (f"Phase-consistency fallback: {phase['larger']} beat {phase['smaller']} in only "
               f"{phase['positive_years']} of {phase['n_years']} Validation years (rule: >= "
               f"{PHASE_RULES['min_positive_years']}), so K={k_sel} is kept. " + why)
    elif phase is not None:
        why = (f"K={k_sel} chosen because smaller-K models are significantly worse on Validation; its advantage over "
               f"{phase['smaller']} holds in {phase['positive_years']} of {phase['n_years']} Validation years. " + why)
    out.update({"selected": selected, "best_validation_model": best, "best_overall_validation_model": best_overall, "selected_k": k_sel,
                "selected_covariance": cov_sel, "reason": why, "table": table, "sticky_assessment": sticky,
                "competitive_set": comp_set["model_id"].tolist(), "standard_counterpart": std_id,
                "phase_consistency": phase, "phase_fallback": fallback})
    return out


# --------------------------------------------------------------------------------------
# Stage 4 GMM reference (frozen; predictions only)
# --------------------------------------------------------------------------------------
def load_gmm_reference(models_dir: Path, scaler_dict: dict) -> tuple[object | None, dict | None, str]:
    path = models_dir / f"{GMM_REFERENCE_ID}.pkl"
    if not path.exists():
        return None, None, f"{path.name} not found (run scripts/06_explore_regimes.py first)"
    with open(path, "rb") as handle:
        obj = pickle.load(handle)
    meta = obj["metadata"]
    pre = meta.get("preprocessing", {})
    same = (pre.get("applied") == scaler_dict["applied"]
            and all(abs(pre["mean"][c] - scaler_dict["mean"][c]) < 1e-8 and
                    abs(pre["scale"][c] - scaler_dict["scale"][c]) < 1e-8 for c in FEATURE_COLUMNS))
    if not same:
        return None, meta, "Stage 4 GMM preprocessing differs from the Stage 5 Train scaler"
    return obj["model"], meta, "ok"


def gmm_outputs(model, meta: dict, x: np.ndarray) -> dict:
    order = np.asarray(meta["state_order_raw_ids"], dtype=int)      # canonical S1..SK -> raw ids
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    raw = model.predict(x)
    proba = model.predict_proba(x)[:, order]
    return {"score": model.score_samples(x), "labels": inverse[raw], "p_stress": proba[:, -1]}


def compare_with_gmm(c: Candidate, gmm_tr: dict, gmm_va: dict) -> pd.DataFrame:
    rows = []
    for split, res, g, labels in (("train", c.train, gmm_tr, c.train_labels),
                                  ("validation", c.validation, gmm_va, c.validation_labels)):
        d = res.log_score - g["score"]
        se = rd.newey_west_se(d, SELECTION_RULES["nw_lags"])
        ps = c.stress_probability(split)
        rows.append({"model_id": c.model_id, "split": split, "hmm_avg_log_score": float(res.log_score.mean()),
                     "gmm_avg_log_score": float(g["score"].mean()), "mean_daily_difference": float(d.mean()),
                     "nw_se": se, "nw_t": float(d.mean() / se) if se and se > 0 else np.nan,
                     "share_days_hmm_better": float((d > 0).mean()),
                     "ari_hmm_vs_gmm_states": float(adjusted_rand_score(g["labels"], labels)),
                     "corr_p_stress": float(np.corrcoef(ps, g["p_stress"])[0, 1]) if ps.std() > 0 else np.nan,
                     "hmm_switches_per_year": rd.switching_stats(labels)["switches_per_year"],
                     "gmm_switches_per_year": rd.switching_stats(g["labels"])["switches_per_year"],
                     "note": "in-sample for both models" if split == "train" else "out-of-sample, fixed Train parameters"})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Late-Train sensitivity (component composition stable)
# --------------------------------------------------------------------------------------
def composition_stable_from(components_path: Path, train_last: pd.Timestamp, train_first: pd.Timestamp
                            ) -> tuple[pd.Timestamp | None, str]:
    """Last date inside Train on which any factor's active component set changes. Rows after the last Train
    date are never parsed: the reader stops at the first such row."""
    from .factors import COMPONENTS

    if not components_path.exists():
        return None, f"{components_path.name} not found"
    cols = [f"{c.factor}__{c.name}" for c in COMPONENTS if c.in_composite]
    previous, last_change = None, None
    with open(components_path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        pos = [header.index(c) for c in cols if c in header]
        if len(pos) != len(cols):
            return None, "component columns not found"
        for line in reader:
            date = pd.Timestamp(line[0])
            if date > train_last:
                break
            if date < train_first:
                continue
            active = tuple(line[i] not in ("", "nan", "NaN") for i in pos)
            if previous is not None and active != previous:
                last_change = date
            previous = active
    if last_change is None:
        return train_first, "no composition change inside Train"
    return last_change, f"last active-component change inside Train on {last_change.date()}"


def late_train_sensitivity(spec_list: list[HMMSpec], full: dict[str, Candidate], x_train: np.ndarray,
                           x_validation: np.ndarray, train_index: pd.DatetimeIndex, start: pd.Timestamp, seeds,
                           mean, scale, train_original, n_init: int = HMM_N_INIT) -> pd.DataFrame:
    mask = train_index >= start
    rows = []
    for spec in spec_list:
        late = run_candidate(spec, x_train[mask], x_validation, seeds, mean, scale, train_original[mask],
                             n_init=n_init)
        base = full[spec.model_id]
        perm = align_by_means(base.reference.params, late.reference.params)
        lp = late.reference.params.permuted(perm)
        rows.append({"model_id": spec.model_id, "late_train_start": str(start.date()), "late_train_rows": int(mask.sum()),
                     "full_validation_avg_loglik": float(base.validation.log_score.mean()),
                     "late_validation_avg_loglik": float(late.validation.log_score.mean()),
                     "full_self_transitions": " / ".join(f"{a:.4f}" for a in np.diag(base.reference.params.transmat)),
                     "late_self_transitions": " / ".join(f"{a:.4f}" for a in np.diag(lp.transmat)),
                     "max_emission_mean_shift_original_units": float(np.abs(
                         (lp.means - base.reference.params.means) * scale).max()),
                     "validation_ari_full_vs_late": float(adjusted_rand_score(base.validation_labels,
                                                                              late.validation_labels)),
                     "full_validation_switches_per_year": rd.switching_stats(base.validation_labels)["switches_per_year"],
                     "late_validation_switches_per_year": rd.switching_stats(late.validation_labels)["switches_per_year"],
                     "late_converged": late.reference.converged})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Protocol checks on the real run
# --------------------------------------------------------------------------------------
def protocol_checks(cands: dict[str, Candidate], selected: str | None, x_train: np.ndarray, x_validation: np.ndarray,
                    files_read: list[str]) -> list[Check]:
    from .hmm_filtering import forward_filter

    checks = []
    test_read = [f for f in files_read if f.startswith("test_")]
    checks.append(Check("No test file read", FAIL if test_read else PASS, ", ".join(files_read)))
    worst_sum = max(max(np.abs(c.train.filtered.sum(1) - 1).max(), np.abs(c.validation.filtered.sum(1) - 1).max(),
                        np.abs(c.validation.predicted.sum(1) - 1).max()) for c in cands.values())
    checks.append(Check("Filtered and predicted probabilities sum to one", PASS if worst_sum < 1e-9 else FAIL,
                        f"max |sum - 1| = {worst_sum:.2e}"))
    worst_row = max(np.abs(c.reference.params.transmat.sum(1) - 1).max() for c in cands.values())
    checks.append(Check("Transition-matrix rows sum to one", PASS if worst_row < 1e-12 else FAIL,
                        f"max |row sum - 1| = {worst_row:.2e}"))
    worst_ll = max(abs(c.train.log_score.sum() - c.reference.train_loglik) / max(abs(c.reference.train_loglik), 1)
                   for c in cands.values())
    checks.append(Check("Filter log scores reproduce the EM Train log-likelihood", PASS if worst_ll < 1e-8 else FAIL,
                        f"max relative difference {worst_ll:.2e}"))
    not_conv = [m for m, c in cands.items() if not c.reference.converged]
    checks.append(Check("Reference fits converged", WARN if not_conv else PASS,
                        ", ".join(not_conv) or f"all {len(cands)} converged"))
    non_mono = [m for m, c in cands.items() if not all(f.monotone for f in c.seed_fits)]
    checks.append(Check("EM objective monotone for every seed", WARN if non_mono else PASS,
                        ", ".join(non_mono) or "all seeds monotone"))
    if selected is None:
        checks.append(Check("Causality and train-only checks on the selected model", NOT_RUN, "no model selected"))
        return checks
    c = cands[selected]
    p = c.reference.params
    # continuation equals filtering the concatenated sequence
    joint = forward_filter(p, np.vstack([x_train, x_validation]))
    diff = np.abs(joint.filtered[len(x_train):] - c.validation.filtered).max()
    checks.append(Check("Validation filter = continuation of the Train filter", PASS if diff < 1e-10 else FAIL,
                        f"max abs difference {diff:.2e}"))
    # future perturbation leaves earlier filtered probabilities unchanged
    cut = len(x_validation) // 2
    shocked = x_validation.copy()
    shocked[cut:] = shocked[cut:] * -5 + 10
    from .hmm_filtering import filter_train_validation as ftv
    _, va2 = ftv(p, x_train, shocked)
    d_before = np.abs(va2.filtered[:cut] - c.validation.filtered[:cut]).max()
    d_after = np.abs(va2.filtered[cut:] - c.validation.filtered[cut:]).max()
    checks.append(Check("Changing later Validation data leaves earlier filtered probabilities unchanged",
                        PASS if d_before == 0 and d_after > 0 else FAIL,
                        f"selected {selected}: max change before cut {d_before:.1e}, after cut {d_after:.2f}"))
    # train-only estimation: refit the reference seed; Validation is not an input of fit_hmm
    refit = fit_hmm_best(x_train, c.spec, c.reference.seed, len(c.reference.start_objectives) or HMM_N_INIT)
    refit.params = refit.params.permuted(align_by_means(p, refit.params))
    pdiff = max(np.abs(refit.params.means - p.means).max(), np.abs(refit.params.transmat - p.transmat).max())
    checks.append(Check("Parameters depend on Train only (refit reproduces the estimate)",
                        PASS if pdiff < 1e-9 else WARN, f"max parameter difference {pdiff:.2e}"))
    return checks


# --------------------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------------------
def model_metadata(c: Candidate, data_manifest: dict, scaler_dict: dict, train_index, input_hashes: dict,
                   version: str) -> dict:
    ref = c.reference
    return {
        "model_id": c.model_id, "model_family": c.spec.family, "n_states": c.spec.k,
        "covariance_type": c.spec.covariance_type,
        "persistence_prior": {"type": "none (maximum likelihood)" if c.spec.family == "standard" else
                              "Dirichlet on each transition row, alpha_ij = 1 + kappa * 1[i=j] (MAP-EM)",
                              "prior_weight_w": c.spec.prior_weight, "kappa": ref.kappa,
                              "kappa_definition": "w * (T_train - 1) / K"},
        "random_seed": ref.seed, "seeds_evaluated": [f.seed for f in c.seed_fits],
        "n_init_per_seed": len(ref.start_objectives), "estimator": "best Train objective of n_init EM starts",
        "reg_covar": HMM_REG_COVAR, "max_iter": HMM_MAX_ITER, "tol_per_obs": HMM_TOL,
        "converged": ref.converged, "n_iter": ref.n_iter, "monotone": ref.monotone,
        "train_loglik": ref.train_loglik, "train_map_objective": ref.objective,
        "train_first_date": str(train_index.min().date()), "train_last_date": str(train_index.max().date()),
        "train_rows": len(train_index), "factor_columns": FEATURE_COLUMNS, "preprocessing": scaler_dict,
        "state_order_rule": "ascending stress score (volatility + liquidity_credit - equity_trend) of emission means "
                            "in original factor units; S1..SK are neutral identifiers",
        "stress_states": [f"S{j + 1}" for j in c.stress_states],
        "input_sha256": input_hashes, "stage3_protocol": data_manifest.get("protocol_version"),
        "model_version": version, "parameters": c.reference.params.to_dict(),
    }


def save_json(obj, path: Path) -> None:
    def default(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return None if math.isnan(o) else float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, pd.DataFrame):
            return o.to_dict(orient="records")
        if isinstance(o, (pd.Timestamp,)):
            return str(o.date())
        return str(o)
    path.write_text(json.dumps(obj, indent=2, default=default), encoding="utf-8")
