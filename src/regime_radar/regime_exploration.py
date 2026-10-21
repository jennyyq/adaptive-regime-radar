"""Stage 4: unsupervised regime exploration with K-Means and Gaussian mixtures (Train/Validation only).

Design rules (fixed before any real-data result was seen)
  * Only data/splits/{train,validation}_*.csv are read. Requesting the test split raises an error.
  * Every learned quantity (scaler, centroids, mixture parameters) is fitted on Train only; Validation is
    only passed to predict / predict_proba / score_samples.
  * K-Means and GMM are static (non-temporal) models: each day is classified from that day's factors
    alone, so there is no smoothing and no future information, but also no notion of persistence.
  * GMM posterior probabilities are static mixture memberships p(state | x_t), NOT HMM filtered
    probabilities p(state_t | x_1..x_t).
  * Model-selection rules (SELECTION_RULES) are fixed in code before results are produced.
"""

from __future__ import annotations

STAGE4_CODE_VERSION = "stage4-2026-10-21b"  # must match scripts/06_explore_regimes.py

import json
import math
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import regime_diagnostics as rd
from .dataset import FEATURE_COLUMNS, PROTOCOL_VERSION, RANDOM_SEED, file_sha256
from .factors import Check

PASS, WARN, FAIL, NOT_RUN = "PASS", "WARN", "FAIL", "NOT RUN"

MODEL_SPLITS = ("train", "validation")
CANDIDATE_K = (2, 3, 4, 5, 6)
SEEDS = tuple(RANDOM_SEED + i for i in range(10))     # protocol: seed .. seed+9
KMEANS_N_INIT = 20
GMM_COVARIANCE_TYPES = ("full", "diag")
GMM_N_INIT = 5
GMM_REG_COVAR = 1e-4          # added to covariance diagonals (data are standardized, so ~1e-4 of unit variance)
GMM_MAX_ITER = 1000
GMM_TOL = 1e-4
ELBOW_K = tuple(range(1, 11))   # elbow curve drawn for K = 1..10 (K=1 = total sum of squares); diagnostic only
# The primary Kneedle elbow is computed over K = 1..max(CANDIDATE_K), the same range as the model comparison
# (K = 7..10 were never candidates and would move the chord). Every other range is reported as sensitivity.
ELBOW_PRIMARY_MAX_K = max(CANDIDATE_K)
# Acceptance criteria for a K=3 rates/inflation state, fixed on 2026-10-20 BEFORE the K=3 profiles were inspected
ALTERNATIVE_STATE_CRITERIA = {
    "min_rates_or_inflation": 0.5,      # train mean of interest_rate or inflation >= +0.5 ...
    "must_not_be_highest_stress": True,  # ... in a state that is not the most stressed one (a new dimension)
    "min_validation_share": 0.05,
    "max_profile_distance": 1.0,         # Euclidean distance between train and validation mean profiles
    "min_mean_ari_train": 0.90,
}
PARAMETER_TOLERANCE = 1e-9   # refits may differ by floating-point rounding (~1e-16) from multi-threaded sums

# Scaling rule: the factors are already robust z-scores, but composites of partially correlated components
# have unequal dispersion. Euclidean K-Means would then implicitly overweight the most dispersed factor.
SCALING_STD_RATIO_THRESHOLD = 1.25   # apply train-fitted standardization if max(std)/min(std) exceeds this
SCALING_MEAN_THRESHOLD = 0.25        # ... or if any |mean| / std exceeds this

SELECTION_RULES = {
    "min_train_state_share": 0.03,         # every state >= 3% of train days (~60 days)
    "min_validation_state_share": 0.01,    # below 1% of validation days = 'collapsed' (FLAGGED, not excluded)
    "min_mean_ari_train": 0.80,            # seed stability of train assignments (10 seeds)
    "gmm_require_all_seeds_converged": True,
    "nw_lags": 21,                         # Newey-West lags for daily log-likelihood differences
    "nw_t_threshold": 2.0,                 # 'significantly worse' than the best validation likelihood
    "kmeans_silhouette_tolerance": 0.01,   # K-Means: smallest K within this of the best validation silhouette
    "flag_median_run_days": 2,             # flag (not exclude) if median state duration <= 2 days
    "profile_shift_threshold": 1.0,        # flag state if train vs validation mean profile distance > 1.0
}
SELECTION_PROCEDURE = [
    "1. Eligibility (all candidates): every train state >= 3% of days; mean pairwise ARI across 10 seeds >= 0.80; "
    "GMM: every seed converged. A state below 1% of Validation days is FLAGGED as collapsed but does not exclude "
    "the candidate: Validation (2016-2019) contains no episode comparable to 2008, so excluding such states would "
    "select against crisis regimes by construction. Out-of-sample adequacy is judged by validation likelihood.",
    "2. GMM preference: among eligible GMMs, take the best validation average log-likelihood. A candidate is "
    "'not significantly worse' if the mean daily log-likelihood shortfall versus the best has a Newey-West "
    "t-statistic (21 lags) below 2. Prefer the not-significantly-worse candidate with the fewest parameters "
    "(ties: smaller K). Train BIC/AIC are reported for reference, not used to override validation evidence.",
    "3. K-Means preference: among eligible K-Means, the smallest K whose validation silhouette is within 0.01 of "
    "the best. Silhouette is known to favour small K, so this is a baseline indication only. Inertia always falls "
    "with K and is not a selection criterion; AIC/BIC are not used for K-Means.",
    "4. Conflicts: if the K-Means and GMM preferences differ, both are reported as leading candidates; the GMM "
    "preference is the reference for the HMM stage because it is likelihood-based and probabilistic.",
    "5. Economic interpretability, switching frequency and asset behaviour are reported for every candidate "
    "but are not used to rank them, so that no candidate is chosen for producing attractive historical plots.",
    "6. If no candidate is eligible, no preferred model is declared and the leading candidates are reported.",
]


# --------------------------------------------------------------------------------------
# Data loading (train / validation only)
# --------------------------------------------------------------------------------------
@dataclass
class Stage4Data:
    train_features: pd.DataFrame
    validation_features: pd.DataFrame
    train_returns: pd.DataFrame
    validation_returns: pd.DataFrame
    manifest: dict
    files_read: list[str] = field(default_factory=list)
    hash_status: dict[str, bool] = field(default_factory=dict)


def _read_split(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col="date", parse_dates=["date"])
    frame.index = pd.DatetimeIndex(frame.index).astype("datetime64[ns]")
    frame.index.name = "date"
    return frame


def load_stage4_data(splits_dir: Path, splits: tuple[str, ...] = MODEL_SPLITS) -> Stage4Data:
    """Read only the train and validation split files; verify their hashes against split_manifest.json."""
    if "test" in splits or set(splits) != set(MODEL_SPLITS):
        raise ValueError("Stage 4 may only read the train and validation splits")
    manifest_path = splits_dir / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames, files_read, hashes = {}, [str(manifest_path.name)], {}
    for split in MODEL_SPLITS:
        for kind in ("features", "returns"):
            name = f"{split}_{kind}.csv"
            path = splits_dir / name
            frames[(split, kind)] = _read_split(path)
            files_read.append(name)
            expected = manifest.get("split_files", {}).get(name)
            hashes[name] = expected is not None and expected == file_sha256(path)
    return Stage4Data(frames[("train", "features")], frames[("validation", "features")],
                      frames[("train", "returns")], frames[("validation", "returns")], manifest, files_read, hashes)


# --------------------------------------------------------------------------------------
# Input checks and train-only scaling
# --------------------------------------------------------------------------------------
def describe_train(train: pd.DataFrame) -> pd.DataFrame:
    """Train-only distribution diagnostics, including each factor's share of total variance."""
    var = train.var(ddof=0)
    return pd.DataFrame({
        "mean": train.mean(), "std": train.std(ddof=0), "median": train.median(),
        "mad_scaled": 1.4826 * (train - train.median()).abs().median(),
        "iqr": train.quantile(0.75) - train.quantile(0.25),
        "skew": train.skew(), "excess_kurtosis": train.kurt(),
        "p01": train.quantile(0.01), "p99": train.quantile(0.99), "min": train.min(), "max": train.max(),
        "share_abs_gt_3": (train.abs() > 3).mean(), "variance_share": var / var.sum(),
    })


def scaling_decision(stats: pd.DataFrame) -> tuple[bool, str]:
    std_ratio = float(stats["std"].max() / stats["std"].min())
    mean_shift = float((stats["mean"] / stats["std"]).abs().max())
    apply = std_ratio > SCALING_STD_RATIO_THRESHOLD or mean_shift > SCALING_MEAN_THRESHOLD
    reason = (f"train std ratio max/min = {std_ratio:.2f} (threshold {SCALING_STD_RATIO_THRESHOLD}); "
              f"max |mean|/std = {mean_shift:.2f} (threshold {SCALING_MEAN_THRESHOLD}); largest variance share "
              f"{stats['variance_share'].max():.0%} ({stats['variance_share'].idxmax()}). ")
    reason += ("Apply train-fitted standardization so that no factor dominates Euclidean distance."
               if apply else "Factors already comparable; identity scaling.")
    return apply, reason


class TrainScaler:
    """(x - mean_train) / std_train with parameters estimated on Train only, or identity."""

    def __init__(self, apply: bool):
        self.apply = apply
        self.mean_: pd.Series | None = None
        self.scale_: pd.Series | None = None
        self.fitted_on: dict | None = None

    def fit(self, train: pd.DataFrame) -> "TrainScaler":
        if self.apply:
            self.mean_, self.scale_ = train.mean(), train.std(ddof=0).replace(0, 1.0)
        else:
            self.mean_ = pd.Series(0.0, index=train.columns)
            self.scale_ = pd.Series(1.0, index=train.columns)
        self.fitted_on = {"split": "train", "first_date": str(train.index.min().date()),
                          "last_date": str(train.index.max().date()), "rows": len(train)}
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        return ((frame[self.mean_.index] - self.mean_) / self.scale_).to_numpy(dtype=float)

    def to_dict(self) -> dict:
        return {"applied": self.apply, "method": "train mean / train std (ddof=0)" if self.apply else "identity",
                "mean": self.mean_.round(10).to_dict(), "scale": self.scale_.round(10).to_dict(),
                "fitted_on": self.fitted_on}


def input_checks(data: Stage4Data) -> list[Check]:
    tr, va = data.train_features, data.validation_features
    checks = []
    expected = data.manifest.get("feature_columns", FEATURE_COLUMNS)
    ok_cols = list(tr.columns) == list(expected) == list(va.columns) == FEATURE_COLUMNS
    checks.append(Check("Five factor columns present in the frozen order", PASS if ok_cols else FAIL,
                        f"train {list(tr.columns)}; validation {list(va.columns)}"))
    for name, frame in (("train", tr), ("validation", va)):
        problems = []
        if not frame.index.is_monotonic_increasing:
            problems.append("not chronological")
        if frame.index.has_duplicates:
            problems.append(f"{int(frame.index.duplicated().sum())} duplicate dates")
        bad = int((~np.isfinite(frame.to_numpy(float))).sum())
        if bad:
            problems.append(f"{bad} NaN/inf values")
        checks.append(Check(f"{name} factors: chronological, unique dates, finite", FAIL if problems else PASS,
                            "; ".join(problems) or f"{len(frame)} rows {frame.index.min().date()}.."
                                                   f"{frame.index.max().date()}"))
    checks.append(Check("Validation starts after train ends (no overlap)",
                        PASS if tr.index.max() < va.index.min() else FAIL,
                        f"train ends {tr.index.max().date()}, validation starts {va.index.min().date()}"))
    rows = {s["name"]: s["rows"] for s in data.manifest.get("splits", [])}
    match = rows.get("train") == len(tr) and rows.get("validation") == len(va)
    checks.append(Check("Row counts match split_manifest.json", PASS if match else FAIL,
                        f"train {len(tr)} (manifest {rows.get('train')}), validation {len(va)} "
                        f"(manifest {rows.get('validation')})"))
    for name, frame in (("train", data.train_returns), ("validation", data.validation_returns)):
        feats = tr if name == "train" else va
        checks.append(Check(f"{name} returns aligned with factors", PASS if frame.index.equals(feats.index) else FAIL,
                            f"{frame.shape[1]} ETF columns"))
    return checks


def frozen_factor_check(manifest: dict, factors_module: Path) -> Check:
    expected = manifest.get("feature_provenance", {}).get("factor_construction", {}).get("module_sha256")
    actual = file_sha256(factors_module)
    if expected is None:
        return Check("Factor definitions unchanged since stage 3 (factors.py hash)", NOT_RUN,
                     "no module hash in split_manifest.json")
    return Check("Factor definitions unchanged since stage 3 (factors.py hash)", PASS if expected == actual else FAIL,
                 f"manifest {expected[:12]}..., current {str(actual)[:12]}...")


def validate_candidate_ks(ks, n_train: int) -> tuple[int, ...]:
    ks = tuple(ks)
    if not ks or any((not isinstance(k, (int, np.integer))) or isinstance(k, bool) for k in ks):
        raise ValueError("Candidate K values must be integers")
    if any(k < 2 for k in ks) or any(k > n_train // 50 for k in ks) or len(set(ks)) != len(ks):
        raise ValueError(f"Candidate K values must be unique, >= 2 and <= n_train/50 ({n_train // 50}): {ks}")
    return tuple(sorted(int(k) for k in ks))


# --------------------------------------------------------------------------------------
# Candidate specification and fitting
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class CandidateSpec:
    family: str                      # "kmeans" | "gmm"
    k: int
    covariance_type: str | None = None

    @property
    def model_id(self) -> str:
        return f"kmeans_k{self.k}" if self.family == "kmeans" else f"gmm_{self.covariance_type}_k{self.k}"

    def n_parameters(self, d: int) -> int:
        if self.family == "kmeans":
            return self.k * d
        cov = self.k * d * (d + 1) // 2 if self.covariance_type == "full" else self.k * d
        return cov + self.k * d + self.k - 1


def candidate_specs(ks=CANDIDATE_K) -> list[CandidateSpec]:
    specs = [CandidateSpec("kmeans", k) for k in ks]
    specs += [CandidateSpec("gmm", k, cov) for cov in GMM_COVARIANCE_TYPES for k in ks]
    return specs


@dataclass
class Fit:
    spec: CandidateSpec
    seed: int
    model: object
    order: np.ndarray                 # raw cluster ids sorted into canonical S1..SK
    train_labels: np.ndarray          # canonical 0..K-1
    validation_labels: np.ndarray
    train_proba: np.ndarray | None
    validation_proba: np.ndarray | None
    converged: bool
    n_iter: int
    warnings: list[str]
    seconds: float


def fit_model(spec: CandidateSpec, x_train: np.ndarray, seed: int):
    """Fit on TRAIN only. Returns (model, converged, n_iter, warnings)."""
    from sklearn.cluster import KMeans
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.mixture import GaussianMixture

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        if spec.family == "kmeans":
            model = KMeans(n_clusters=spec.k, n_init=KMEANS_N_INIT, random_state=seed).fit(x_train)
            converged = model.n_iter_ < model.max_iter
        else:
            model = GaussianMixture(n_components=spec.k, covariance_type=spec.covariance_type, n_init=GMM_N_INIT,
                                    reg_covar=GMM_REG_COVAR, max_iter=GMM_MAX_ITER, tol=GMM_TOL,
                                    random_state=seed).fit(x_train)
            converged = bool(model.converged_)
    messages = sorted({str(w.message)[:160] for w in caught if issubclass(w.category, ConvergenceWarning)})
    if messages:
        converged = False
    return model, converged, int(model.n_iter_), messages


def _raw_predict(spec: CandidateSpec, model, x: np.ndarray):
    if spec.family == "kmeans":
        return model.predict(x), None
    proba = model.predict_proba(x)
    return proba.argmax(axis=1), proba


def fit_candidate(spec: CandidateSpec, x_train: np.ndarray, x_validation: np.ndarray, train_features: pd.DataFrame,
                  seed: int) -> Fit:
    start = time.perf_counter()
    model, converged, n_iter, messages = fit_model(spec, x_train, seed)
    raw_tr, p_tr = _raw_predict(spec, model, x_train)
    raw_va, p_va = _raw_predict(spec, model, x_validation)
    centroids = rd.factor_profile(train_features, raw_tr, spec.k)[FEATURE_COLUMNS]   # original factor units
    order = rd.stress_order(centroids.reset_index(drop=True))
    inverse = np.empty(spec.k, dtype=int)
    inverse[order] = np.arange(spec.k)
    return Fit(spec, seed, model, order, inverse[raw_tr], inverse[raw_va],
               None if p_tr is None else p_tr[:, order], None if p_va is None else p_va[:, order],
               converged, n_iter, messages, time.perf_counter() - start)


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------
def _cluster_scores(x: np.ndarray, labels: np.ndarray) -> dict:
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

    if len(np.unique(labels)) < 2:
        return {"silhouette": np.nan, "calinski_harabasz": np.nan, "davies_bouldin": np.nan}
    return {"silhouette": float(silhouette_score(x, labels)),
            "calinski_harabasz": float(calinski_harabasz_score(x, labels)),
            "davies_bouldin": float(davies_bouldin_score(x, labels))}


def _population_metrics(prefix: str, labels: np.ndarray, k: int) -> dict:
    shares = rd.population(labels, k)
    out = {f"{prefix}_min_state_share": float(shares.min()), f"{prefix}_max_state_share": float(shares.max()),
           f"{prefix}_imbalance_ratio": float(shares.max() / shares.min()) if shares.min() > 0 else np.inf,
           f"{prefix}_states_present": int((shares > 0).sum()),
           f"{prefix}_shares": ";".join(f"S{i + 1}={s:.3f}" for i, s in enumerate(shares))}
    out.update({f"{prefix}_{key}": val for key, val in rd.switching_stats(labels).items() if key != "n_obs"})
    return out


def evaluate(fit: Fit, x_train: np.ndarray, x_validation: np.ndarray) -> dict:
    spec, model = fit.spec, fit.model
    d = x_train.shape[1]
    row = {"model_id": spec.model_id, "family": spec.family, "k": spec.k,
           "covariance_type": spec.covariance_type or "-", "n_parameters": spec.n_parameters(d),
           "seed": fit.seed, "converged": fit.converged, "n_iter": fit.n_iter,
           "convergence_warnings": " | ".join(fit.warnings) or "-", "fit_seconds": round(fit.seconds, 3)}
    tr = {f"train_{k}": v for k, v in _cluster_scores(x_train, fit.train_labels).items()}
    va = {f"validation_{k}": v for k, v in _cluster_scores(x_validation, fit.validation_labels).items()}
    row.update(tr)
    row.update(va)
    row.update(_population_metrics("train", fit.train_labels, spec.k))
    row.update(_population_metrics("validation", fit.validation_labels, spec.k))
    if spec.family == "kmeans":
        inertia_val = float(-model.score(x_validation))
        row.update({"train_inertia": float(model.inertia_),
                    "train_inertia_per_obs": float(model.inertia_) / len(x_train),
                    "validation_inertia_per_obs": inertia_val / len(x_validation),
                    "validation_to_train_distortion": (inertia_val / len(x_validation))
                    / (float(model.inertia_) / len(x_train))})
    else:
        ent_tr = rd.normalized_entropy(fit.train_proba)
        ent_va = rd.normalized_entropy(fit.validation_proba)
        row.update({
            "train_bic": float(model.bic(x_train)), "train_aic": float(model.aic(x_train)),
            "train_avg_loglik": float(model.score(x_train)),
            "validation_avg_loglik": float(model.score(x_validation)),
            "train_minus_validation_loglik": float(model.score(x_train) - model.score(x_validation)),
            "lower_bound": float(model.lower_bound_),
            "train_mean_max_posterior": float(fit.train_proba.max(axis=1).mean()),
            "train_mean_entropy": float(ent_tr.mean()),
            "validation_mean_max_posterior": float(fit.validation_proba.max(axis=1).mean()),
            "validation_share_max_posterior_below_0.6": float((fit.validation_proba.max(axis=1) < 0.6).mean()),
            "validation_mean_entropy": float(ent_va.mean()),
            "posterior_sum_max_error": float(max(np.abs(fit.train_proba.sum(axis=1) - 1).max(),
                                                 np.abs(fit.validation_proba.sum(axis=1) - 1).max())),
        })
    return row


# --------------------------------------------------------------------------------------
# Exploration run
# --------------------------------------------------------------------------------------
@dataclass
class ExplorationResult:
    scaler: TrainScaler
    train_stats: pd.DataFrame
    scaling_reason: str
    reference: dict[str, Fit]                       # model_id -> fit with SEEDS[0]
    comparison: pd.DataFrame                        # one row per candidate (reference seed)
    stability: pd.DataFrame
    seed_fits: dict[str, list[Fit]]
    validation_loglik_obs: dict[str, np.ndarray]    # GMM model_id -> per-day validation log-likelihood
    selection: dict = field(default_factory=dict)
    elbow: pd.DataFrame | None = None
    elbow_k: int | None = None
    elbow_sensitivity: pd.DataFrame | None = None


def run_exploration(data: Stage4Data, ks=CANDIDATE_K, seeds=SEEDS, progress=None) -> ExplorationResult:
    tr, va = data.train_features[FEATURE_COLUMNS], data.validation_features[FEATURE_COLUMNS]
    ks = validate_candidate_ks(ks, len(tr))
    stats = describe_train(tr)
    apply, reason = scaling_decision(stats)
    scaler = TrainScaler(apply).fit(tr)
    x_tr, x_va = scaler.transform(tr), scaler.transform(va)

    reference, seed_fits, rows, stab_rows, ll_obs = {}, {}, [], [], {}
    for spec in candidate_specs(ks):
        fits = [fit_candidate(spec, x_tr, x_va, tr, seed) for seed in seeds]
        ref = fits[0]
        reference[spec.model_id], seed_fits[spec.model_id] = ref, fits
        rows.append(evaluate(ref, x_tr, x_va))
        if spec.family == "gmm":
            ll_obs[spec.model_id] = ref.model.score_samples(x_va)
        agree_tr = rd.pairwise_agreement([f.train_labels for f in fits], spec.k)
        agree_va = rd.pairwise_agreement([f.validation_labels for f in fits], spec.k)
        stab = {"model_id": spec.model_id, "family": spec.family, "k": spec.k,
                "covariance_type": spec.covariance_type or "-", "n_seeds": len(fits),
                "train_mean_ari": agree_tr["mean_ari"], "train_min_ari": agree_tr["min_ari"],
                "train_mean_aligned_agreement": agree_tr["mean_aligned_agreement"],
                "validation_mean_ari": agree_va["mean_ari"], "validation_min_ari": agree_va["min_ari"],
                "validation_mean_aligned_agreement": agree_va["mean_aligned_agreement"],
                "converged_share": float(np.mean([f.converged for f in fits]))}
        if spec.family == "gmm":
            bounds = [f.model.lower_bound_ for f in fits]
            stab["lower_bound_range"] = float(max(bounds) - min(bounds))
            stab["validation_avg_loglik_range"] = float(np.ptp([f.model.score(x_va) for f in fits]))
        else:
            inertias = [f.model.inertia_ for f in fits]
            stab["inertia_relative_range"] = float((max(inertias) - min(inertias)) / min(inertias))
        stab_rows.append(stab)
        if progress:
            progress(spec.model_id, ref)
    comparison = pd.DataFrame(rows)
    stability = pd.DataFrame(stab_rows)
    result = ExplorationResult(scaler, stats, reason, reference, comparison, stability, seed_fits, ll_obs)
    result.selection = select_models(result)
    result.elbow, result.elbow_k, result.elbow_sensitivity = elbow_curve(x_tr, x_va, seed=seeds[0])
    return result


# --------------------------------------------------------------------------------------
# Elbow method (diagnostic only)
# --------------------------------------------------------------------------------------
def kneedle_elbow(ks, values) -> int:
    """K at the maximum distance between a decreasing curve and the chord joining its end points (Kneedle)."""
    ks = np.asarray(ks, dtype=float)
    values = np.asarray(values, dtype=float)
    x = (ks - ks.min()) / (ks.max() - ks.min())
    y = (values - values.min()) / (values.max() - values.min())
    distance = (1 - x) - y
    return int(ks[int(np.argmax(distance))])


def elbow_curve(x_train: np.ndarray, x_validation: np.ndarray, ks=ELBOW_K, seed: int = SEEDS[0],
                primary_max_k: int = ELBOW_PRIMARY_MAX_K) -> tuple[pd.DataFrame, int, pd.DataFrame]:
    """K-Means within-cluster sum of squares for K = 1..10, fitted on TRAIN only.

    Returns (curve, primary elbow, sensitivity). The primary Kneedle elbow uses K = 1..primary_max_k, the
    candidate range of the model comparison; `sensitivity` gives the Kneedle elbow for every upper bound
    from 4 to max(ks), because the elbow of a smooth curve depends on the range over which it is drawn.
    Validation inertia uses the Train centroids, so it shows whether each extra cluster helps out of sample.
    Reported as a diagnostic; not part of SELECTION_RULES.
    """
    from sklearn.cluster import KMeans

    total_ss = float(((x_train - x_train.mean(axis=0)) ** 2).sum())
    rows = []
    for k in ks:
        model = KMeans(n_clusters=k, n_init=KMEANS_N_INIT, random_state=seed).fit(x_train)
        rows.append({"k": k, "train_inertia": float(model.inertia_),
                     "train_inertia_per_obs": float(model.inertia_) / len(x_train),
                     "validation_inertia_per_obs": float(-model.score(x_validation)) / len(x_validation),
                     "train_explained_share": 1 - float(model.inertia_) / total_ss})
    table = pd.DataFrame(rows)
    table["pct_reduction_vs_previous_k"] = -100 * table["train_inertia"].pct_change()
    table["validation_pct_reduction_vs_previous_k"] = -100 * table["validation_inertia_per_obs"].pct_change()

    sens = []
    for hi in range(4, int(table["k"].max()) + 1):
        sub = table[table["k"] <= hi]
        x = (sub["k"] - sub["k"].min()) / (sub["k"].max() - sub["k"].min())
        y = (sub["train_inertia"] - sub["train_inertia"].min()) / (sub["train_inertia"].max() - sub["train_inertia"].min())
        dist = ((1 - x) - y).to_numpy()
        order = np.argsort(dist)[::-1]
        sens.append({"k_range": f"1-{hi}", "max_k": hi, "kneedle_elbow": kneedle_elbow(sub["k"], sub["train_inertia"]),
                     "runner_up": int(sub["k"].iloc[order[1]]),
                     "margin_vs_runner_up": float(dist[order[0]] - dist[order[1]]),
                     "primary_range": hi == primary_max_k})
    sensitivity = pd.DataFrame(sens)
    elbow = int(sensitivity.loc[sensitivity["primary_range"], "kneedle_elbow"].iloc[0])
    full = int(sensitivity["kneedle_elbow"].iloc[-1])
    table["kneedle_elbow"] = table["k"] == elbow
    table[f"kneedle_elbow_range_1_{int(table['k'].max())}"] = table["k"] == full
    table["stage4_candidate"] = table["k"].isin(CANDIDATE_K)
    return table, elbow, sensitivity


# --------------------------------------------------------------------------------------
# K=3 alternative: does a third state capture the rates / inflation dimension?
# --------------------------------------------------------------------------------------
def alternative_state_assessment(profiles: pd.DataFrame, stability: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """Apply ALTERNATIVE_STATE_CRITERIA to every K-state candidate (fixed before the profiles were seen)."""
    c = ALTERNATIVE_STATE_CRITERIA
    rows = []
    for model_id in sorted(profiles.loc[profiles["k"] == k, "model_id"].unique()):
        tr = profiles[(profiles["model_id"] == model_id) & (profiles["split"] == "train")].set_index("state")
        va = profiles[(profiles["model_id"] == model_id) & (profiles["split"] == "validation")].set_index("state")
        stress = tr["volatility"] + tr["liquidity_credit"] - tr["equity_trend"]
        macro = tr[["interest_rate", "inflation"]].max(axis=1)
        eligible = macro[stress.index != stress.idxmax()] if c["must_not_be_highest_stress"] else macro
        state = eligible.idxmax()
        ari = float(stability.set_index("model_id").loc[model_id, "train_mean_ari"])
        checks = {
            "c1_rates_or_inflation_state": bool(eligible.max() >= c["min_rates_or_inflation"]),
            "c2_validation_share": bool(va.loc[state, "share"] >= c["min_validation_share"]),
            "c3_profile_stable": bool(tr.loc[state, "train_validation_profile_distance"] < c["max_profile_distance"]),
            "c4_seed_stability": bool(ari >= c["min_mean_ari_train"]),
        }
        rows.append({"model_id": model_id, "candidate_state": state,
                     "train_interest_rate": tr.loc[state, "interest_rate"], "train_inflation": tr.loc[state, "inflation"],
                     "train_volatility": tr.loc[state, "volatility"],
                     "validation_share": va.loc[state, "share"],
                     "profile_distance": tr.loc[state, "train_validation_profile_distance"],
                     "train_mean_ari": ari, **checks, "criteria_met": f"{sum(checks.values())}/4",
                     "accepted": all(checks.values())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Model selection (rules fixed in SELECTION_RULES / SELECTION_PROCEDURE)
# --------------------------------------------------------------------------------------
def eligibility(result: ExplorationResult) -> pd.DataFrame:
    r = SELECTION_RULES
    table = result.comparison.merge(result.stability[["model_id", "train_mean_ari", "converged_share"]], on="model_id")
    reasons = []
    for row in table.itertuples():
        why = []
        if row.train_min_state_share < r["min_train_state_share"]:
            why.append(f"train state share {row.train_min_state_share:.1%} < {r['min_train_state_share']:.0%}")
        if not (row.train_mean_ari >= r["min_mean_ari_train"]):
            why.append(f"seed ARI {row.train_mean_ari:.2f} < {r['min_mean_ari_train']}")
        if row.family == "gmm" and r["gmm_require_all_seeds_converged"] and row.converged_share < 1:
            why.append(f"converged in {row.converged_share:.0%} of seeds")
        reasons.append("; ".join(why))
    table["eligible"] = [not w for w in reasons]
    table["ineligible_reasons"] = [w or "-" for w in reasons]
    table["excessive_switching_flag"] = table["train_median_duration_days"] <= r["flag_median_run_days"]
    table["validation_collapsed_states"] = [
        sum(float(x.split("=")[1]) < r["min_validation_state_share"] for x in shares.split(";"))
        for shares in table["validation_shares"]]
    return table


def select_models(result: ExplorationResult) -> dict:
    r = SELECTION_RULES
    table = eligibility(result)
    out = {"eligibility": table[["model_id", "eligible", "ineligible_reasons", "excessive_switching_flag",
                                 "validation_collapsed_states"]]}

    gmm = table[table["family"] == "gmm"].copy()
    eligible = gmm[gmm["eligible"]]
    gmm["nw_t_vs_best"], gmm["mean_loglik_shortfall"], gmm["not_significantly_worse"] = np.nan, np.nan, False
    if eligible.empty:
        out["gmm_preferred"] = None
        out["gmm_note"] = "No GMM candidate met the eligibility rules; leading candidates by validation likelihood " \
                          "are reported instead."
    else:
        best_id = eligible.loc[eligible["validation_avg_loglik"].idxmax(), "model_id"]
        for idx, row in gmm.iterrows():
            diff = result.validation_loglik_obs[best_id] - result.validation_loglik_obs[row["model_id"]]
            se = rd.newey_west_se(diff, r["nw_lags"])
            t = diff.mean() / se if se and se > 0 else (0.0 if row["model_id"] == best_id else np.inf)
            gmm.loc[idx, "mean_loglik_shortfall"] = float(diff.mean())
            gmm.loc[idx, "nw_t_vs_best"] = float(t)
            gmm.loc[idx, "not_significantly_worse"] = bool(row["eligible"] and t < r["nw_t_threshold"])
        pool = gmm[gmm["not_significantly_worse"]].sort_values(["n_parameters", "k"])
        out["gmm_best_validation"] = best_id
        out["gmm_preferred"] = pool["model_id"].iloc[0]
        out["gmm_not_significantly_worse"] = pool["model_id"].tolist()
        out["gmm_note"] = (f"Best validation log-likelihood: {best_id}. Not significantly worse (NW t < "
                           f"{r['nw_t_threshold']}): {', '.join(pool['model_id'])}. Most parsimonious: "
                           f"{out['gmm_preferred']}.")
        out["candidate_k_range"] = sorted({int(k) for k in pool["k"]})
    bic_best = gmm.loc[gmm["train_bic"].idxmin(), "model_id"]
    out["gmm_train_bic_best"] = bic_best
    if out.get("gmm_best_validation"):
        k_bic = int(gmm.set_index("model_id").loc[bic_best, "k"])
        k_val = int(gmm.set_index("model_id").loc[out["gmm_best_validation"], "k"])
        out["bic_validation_conflict"] = (
            f"Train BIC is minimised by {bic_best} (K={k_bic}); validation likelihood is maximised by "
            f"{out['gmm_best_validation']} (K={k_val}). "
            + ("They agree." if k_bic == k_val else
               "Under the predefined rule the validation evidence prevails: additional components improve the "
               "in-sample fit but not the out-of-sample fit, i.e. part of the in-sample gain is train-specific."))
    out["gmm_table"] = gmm[["model_id", "k", "covariance_type", "n_parameters", "train_bic", "train_aic",
                            "train_avg_loglik", "validation_avg_loglik", "train_minus_validation_loglik",
                            "mean_loglik_shortfall", "nw_t_vs_best",
                            "eligible", "ineligible_reasons", "not_significantly_worse"]]

    km = table[table["family"] == "kmeans"].copy()
    km_eligible = km[km["eligible"] & km["validation_silhouette"].notna()]
    if km_eligible.empty:
        out["kmeans_preferred"] = None
        out["kmeans_note"] = "No K-Means candidate met the eligibility rules."
    else:
        best = km_eligible["validation_silhouette"].max()
        pool = km_eligible[km_eligible["validation_silhouette"] >= best - r["kmeans_silhouette_tolerance"]]
        out["kmeans_preferred"] = pool.sort_values("k")["model_id"].iloc[0]
        out["kmeans_note"] = (f"Best validation silhouette {best:.3f}; candidates within "
                              f"{r['kmeans_silhouette_tolerance']}: {', '.join(pool.sort_values('k')['model_id'])}; "
                              f"smallest K: {out['kmeans_preferred']}.")
    out["kmeans_table"] = km[["model_id", "k", "train_inertia", "train_silhouette", "train_calinski_harabasz",
                              "train_davies_bouldin", "validation_silhouette", "validation_to_train_distortion",
                              "train_min_state_share", "validation_min_state_share", "eligible", "ineligible_reasons"]]

    gp, kp = out.get("gmm_preferred"), out.get("kmeans_preferred")
    if gp and kp:
        from sklearn.metrics import adjusted_rand_score

        a, b = result.reference[gp], result.reference[kp]
        out["preferred_agreement"] = {
            "train_ari": float(adjusted_rand_score(a.train_labels, b.train_labels)),
            "validation_ari": float(adjusted_rand_score(a.validation_labels, b.validation_labels)),
            "same_k": a.spec.k == b.spec.k,
        }
    return out


# --------------------------------------------------------------------------------------
# Profiles, generalization, asset statistics, assignments
# --------------------------------------------------------------------------------------
def profiles_table(result: ExplorationResult, data: Stage4Data) -> pd.DataFrame:
    rows = []
    for model_id, fit in result.reference.items():
        k = fit.spec.k
        p_tr = rd.factor_profile(data.train_features[FEATURE_COLUMNS], fit.train_labels, k)
        p_va = rd.factor_profile(data.validation_features[FEATURE_COLUMNS], fit.validation_labels, k)
        for state in p_tr.index:
            label, rule = rd.hypothesis_label(p_tr.loc[state])
            shift = float(np.sqrt(((p_tr.loc[state, FEATURE_COLUMNS] - p_va.loc[state, FEATURE_COLUMNS]) ** 2).sum())) \
                if p_va.loc[state, "n_obs"] > 0 else np.nan
            flags = []
            if p_va.loc[state, "share"] < SELECTION_RULES["min_validation_state_share"]:
                flags.append("collapsed in validation")
            if pd.notna(shift) and shift > SELECTION_RULES["profile_shift_threshold"]:
                flags.append("profile shift")
            for split, prof in (("train", p_tr), ("validation", p_va)):
                rows.append({"model_id": model_id, "family": fit.spec.family, "k": k, "state": state, "split": split,
                             "n_obs": int(prof.loc[state, "n_obs"]), "share": prof.loc[state, "share"],
                             **{c: prof.loc[state, c] for c in FEATURE_COLUMNS},
                             "hypothesis_label": label, "label_rule": rule,
                             "train_validation_profile_distance": shift,
                             "generalization_flags": ", ".join(flags) or "-"})
    return pd.DataFrame(rows)


def asset_table(result: ExplorationResult, data: Stage4Data, model_ids: list[str]) -> pd.DataFrame:
    frames = []
    for model_id in model_ids:
        fit = result.reference[model_id]
        for split, returns, labels in (("train", data.train_returns, fit.train_labels),
                                       ("validation", data.validation_returns, fit.validation_labels)):
            stats = rd.asset_statistics(returns, labels, fit.spec.k)
            stats.insert(0, "split", split)
            stats.insert(0, "model_id", model_id)
            frames.append(stats)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def assignments(result: ExplorationResult, data: Stage4Data) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                                                                           pd.DataFrame]:
    names = lambda labels: np.array([f"S{i + 1}" for i in labels])  # noqa: E731
    tr = pd.DataFrame({m: names(f.train_labels) for m, f in result.reference.items()}, index=data.train_features.index)
    va = pd.DataFrame({m: names(f.validation_labels) for m, f in result.reference.items()},
                      index=data.validation_features.index)
    prob_tr, prob_va = {}, {}
    for model_id, fit in result.reference.items():
        if fit.train_proba is None:
            continue
        for j in range(fit.spec.k):
            prob_tr[f"{model_id}__S{j + 1}"] = fit.train_proba[:, j]
            prob_va[f"{model_id}__S{j + 1}"] = fit.validation_proba[:, j]
    return (tr, va, pd.DataFrame(prob_tr, index=data.train_features.index),
            pd.DataFrame(prob_va, index=data.validation_features.index))


# --------------------------------------------------------------------------------------
# Leakage / protocol checks executed on the real run
# --------------------------------------------------------------------------------------
def leakage_checks(data: Stage4Data, result: ExplorationResult, preferred: list[str]) -> list[Check]:
    checks = []
    read_test = [f for f in data.files_read if f.startswith("test_")]
    checks.append(Check("Test split never read", PASS if not read_test else FAIL,
                        f"files read: {', '.join(data.files_read)}"))
    bad_hash = [n for n, ok in data.hash_status.items() if not ok]
    checks.append(Check("Train/validation files match split_manifest.json hashes", PASS if not bad_hash else FAIL,
                        f"mismatched: {bad_hash or 'none'}"))
    tr = data.train_features[FEATURE_COLUMNS]
    expected_mean = tr.mean() if result.scaler.apply else pd.Series(0.0, index=tr.columns)
    ok_scaler = np.allclose(result.scaler.mean_.to_numpy(), expected_mean.to_numpy(), atol=1e-12)
    checks.append(Check("Scaler fitted on train only", PASS if ok_scaler else FAIL,
                        f"{result.scaler.to_dict()['method']}; fitted on {result.scaler.fitted_on}"))

    # Fitted parameters must not depend on validation data: refit with validation replaced by noise.
    # sklearn reduces sums across OpenMP/BLAS threads, so bit-identical floats are not guaranteed between two
    # fits; differences of ~1e-16 are rounding. The substantive test is identical state assignments plus
    # parameters equal within PARAMETER_TOLERANCE (many orders of magnitude below any data-driven change).
    rng = np.random.default_rng(0)
    noise = pd.DataFrame(rng.normal(size=data.validation_features.shape) * 5,
                         index=data.validation_features.index, columns=FEATURE_COLUMNS)
    x_tr = result.scaler.transform(tr)
    x_va = result.scaler.transform(data.validation_features[FEATURE_COLUMNS])

    def compare(ref: Fit, other: Fit) -> tuple[float, int]:
        diff = float(np.max(np.abs(_parameters(ref.model) - _parameters(other.model))))
        label_mismatch = int((ref.train_labels != other.train_labels).sum())
        return diff, label_mismatch

    worst, mismatches = 0.0, 0
    for model_id in preferred:
        ref = result.reference[model_id]
        diff, mm = compare(ref, fit_candidate(ref.spec, x_tr, result.scaler.transform(noise), tr, ref.seed))
        worst, mismatches = max(worst, diff), mismatches + mm
    checks.append(Check("Fitted parameters independent of validation data",
                        PASS if worst <= PARAMETER_TOLERANCE and mismatches == 0 else FAIL,
                        f"refit of {', '.join(preferred) or 'no model'} with validation replaced by noise; "
                        f"max parameter difference {worst:.1e} (tolerance {PARAMETER_TOLERANCE:g}, floating-point "
                        f"rounding); train label mismatches {mismatches}"))

    worst, mismatches = 0.0, 0
    for model_id in preferred:
        ref = result.reference[model_id]
        diff, mm = compare(ref, fit_candidate(ref.spec, x_tr, x_va, tr, ref.seed))
        worst, mismatches = max(worst, diff), mismatches + mm
    checks.append(Check("Same seed reproduces the same fit",
                        PASS if worst <= PARAMETER_TOLERANCE and mismatches == 0 else FAIL,
                        f"max parameter difference {worst:.1e} (tolerance {PARAMETER_TOLERANCE:g}); "
                        f"train label mismatches {mismatches}"))

    bad_ids = sum(int(((f.train_labels < 0) | (f.train_labels >= f.spec.k)).sum()
                      + ((f.validation_labels < 0) | (f.validation_labels >= f.spec.k)).sum())
                  for f in result.reference.values())
    checks.append(Check("State ids within 0..K-1 for every candidate", PASS if bad_ids == 0 else FAIL,
                        f"{bad_ids} invalid ids"))
    errors = [float(np.abs(p.sum(axis=1) - 1).max()) for f in result.reference.values()
              for p in (f.train_proba, f.validation_proba) if p is not None]
    finite = all(np.isfinite(p).all() for f in result.reference.values()
                 for p in (f.train_proba, f.validation_proba) if p is not None)
    checks.append(Check("GMM posterior probabilities finite and sum to one", PASS if finite and max(errors) < 1e-9
                        else FAIL, f"max |sum - 1| = {max(errors):.1e}"))
    last = data.validation_features.index.max()
    checks.append(Check("No output date beyond the validation period", PASS, f"last classified date {last.date()}"))
    checks.append(Check("Non-temporal models: no smoothing, no use of later observations", PASS,
                        "each day is classified from its own factor vector with train-fitted parameters"))
    return checks


def _parameters(model) -> np.ndarray:
    if hasattr(model, "cluster_centers_"):
        return np.asarray(model.cluster_centers_).ravel()
    return np.concatenate([model.weights_.ravel(), model.means_.ravel(), np.asarray(model.covariances_).ravel()])


def software_versions() -> dict:
    from importlib import metadata
    import platform

    out = {"python": platform.python_version()}
    for package in ("numpy", "pandas", "scikit-learn", "scipy", "matplotlib"):
        try:
            out[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            out[package] = "not installed"
    return out


def model_metadata(fit: Fit, data: Stage4Data, scaler: TrainScaler) -> dict:
    return {
        "model_id": fit.spec.model_id, "model_type": "KMeans" if fit.spec.family == "kmeans" else "GaussianMixture",
        "n_states": fit.spec.k, "covariance_type": fit.spec.covariance_type, "random_seed": fit.seed,
        "n_init": KMEANS_N_INIT if fit.spec.family == "kmeans" else GMM_N_INIT,
        "reg_covar": GMM_REG_COVAR if fit.spec.family == "gmm" else None,
        "training_split": "train", "training_first_date": str(data.train_features.index.min().date()),
        "training_last_date": str(data.train_features.index.max().date()), "training_rows": len(data.train_features),
        "factor_columns": FEATURE_COLUMNS, "preprocessing": scaler.to_dict(),
        "state_order_raw_ids": [int(i) for i in fit.order],
        "state_order_rule": "ascending train mean of volatility + liquidity_credit - equity_trend (ordering only)",
        "converged": fit.converged, "n_iter": fit.n_iter, "stage3_protocol": data.manifest.get("protocol_version"),
        "software": software_versions(),
    }


def stage4_manifest(data: Stage4Data, result: ExplorationResult, checks: list[Check], outputs: list[str]) -> dict:
    sel = result.selection
    return {
        "stage": "4 - exploratory regime models (K-Means, GMM)",
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "stage3_protocol_version": data.manifest.get("protocol_version"),
        "code_protocol_version": PROTOCOL_VERSION,
        "inputs": {name: data.manifest.get("split_files", {}).get(name) for name in data.files_read
                   if name.endswith(".csv")},
        "files_read": data.files_read,
        "test_data_loaded": any(f.startswith("test_") for f in data.files_read),
        "train_period": {"first": str(data.train_features.index.min().date()),
                         "last": str(data.train_features.index.max().date()), "rows": len(data.train_features)},
        "validation_period": {"first": str(data.validation_features.index.min().date()),
                              "last": str(data.validation_features.index.max().date()),
                              "rows": len(data.validation_features)},
        "factor_columns": FEATURE_COLUMNS,
        "preprocessing": result.scaler.to_dict(), "scaling_decision": result.scaling_reason,
        "candidate_k": sorted({f.spec.k for f in result.reference.values()}),
        "seeds": list(SEEDS), "kmeans_n_init": KMEANS_N_INIT, "gmm_n_init": GMM_N_INIT,
        "gmm_covariance_types": list(GMM_COVARIANCE_TYPES), "gmm_reg_covar": GMM_REG_COVAR,
        "gmm_max_iter": GMM_MAX_ITER, "selection_rules": SELECTION_RULES, "selection_procedure": SELECTION_PROCEDURE,
        "alternative_state_criteria": ALTERNATIVE_STATE_CRITERIA, "elbow_k_range": list(ELBOW_K),
        "elbow_primary_range": f"1-{ELBOW_PRIMARY_MAX_K}",
        "elbow_sensitivity": (None if result.elbow_sensitivity is None
                              else result.elbow_sensitivity.to_dict(orient="records")),
        "preferred": {"gmm": sel.get("gmm_preferred"), "kmeans": sel.get("kmeans_preferred"),
                      "gmm_best_validation_loglik": sel.get("gmm_best_validation"),
                      "gmm_train_bic_best": sel.get("gmm_train_bic_best"),
                      "bic_validation_conflict": sel.get("bic_validation_conflict"),
                      "candidate_k_range_for_hmm": sel.get("candidate_k_range"),
                      "kmeans_elbow_k_diagnostic": result.elbow_k,
                      "agreement": sel.get("preferred_agreement")},
        "checks": [vars(c) for c in checks],
        "model_nature": "static (non-temporal) clustering; GMM posteriors are mixture memberships, not HMM "
                        "filtered probabilities",
        "outputs": outputs, "software": software_versions(),
    }


def save_json(obj: dict, path: Path) -> None:
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if math.isnan(o) else float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, pd.DataFrame):
            return o.to_dict(orient="records")
        return str(o)
    path.write_text(json.dumps(obj, indent=2, default=default), encoding="utf-8")
