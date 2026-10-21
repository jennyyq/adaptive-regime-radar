"""Stage 5: robustness of the HMM results to the training window, the initialization and the market phase,
and a version-independent rebuild of the Stage 4 GMM reference.

Training windows (rules fixed before these results were seen; all inside Train, never Validation/Test):
    full              first..last Train date (the main fit)
    start_plus_2y     first Train date + 2 years .. last Train date (drops the first two years, incl. late 2007-09)
    post_composition  first date after the last factor-composition change .. last Train date
    end_minus_2y      first Train date .. last Train date - 2 years (2014-2015 are filtered, not fitted)
For every window the parameters are estimated on the window only; the causal filter then runs from the window
start through the end of Train and continues into Validation, so Validation is always scored the same way.

Market phases: the Validation daily log-score differences are split by calendar year and by whether the factor
stress score is above its Train median (high / low stress). Train years are shown too, flagged in-sample.

GMM reference: the Stage 4 model (GMM, diag, K=2, seed 20240531, n_init 5) is refitted on Train with the
installed scikit-learn instead of unpickling a file written by another version. The saved pickle is only
compared when its recorded scikit-learn version equals the installed one.
"""

from __future__ import annotations

STAGE5_CODE_VERSION = "stage5-2026-10-21c"  # must match scripts/07_fit_hmm.py

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from . import regime_diagnostics as rd
from .hmm_filtering import forward_filter
from .hmm_models import align_by_means

PHASE_RULES = {"min_positive_years": 3, "n_years": 4}   # a larger K must beat the smaller K in >= 3 of 4 years
WINDOW_SEEDS = 5


# --------------------------------------------------------------------------------------
# GMM reference rebuilt from Train
# --------------------------------------------------------------------------------------
def rebuild_gmm_reference(x_train, x_validation, train_features, models_dir: Path) -> tuple[dict, dict, str]:
    from importlib import metadata

    from . import regime_exploration as rx

    spec = rx.CandidateSpec("gmm", 2, "diag")
    fit = rx.fit_candidate(spec, x_train, x_validation, train_features, rx.SEEDS[0])
    out = {}
    for split, x, labels, proba in (("train", x_train, fit.train_labels, fit.train_proba),
                                    ("validation", x_validation, fit.validation_labels, fit.validation_proba)):
        out[split] = {"score": fit.model.score_samples(x), "labels": labels, "p_stress": proba[:, -1]}
    installed = metadata.version("scikit-learn")
    note = f"refitted on Train with scikit-learn {installed} (Stage 4 spec: diag, K=2, seed {rx.SEEDS[0]}, n_init 5)"
    meta_path, pkl_path = models_dir / "gmm_diag_k2.json", models_dir / "gmm_diag_k2.pkl"
    info = {"installed_sklearn": installed, "pickle_sklearn": None, "max_abs_score_diff": None}
    if meta_path.exists():
        saved = json.loads(meta_path.read_text(encoding="utf-8")).get("software", {}).get("scikit-learn")
        info["pickle_sklearn"] = saved
        if saved == installed and pkl_path.exists():
            with warnings.catch_warnings():
                warnings.simplefilter("error")       # any unpickling warning -> do not use the pickle
                try:
                    with open(pkl_path, "rb") as handle:
                        model = pickle.load(handle)["model"]
                    diff = float(np.abs(model.score_samples(x_validation)
                                        - out["validation"]["score"]).max())
                    info["max_abs_score_diff"] = diff
                    note += f"; matches the Stage 4 pickle (max |log-density difference| {diff:.1e})"
                except Exception as exc:  # noqa: BLE001
                    note += f"; Stage 4 pickle not compared ({type(exc).__name__})"
        else:
            note += (f"; Stage 4 pickle written by scikit-learn {saved}, so it is not unpickled (avoids the "
                     "version-mismatch warning)")
    return out, info, note


# --------------------------------------------------------------------------------------
# Training-window sensitivity
# --------------------------------------------------------------------------------------
def window_definitions(train_index: pd.DatetimeIndex, stable_from: pd.Timestamp | None) -> dict:
    first, last = train_index.min(), train_index.max()
    wins = {"full": (first, last), "start_plus_2y": (first + pd.DateOffset(years=2), last),
            "end_minus_2y": (first, last - pd.DateOffset(years=2))}
    if stable_from is not None and stable_from > first:
        wins["post_composition"] = (stable_from, last)
    return wins


def window_sensitivity(cands: dict, model_ids: list[str], x_train, x_validation, train_index, windows: dict, seeds,
                       mean, scale, train_original, n_init: int, progress=None) -> pd.DataFrame:
    from .hmm_diagnostics import run_candidate

    rows = []
    for name, (start, end) in windows.items():
        fit_mask = np.asarray((train_index >= start) & (train_index <= end))
        run_mask = np.asarray(train_index >= start)
        for m in model_ids:
            base = cands[m]
            if name == "full":
                c, params = base, base.reference.params
                val = base.validation
                vlabels = base.validation_labels
                converged = base.reference.converged
            else:
                c = run_candidate(base.spec, x_train[fit_mask], x_validation, seeds[:WINDOW_SEEDS], mean, scale,
                                  train_original[fit_mask], n_init=n_init)
                params = c.reference.params
                pre = forward_filter(params, x_train[run_mask])                      # window start .. Train end
                val = forward_filter(params, x_validation, previous_filtered=pre.last_filtered)
                vlabels = val.filtered.argmax(axis=1)
                converged = c.reference.converged
            perm = align_by_means(base.reference.params, params)
            aligned = params.permuted(perm)
            rows.append({"window": name, "start": str(pd.Timestamp(start).date()), "end": str(pd.Timestamp(end).date()),
                         "fit_rows": int(fit_mask.sum()), "model_id": m, "k": base.spec.k,
                         "converged": converged,
                         "validation_avg_loglik": float(val.log_score.mean()),
                         "validation_shares": " / ".join(f"S{i + 1} {s:.0%}" for i, s in
                                                         enumerate(rd.population(vlabels, base.spec.k))),
                         "validation_switches_per_year": rd.switching_stats(vlabels)["switches_per_year"],
                         "validation_ari_vs_full": float(adjusted_rand_score(base.validation_labels, vlabels)),
                         "max_emission_mean_shift_original_units": float(np.abs(
                             (aligned.means - base.reference.params.means) * scale).max()),
                         "self_transitions": " / ".join(f"{a:.4f}" for a in np.diag(aligned.transmat)),
                         "validation_mean_max_prob": float(val.filtered.max(axis=1).mean())})
            if progress:
                progress(name, m)
    out = pd.DataFrame(rows)
    out["rank_in_window"] = out.groupby("window")["validation_avg_loglik"].rank(ascending=False).astype(int)
    return out


# --------------------------------------------------------------------------------------
# Market phases
# --------------------------------------------------------------------------------------
def phase_table(cands: dict, model_ids: list[str], train_index, va_index, train_feat, val_feat,
                gmm: dict | None = None) -> pd.DataFrame:
    from .hmm_diagnostics import stress_score_series

    s_tr, s_va = stress_score_series(train_feat), stress_score_series(val_feat)
    median = float(s_tr.median())
    rows = []
    for split, index, s in (("train", train_index, s_tr), ("validation", va_index, s_va)):
        scores = {m: (cands[m].train if split == "train" else cands[m].validation).log_score for m in model_ids}
        if gmm is not None:
            scores["gmm_diag_k2"] = gmm[split]["score"]
        frame = pd.DataFrame(scores, index=index)
        phases = [(f"{y}", np.asarray(index.year == y)) for y in sorted(set(index.year))]
        phases += [("high stress (score > Train median)", np.asarray(s.to_numpy() > median)),
                   ("low stress (score <= Train median)", np.asarray(s.to_numpy() <= median))]
        for phase, mask in phases:
            for m in frame.columns:
                rows.append({"split": split, "phase": phase, "model_id": m, "n_days": int(mask.sum()),
                             "mean_log_score": float(frame[m][mask].mean()) if mask.any() else np.nan,
                             "in_sample": split == "train"})
    return pd.DataFrame(rows)


def phase_consistency(scores: pd.DataFrame, larger: str, smaller: str) -> dict:
    """Yearly mean daily log-score advantage of `larger` over `smaller` on Validation."""
    d = (scores[larger] - scores[smaller])
    yearly = d.groupby(d.index.year).mean()
    positive = int((yearly > 0).sum())
    return {"larger": larger, "smaller": smaller, "yearly_advantage": {int(y): float(v) for y, v in yearly.items()},
            "positive_years": positive, "n_years": int(len(yearly)),
            "consistent": positive >= PHASE_RULES["min_positive_years"]}
