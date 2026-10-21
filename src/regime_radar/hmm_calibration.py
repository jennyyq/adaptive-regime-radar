"""Stage 5: calibration diagnostics for HMM predictive densities and filtered state probabilities.

Why: on Validation the filtered state probabilities are close to 0 or 1 almost every day. That can be correct
(well separated regimes) or a symptom of misspecification. These diagnostics test it with observed data only;
nothing here smooths or edits probabilities.

1. PIT of the one-step-ahead predictive distribution, per factor:
       u_td = sum_k P(Z_t = k | x_1..x_t-1) Phi((x_td - mu_kd) / sigma_kd)
   If the model's predictive distribution is correct, u_td ~ U(0, 1) and is serially independent.
2. Probability reliability (martingale property): under a correctly specified HMM,
       E[ P(Z_t | x_1..x_t) | x_1..x_t-1 ] = P(Z_t | x_1..x_t-1),
   so the filtered stress probability should on average equal the predicted one, in every bin of the latter.
   The model-implied switching rate sum_i P(Z_t = i | x_1..x_t)(1 - A_ii) is compared with realized switching.
3. Within-state residual autocorrelation: the HMM assumes x_t independent given Z_t. Lag-1 autocorrelation of
   standardized residuals on consecutive same-state days measures how badly this holds (overlapping rolling
   windows make it large). Each day is then counted as fresh evidence, which makes the filter overconfident.
4. Predictive recalibration (a candidate fix, tested out of sample): emission covariances are multiplied by a
   scalar c >= 1 inside the filter only (Train parameters unchanged otherwise). c is chosen on Validation-A
   (2016-2017) by the mean predictive log score and evaluated on Validation-B (2018-2019). It is adopted for
   decision outputs only if the Validation-B log score improves with a Newey-West t-statistic >= 2.
"""

from __future__ import annotations

STAGE5_CODE_VERSION = "stage5-2026-10-21c"  # must match scripts/07_fit_hmm.py

import numpy as np
import pandas as pd
from scipy.stats import kstest, norm

from . import regime_diagnostics as rd
from .hmm_filtering import FilterResult, filter_train_validation
from .hmm_models import HMMParams, inflate_covariance

TRADING_DAYS = 252
CALIBRATION_RULES = {
    "inflation_grid": (1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0),
    "selection_period": ("2016-01-01", "2017-12-31"),     # Validation-A
    "evaluation_period": ("2018-01-01", "2019-12-31"),    # Validation-B
    "adopt_min_nw_t": 2.0,
    "nw_lags": 21,
    "reliability_bins": (0.0, 0.02, 0.1, 0.3, 0.5, 0.7, 0.9, 0.98, 1.0),
}


# --------------------------------------------------------------------------------------
# 1. PIT of the predictive distribution
# --------------------------------------------------------------------------------------
def predictive_pit(params: HMMParams, res: FilterResult, x: np.ndarray) -> np.ndarray:
    """(T, D) PIT values of each factor under the one-step-ahead predictive mixture (marginal per factor)."""
    sd = np.sqrt(np.stack([np.diag(c) for c in params.full_covars()]))          # (K, D)
    z = (np.asarray(x)[:, None, :] - params.means[None]) / sd[None]                 # (T, K, D)
    return (res.predicted[:, :, None] * norm.cdf(z)).sum(axis=1)


def _lag1(a: np.ndarray) -> float:
    a = np.asarray(a, float)
    if len(a) < 3 or a.std() == 0:
        return np.nan
    return float(np.corrcoef(a[1:], a[:-1])[0, 1])


def pit_summary(pit: np.ndarray, columns: list[str]) -> pd.DataFrame:
    rows = []
    for j, name in enumerate(list(columns) + ["all factors"]):
        u = pit[:, j] if j < len(columns) else pit.ravel()
        rows.append({"factor": name, "n": len(u), "mean": float(u.mean()), "std": float(u.std()),
                     "uniform_std": 1 / np.sqrt(12),
                     "ks_statistic": float(kstest(u, "uniform").statistic),
                     "coverage_50": float(((u > 0.25) & (u < 0.75)).mean()),
                     "coverage_90": float(((u > 0.05) & (u < 0.95)).mean()),
                     "share_below_0.01": float((u < 0.01).mean()), "share_above_0.99": float((u > 0.99).mean()),
                     "lag1_autocorrelation": _lag1(u) if j < len(columns) else
                     float(np.nanmean([_lag1(pit[:, i]) for i in range(len(columns))]))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# 2. Reliability of state probabilities
# --------------------------------------------------------------------------------------
def reliability_table(predicted: np.ndarray, filtered: np.ndarray, bins=CALIBRATION_RULES["reliability_bins"]
                      ) -> pd.DataFrame:
    """Bins of the predicted stress probability; under correct specification mean filtered = mean predicted."""
    cut = pd.cut(predicted, bins=list(bins), include_lowest=True)
    frame = pd.DataFrame({"bin": cut, "predicted": predicted, "filtered": filtered})
    out = frame.groupby("bin", observed=False).agg(n=("predicted", "size"), mean_predicted=("predicted", "mean"),
                                                   mean_filtered=("filtered", "mean")).reset_index()
    out["bin"] = out["bin"].astype(str)
    out["gap"] = out["mean_filtered"] - out["mean_predicted"]
    return out


def reliability_summary(predicted: np.ndarray, filtered: np.ndarray, filtered_all: np.ndarray,
                        transmat: np.ndarray, labels: np.ndarray) -> dict:
    d = filtered - predicted
    se = rd.newey_west_se(d, CALIBRATION_RULES["nw_lags"])
    slope = float(np.polyfit(predicted, filtered, 1)[0]) if predicted.std() > 0 else np.nan
    implied = float(TRADING_DAYS * (filtered_all[:-1] * (1 - np.diag(transmat))[None, :]).sum(axis=1).mean())
    realized = rd.switching_stats(labels)["switches_per_year"]
    return {"mean_filtered_minus_predicted": float(d.mean()), "nw_t": float(d.mean() / se) if se and se > 0 else np.nan,
            "mean_abs_update": float(np.abs(d).mean()), "calibration_slope": slope,
            "implied_switches_per_year": implied, "realized_switches_per_year": realized,
            "realized_to_implied_switch_ratio": realized / implied if implied > 0 else np.nan}


# --------------------------------------------------------------------------------------
# 3. Conditional independence check
# --------------------------------------------------------------------------------------
def residual_autocorrelation(params: HMMParams, x: np.ndarray, labels: np.ndarray, columns: list[str]) -> dict:
    """Lag-1 autocorrelation of standardized residuals (x_t - mu_z) / sigma_z on consecutive same-state days."""
    labels = np.asarray(labels, int)
    sd = np.sqrt(np.stack([np.diag(c) for c in params.full_covars()]))
    r = (np.asarray(x) - params.means[labels]) / sd[labels]
    same = labels[1:] == labels[:-1]
    out = {}
    for j, c in enumerate(columns):
        a, b = r[1:, j][same], r[:-1, j][same]
        out[c] = float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else np.nan
    out["mean"] = float(np.nanmean(list(out.values())))
    return out


# --------------------------------------------------------------------------------------
# 4. Covariance-inflation recalibration (selected on Validation-A, evaluated on Validation-B)
# --------------------------------------------------------------------------------------
def _period_mask(index: pd.DatetimeIndex, period) -> np.ndarray:
    return np.asarray((index >= pd.Timestamp(period[0])) & (index <= pd.Timestamp(period[1])))


def inflation_curve(params: HMMParams, stress_states: list[int], x_train: np.ndarray, x_validation: np.ndarray,
                    va_index: pd.DatetimeIndex, grid=CALIBRATION_RULES["inflation_grid"]) -> tuple[pd.DataFrame, dict]:
    a_mask = _period_mask(va_index, CALIBRATION_RULES["selection_period"])
    b_mask = _period_mask(va_index, CALIBRATION_RULES["evaluation_period"])
    rows, store = [], {}
    for c in grid:
        p = inflate_covariance(params, c)
        _, va = filter_train_validation(p, x_train, x_validation)
        f = va.filtered
        ps = f[:, stress_states].sum(axis=1)
        pit = predictive_pit(p, va, x_validation)
        store[c] = (va, ps)
        row = {"inflation": c}
        for tag, m in (("val_a", a_mask), ("val_b", b_mask)):
            row[f"{tag}_mean_log_score"] = float(va.log_score[m].mean())
            row[f"{tag}_mean_max_prob"] = float(f[m].max(axis=1).mean())
            row[f"{tag}_mean_entropy"] = float(rd.normalized_entropy(f[m]).mean())
            row[f"{tag}_share_uncertain_days"] = float(((ps[m] > 0.2) & (ps[m] < 0.8)).mean())
            row[f"{tag}_pit_coverage_90"] = float(((pit[m] > 0.05) & (pit[m] < 0.95)).mean())
            row[f"{tag}_switches_per_year"] = rd.switching_stats(f[m].argmax(axis=1))["switches_per_year"]
        rows.append(row)
    curve = pd.DataFrame(rows)
    best = float(curve.loc[curve["val_a_mean_log_score"].idxmax(), "inflation"])
    d = store[best][0].log_score[b_mask] - store[1.0][0].log_score[b_mask]
    se = rd.newey_west_se(d, CALIBRATION_RULES["nw_lags"])
    t = float(d.mean() / se) if se and se > 0 else (0.0 if best == 1.0 else np.nan)
    adopt = bool(best != 1.0 and d.mean() > 0 and t >= CALIBRATION_RULES["adopt_min_nw_t"])
    decision = {"chosen_inflation_on_val_a": best, "val_b_mean_gain": float(d.mean()), "val_b_nw_t": t,
                "adopted": adopt, "decision_inflation": best if adopt else 1.0,
                "_p_stress_chosen": store[best][1], "_p_stress_raw": store[1.0][1],
                "rule": f"choose c on {CALIBRATION_RULES['selection_period']} by mean log score; adopt only if the "
                        f"{CALIBRATION_RULES['evaluation_period']} gain has NW t >= {CALIBRATION_RULES['adopt_min_nw_t']}"}
    return curve, decision


def calibration_report(cands: dict, model_ids: list[str], x_train: np.ndarray, x_validation: np.ndarray,
                       va_index: pd.DatetimeIndex, columns: list[str]) -> dict:
    """All calibration tables for the listed models (Validation only; parameters fixed at Train)."""
    pit_rows, rel_rows, rel_sum, auto_rows, curves, decisions, pit_values = [], [], [], [], [], {}, {}
    for m in model_ids:
        c = cands[m]
        p = c.reference.params
        pit = predictive_pit(p, c.validation, x_validation)
        pit_values[m] = pit
        s = pit_summary(pit, columns)
        s.insert(0, "model_id", m)
        pit_rows.append(s)
        pred = c.validation.predicted[:, c.stress_states].sum(axis=1)
        filt = c.stress_probability("validation")
        r = reliability_table(pred, filt)
        r.insert(0, "model_id", m)
        rel_rows.append(r)
        rel_sum.append({"model_id": m, **reliability_summary(pred, filt, c.validation.filtered, p.transmat,
                                                             c.validation_labels)})
        ac_tr = residual_autocorrelation(p, x_train, c.train_labels, columns)
        ac_va = residual_autocorrelation(p, x_validation, c.validation_labels, columns)
        auto_rows += [{"model_id": m, "split": "train", **ac_tr}, {"model_id": m, "split": "validation", **ac_va}]
        curve, dec = inflation_curve(p, c.stress_states, x_train, x_validation, va_index)
        curve.insert(0, "model_id", m)
        curves.append(curve)
        decisions[m] = dec
    return {"pit": pd.concat(pit_rows, ignore_index=True), "reliability": pd.concat(rel_rows, ignore_index=True),
            "reliability_summary": pd.DataFrame(rel_sum), "residual_autocorrelation": pd.DataFrame(auto_rows),
            "inflation_curve": pd.concat(curves, ignore_index=True),
            "inflation_decision": pd.DataFrame([{"model_id": m, **{k: v for k, v in d.items() if not k.startswith("_")}}
                                                for m, d in decisions.items()]),
            "decisions": decisions, "pit_values": pit_values}
