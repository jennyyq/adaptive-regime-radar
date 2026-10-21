"""Stage 8: sequential frozen-model signals for Train + Validation + Test (no refit, no smoothing, no reset).

The Stage 6 signal generator `early_warning.build_signals` is reused unchanged. Its second segment is simply the
chronological continuation Validation -> Test, so for every date t
  * the HMM forward filter continues from its filtered distribution of t-1 (frozen hmm_std_diag_k3 parameters);
  * AR(1) innovations and the AR-HMM predictive surprise use the frozen Stage 5.5 models and the previous factor row;
  * BOCPD continues its run-length posterior (frozen Train NIG priors, hazard 1/126; never reset at 2020-01-02);
  * the alert state machine continues with its persistence, confirmation, hysteresis, recovery and emergency rules
    (frozen Train thresholds; the Validation state carries into Test).
Train-only constants (factor scaler, innovation normalization, BOCPD priors, component ECDFs, alert thresholds,
the baseline q80 threshold) are recomputed from the Train rows exactly as in Stages 6 / 6.1 and verified against
the frozen handoff values. Factors after t never enter a date-t signal; returns never enter any signal.

The output is ROLE-SEPARATED: `SignalRun.roles['baseline_regime']` carries only baseline_* columns,
`SignalRun.roles['challenger_ews']` only challenger_* columns (same timing columns as the Stage 6.1 handoff).
"""

from __future__ import annotations

STAGE8_CODE_VERSION = "stage8-2026-10-21"  # must match scripts/10_backtest_portfolios.py

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import alert_state_machine as asm
from . import early_warning as ew
from . import regime_exploration as rx
from . import stage7_handoff as h7
from .ar_hmm import ARParams
from .dataset import FEATURE_COLUMNS
from .hmm_models import HMMParams

SIGNAL_COLUMNS = {"baseline_regime": ("baseline_p_non_normal", "baseline_alert_ge_q80"),
                  "challenger_ews": ("challenger_ews_score", "challenger_alert_level", "challenger_alert_ge_amber",
                                     "challenger_alert_ge_orange")}


@dataclass
class FrozenSignalModels:
    hmm: HMMParams
    ar1: ARParams
    arhmm: ARParams
    metas: dict
    files: dict
    baseline_threshold: float
    alert_thresholds: dict
    ews_weights: dict
    normalization: dict
    bocpd_priors: dict


def load_frozen_models(root: Path, handoff: dict) -> FrozenSignalModels:
    """Frozen model files named in the Stage 6.1 handoff manifest (paths and hashes are verified by the caller)."""
    fm = handoff["input_hashes"]["frozen_models"]
    metas = {k: json.loads((Path(root) / v["file"]).read_text(encoding="utf-8")) for k, v in fm.items()}
    ch = handoff["roles"]["challenger_ews"]["frozen_parameters"]
    base = handoff["roles"]["baseline_regime"]["frozen_parameters"]["baseline_alert_ge_q80"]
    return FrozenSignalModels(HMMParams.from_dict(metas["hmm"]["parameters"]), ARParams.from_dict(metas["ar1"]["parameters"]),
                              ARParams.from_dict(metas["ar_hmm_k3"]["parameters"]), metas,
                              {k: v["file"] for k, v in fm.items()}, float(base["threshold"]),
                              dict(ch["alert_thresholds"]), dict(ch["ews_weights"]), dict(ch["train_normalization"]),
                              dict(ch["bocpd_priors"]))


@dataclass
class SignalRun:
    roles: dict
    descriptive: pd.DataFrame
    thresholds: dict
    normalization: dict
    bocpd_priors: dict
    scaler: dict
    train_mask: np.ndarray
    checks: list = field(default_factory=list)


def train_scaler(train_features: pd.DataFrame):
    apply, _ = rx.scaling_decision(rx.describe_train(train_features[FEATURE_COLUMNS]))
    return rx.TrainScaler(apply).fit(train_features[FEATURE_COLUMNS])


def sequential_signals(models: FrozenSignalModels, train_features: pd.DataFrame, continuation: pd.DataFrame,
                       split_labels: pd.Series) -> SignalRun:
    """Signals for Train followed by `continuation` (Validation, or Validation + Test) in one causal pass."""
    tr = train_features[FEATURE_COLUMNS]
    co = continuation[FEATURE_COLUMNS]
    scaler = train_scaler(tr)
    x_tr, x_co = scaler.transform(tr), scaler.transform(co)
    lv_tr = pd.DataFrame(x_tr, index=tr.index, columns=FEATURE_COLUMNS)
    lv_co = pd.DataFrame(x_co, index=co.index, columns=FEATURE_COLUMNS)
    daily, meta = ew.build_signals(models.hmm, models.ar1, models.arhmm, x_tr, x_co, tr.index, co.index, lv_tr, lv_co)
    daily["split"] = split_labels.reindex(daily.index).to_numpy()
    train_mask = (daily["split"] == "train").to_numpy()
    comp = ew.components(daily, ew.fit_scaler(daily, train_mask))
    weights, subs = ew.ews_specs()["full"]
    score, _ = ew.ews_score(comp, weights, subs)
    th = asm.Thresholds.from_train(score[train_mask])
    states = asm.run_state_machine(score, th, daily["p_stress"], comp["confirmation"], comp["market_innovation_1d"])
    frames = h7.build_role_frames(daily, score, states["alert_level"], models.baseline_threshold)
    roles = {}
    for role, f in frames.items():
        g = f.copy()
        g.attrs["stage7_role"] = role
        h7.assert_single_role(g)
        roles[role] = g
    desc = daily[["split", "p_s1", "p_s2", "p_s3", "p_stress", "p_non_normal", "hmm_state", "normalized_entropy",
                  "market_innovation", "bocpd_primary_recent"]].copy()
    desc["alert_rule"] = states["alert_rule"]
    return SignalRun(roles, desc, th.to_dict(), meta["normalization"].to_dict(),
                     {k: v for k, v in meta["bocpd_priors"].items()}, scaler.to_dict(), train_mask)


def role_frame(run: SignalRun, role: str) -> pd.DataFrame:
    """One role only (timing + that role's columns). The caller never receives both roles in one frame."""
    if role not in h7.ROLES:
        raise h7.HandoffError(f"unknown role {role!r}")
    f = run.roles[role]
    out = f[[c for c in f.columns if c == "split" or c in h7.TIMING_COLUMNS or c in SIGNAL_COLUMNS[role]]].copy()
    out.attrs["stage7_role"] = role
    if h7.assert_single_role(out) != role:
        raise h7.HandoffError("role frame mixes roles")
    return out


# --------------------------------------------------------------------------------------
# Verification against the frozen Stage 6.1 handoff
# --------------------------------------------------------------------------------------
def _same(a: pd.Series, b: pd.Series) -> tuple[bool, float]:
    """Exact equality (NaN == NaN) for floats; equality for strings / ints. Returns (identical, max abs diff)."""
    if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
        x, y = a.to_numpy(float), b.to_numpy(float)
        nan_ok = bool((np.isnan(x) == np.isnan(y)).all())
        m = ~np.isnan(x) & ~np.isnan(y)
        diff = float(np.max(np.abs(x[m] - y[m]))) if m.any() else 0.0
        return nan_ok and diff == 0.0, diff if nan_ok else np.inf
    same = bool((a.astype(str).to_numpy() == b.astype(str).to_numpy()).all())
    return same, 0.0 if same else np.inf


def compare_with_handoff(run: SignalRun, handoff_frames: dict) -> dict:
    """handoff_frames[role] = Train + Validation frame read with stage7_handoff.load_stage7_inputs.
    Signal columns must be bit-identical; timing columns identical except the last two Validation rows, whose
    execution / holding dates were unknown in Stage 6.1 (no Test calendar) and are now filled from Test."""
    out = {}
    for role, hf in handoff_frames.items():
        f = run.roles[role].loc[hf.index]
        for c in SIGNAL_COLUMNS[role]:
            out[f"{role}:{c}"] = _same(f[c], hf[c])
        for c in h7.TIMING_COLUMNS:
            a, b = f[c].iloc[:-2], hf[c].iloc[:-2]
            out[f"{role}:{c} (all but last 2 rows)"] = _same(a.fillna(""), b.fillna(""))
    return out


def compare_frozen_constants(run: SignalRun, models: FrozenSignalModels) -> dict:
    """Max absolute differences between recomputed Train constants and the frozen handoff values."""
    th = max(abs(run.thresholds[k] - float(v)) for k, v in models.alert_thresholds.items())
    norm = max(abs(float(run.normalization[k]) - float(v)) for k, v in models.normalization.items()
               if isinstance(v, (int, float)))
    pri = max(abs(float(run.bocpd_priors[n][k]) - float(v)) for n, d in models.bocpd_priors.items() for k, v in d.items())
    pre = models.metas["hmm"]["preprocessing"]
    sc = run.scaler
    scl = max(max(abs(pre["mean"][c] - sc["mean"][c]), abs(pre["scale"][c] - sc["scale"][c])) for c in FEATURE_COLUMNS)
    same_apply = pre["applied"] == sc["applied"]
    return {"alert_thresholds": th, "innovation_normalization": norm, "bocpd_priors": pri,
            "factor_scaler": scl if same_apply else np.inf}


def prefix_difference(full: SignalRun, part: SignalRun, upto: pd.Timestamp | None = None) -> float:
    """Max difference between two runs over the dates of `part` (or over dates <= `upto`): role signal columns and
    the descriptive HMM / BOCPD columns. np.inf if an alert level or flag differs."""
    worst = 0.0
    idx = part.descriptive.index if upto is None else part.descriptive.index[part.descriptive.index <= upto]
    for role in h7.ROLES:
        a = full.roles[role].loc[idx]
        b = part.roles[role].loc[idx]
        for c in SIGNAL_COLUMNS[role]:
            ok, d = _same(a[c], b[c])
            worst = max(worst, 0.0 if ok else (d if np.isfinite(d) and d > 0 else np.inf))
    for c in ("p_s1", "p_s2", "p_s3", "p_non_normal", "bocpd_primary_recent", "market_innovation"):
        ok, d = _same(full.descriptive.loc[idx, c], part.descriptive.loc[idx, c])
        worst = max(worst, 0.0 if ok else (d if np.isfinite(d) and d > 0 else np.inf))
    return worst


def defensiveness_series(role_frames: dict, scale: float, level_map: dict, mapping: str = "primary",
                         thresholds: dict | None = None) -> dict[str, pd.Series]:
    """{portfolio: d series}. Each series is built from ONE role frame only.
    mapping 'primary': d1 = min(1, a p), d2 = level map; 'full_range_continuous': d1 = p, d2 = continuous score map."""
    from . import risk_budgeting as rb
    base = role_frames["baseline_regime"]
    chal = role_frames["challenger_ews"]
    if h7.assert_single_role(base) != "baseline_regime" or h7.assert_single_role(chal) != "challenger_ews":
        raise h7.HandoffError("defensiveness must be built from one role frame each")
    if mapping == "primary":
        d1 = base["baseline_p_non_normal"].map(lambda p: rb.p1_defensiveness(p, scale))
        d2 = chal["challenger_alert_level"].map(lambda lv: rb.p2_defensiveness(lv, level_map))
    elif mapping == "full_range_continuous":
        d1 = base["baseline_p_non_normal"].map(lambda p: rb.p1_defensiveness(p, 1.0))
        d2 = chal["challenger_ews_score"].map(
            lambda s: rb.p2_defensiveness_continuous(s, thresholds["amber_out"], thresholds["red_in"]))
    else:
        raise ValueError(mapping)
    return {"P1_regime_aware": d1.astype(float), "P2_ews_enhanced": d2.astype(float)}
