"""Stage 6: Early Warning Score (EWS) from the slow regime layer, fast innovations and BOCPD.

Every component is mapped to [0, 1] with a scale fixed on Train:
    regime        = 0.5 * p_non_normal + 0.5 * p_stress             (HMM filtered probabilities; already [0, 1])
    confirmation  = F_train(stress_index_level)                     (factor LEVELS: -equity + volatility + credit)
    innovation    = F_train(ar_innovation_stress)                   (positive 5-session AR(1) market innovation)
    bocpd         = F_train(P(r_t < 10)) * 1[market_innovation_5d > 0]   (structural change, adverse-gated)
    velocity      = F_train(positive 5-session change of p_non_normal)
    macro         = F_train(max(interest_rate, inflation) level)    (rates / inflation pressure)
    entropy       = F_train(normalized HMM entropy)                 (candidate only; weight 0 in the primary spec)
F_train(v) = share of Train values strictly below v (so a value at the Train minimum, e.g. zero velocity, is 0).

EWS = sum_i w_i * component_i with fixed weights (sum 1). The weights were set in advance from the design
logic (slow regime and confirmed level stress carry the most weight; fast signals less; macro least) and are
NOT optimized. The EWS is a ranking score, not a probability of loss.
"""

from __future__ import annotations

STAGE6_CODE_VERSION = "stage6-2026-10-21"  # must match scripts/08_build_early_warning.py

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

PRIMARY_WEIGHTS = {"regime": 0.25, "confirmation": 0.25, "innovation": 0.20, "bocpd": 0.15, "velocity": 0.10,
                   "macro": 0.05, "entropy": 0.0}
COMPONENT_LABELS = {"regime": "HMM non-normal / stress regime", "confirmation": "factor-level market stress",
                    "innovation": "adverse AR(1) innovations", "bocpd": "structural change (BOCPD)",
                    "velocity": "rising non-normal probability", "macro": "rates / inflation pressure",
                    "entropy": "regime uncertainty"}
CHANNEL_RULES = {"equity weakness": ("level_pct_equity_weakness", 0.80),
                 "volatility stress": ("level_pct_volatility", 0.80),
                 "credit stress": ("level_pct_liquidity_credit", 0.80),
                 "rates/inflation pressure": ("macro_component", 0.90)}


class TrainECDF:
    """F(v) = share of Train values strictly below v."""

    def __init__(self, train_values):
        v = np.asarray(train_values, float)
        self.sorted = np.sort(v[np.isfinite(v)])

    def __call__(self, values) -> np.ndarray:
        v = np.asarray(values, float)
        out = np.searchsorted(self.sorted, v, side="left") / max(len(self.sorted), 1)
        out = out.astype(float)
        out[~np.isfinite(v)] = np.nan
        return out

    def quantile(self, q: float) -> float:
        return float(np.quantile(self.sorted, q))


@dataclass
class ComponentScaler:
    ecdf: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {k: {"train_n": len(e.sorted), "q50": e.quantile(0.5), "q80": e.quantile(0.8), "q95": e.quantile(0.95)}
                for k, e in self.ecdf.items()}


RAW_INPUTS = {"confirmation": "stress_index_level", "confirmation_no_credit": "stress_index_no_credit",
              "innovation": "ar_innovation_stress", "bocpd_raw": "bocpd_primary_recent",
              "velocity": "non_normal_probability_velocity_positive", "macro": "macro_level",
              "entropy": "normalized_entropy", "level_pct_equity_weakness": "equity_weakness_level",
              "level_pct_volatility": "volatility", "level_pct_liquidity_credit": "liquidity_credit",
              "market_innovation_1d": "market_innovation"}


def add_raw_inputs(daily: pd.DataFrame, levels: pd.DataFrame) -> pd.DataFrame:
    d = daily.copy()
    d["macro_level"] = levels[["interest_rate", "inflation"]].max(axis=1)
    d["equity_weakness_level"] = -levels["equity_trend"]
    d["volatility"] = levels["volatility"]
    d["liquidity_credit"] = levels["liquidity_credit"]
    s = -levels["equity_trend"] + levels["volatility"]
    d["stress_index_no_credit"] = s
    return d


def fit_scaler(daily: pd.DataFrame, train_mask) -> ComponentScaler:
    sc = ComponentScaler()
    for name, col in RAW_INPUTS.items():
        sc.ecdf[name] = TrainECDF(daily.loc[train_mask, col])
    return sc


def components(daily: pd.DataFrame, scaler: ComponentScaler) -> pd.DataFrame:
    """All candidate components in [0, 1] (plus channel percentiles used for explanations)."""
    c = pd.DataFrame(index=daily.index)
    c["regime"] = 0.5 * daily["p_non_normal"] + 0.5 * daily["p_stress"]
    c["confirmation"] = scaler.ecdf["confirmation"](daily["stress_index_level"])
    c["confirmation_no_credit"] = scaler.ecdf["confirmation_no_credit"](daily["stress_index_no_credit"])
    c["innovation"] = scaler.ecdf["innovation"](daily["ar_innovation_stress"])
    gate = (daily["market_innovation_5d"] > 0).astype(float).where(daily["market_innovation_5d"].notna())
    c["bocpd"] = scaler.ecdf["bocpd_raw"](daily["bocpd_primary_recent"]) * gate
    c["velocity"] = scaler.ecdf["velocity"](daily["non_normal_probability_velocity_positive"])
    c["macro"] = scaler.ecdf["macro"](daily["macro_level"])
    c["entropy"] = scaler.ecdf["entropy"](daily["normalized_entropy"])
    for k in ("level_pct_equity_weakness", "level_pct_volatility", "level_pct_liquidity_credit", "market_innovation_1d"):
        c[k] = scaler.ecdf[k](daily[RAW_INPUTS[k]])
    return c


def ews_score(comp: pd.DataFrame, weights: dict, substitutions: dict | None = None) -> tuple[pd.Series, pd.DataFrame]:
    """Weighted sum of components with weights renormalized to 1 over the components used.
    `substitutions` swaps a component column (e.g. confirmation -> confirmation_no_credit)."""
    use = {k: v for k, v in weights.items() if v > 0}
    total = sum(use.values())
    contrib = pd.DataFrame(index=comp.index)
    for k, w in use.items():
        col = (substitutions or {}).get(k, k)
        contrib[k] = comp[col] * (w / total)
    score = contrib.sum(axis=1, min_count=len(use))
    return score, contrib


def ews_specs() -> dict:
    """Primary specification and the predefined ablations / additions (no weight search)."""
    base = dict(PRIMARY_WEIGHTS)
    specs = {"full": (base, None)}
    for comp_name in ("bocpd", "innovation", "velocity", "macro"):
        w = dict(base)
        w[comp_name] = 0.0
        specs[f"no_{comp_name}"] = (w, None)
    specs["no_credit_confirmation"] = (dict(base), {"confirmation": "confirmation_no_credit"})
    w = dict(base)
    w["entropy"] = 0.10
    specs["with_entropy"] = (w, None)
    return specs


def dominant_drivers(contrib: pd.DataFrame, comp: pd.DataFrame, min_contribution: float = 0.05, top: int = 2
                     ) -> pd.Series:
    """Top contributing components and the factor channels behind them, as a readable string."""
    out = []
    for t in contrib.index:
        row = contrib.loc[t].dropna().sort_values(ascending=False)
        names = [COMPONENT_LABELS[k] for k, v in row.head(top).items() if v >= min_contribution]
        tags = [label for label, (col, thr) in CHANNEL_RULES.items()
                if col in comp and pd.notna(comp.at[t, col]) and comp.at[t, col] >= thr]
        b = comp.at[t, "bocpd"]
        if pd.notna(b) and b >= 0.9:
            tags.append("structural change")
        if pd.notna(comp.at[t, "market_innovation_1d"]) and comp.at[t, "market_innovation_1d"] >= 0.99:
            tags.append("abrupt market innovation")
        text = "; ".join(names) if names else "no material driver"
        if tags:
            text += " [" + ", ".join(tags) + "]"
        out.append(text)
    return pd.Series(out, index=contrib.index)


def equal_weight_score(comp: pd.DataFrame, names: list[str]) -> pd.Series:
    """Baselines G and H: simple average of the listed components."""
    return comp[names].mean(axis=1, skipna=False)


# --------------------------------------------------------------------------------------
# Daily signal construction (pure function of factors and frozen parameters; no returns)
# --------------------------------------------------------------------------------------
BOCPD_INPUTS = {"B1_level": "stress_index_level", "B2_innovation": "market_innovation",
                "B3_arhmm_surprise": "arhmm_surprise_probit"}
BOCPD_CHANNELS = {"B2_volatility": "directional_z_volatility", "B2_credit": "directional_z_liquidity_credit"}


def build_signals(hmm_params, ar1_params, arhmm_params, x_train, x_validation, train_index, val_index,
                  levels_train: pd.DataFrame, levels_validation: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """All date-t signals from factors through t and frozen Train-estimated models. Investment returns are not an
    argument, so they cannot enter any signal. Train-only constants are estimated from the Train rows."""
    from . import bocpd as bo
    from . import innovation_signals as ins

    n_tr = len(train_index)
    hmm = ins.hmm_features(hmm_params, x_train, x_validation, train_index, val_index)
    z = ins.ar_innovations(ar1_params, x_train, x_validation, train_index, val_index)
    surprise = ins.arhmm_surprise(arhmm_params, x_train, x_validation, train_index, val_index)
    levels = pd.concat([levels_train, levels_validation])
    norm = ins.fit_normalization(z.iloc[:n_tr], levels.iloc[:n_tr], surprise.iloc[:n_tr])
    inn = ins.innovation_frame(z, levels, surprise, norm)
    daily = pd.concat([hmm, z, inn], axis=1)
    priors, runs = {}, {}
    for name, col in {**BOCPD_INPUTS, **BOCPD_CHANNELS}.items():
        prior = bo.NIGPrior.from_train(daily[col].iloc[:n_tr].to_numpy())
        priors[name] = prior
        hazards = bo.HAZARD_RUN_LENGTHS if name in BOCPD_INPUTS else (bo.PRIMARY_RUN_LENGTH,)
        for lam in hazards:
            res = bo.bocpd(daily[col].to_numpy(), prior, lam, index=daily.index)
            runs[f"{name}_h{lam}"] = res
            daily = pd.concat([daily, res.frame(f"bocpd_{name}_h{lam}")], axis=1)
    primary = f"bocpd_B2_innovation_h{bo.PRIMARY_RUN_LENGTH}_p_change_within_{bo.PRIMARY_WINDOW}d"
    daily["bocpd_primary_recent"] = daily[primary]
    daily = add_raw_inputs(daily, levels)
    meta = {"normalization": norm, "bocpd_priors": {k: v.to_dict() for k, v in priors.items()}, "bocpd_runs": runs,
            "bocpd_primary_column": primary}
    return daily, meta
