"""Markdown diagnostics and presentation figures for the five-factor layer.

The event windows below are DIAGNOSTIC ONLY: they are used to eyeball whether factors
react sensibly to well-known stress periods. They are never used as regime labels,
never used to fit anything, and play no role in factor construction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .factors import (
    COMPONENT_CLIP, COMPONENTS, DEFAULT_MIN_PERIODS, DEFAULT_WINDOW, FACTOR_DIRECTION, FACTORS,
    MIN_COMPONENT_SHARE, MONTHLY_MIN_PERIODS, MONTHLY_WINDOW, Check, component_dominance,
)
from .quality import _md_table

EVENT_WINDOWS = {
    "2008 Global Financial Crisis": ("2008-09-01", "2009-03-31"),
    "2020 COVID shock": ("2020-02-19", "2020-04-30"),
    "2022 inflation / rates shock": ("2022-01-03", "2022-10-31"),
}

# Chart styling: reference categorical slots 1-5 (one fixed color per factor), light surface.
FACTOR_COLORS = {
    "equity_trend": "#2a78d6",
    "interest_rate": "#eb6834",
    "inflation": "#1baf7a",
    "volatility": "#eda100",
    "liquidity_credit": "#e87ba4",
}
INK, INK_2, MUTED, GRID, AXIS, SURFACE, EVENT_SHADE = (
    "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb", "#ecebe6")


# --------------------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------------------
def _first_valid(series: pd.Series) -> str:
    idx = series.first_valid_index()
    return idx.date().isoformat() if idx is not None else "never"


def summary_statistics(factors: pd.DataFrame, sample_start: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for f in factors:
        s = factors[f].dropna()
        rows.append({
            "factor": f,
            "first_valid": _first_valid(factors[f]),
            "missing_full_pct": round(100 * factors[f].isna().mean(), 1),
            "missing_in_sample_pct": round(100 * factors[f].loc[sample_start:].isna().mean(), 1),
            "mean": s.mean(), "std": s.std(), "p05": s.quantile(0.05), "median": s.median(),
            "p95": s.quantile(0.95), "min": s.min(), "max": s.max(),
        })
    return pd.DataFrame(rows).set_index("factor").round(2)


def component_table(raw: pd.DataFrame, comps: pd.DataFrame, sample_start: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for c in COMPONENTS:
        col = f"{c.factor}__{c.name}"
        rows.append({
            "factor": c.factor,
            "component": c.name,
            "definition": c.definition,
            "source": c.source,
            "sign": f"{c.sign:+d}",
            "window": c.window,
            "in_composite": "yes" if c.in_composite else "monitor only",
            "raw_first_valid": _first_valid(raw[c.name]),
            "z_first_valid": _first_valid(comps[col]),
            "z_missing_in_sample_pct": round(100 * comps[col].loc[sample_start:].isna().mean(), 1),
        })
    return pd.DataFrame(rows).set_index("component")


def largest_episodes(series: pd.Series, n: int = 5, highest: bool = True, separation: int = 63) -> pd.DataFrame:
    """Top-n extreme days, keeping only one date per `separation`-day episode."""
    s = series.dropna()
    order = s.sort_values(ascending=not highest)
    picked: list[pd.Timestamp] = []
    positions = {d: i for i, d in enumerate(s.index)}
    for date in order.index:
        if all(abs(positions[date] - positions[p]) > separation for p in picked):
            picked.append(date)
        if len(picked) == n:
            break
    return pd.DataFrame({"date": [d.date().isoformat() for d in picked], "value": [round(s[d], 2) for d in picked]})


def event_window_table(factors: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for event, (start, end) in EVENT_WINDOWS.items():
        for f in factors:
            full = factors[f].dropna()
            window = factors[f].loc[start:end].dropna()
            if window.empty or full.empty:
                rows.append({"event": event, "factor": f, "window_mean": np.nan})
                continue
            rows.append({
                "event": event,
                "factor": f,
                "window_mean": round(window.mean(), 2),
                "window_min": round(window.min(), 2),
                "min_date": window.idxmin().date().isoformat(),
                "window_max": round(window.max(), 2),
                "max_date": window.idxmax().date().isoformat(),
                "mean_percentile_vs_history": round(100 * (full < window.mean()).mean(), 0),
            })
    return pd.DataFrame(rows).set_index(["event", "factor"])


# --------------------------------------------------------------------------------------
# Markdown report
# --------------------------------------------------------------------------------------
def markdown_report(
    raw: pd.DataFrame,
    comps: pd.DataFrame,
    factors: pd.DataFrame,
    checks: list[Check],
    sample_start: pd.Timestamp,
    release_log: pd.DataFrame,
) -> str:
    in_sample = factors.loc[sample_start:]
    lines = ["# Factor diagnostics: five economic risk factors", ""]
    lines.append(f"Calendar: {factors.index.min().date()} to {factors.index.max().date()} "
                 f"({len(factors)} trading days). Investment sample starts {sample_start.date()}; "
                 "statistics marked 'in sample' use dates from that point on.")

    lines += ["", "## Validation checks", ""]
    lines.append(_md_table(pd.DataFrame([vars(c) for c in checks]).set_index("name")))

    lines += ["", "## Construction", ""]
    lines += [
        f"- Each component is standardized with a rolling robust z-score against the previous "
        f"{DEFAULT_WINDOW} trading days (minimum {DEFAULT_MIN_PERIODS} observations): "
        "`z_t = (x_t - median[t-W..t-1]) / (1.4826 * MAD[t-W..t-1])`. Day t never enters its own "
        "center or scale; no future data are used; no full-sample mean/std anywhere.",
        f"- Monthly CPI components use a {MONTHLY_WINDOW}-day window (minimum {MONTHLY_MIN_PERIODS}) "
        "because a monthly series changes only ~12 times per year and a 1-year MAD is unstable.",
        "- If MAD is zero the window standard deviation is used; if that is also zero the value is missing.",
        f"- Components are signed so that higher = more stress (Equity Trend: higher = healthier), "
        f"then winsorized at +/-{COMPONENT_CLIP:g}.",
        f"- Factor = equal-weight mean of available signed components; a factor needs at least "
        f"{MIN_COMPONENT_SHARE:.0%} of its composite components (rounded up) on that day. No weights are estimated.",
        "- Macro inputs come from the point-in-time panel (`macro_pit.csv`); CPI year-over-year is computed "
        "from the release log and becomes usable only once both months were published.",
    ]

    lines += ["", "## Factor definitions", ""]
    fdef = pd.DataFrame([
        {"factor": k, "name": v, "direction": FACTOR_DIRECTION[k],
         "composite components": ", ".join(c.name for c in COMPONENTS if c.factor == k and c.in_composite)}
        for k, v in FACTORS.items()
    ]).set_index("factor")
    lines.append(_md_table(fdef))

    lines += ["", "## Components and sign conventions", ""]
    ctab = component_table(raw, comps, sample_start)
    lines.append(_md_table(ctab))
    lines += ["", "Sign rationale:", ""]
    lines += [f"- `{c.name}` ({c.sign:+d}): {c.sign_rationale}" for c in COMPONENTS]

    lines += ["", "## Factor summary statistics", ""]
    lines.append(_md_table(summary_statistics(factors, sample_start)))

    lines += ["", "## Correlations between factors (in sample)", ""]
    lines.append(_md_table(in_sample.corr().round(2)))

    lines += ["", "## Component contribution (in sample)", ""]
    lines.append("`avg_abs_share` = average share of the factor's absolute signal coming from the component; "
                 "`corr_with_other_components` = correlation with the mean of the factor's other components.")
    lines += ["", _md_table(component_dominance(comps.loc[sample_start:]).set_index("component"))]

    lines += ["", "## Largest historical readings", ""]
    lines.append("Highest values for stress factors; lowest values for Equity Trend. "
                 "One date per 63-trading-day episode.")
    for f in factors:
        top = largest_episodes(factors[f], highest=(f != "equity_trend"))
        lines += ["", f"**{FACTORS[f]}**", "", _md_table(top.set_index("date"))]

    lines += ["", "## Diagnostic event windows (not regime labels)", ""]
    lines.append("Percentile = share of all historical factor values below the window mean. "
                 "Both tails are shown: e.g. Interest Rate Conditions typically falls (easing) in "
                 "growth shocks and rises in rate shocks.")
    lines += ["", _md_table(event_window_table(factors).reset_index().set_index("event"))]

    lines += ["", "## Data limitations", ""]
    lines += data_limitations(raw, release_log, sample_start)
    return "\n".join(lines) + "\n"


def data_limitations(raw: pd.DataFrame, release_log: pd.DataFrame, sample_start: pd.Timestamp) -> list[str]:
    notes = []
    hy = raw["hy_oas"].dropna()
    notes.append(
        f"- **HY OAS (BAMLH0A0HYM2)**: {'starts ' + hy.index.min().date().isoformat() if len(hy) else 'not available'}. "
        "FRED now serves only a rolling ~3-year window, so it is monitor-only and excluded from the "
        "Liquidity / Credit composite (BAA10Y and HYG/LQD carry the credit-spread signal over the full sample).")
    if not release_log.empty and "availability_method" in release_log:
        extra = {
            "NFCI": "NFCI is re-estimated every week, so current-vintage history is smoother than what was "
                    "known in real time.",
            "CPIAUCSL": "CPI revisions are small (mainly seasonal factors), so the effect is minor.",
            "CPILFESL": "Core CPI revisions are small (mainly seasonal factors), so the effect is minor.",
        }
        for sid, label in (("NFCI", "NFCI"), ("CPIAUCSL", "CPI"), ("CPILFESL", "Core CPI")):
            methods = release_log.loc[release_log["series_id"] == sid, "availability_method"].value_counts()
            if methods.empty:
                notes.append(f"- **{label}**: not downloaded; the component is missing.")
            else:
                desc = ", ".join(f"{k}={v}" for k, v in methods.items())
                notes.append(f"- **{label}** release dates: {desc}. 'fallback' rows use rule-based release dates "
                             f"and current-vintage values. {extra[sid]}")
    vix3m = raw["vix_term_structure"].dropna()
    if len(vix3m):
        notes.append(f"- **VIX term structure** starts {vix3m.index.min().date()} (VIX3M history); before that "
                     "Volatility Stress averages the remaining components.")
    hyg = raw["hyg_vs_lqd_63d"].dropna()
    if len(hyg):
        notes.append(f"- **HYG-based components** start {hyg.index.min().date()}; Liquidity / Credit Stress uses "
                     "BAA10Y and NFCI before that.")
    notes.append("- **HYG vs LQD** also contains a duration difference (LQD is longer duration), so in rising-rate "
                 "periods it can understate credit stress.")
    notes.append("- **Rolling 1-year standardization** measures stress relative to the trailing year: in a long "
                 "stress period (e.g. 2008-2009) z-scores drift back toward zero as the window absorbs the stress. "
                 "Keep this in mind when interpreting persistence in the regime model.")
    notes.append(f"- Statistics in this report cover all valid dates; the investment sample starts {sample_start.date()}.")
    return notes


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
def _style_axis(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.grid(axis="x", visible=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.axhline(0, color=AXIS, linewidth=0.8, zorder=1)


def plot_timeseries(factors: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(FACTORS), 1, figsize=(12, 11), sharex=True, facecolor=SURFACE)
    for ax, f in zip(axes, FACTORS):
        _style_axis(ax)
        for start, end in EVENT_WINDOWS.values():
            ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color=EVENT_SHADE, zorder=0, linewidth=0)
        ax.plot(factors.index, factors[f], color=FACTOR_COLORS[f], linewidth=1.0, zorder=2)
        ax.set_title(f"{FACTORS[f]}  ({FACTOR_DIRECTION[f]})", loc="left", fontsize=10, color=INK)
        ax.set_ylabel("z-score", fontsize=8, color=INK_2)
    axes[0].text(0.995, 1.12, "Shaded: diagnostic event windows (GFC, COVID, 2022)", transform=axes[0].transAxes,
                 ha="right", fontsize=8, color=MUTED)
    axes[-1].set_xlabel("Date", fontsize=9, color=INK_2)
    fig.suptitle("Five economic risk factors (rolling robust z-scores)", x=0.01, ha="left",
                 fontsize=13, color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_correlation_heatmap(factors: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    corr = factors.corr()
    labels = [FACTORS[f] for f in corr.columns]
    cmap = LinearSegmentedColormap.from_list(
        "diverging_blue_red", ["#184f95", "#5598e7", "#f0efec", "#e66767", "#a72f2f"])
    fig, ax = plt.subplots(figsize=(7.5, 6.2), facecolor=SURFACE)
    image = ax.imshow(corr.to_numpy(), cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right", fontsize=9, color=INK_2)
    ax.set_yticks(range(len(labels)), labels, fontsize=9, color=INK_2)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = corr.iat[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=9,
                    color="#ffffff" if abs(value) > 0.6 else INK)
    bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    bar.set_label("Pearson correlation", fontsize=8, color=INK_2)
    bar.ax.tick_params(labelsize=8, colors=MUTED, length=0)
    bar.outline.set_visible(False)
    start, end = factors.dropna().index.min(), factors.dropna().index.max()
    ax.set_title(f"Correlation between factors ({start.year}-{end.year}, daily)", loc="left",
                 fontsize=12, color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_event_windows(factors: pd.DataFrame, path: Path, pad_days: int = 90) -> None:
    """Small multiples: rows = factors, columns = events; shared y-axis per row."""
    import matplotlib.pyplot as plt

    events = list(EVENT_WINDOWS.items())
    fig, axes = plt.subplots(len(FACTORS), len(events), figsize=(13, 11), sharey="row", facecolor=SURFACE)
    for col, (event, (start, end)) in enumerate(events):
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        view = factors.loc[start - pd.Timedelta(days=pad_days): end + pd.Timedelta(days=pad_days)]
        for row, f in enumerate(FACTORS):
            ax = axes[row, col]
            _style_axis(ax)
            ax.axvspan(start, end, color=EVENT_SHADE, zorder=0, linewidth=0)
            ax.plot(view.index, view[f], color=FACTOR_COLORS[f], linewidth=1.4, zorder=2)
            ax.tick_params(axis="x", labelrotation=0)
            ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=3))
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %y"))
            if row == 0:
                ax.set_title(event, fontsize=10, color=INK, loc="left")
            if col == 0:
                ax.set_ylabel(FACTORS[f], fontsize=8.5, color=INK_2)
    fig.suptitle("Factor behaviour around diagnostic stress windows (shaded; +/-90 days shown)",
                 x=0.01, ha="left", fontsize=13, color=INK, fontweight="bold")
    fig.text(0.01, 0.005, "Diagnostic windows only - not used as regime labels or for any estimation.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.015, 1, 1))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def save_figures(factors: pd.DataFrame, figure_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = [figure_dir / "five_factor_timeseries.png",
             figure_dir / "factor_correlation_heatmap.png",
             figure_dir / "factor_event_windows.png"]
    plot_timeseries(factors, paths[0])
    plot_correlation_heatmap(factors, paths[1])
    plot_event_windows(factors, paths[2])
    return paths
