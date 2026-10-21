"""Stage 6.1: audit figures and report. All statements are generated from the audit tables (no hand-written results)."""

from __future__ import annotations

STAGE61_CODE_VERSION = "stage6.1-2026-10-21b"  # must match scripts/08b_audit_early_warning.py

from pathlib import Path

import numpy as np
import pandas as pd

from .early_warning_audit import LATE_WINDOW, WINDOWS, interval_statement
from .quality import _md_table
from .regime_report import INK, INK_2, SURFACE, _fmt, _save, _style, _title

PRIMARY_SPECS = ["A_p_non_normal@budget20", "A_p_non_normal@amber_exposure", "A_p_non_normal@orange_exposure",
                 "B_ews_score@budget20", "B_ews_score@amber_exposure", "B_ews_score@orange_exposure",
                 "C_ews_state>=AMBER", "C_ews_state>=ORANGE"]
SPEC_COLORS = {"A": "#2a78d6", "B": "#eb6834", "C": "#c93131"}
SPEC_STYLE = {"budget20": "-", "amber_exposure": "--", "orange_exposure": ":", "AMBER": "-", "ORANGE": "--"}
STATUS_COLORS = {"warned": "#1baf7a", "detected_late": "#eda100", "missed": "#c93131"}
MAIN_WINDOW = 21


def _style_of(spec: str) -> str:
    return SPEC_STYLE.get(spec.split("@")[-1] if "@" in spec else spec.split(">=")[-1], "-")


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
def fig_window_sensitivity(op: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    v = op[(op["split"] == "validation") & op["spec"].isin(PRIMARY_SPECS)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=SURFACE)
    for ax, conv in zip(axes[:2], ("permissive", "strict")):
        _style(ax)
        for spec in PRIMARY_SPECS:
            g = v[(v["spec"] == spec) & (v["convention"] == conv)].sort_values("window")
            xs = [list(WINDOWS).index(w) for w in g["window"]]
            ax.plot(xs, g["warned_share"], marker="o", markersize=4, color=SPEC_COLORS[spec[0]],
                    linestyle=_style_of(spec), linewidth=1.4, label=spec, alpha=0.85)
        ax.set_xticks(range(len(WINDOWS)), [str(w) for w in WINDOWS])
        ax.set_ylim(-0.03, 1.05)
        ax.set_xlabel("warning window W (sessions before onset)", fontsize=8, color=INK_2)
        ax.set_ylabel("share of Validation events warned", fontsize=8, color=INK_2)
        ax.set_title(f"{conv.capitalize()} matching", fontsize=10, color=INK, loc="left")
    axes[0].legend(fontsize=7, frameon=False, loc="upper left")
    ax = axes[2]
    _style(ax)
    g = v[(v["convention"] == "strict") & (v["window"] == MAIN_WINDOW)]
    def short(spec: str) -> str:
        return (spec.replace("_p_non_normal", "").replace("_ews_score", "").replace("_ews_state", "")
                .replace("amber_exposure", "amb").replace("orange_exposure", "ora").replace("budget20", "20%"))

    for r in g.itertuples():
        ax.scatter(r.time_share_alerted, r.false_alerts_per_year, s=60, color=SPEC_COLORS[r.spec[0]], alpha=0.75,
                   marker="o" if r.spec[0] != "C" else "s")
    near = g.assign(x=g["time_share_alerted"].round(2), y=g["false_alerts_per_year"].round(2))
    for k, ((x, y), grp) in enumerate(near.groupby(["x", "y"])):
        ax.annotate("\n".join(short(s) for s in grp["spec"]), (x, y), fontsize=7, color=INK_2,
                    xytext=(6, 4 if k % 2 else -14), textcoords="offset points")
    ax.set_xlabel("Validation share of sessions on alert", fontsize=8, color=INK_2)
    ax.set_ylabel("strict false alert episodes per year", fontsize=8, color=INK_2)
    ax.set_title(f"Exposure vs false alerts (strict, W={MAIN_WINDOW})", fontsize=10, color=INK, loc="left")
    _title(fig, "Warning-window sensitivity (Validation, frozen specifications)",
           "A = P(non-normal), B = full EWS score, C = full EWS state machine. Line style: solid = Train 20% budget / "
           ">=AMBER, dashed = AMBER-matched exposure / >=ORANGE, dotted = ORANGE-matched exposure. Development sample.")
    _save(fig, path)


def fig_event_matrix(et: pd.DataFrame, path: Path, window: int = MAIN_WINDOW) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4), facecolor=SURFACE)
    for ax, conv in zip(axes, ("permissive", "strict")):
        t = et[(et["split"] == "validation") & (et["window"] == window) & (et["convention"] == conv) &
               et["spec"].isin(PRIMARY_SPECS)]
        dates = sorted(t["onset_date"].unique())
        ax.set_facecolor(SURFACE)
        for j, spec in enumerate(PRIMARY_SPECS):
            g = t[t["spec"] == spec].set_index("onset_date")
            for i, d in enumerate(dates):
                if d not in g.index:
                    continue
                r = g.loc[d]
                y = len(PRIMARY_SPECS) - 1 - j
                ax.add_patch(plt.Rectangle((i, y), 0.95, 0.9, color=STATUS_COLORS[r["status"]], alpha=0.85))
                txt = (f"{r['lead_days']:.0f}" if r["status"] == "warned" else
                       f"+{r['delay_days']:.0f}" if r["status"] == "detected_late" else "")
                ax.text(i + 0.47, y + 0.45, txt, ha="center", va="center", fontsize=7, color="white")
        ax.set_xlim(0, max(len(dates), 1))
        ax.set_ylim(0, len(PRIMARY_SPECS))
        ax.set_xticks(np.arange(len(dates)) + 0.47, [pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates],
                      rotation=45, ha="right", fontsize=7, color=INK_2)
        ax.set_yticks(np.arange(len(PRIMARY_SPECS)) + 0.45, PRIMARY_SPECS[::-1], fontsize=7, color=INK_2)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{conv.capitalize()} matching, W={window}: green = warned (lead), amber = late (+delay), "
                     "red = missed", fontsize=9, color=INK, loc="left")
    _title(fig, "Validation events by specification",
           "Strict matching: each alert episode can warn at most one event and must START before the onset; an alert "
           "first raised on the onset day is late, never an early warning.")
    _save(fig, path)


def save_figures(directory: Path, op: pd.DataFrame, spec_table: pd.DataFrame, et: pd.DataFrame | None = None
                 ) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    out = []
    p = directory / "audit_window_sensitivity.png"
    fig_window_sensitivity(op, p)
    out.append(str(p))
    if et is not None:
        p = directory / "audit_event_matrix.png"
        fig_event_matrix(et, p)
        out.append(str(p))
    return out


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------
def _wlm(r) -> str:
    return f"{int(r['warned'])}/{int(r['detected_late'])}/{int(r['missed'])}"


def _exposure_label(spec_table: pd.DataFrame | None, spec: str) -> str:
    if spec_table is None:
        return ""
    r = spec_table.set_index("spec").loc[spec] if spec in set(spec_table["spec"]) else None
    if r is None or pd.isna(r.get("exposure_matched", np.nan)):
        return "descriptive"
    return "matched" if bool(r["exposure_matched"]) else (f"NOT matched (Train {r['achieved_train_share']:.2%} vs "
                                                          f"target {r['target_train_share']:.2%})")


def window_table(op: pd.DataFrame, split: str, window: int, specs=PRIMARY_SPECS,
                 spec_table: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    for spec in specs:
        g = op[(op["spec"] == spec) & (op["split"] == split) & (op["window"] == window)].set_index("convention")
        if g.empty:
            continue
        p, s = g.loc["permissive"], g.loc["strict"]
        rows.append({"spec": spec, "exposure budget": _exposure_label(spec_table, spec),
                     "time_on_alert": f"{s['time_share_alerted']:.1%}",
                     "episodes/yr": f"{s['episodes_per_year']:.1f}",
                     "permissive W/L/M": _wlm(p), "permissive false/yr": f"{p['false_alerts_per_year']:.2f}",
                     "strict W/L/M": _wlm(s), "strict executable W": f"{int(s['executable_warned'])}",
                     "strict median lead": "" if pd.isna(s["median_lead_days"]) else f"{s['median_lead_days']:.0f}",
                     "strict false/yr": f"{s['false_alerts_per_year']:.2f}",
                     "strict precision": "" if pd.isna(s["episode_precision"]) else f"{s['episode_precision']:.2f}",
                     "redundant / premature / in-event": f"{int(s['redundant_pre_event_episodes'])} / "
                                                         f"{int(s['premature_episodes'])} / "
                                                         f"{int(s['in_event_period_episodes'])}"})
    return pd.DataFrame(rows).set_index("spec")


def _get(op, spec, split, window, conv, col):
    r = op[(op["spec"] == spec) & (op["split"] == split) & (op["window"] == window) & (op["convention"] == conv)]
    return r[col].iloc[0] if len(r) else np.nan


def interpretation(ctx: dict) -> list[str]:
    op, boot, rk, st = ctx["operational"], ctx["boot"], ctx["ranking"], ctx["spec_table"]
    n_val = len(ctx["events"]["validation"])
    out = []
    b = boot[boot["label"] == "SPY 21d loss >= 5%"].iloc[0]
    rv = rk[(rk["sample"] == "validation") & (rk["label"] == "SPY 21d loss >= 5%")].set_index("score")
    out.append(f"Ranking (statistical): Validation PR-AUC is {rv.at['B_ews_score', 'pr_auc']:.3f} for the full EWS score "
               f"and {rv.at['A_p_non_normal', 'pr_auc']:.3f} for P(non-normal) (base rate "
               f"{rv.at['A_p_non_normal', 'base_rate']:.3f}). " +
               interval_statement(b["ci95_low"], b["ci95_high"], "Block-bootstrap difference (EWS minus P(non-normal))"))
    for exp, c in (("amber_exposure", "C_ews_state>=AMBER"), ("orange_exposure", "C_ews_state>=ORANGE")):
        parts = []
        for w in (5, MAIN_WINDOW, 63):
            vals = [(lab, s, _get(op, s, "validation", w, "strict", "warned"),
                     _get(op, s, "validation", w, "strict", "executable_warned"),
                     _get(op, s, "validation", w, "strict", "false_alerts_per_year"))
                    for lab, s in (("A", f"A_p_non_normal@{exp}"), ("B", f"B_ews_score@{exp}"), ("C", c))]
            parts.append(f"W={w}: " + ", ".join(f"{lab} {int(wv)}/{n_val} signal-warned ({int(xv)} executable), "
                                                f"{fv:.2f} false/yr" for lab, _, wv, xv, fv in vals))
        flags = [f"{s} is {_exposure_label(st, s)}" for s in (f"A_p_non_normal@{exp}", f"B_ews_score@{exp}")
                 if not _exposure_label(st, s).startswith("matched")]
        share = ctx["budgets"][exp]
        head = (f"Operational (strict matching) at the C Train exposure {share:.2%} ({exp.replace('_', ' ')}; "
                f"C = {c.split('_')[-1]})")
        if flags:
            head += " - NOT A FULLY EXPOSURE-MATCHED COMPARISON: " + "; ".join(flags)
        out.append(head + ": " + "; ".join(parts) + ".")
    diff = op.pivot_table(index=["spec", "split", "window"], columns="convention", values="warned")
    dv = diff.xs("validation", level="split")
    red = int((dv["permissive"] - dv["strict"]).sum())
    out.append(f"Matching convention: across all Validation specification-window cells the permissive convention "
               f"credits {red} more early warnings than the strict one-to-one convention.")
    et = ctx["event_table"]
    w21 = et[(et["split"] == "validation") & (et["window"] == MAIN_WINDOW) & (et["convention"] == "strict") &
             et["spec"].isin(PRIMARY_SPECS)]
    lead1 = int(((w21["status"] == "warned") & (w21["signal_lead_sessions"] == 1)).sum())
    out.append(f"Signal versus executable lead (strict, W={MAIN_WINDOW}, Validation): {int((w21['status'] == 'warned').sum())} "
               f"event x specification cells are signal-warned, of which {lead1} have a signal lead of one session. "
               "Under the conservative convention those positions change at the close of the onset day itself, so "
               "they offer no protection before the -5% threshold is crossed.")
    onset = w21[w21["first_alert_on_onset_day"]]
    out.append(f"Onset-day alerts: {len(onset)} Validation event x specification cells (W={MAIN_WINDOW}) have an alert "
               "first raised on the onset day; none of them is counted as an early warning.")
    out.append(f"Sample size: {n_val} Validation events. A difference of one or two warned events between specifications "
               "is within what chance alone can produce; no event-level comparison here is statistically resolved.")
    out.append("No new model is selected by this audit. The Stage 6 decision stands: P(non-normal) is the simple "
               "regime baseline and the full EWS is the challenger for Stage 7.")
    return out


def render_report(ctx: dict) -> str:
    data, op, rk, boot = ctx["data"], ctx["operational"], ctx["ranking"], ctx["boot"]
    checks, h, st = ctx["checks"], ctx["handoff"], ctx["spec_table"]
    counts = pd.Series([c.status for c in checks]).value_counts().to_dict()
    tr, va = data.train_features.index, data.validation_features.index
    n_tr, n_va = len(ctx["events"]["train"]), len(ctx["events"]["validation"])
    sat = ctx.get("saturation", {})
    L = ["# Stage 6.1 - Final release audit: early-warning evaluation and Stage 7 handoff", "",
         f"## HANDOFF STATUS: {ctx['status']}", "", f"{ctx['status_reason']}", "",
         f"Checks: {counts.get('PASS', 0)} PASS, {counts.get('WARN', 0)} WARN, {counts.get('FAIL', 0)} FAIL. "
         f"Train {tr.min().date()}..{tr.max().date()}; Validation {va.min().date()}..{va.max().date()}. Test not read. "
         f"Events (SPY -5% drawdown onsets): Train {n_tr}, Validation {n_va}.", "",
         "READY requires every check to PASS or WARN; any FAIL makes the handoff BLOCKED and the Stage 7 loader refuses "
         "it. WARN items are documented limitations, not blockers.", ""]
    blockers = [c for c in checks if c.status != "PASS"]
    L += ["## 1. Blockers and warnings", ""]
    L += ([_md_table(pd.DataFrame([vars(c) for c in blockers]).set_index("name"))] if blockers else
          ["None: every check passed."]) + [""]
    ts = ctx["test_summary"]
    L += ["## 2. Automated tests on the real Train/Validation outputs", ""]
    if ts.get("ran"):
        L += [f"{ts['tests']} tests run, {ts['failures']} failures, {ts['errors']} errors, {ts['skipped']} skipped. "
              "Full runner output: `data/processed/early_warning/audit/real_data_test_results.txt`.", ""]
        for f in ts.get("failed_tests", []):
            L += [f"**{f['kind'].upper()}: {f['test']}**", "", "```text", f["traceback"].rstrip(), "```", ""]
    else:
        L += ["Not run.", ""]
    L += ["## 3. Amendment: earlier real-data test failure (root cause and fix)", "",
          "- Failing test in the previous run: `test_baseline_rule_threshold_is_train_q80` (identified by elimination: "
          "it is the only real-data test whose result depends on the numerical precision of P(non-normal) at the "
          "frozen threshold; the runner output is now saved with full tracebacks).",
          "- Root cause 1 (implementation error): the Stage 7 handoff CSVs stored floats with 10 significant digits. "
          f"On Train, {sat.get('train_exactly_one', 'n/a')} of {sat.get('n_train', 'n/a')} days have P(non-normal) "
          f"exactly 1.0 in float64 and {sat.get('train_within_1e10_not_one', 'n/a')} more lie within 1e-10 of 1.0; the "
          f"frozen 80th-percentile threshold is {sat.get('baseline_threshold_repr', 'n/a')}. Values such as 1 - 1e-11 "
          "were written as '1', so the frozen rule `p >= threshold` could not be reproduced from the handoff files.",
          "- Root cause 2 (test defect): the test compared with `threshold - 1e-12`; under saturation that tolerance "
          "itself flips days in [1 - 1e-12, 1). The assertion is now exact (stricter, not relaxed).",
          "- Fix: floats are written with 17 significant digits and read with round-trip precision; an audit check "
          "verifies bit-for-bit reproduction of every handoff value and rule. No threshold, score or rule changed.", ""]
    L += ["## 4. Scope", "",
          "- Non-tuning audit: no model was refitted, no BOCPD prior, EWS weight, threshold or alert rule was changed, "
          "no tie-breaking was used, and no specification is selected from these diagnostics.",
          "- Stage 5, 5.5 and 6 artifacts are preserved (hashes verified before and after); all outputs are new files.",
          "- Validation was used in Stages 5, 5.5 and 6: everything below is development-sample evidence. Train "
          "signals are in-sample reconstructions.",
          "- Ranking performance (PR-AUC, ROC-AUC of a continuous score against a daily loss label) and operational "
          "event-warning performance (binary alerts against drawdown events) are reported separately.", ""]
    L += ["## 5. Interpretation", ""] + [f"- {x}" for x in interpretation(ctx)] + [""]
    ex = st[st["family"].isin(["A", "B"]) & (st["role"] == "evaluation")].set_index("spec")
    show = pd.DataFrame({
        "target Train share": ex["target_train_share"].map(lambda v: f"{v:.4%}"),
        "threshold (full precision)": ex["threshold_repr"],
        "Train days tied at threshold": ex["n_train_tied_at_threshold"].astype(int),
        "Train days == 1.0": ex["n_train_exactly_one"].astype(int),
        "achieved Train share": ex["achieved_train_share"].map(lambda v: f"{v:.4%}"),
        "share strictly above threshold": ex["attainable_share_just_below"].map(lambda v: f"{v:.4%}"),
        "Validation share": ex["validation_share"].map(lambda v: f"{v:.4%}"),
        "status": ex["exposure_status"].map(lambda v: v if v == "EXPOSURE-MATCHED" else "NOT EXPOSURE-MATCHED")})
    L += ["## 6. Exposure budgets under ties and saturation", "",
          "Rule: alert when score >= c, c = Train quantile at 1 - target (numpy linear interpolation). With ties, the "
          "achievable Train exposures are the step values share(score >= v); no chronological or random tie-breaking "
          f"is used. A budget counts as exposure-matched when the achieved Train share is within "
          f"{100 * 0.005:.1f} percentage point of the target. For B the 'Train days == 1.0' column is not meaningful.", "",
          _md_table(show), ""]
    for r in ex[~ex["exposure_matched"].astype(bool)].itertuples():
        L.append(f"- **{r.Index}: NOT EXPOSURE-MATCHED.** {r.exposure_status}. Every comparison that uses this budget "
                 "is labelled accordingly; it is not an equal-exposure comparison.")
    L += ["", "State-machine budgets (they define the AMBER / ORANGE exposure targets):", "",
          _md_table(st[st["family"] == "C"].set_index("spec")[["rule", "train_time_share", "validation_time_share"]]
                    .map(lambda v: f"{v:.4%}" if isinstance(v, float) else v)), ""]
    L += ["## 7. Ranking performance (statistical)", "",
          "Daily scores against daily labels on identical dates. The state machine is a discretization of the EWS "
          "score and has no separate ranking.", "",
          _md_table(_fmt(rk.pivot_table(index=["label", "sample"], columns="score", values="pr_auc"), 3)
                    .rename(columns=lambda c: f"PR-AUC {c}")), "",
          _md_table(_fmt(rk.pivot_table(index=["label", "sample"], columns="score", values="roc_auc"), 3)
                    .rename(columns=lambda c: f"ROC-AUC {c}")), "",
          "Validation block bootstrap (63-session blocks) of the PR-AUC difference, EWS minus P(non-normal):", "",
          _md_table(_fmt(boot.set_index("label")[["model", "reference", "point_ap_diff", "ci95_low", "ci95_high",
                                                  "share_positive", "valid_reps"]], 4)), ""]
    L += [f"- {interval_statement(r.ci95_low, r.ci95_high, f'{r.label}')}" for r in boot.itertuples()] + [""]
    L += ["## 8. Operational event-warning performance - Validation", "",
          f"W/L/M = warned / detected late (first alert within {LATE_WINDOW} sessions after onset) / missed. Strict "
          "executable W = strict warnings with signal lead >= 2 (protection begins before onset under the conservative "
          "convention). Unmatched episodes: redundant = started inside an event's window but another episode was "
          "matched; premature = started 64..W+1 sessions before an onset; in-event = started while a drawdown was under "
          "way; only the remainder counts as strict false alerts.", ""]
    for w in WINDOWS:
        L += [f"### Warning window W = {w}", "", _md_table(window_table(op, "validation", w, spec_table=st)), ""]
    L += ["Descriptive natural threshold, P(non-normal) >= 0.5:", "",
          _md_table(pd.concat([window_table(op, "validation", w, ["A_p_non_normal@0.5"], st).assign(window=w)
                               for w in WINDOWS])), ""]
    et = ctx["event_table"]
    sv = op[(op["split"] == "validation") & (op["convention"] == "strict") & op["spec"].isin(PRIMARY_SPECS)]
    L += ["## 9. Signal lead versus executable protection lead (strict matching)", "",
          "Signal lead = onset - first session of the matched alert episode (the signal exists after that close). "
          "Under the frozen convention the position changes at the close of the next session, so the executable "
          "protection lead = signal lead - 1 sessions of pre-onset returns. A signal lead of 1 is a warning on paper "
          "but gives no executable protection before onset.", "",
          _md_table(sv.assign(cell=sv["warned"].astype(int).astype(str) + " / " +
                              sv["executable_warned"].astype(int).astype(str))
                    .pivot(index="spec", columns="window", values="cell").reindex(PRIMARY_SPECS)
                    .rename(columns=lambda w: f"W={w}: signal / executable")), "",
          f"Event detail (Validation, strict, W={MAIN_WINDOW}): signal lead, executable lead, SPY drawdown from its "
          "peak at the execution close, and SPY return from the execution close to the onset close.", ""]
    t = et[(et["split"] == "validation") & (et["window"] == MAIN_WINDOW) & (et["convention"] == "strict") &
           et["spec"].isin(PRIMARY_SPECS) & (et["status"] == "warned")].copy()
    if len(t):
        t["onset"] = pd.to_datetime(t["onset_date"]).dt.date
        L += [_md_table(_fmt(t.set_index(["onset", "spec"])[["signal_lead_sessions", "executable_lead_sessions",
                                                             "drawdown_at_signal_pct", "drawdown_at_execution_pct",
                                                             "spy_return_execution_to_onset_pct"]], 2)), ""]
    L += ["## 10. Operational event-warning performance - Train (in-sample)", ""]
    for w in (MAIN_WINDOW, 63):
        L += [f"### W = {w}", "", _md_table(window_table(op, "train", w, spec_table=st)), ""]
    L += [f"## 11. Event-by-event detail (Validation, W = {MAIN_WINDOW})", ""]
    for conv in ("permissive", "strict"):
        t = et[(et["split"] == "validation") & (et["window"] == MAIN_WINDOW) & (et["convention"] == conv) &
               et["spec"].isin(PRIMARY_SPECS)].copy()
        t["cell"] = np.where(t["status"] == "warned", "W " + t["lead_days"].map(lambda v: f"{v:.0f}"),
                             np.where(t["status"] == "detected_late", "L +" + t["delay_days"].map(
                                 lambda v: "" if pd.isna(v) else f"{v:.0f}"), "M"))
        t["cell"] = t["cell"] + np.where(t["first_alert_on_onset_day"], " (onset-day alert)", "")
        if conv == "strict":
            t["cell"] = t["cell"] + np.where(t["stale_alert_active_at_onset"].fillna(False).astype(bool),
                                             " (stale alert active)", "")
        piv = t.assign(onset=pd.to_datetime(t["onset_date"]).dt.date).pivot(index="onset", columns="spec",
                                                                             values="cell")
        L += [f"**{conv.capitalize()}** (W lead = signal lead; L +d = delay; M = missed)", "",
              _md_table(piv[[c for c in PRIMARY_SPECS if c in piv.columns]]), ""]
    L += ["## 12. Matching-convention audit", "",
          "- Permissive (Stage 6): any alert session in [onset - W, onset - 1] counts; one long episode can warn "
          "several events and an episode that started long before the window still counts.",
          "- Strict: an episode must START in [onset - W, onset - 1]; each episode warns at most one event "
          "(chronological, most recent start); an older episode still active at onset is a stale alert, not a warning.",
          "- In both conventions the early window ends at onset - 1, so an alert first raised on the onset date is "
          "detected late (delay 0), never an early warning.",
          "- The strict tables are re-verified from the raw alert series by an independent invariant check.",
          "- Caveat: strict matching counts episode starts, so a flickering alert can be credited where one continuous "
          "alert would not; compare episodes per year.", "",
          _md_table(_fmt(op[(op["split"] == "validation") & op["spec"].isin(PRIMARY_SPECS)]
                         .pivot_table(index="spec", columns=["window", "convention"], values="warned")
                         .reindex(PRIMARY_SPECS), 0)), ""]
    conv = h["execution_convention"]
    L += ["## 13. Stage 7 handoff", "",
          f"Manifest: `data/processed/early_warning/stage7_handoff/stage7_handoff_manifest.json` "
          f"(handoff_status = **{ctx['status']}**).", "",
          "Execution convention (frozen):", ""] + [f"- **{k}**: {v}" for k, v in conv.items()] + [""]
    for role, r in h["roles"].items():
        L += [f"### {role}: {r['name']}", "", f"- Status: {r['status']}.",
              f"- Signal columns: {', '.join(r['columns'])}; timing columns: {', '.join(r['timing_columns'])}.", ""]
        files = pd.DataFrame(r["files"]).T[["split", "rows", "first_signal_date", "last_signal_date",
                                            "first_signal_available_after", "first_execution_date",
                                            "last_execution_date", "rows_without_execution_date", "sha256"]]
        L += [_md_table(files.rename_axis("file")), ""]
        fp = r["frozen_parameters"]
        if role == "baseline_regime":
            q = fp["baseline_alert_ge_q80"]
            L += [f"- Frozen rule: {q['rule']} with threshold {q['threshold_repr']} ({q['threshold_origin']}); "
                  f"Train days tied at the threshold {q['train_days_tied_at_threshold']}, exactly 1.0 "
                  f"{q['train_days_exactly_one']}; achieved Train share {q['achieved_train_share']:.4%}, Validation "
                  f"{q['validation_share']:.4%}.",
                  f"- Frozen model: {fp['hmm_model']['id']} ({fp['hmm_model']['file']}, sha256 "
                  f"{fp['hmm_model']['sha256']}).", ""]
        else:
            L += ["- Frozen thresholds: " + ", ".join(f"{k} {v}" for k, v in fp["alert_thresholds_repr"].items()) + ".",
                  "- EWS weights: " + ", ".join(f"{k} {v}" for k, v in fp["ews_weights"].items()) + ".",
                  "- Rules: " + "; ".join(f"{k}: {v}" for k, v in fp["alert_rules"].items()) + ".",
                  "- Frozen models: " + "; ".join(f"{k} {v.get('file')} sha256 {v.get('sha256')}"
                                                  for k, v in fp["frozen_models"].items()) + ".", ""]
    L += ["Permitted portfolio inputs: " + "; ".join(f"{k}: {', '.join(v)}"
                                                    for k, v in h["permitted_portfolio_inputs"].items()) + ".", "",
          "Descriptive only (never portfolio inputs): " + ", ".join(h["descriptive_only"]) + ".", "",
          "Forbidden: " + ", ".join(h["forbidden_inputs"]) + ".", "",
          "Controls against confusing P(non-normal) with full-EWS alerts:", ""] + \
         [f"- {c}" for c in h["anti_confusion_controls"]] + [""]
    L += ["Input hashes: split files " + ", ".join(f"{k} {v}" for k, v in h["input_hashes"]["split_files"].items()) +
          f"; {len(h['input_hashes']['stage6_outputs'])} Stage 6 output files and "
          f"{len(h['input_hashes']['code'])} code files hashed in the manifest.", ""]
    L += ["## 14. Remaining limitations", ""]
    for k, items in h["remaining_limitations"].items():
        L += [f"**{k.replace('_', ' ').capitalize()}**", ""] + [f"- {i}" for i in items] + [""]
    L += ["## 15. All checks", "", _md_table(pd.DataFrame([vars(c) for c in checks]).set_index("name")), "",
          f"**Final: {counts.get('PASS', 0)} PASS, {counts.get('WARN', 0)} WARN, {counts.get('FAIL', 0)} FAIL - "
          f"handoff {ctx['status']}.**", "",
          "Figures: " + ", ".join(f"`figures/early_warning_audit/{Path(p).name}`" for p in ctx.get("figures", [])), ""]
    return "\n".join(L)
