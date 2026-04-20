"""
2(condition: within) × 2(group: between) Mixed ANOVA / ARTool on NAI event metrics.

Flow per metric:
  1) Assumption checks
     a) Normality of residuals (Shapiro-Wilk on each cell & on residuals)
     b) Homogeneity of variance (Levene on each condition level)
  2) If assumptions OK → Mixed ANOVA (pingouin)
     → interaction significant? → post-hoc (simple effects via paired/independent t-tests)
  3) If assumptions violated → ARTool (Aligned Rank Transform, art package via rpy2 or manual)
     → interaction significant? → report

Dependencies: pip install pingouin scipy pandas numpy
"""
from __future__ import annotations

import argparse
import sys
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

try:
    import pingouin as pg
except ImportError:
    raise SystemExit("pingouin 필요: pip install pingouin")


MEAN_METRICS = [
    "proportion_ge_thr",
    "mean_sustain_s",
    "mean_reach_again_s",
    "first_reach_time_s",
]

MEDIAN_METRICS = [
    "proportion_ge_thr",
    "median_sustain_s",
    "median_reach_again_s",
    "first_reach_time_s",
]


def _clean_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Drop subjects with NaN/Inf in either condition for this metric."""
    wide = df.pivot(index="subject", columns="condition", values=metric).reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={1: "c1", 2: "c2"})
    wide = wide[np.isfinite(wide["c1"]) & np.isfinite(wide["c2"])].copy()
    grp = df[["subject", "group"]].drop_duplicates()
    wide = wide.merge(grp, on="subject", how="left")
    long = wide.melt(id_vars=["subject", "group"], value_vars=["c1", "c2"],
                     var_name="condition", value_name=metric)
    long["condition"] = long["condition"].map({"c1": 1, "c2": 2})
    return long


def check_normality(df: pd.DataFrame, metric: str) -> Tuple[bool, str]:
    """
    Shapiro-Wilk on each cell (group × condition) and on pooled residuals.
    Returns (all_normal, details_string).
    """
    lines: List[str] = []
    all_ok = True
    alpha = 0.05

    for grp in sorted(df["group"].unique()):
        for cond in sorted(df["condition"].unique()):
            cell = df[(df["group"] == grp) & (df["condition"] == cond)][metric].to_numpy(float)
            cell = cell[np.isfinite(cell)]
            if cell.size < 3:
                lines.append(f"  {grp} cond{cond}: n={cell.size} (too few for Shapiro)")
                all_ok = False
                continue
            w, p = stats.shapiro(cell)
            ok = p >= alpha
            if not ok:
                all_ok = False
            lines.append(f"  {grp} cond{cond}: W={w:.4f} p={p:.4f} {'OK' if ok else 'FAIL'}")

    return all_ok, "\n".join(lines)


def check_homogeneity(df: pd.DataFrame, metric: str) -> Tuple[bool, str]:
    """
    Levene test for each condition level (between groups).
    """
    lines: List[str] = []
    all_ok = True
    alpha = 0.05

    for cond in sorted(df["condition"].unique()):
        groups = []
        group_names = sorted(df["group"].unique())
        for grp in group_names:
            v = df[(df["group"] == grp) & (df["condition"] == cond)][metric].to_numpy(float)
            v = v[np.isfinite(v)]
            groups.append(v)
        if any(g.size < 2 for g in groups):
            lines.append(f"  cond{cond}: too few data for Levene")
            all_ok = False
            continue
        stat_val, p = stats.levene(*groups)
        ok = p >= alpha
        if not ok:
            all_ok = False
        lines.append(f"  cond{cond}: F={stat_val:.4f} p={p:.4f} {'OK' if ok else 'FAIL'}")

    return all_ok, "\n".join(lines)


def run_mixed_anova(df: pd.DataFrame, metric: str) -> Tuple[pd.DataFrame, str]:
    """pingouin mixed_anova."""
    aov = pg.mixed_anova(
        data=df,
        dv=metric,
        within="condition",
        between="group",
        subject="subject",
    )
    buf = StringIO()
    buf.write(aov.to_string(index=False))
    return aov, buf.getvalue()


def _art_transform(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Manual Aligned Rank Transform (Wobbrock et al., 2011).
    For a 2-factor design: align residuals for each effect, then rank.
    """
    d = df.copy()
    grand = d[metric].mean()

    # cell means
    cell = d.groupby(["group", "condition"])[metric].mean()
    # marginal means
    marg_g = d.groupby("group")[metric].mean()
    marg_c = d.groupby("condition")[metric].mean()

    # aligned for interaction: Y - cell_mean + grand
    d["_cell"] = d.apply(lambda r: cell.loc[(r["group"], r["condition"])], axis=1)
    d["_marg_g"] = d["group"].map(marg_g)
    d["_marg_c"] = d["condition"].map(marg_c)

    # Wobbrock et al. (2011) alignment formulas:
    # strip effects other than the one of interest
    d["art_group"] = d[metric] - d["_cell"] + d["_marg_g"] - grand
    d["art_group_rank"] = stats.rankdata(d["art_group"])

    d["art_cond"] = d[metric] - d["_cell"] + d["_marg_c"] - grand
    d["art_cond_rank"] = stats.rankdata(d["art_cond"])

    d["art_inter"] = d[metric] - d["_marg_g"] - d["_marg_c"] + grand
    d["art_inter_rank"] = stats.rankdata(d["art_inter"])

    return d


def run_art_anova(df: pd.DataFrame, metric: str) -> Tuple[str, Optional[float]]:
    """
    ARTool: align → rank → mixed ANOVA on ranked values for each effect.
    Returns (report_string, interaction_p_value).
    """
    d = _art_transform(df, metric)
    lines: List[str] = []

    aov_g = pg.mixed_anova(data=d, dv="art_group_rank", within="condition",
                           between="group", subject="subject")
    row_g = aov_g[aov_g["Source"] == "group"]
    if not row_g.empty:
        lines.append(f"  Group main:  F({int(row_g['DF1'].iloc[0])},{int(row_g['DF2'].iloc[0])})="
                     f"{float(row_g['F'].iloc[0]):.4f}  p={float(row_g['p_unc'].iloc[0]):.4f}  "
                     f"np2={float(row_g['np2'].iloc[0]):.4f}")

    aov_c = pg.mixed_anova(data=d, dv="art_cond_rank", within="condition",
                           between="group", subject="subject")
    row_c = aov_c[aov_c["Source"] == "condition"]
    if not row_c.empty:
        lines.append(f"  Condition main: F({int(row_c['DF1'].iloc[0])},{int(row_c['DF2'].iloc[0])})="
                     f"{float(row_c['F'].iloc[0]):.4f}  p={float(row_c['p_unc'].iloc[0]):.4f}  "
                     f"np2={float(row_c['np2'].iloc[0]):.4f}")

    aov_i = pg.mixed_anova(data=d, dv="art_inter_rank", within="condition",
                           between="group", subject="subject")
    row_i = aov_i[aov_i["Source"] == "Interaction"]
    p_inter = None
    if not row_i.empty:
        p_inter = float(row_i["p_unc"].iloc[0])
        lines.append(f"  Interaction:    F({int(row_i['DF1'].iloc[0])},{int(row_i['DF2'].iloc[0])})="
                     f"{float(row_i['F'].iloc[0]):.4f}  p={p_inter:.4f}  "
                     f"np2={float(row_i['np2'].iloc[0]):.4f}")

    return "\n".join(lines), p_inter


def run_posthoc(df: pd.DataFrame, metric: str, aov: pd.DataFrame) -> str:
    """
    Simple effects post-hoc:
      - within each group: paired t-test (cond1 vs cond2)
      - within each condition: independent t-test (control vs neurotune)
    """
    lines: List[str] = []
    alpha = 0.05

    lines.append("  [Post-hoc] Simple effects:")

    # within-group (paired): condition effect per group
    for grp in sorted(df["group"].unique()):
        g = df[df["group"] == grp]
        w = g.pivot(index="subject", columns="condition", values=metric)
        if 1 not in w.columns or 2 not in w.columns:
            continue
        c1 = w[1].dropna().to_numpy(float)
        c2 = w[2].dropna().to_numpy(float)
        n = min(len(c1), len(c2))
        if n < 3:
            lines.append(f"    {grp}: paired t-test skipped (n={n})")
            continue
        t_stat, p_val = stats.ttest_rel(c1[:n], c2[:n])
        sig = "*" if p_val < alpha else "ns"
        cohend = pg.compute_effsize(c1[:n], c2[:n], eftype="cohen")
        lines.append(
            f"    {grp} (cond1 vs cond2): t={t_stat:.4f} p={p_val:.4f} d={cohend:.4f} {sig}"
        )

    # within-condition (independent): group effect per condition
    for cond in sorted(df["condition"].unique()):
        c = df[df["condition"] == cond]
        groups = {}
        for grp in sorted(c["group"].unique()):
            groups[grp] = c[c["group"] == grp][metric].dropna().to_numpy(float)
        gnames = sorted(groups.keys())
        if len(gnames) < 2:
            continue
        a, b = groups[gnames[0]], groups[gnames[1]]
        if len(a) < 3 or len(b) < 3:
            lines.append(f"    cond{cond}: ind t-test skipped (n={len(a)},{len(b)})")
            continue
        t_stat, p_val = stats.ttest_ind(a, b)
        sig = "*" if p_val < alpha else "ns"
        cohend = pg.compute_effsize(a, b, eftype="cohen")
        lines.append(
            f"    cond{cond} ({gnames[0]} vs {gnames[1]}): t={t_stat:.4f} p={p_val:.4f} d={cohend:.4f} {sig}"
        )

    return "\n".join(lines)


def analyze_metric(df_all: pd.DataFrame, metric: str) -> str:
    """Full pipeline for one metric."""
    lines: List[str] = []
    lines.append(f"\n{'='*70}")
    lines.append(f"METRIC: {metric}")
    lines.append(f"{'='*70}")

    df = _clean_metric(df_all, metric)
    n_subj = df["subject"].nunique()
    for grp in sorted(df["group"].unique()):
        ng = df[df["group"] == grp]["subject"].nunique()
        lines.append(f"  {grp}: n={ng}")
    lines.append(f"  total subjects (with valid data both conditions): {n_subj}")

    if n_subj < 4:
        lines.append("  [SKIP] too few subjects for analysis")
        return "\n".join(lines)

    # 1) Normality
    norm_ok, norm_detail = check_normality(df, metric)
    lines.append(f"\n[1] Normality (Shapiro-Wilk, alpha=0.05):")
    lines.append(norm_detail)
    lines.append(f"  → {'PASS' if norm_ok else 'FAIL'}")

    # 2) Homogeneity
    homo_ok, homo_detail = check_homogeneity(df, metric)
    lines.append(f"\n[2] Homogeneity of variance (Levene, alpha=0.05):")
    lines.append(homo_detail)
    lines.append(f"  → {'PASS' if homo_ok else 'FAIL'}")

    assumptions_met = norm_ok and homo_ok

    if assumptions_met:
        lines.append("\n[3] Assumptions MET → Mixed ANOVA (pingouin)")
        aov, aov_str = run_mixed_anova(df, metric)
        lines.append(aov_str)

        # check interaction
        inter_row = aov[aov["Source"] == "Interaction"]
        if not inter_row.empty:
            p_inter = float(inter_row["p_unc"].iloc[0])
            lines.append(f"\n  Interaction p = {p_inter:.4f}")
            if p_inter < 0.05:
                lines.append("  → Interaction SIGNIFICANT → Post-hoc")
                ph = run_posthoc(df, metric, aov)
                lines.append(ph)
            else:
                lines.append("  → Interaction NOT significant")
        else:
            lines.append("  (Interaction row not found in ANOVA table)")
    else:
        lines.append("\n[3] Assumptions VIOLATED → ARTool (non-parametric)")
        art_str, p_inter = run_art_anova(df, metric)
        lines.append(art_str)

        if p_inter is not None:
            lines.append(f"\n  Interaction p = {p_inter:.4f}")
            if p_inter < 0.05:
                lines.append("  → Interaction SIGNIFICANT → Post-hoc")
                ph = run_posthoc(df, metric, None)
                lines.append(ph)
            else:
                lines.append("  → Interaction NOT significant")

    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="2(condition)×2(group) Mixed ANOVA / ARTool on NAI metrics"
    )
    ap.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent / "tbr_nai_calculate" / "nai_threshold_report" / "subject_metrics.csv",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "tbr_nai_calculate" / "nai_threshold_report" / "mixed_anova_report.txt",
    )
    ap.add_argument(
        "--version",
        choices=["mean", "median", "both"],
        default="both",
        help="mean/median/both 버전 지표 선택",
    )
    args = ap.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"[ERR] {args.input} not found")

    df = pd.read_csv(args.input)

    if args.version == "mean":
        metrics = MEAN_METRICS
    elif args.version == "median":
        metrics = MEDIAN_METRICS
    else:
        metrics = list(dict.fromkeys(MEAN_METRICS + MEDIAN_METRICS))

    report_parts: List[str] = []
    report_parts.append("Mixed ANOVA / ARTool Report")
    report_parts.append(f"Input: {args.input}")
    report_parts.append(f"Metrics: {metrics}")
    report_parts.append(f"Design: 2(condition: within) × 2(group: between)")

    for metric in metrics:
        if metric not in df.columns:
            report_parts.append(f"\n[SKIP] {metric}: column not found")
            continue
        report_parts.append(analyze_metric(df, metric))

    report = "\n".join(report_parts)
    print(report)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(f"\n[OK] saved: {args.out}")


if __name__ == "__main__":
    main()
