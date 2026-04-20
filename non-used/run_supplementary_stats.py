"""
Supplementary statistical analyses on NAI event metrics.

1) Log-transformed time metrics → Mixed ANOVA / ARTool (same pipeline)
2) Delta scores (cond2 - cond1) → Mann-Whitney U between groups

Dependencies: pip install pingouin scipy pandas numpy
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats

try:
    import pingouin as pg
except ImportError:
    raise SystemExit("pingouin 필요: pip install pingouin")


TIME_METRICS = [
    "mean_sustain_s",
    "mean_reach_again_s",
    "first_reach_time_s",
    "median_sustain_s",
    "median_reach_again_s",
]

ALL_METRICS = [
    "proportion_ge_thr",
    "mean_sustain_s",
    "mean_reach_again_s",
    "first_reach_time_s",
    "median_sustain_s",
    "median_reach_again_s",
]


# ──────────────────────────────────────────────
#  Shared helpers
# ──────────────────────────────────────────────

def _clean_wide(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Pivot to wide (c1, c2) + group, drop NaN/Inf rows."""
    wide = df.pivot(index="subject", columns="condition", values=metric).reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={1: "c1", 2: "c2"})
    wide = wide[np.isfinite(wide["c1"]) & np.isfinite(wide["c2"])].copy()
    grp = df[["subject", "group"]].drop_duplicates()
    wide = wide.merge(grp, on="subject", how="left")
    return wide


def _wide_to_long(wide: pd.DataFrame, metric: str) -> pd.DataFrame:
    long = wide.melt(id_vars=["subject", "group"], value_vars=["c1", "c2"],
                     var_name="condition", value_name=metric)
    long["condition"] = long["condition"].map({"c1": 1, "c2": 2})
    return long


# ──────────────────────────────────────────────
#  Part 1: Log-transformed Mixed ANOVA / ARTool
# ──────────────────────────────────────────────

def check_normality(df: pd.DataFrame, metric: str) -> Tuple[bool, str]:
    lines, all_ok = [], True
    for grp in sorted(df["group"].unique()):
        for cond in sorted(df["condition"].unique()):
            cell = df[(df["group"] == grp) & (df["condition"] == cond)][metric].to_numpy(float)
            cell = cell[np.isfinite(cell)]
            if cell.size < 3:
                lines.append(f"  {grp} cond{cond}: n={cell.size} (too few)")
                all_ok = False
                continue
            w, p = stats.shapiro(cell)
            ok = p >= 0.05
            if not ok:
                all_ok = False
            lines.append(f"  {grp} cond{cond}: W={w:.4f} p={p:.4f} {'OK' if ok else 'FAIL'}")
    return all_ok, "\n".join(lines)


def check_homogeneity(df: pd.DataFrame, metric: str) -> Tuple[bool, str]:
    lines, all_ok = [], True
    for cond in sorted(df["condition"].unique()):
        groups = []
        for grp in sorted(df["group"].unique()):
            v = df[(df["group"] == grp) & (df["condition"] == cond)][metric].to_numpy(float)
            groups.append(v[np.isfinite(v)])
        if any(g.size < 2 for g in groups):
            lines.append(f"  cond{cond}: too few")
            all_ok = False
            continue
        stat_val, p = stats.levene(*groups)
        ok = p >= 0.05
        if not ok:
            all_ok = False
        lines.append(f"  cond{cond}: F={stat_val:.4f} p={p:.4f} {'OK' if ok else 'FAIL'}")
    return all_ok, "\n".join(lines)


def _art_transform(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    d = df.copy()
    grand = d[metric].mean()
    cell = d.groupby(["group", "condition"])[metric].mean()
    marg_g = d.groupby("group")[metric].mean()
    marg_c = d.groupby("condition")[metric].mean()
    d["_cell"] = d.apply(lambda r: cell.loc[(r["group"], r["condition"])], axis=1)
    d["_marg_g"] = d["group"].map(marg_g)
    d["_marg_c"] = d["condition"].map(marg_c)

    d["art_group"] = d[metric] - d["_cell"] + d["_marg_g"] - grand
    d["art_group_rank"] = stats.rankdata(d["art_group"])
    d["art_cond"] = d[metric] - d["_cell"] + d["_marg_c"] - grand
    d["art_cond_rank"] = stats.rankdata(d["art_cond"])
    d["art_inter"] = d[metric] - d["_marg_g"] - d["_marg_c"] + grand
    d["art_inter_rank"] = stats.rankdata(d["art_inter"])
    return d


def run_art_anova(df: pd.DataFrame, metric: str) -> Tuple[str, Optional[float]]:
    d = _art_transform(df, metric)
    lines: List[str] = []

    for label, dv, src in [
        ("Group main", "art_group_rank", "group"),
        ("Condition main", "art_cond_rank", "condition"),
        ("Interaction", "art_inter_rank", "Interaction"),
    ]:
        aov = pg.mixed_anova(data=d, dv=dv, within="condition",
                             between="group", subject="subject")
        row = aov[aov["Source"] == src]
        if not row.empty:
            lines.append(
                f"  {label:18s} F({int(row['DF1'].iloc[0])},{int(row['DF2'].iloc[0])})="
                f"{float(row['F'].iloc[0]):.4f}  p={float(row['p_unc'].iloc[0]):.4f}  "
                f"np2={float(row['np2'].iloc[0]):.4f}"
            )
    p_inter = None
    aov_i = pg.mixed_anova(data=d, dv="art_inter_rank", within="condition",
                           between="group", subject="subject")
    row_i = aov_i[aov_i["Source"] == "Interaction"]
    if not row_i.empty:
        p_inter = float(row_i["p_unc"].iloc[0])
    return "\n".join(lines), p_inter


def run_posthoc(df: pd.DataFrame, metric: str) -> str:
    lines = ["  [Post-hoc] Simple effects:"]
    for grp in sorted(df["group"].unique()):
        g = df[df["group"] == grp]
        w = g.pivot(index="subject", columns="condition", values=metric)
        if 1 not in w.columns or 2 not in w.columns:
            continue
        c1 = w[1].dropna().to_numpy(float)
        c2 = w[2].dropna().to_numpy(float)
        n = min(len(c1), len(c2))
        if n < 3:
            continue
        t_stat, p_val = stats.ttest_rel(c1[:n], c2[:n])
        d_val = pg.compute_effsize(c1[:n], c2[:n], eftype="cohen")
        lines.append(f"    {grp} (cond1→2): t={t_stat:.4f} p={p_val:.4f} d={d_val:.4f} "
                     f"{'*' if p_val < 0.05 else 'ns'}")
    for cond in sorted(df["condition"].unique()):
        c = df[df["condition"] == cond]
        groups = {grp: c[c["group"] == grp][metric].dropna().to_numpy(float)
                  for grp in sorted(c["group"].unique())}
        gnames = sorted(groups.keys())
        if len(gnames) < 2:
            continue
        a, b = groups[gnames[0]], groups[gnames[1]]
        if len(a) < 3 or len(b) < 3:
            continue
        t_stat, p_val = stats.ttest_ind(a, b)
        d_val = pg.compute_effsize(a, b, eftype="cohen")
        lines.append(f"    cond{cond} ({gnames[0]} vs {gnames[1]}): t={t_stat:.4f} p={p_val:.4f} "
                     f"d={d_val:.4f} {'*' if p_val < 0.05 else 'ns'}")
    return "\n".join(lines)


def analyze_log_metric(df_all: pd.DataFrame, metric: str) -> str:
    lines: List[str] = []
    col = f"log_{metric}"
    lines.append(f"\n{'='*70}")
    lines.append(f"LOG-TRANSFORMED: {metric}  →  ln({metric})")
    lines.append(f"{'='*70}")

    wide = _clean_wide(df_all, metric)
    # 0 이하인 값은 로그 불가 → 작은 양수로 clamp
    eps = 1e-6
    wide["c1"] = np.log(np.maximum(wide["c1"], eps))
    wide["c2"] = np.log(np.maximum(wide["c2"], eps))
    long = wide.melt(id_vars=["subject", "group"], value_vars=["c1", "c2"],
                     var_name="condition", value_name=col)
    long["condition"] = long["condition"].map({"c1": 1, "c2": 2})

    n = long["subject"].nunique()
    for grp in sorted(long["group"].unique()):
        ng = long[long["group"] == grp]["subject"].nunique()
        lines.append(f"  {grp}: n={ng}")
    lines.append(f"  total: {n}")

    if n < 4:
        lines.append("  [SKIP]")
        return "\n".join(lines)

    norm_ok, norm_detail = check_normality(long, col)
    lines.append(f"\n[1] Normality (Shapiro-Wilk):")
    lines.append(norm_detail)
    lines.append(f"  → {'PASS' if norm_ok else 'FAIL'}")

    homo_ok, homo_detail = check_homogeneity(long, col)
    lines.append(f"\n[2] Homogeneity (Levene):")
    lines.append(homo_detail)
    lines.append(f"  → {'PASS' if homo_ok else 'FAIL'}")

    if norm_ok and homo_ok:
        lines.append("\n[3] Assumptions MET → Mixed ANOVA")
        aov = pg.mixed_anova(data=long, dv=col, within="condition",
                             between="group", subject="subject")
        lines.append(aov.to_string(index=False))
        inter_row = aov[aov["Source"] == "Interaction"]
        if not inter_row.empty:
            p_i = float(inter_row["p_unc"].iloc[0])
            lines.append(f"\n  Interaction p = {p_i:.4f}")
            if p_i < 0.05:
                lines.append("  → SIGNIFICANT → Post-hoc")
                lines.append(run_posthoc(long, col))
            else:
                lines.append("  → NOT significant")
    else:
        lines.append("\n[3] Assumptions VIOLATED → ARTool")
        art_str, p_i = run_art_anova(long, col)
        lines.append(art_str)
        if p_i is not None:
            lines.append(f"\n  Interaction p = {p_i:.4f}")
            if p_i < 0.05:
                lines.append("  → SIGNIFICANT → Post-hoc")
                lines.append(run_posthoc(long, col))
            else:
                lines.append("  → NOT significant")

    return "\n".join(lines)


# ──────────────────────────────────────────────
#  Part 2: Delta scores → Mann-Whitney U
# ──────────────────────────────────────────────

def analyze_delta(df_all: pd.DataFrame, metric: str) -> str:
    lines: List[str] = []
    lines.append(f"\n{'='*70}")
    lines.append(f"DELTA (cond2 − cond1): {metric}")
    lines.append(f"{'='*70}")

    wide = _clean_wide(df_all, metric)
    wide["delta"] = wide["c2"] - wide["c1"]

    for grp in sorted(wide["group"].unique()):
        g = wide[wide["group"] == grp]
        d = g["delta"]
        lines.append(f"  {grp} (n={len(g)}):  M={d.mean():.4f}  SD={d.std():.4f}  "
                     f"Mdn={d.median():.4f}")

    ctrl = wide[wide["group"] == "control"]["delta"].to_numpy(float)
    nt = wide[wide["group"] == "neurotune"]["delta"].to_numpy(float)

    if len(ctrl) < 3 or len(nt) < 3:
        lines.append("  [SKIP] too few")
        return "\n".join(lines)

    # Mann-Whitney U
    u_stat, p_mw = stats.mannwhitneyu(ctrl, nt, alternative="two-sided")
    n1, n2 = len(ctrl), len(nt)
    r_effect = 1 - (2 * u_stat) / (n1 * n2)  # rank-biserial r
    lines.append(f"\n  Mann-Whitney U = {u_stat:.1f},  p = {p_mw:.4f},  r = {r_effect:.4f}  "
                 f"{'*' if p_mw < 0.05 else 'ns'}")

    # supplementary: Wilcoxon signed-rank within each group (cond2 vs cond1)
    lines.append("\n  Within-group Wilcoxon signed-rank (cond1 → cond2):")
    for grp in sorted(wide["group"].unique()):
        g = wide[wide["group"] == grp]
        c1 = g["c1"].to_numpy(float)
        c2 = g["c2"].to_numpy(float)
        if len(c1) < 6:
            lines.append(f"    {grp}: skipped (n={len(c1)})")
            continue
        try:
            w_stat, p_w = stats.wilcoxon(c1, c2)
            lines.append(f"    {grp} (n={len(c1)}): W={w_stat:.1f}  p={p_w:.4f}  "
                         f"{'*' if p_w < 0.05 else 'ns'}")
        except ValueError as e:
            lines.append(f"    {grp}: {e}")

    return "\n".join(lines)


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path,
                    default=Path(__file__).resolve().parent
                    / "tbr_nai_calculate" / "nai_threshold_report" / "subject_metrics.csv")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent
                    / "tbr_nai_calculate" / "nai_threshold_report" / "supplementary_stats_report.txt")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    parts: List[str] = []

    # ── Part 1: Log-transform time metrics ──
    parts.append("=" * 70)
    parts.append("PART 1:  Log-transformed time metrics → Mixed ANOVA / ARTool")
    parts.append("=" * 70)
    for m in TIME_METRICS:
        if m in df.columns:
            parts.append(analyze_log_metric(df, m))

    # ── Part 2: Delta → Mann-Whitney U ──
    parts.append("\n\n" + "=" * 70)
    parts.append("PART 2:  Delta (cond2 − cond1) → Mann-Whitney U between groups")
    parts.append("=" * 70)
    for m in ALL_METRICS:
        if m in df.columns:
            parts.append(analyze_delta(df, m))

    report = "\n".join(parts)
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    print(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(f"\n[OK] saved: {args.out}")


if __name__ == "__main__":
    main()
