"""
TBR Delta (condition2 − condition1) focused analysis.

Data source: tbr_nai_calculate_7ch_mne (7ch MNE, ratio-of-means TBR)
Warmup: 10s removed
S23: excluded

Analyses:
  1) Cell descriptives (group × condition) for TBR mean
  2) Baseline (c1) balance check
  3) Delta descriptives + direction
  4) Mann-Whitney U on delta (two-tailed + one-tailed)
  5) Within-group Wilcoxon signed-rank (c2 vs c1)
  6) ANCOVA-style: compare c2 controlling for c1 (residual approach)
  7) ARTool nonparametric mixed ANOVA (group × condition interaction)
  8) Bootstrap 95% CI for interaction contrast (Δ_neurotune − Δ_control)
  9) Permutation test for interaction
 10) Cliff's delta effect size
 11) Individual-level change classification
 12) Robustness: log-transformed TBR analysis
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats

try:
    import pingouin as pg
except ImportError:
    pg = None

np.random.seed(42)

CONTROL_SUBJECTS = {4, 6, 8, 9, 15, 17, 18, 22, 24, 25, 30, 31, 32}
NEUROTUNE_SUBJECTS = {10, 11, 12, 13, 14, 16, 19, 20, 21, 27, 28, 29}

TBR_ROOT = Path(__file__).resolve().parent / "tbr_nai_calculate_7ch_mne"
WARMUP_S = 10.0
N_BOOT = 10_000
N_PERM = 10_000

group_map = {}
for _n in CONTROL_SUBJECTS:
    group_map[f"S{_n:02d}"] = "Sori"
for _n in NEUROTUNE_SUBJECTS:
    group_map[f"S{_n:02d}"] = "NeuroTune"
SUBJECTS = sorted(group_map.keys(), key=lambda s: int(s[1:]))


def _warmup_cut(t, *arrs):
    if len(t) == 0:
        return (t, *arrs)
    t0 = t[0] + WARMUP_S
    m = t >= t0
    return (t[m] - t0, *(a[m] for a in arrs))


def load_tbr_wide() -> pd.DataFrame:
    """Load per-subject TBR mean for c1/c2, return wide DataFrame."""
    rows = []
    for s in SUBJECTS:
        path = TBR_ROOT / f"{s}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        g = group_map[s]
        row = {"subject": s, "group": g}
        for c in (1, 2):
            t = df[f"condition {c} time (s)"].dropna().to_numpy()
            tbr = df[f"condition {c} tbr"].dropna().to_numpy()
            n = min(len(t), len(tbr))
            t, tbr = t[:n], tbr[:n]
            t, tbr = _warmup_cut(t, tbr)
            row[f"c{c}_tbr_mean"] = float(np.mean(tbr))
            row[f"c{c}_tbr_median"] = float(np.median(tbr))
            row[f"c{c}_tbr_std"] = float(np.std(tbr))
            row[f"c{c}_tbr_p25"] = float(np.percentile(tbr, 25))
            row[f"c{c}_tbr_p75"] = float(np.percentile(tbr, 75))
        rows.append(row)
    return pd.DataFrame(rows)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    n1, n2 = len(a), len(b)
    more = sum(1 for ai in a for bi in b if ai > bi)
    less = sum(1 for ai in a for bi in b if ai < bi)
    return (more - less) / (n1 * n2)


# ──────────────────────────────────────────────────────────────
#  1. Cell descriptives
# ──────────────────────────────────────────────────────────────
def section_cell_descriptives(W: pd.DataFrame) -> str:
    lines = [
        "=" * 70,
        "1. CELL DESCRIPTIVES: TBR mean  (lower = more focused)",
        "   Source: 7ch MNE, warmup 10s removed, S23 excluded",
        "=" * 70,
    ]
    for grp in ["Sori", "NeuroTune"]:
        g = W[W["group"] == grp]
        for cond in ["c1", "c2"]:
            v = g[f"{cond}_tbr_mean"]
            lines.append(
                f"  {grp:10s} {cond}: "
                f"M={v.mean():.4f}  Mdn={v.median():.4f}  SD={v.std():.4f}  n={len(v)}"
            )
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
#  2. Baseline balance check
# ──────────────────────────────────────────────────────────────
def section_baseline(W: pd.DataFrame) -> str:
    lines = [
        "",
        "=" * 70,
        "2. BASELINE (C1) BALANCE CHECK",
        "   H0: no group difference at baseline",
        "=" * 70,
    ]
    sori = W[W["group"] == "Sori"]["c1_tbr_mean"].to_numpy()
    nt = W[W["group"] == "NeuroTune"]["c1_tbr_mean"].to_numpy()

    u, p = stats.mannwhitneyu(sori, nt, alternative="two-sided")
    cd = cliffs_delta(sori, nt)

    lines.append(f"  Sori      c1: M={sori.mean():.4f}  Mdn={np.median(sori):.4f}  SD={sori.std():.4f}")
    lines.append(f"  NeuroTune c1: M={nt.mean():.4f}  Mdn={np.median(nt):.4f}  SD={nt.std():.4f}")
    lines.append(f"  Mann-Whitney U={u:.1f}, p={p:.4f}, Cliff's δ={cd:.4f}  "
                 f"{'*' if p < 0.05 else 'ns'}")

    if p < 0.05:
        lines.append("  ⚠ Groups differ at baseline → delta analysis is especially important")
    else:
        lines.append("  ✓ No significant baseline difference")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
#  3. Delta descriptives
# ──────────────────────────────────────────────────────────────
def section_delta_descriptives(W: pd.DataFrame) -> str:
    lines = [
        "",
        "=" * 70,
        "3. DELTA (c2 − c1) DESCRIPTIVES",
        "   TBR: negative delta = decreased = better attention",
        "=" * 70,
    ]
    W = W.copy()
    W["delta"] = W["c2_tbr_mean"] - W["c1_tbr_mean"]

    for grp in ["Sori", "NeuroTune"]:
        g = W[W["group"] == grp]
        d = g["delta"]
        direction = "↓ decreased (BETTER)" if d.mean() < 0 else "↑ increased (worse)"
        n_dec = (d < 0).sum()
        n_inc = (d > 0).sum()
        lines.append(
            f"  {grp:10s}: M={d.mean():+.4f}  Mdn={d.median():+.4f}  SD={d.std():.4f}  "
            f"↓{n_dec} ↑{n_inc}  {direction}"
        )

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
#  4. Mann-Whitney U on delta
# ──────────────────────────────────────────────────────────────
def section_mw_delta(W: pd.DataFrame) -> str:
    lines = [
        "",
        "=" * 70,
        "4. MANN-WHITNEY U ON DELTA (between groups)",
        "   H1 (one-tailed): NeuroTune delta < Sori delta",
        "   (NeuroTune shows more TBR decrease = more improvement)",
        "=" * 70,
    ]
    W = W.copy()
    W["delta"] = W["c2_tbr_mean"] - W["c1_tbr_mean"]

    sori_d = W[W["group"] == "Sori"]["delta"].to_numpy()
    nt_d = W[W["group"] == "NeuroTune"]["delta"].to_numpy()

    u_two, p_two = stats.mannwhitneyu(sori_d, nt_d, alternative="two-sided")
    _, p_nt_less = stats.mannwhitneyu(nt_d, sori_d, alternative="less")

    n1, n2 = len(sori_d), len(nt_d)
    r_rb = 1 - (2 * u_two) / (n1 * n2)
    cd = cliffs_delta(sori_d, nt_d)

    lines.append(f"  Two-tailed: U={u_two:.1f}, p={p_two:.4f}")
    lines.append(f"  One-tailed (H1: Δ_NeuroTune < Δ_Sori): p={p_nt_less:.4f}  "
                 f"{'*' if p_nt_less < 0.05 else 'ns'}")
    lines.append(f"  rank-biserial r={r_rb:.4f},  Cliff's δ={cd:.4f}")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
#  5. Within-group Wilcoxon (c2 vs c1)
# ──────────────────────────────────────────────────────────────
def section_wilcoxon(W: pd.DataFrame) -> str:
    lines = [
        "",
        "=" * 70,
        "5. WITHIN-GROUP WILCOXON SIGNED-RANK (c2 vs c1)",
        "   H1: c2 < c1  (TBR decreased = improved)",
        "=" * 70,
    ]
    for grp in ["Sori", "NeuroTune"]:
        g = W[W["group"] == grp]
        c1 = g["c1_tbr_mean"].to_numpy()
        c2 = g["c2_tbr_mean"].to_numpy()
        delta = c2 - c1
        n_dec = (delta < 0).sum()
        n_inc = (delta > 0).sum()

        w, p_two = stats.wilcoxon(c1, c2)
        _, p_less = stats.wilcoxon(c2, c1, alternative="less")

        if pg:
            d_val = pg.compute_effsize(c1, c2, paired=True, eftype="cohen")
        else:
            d_val = (c2 - c1).mean() / (c2 - c1).std()

        lines.append(
            f"\n  {grp} (n={len(c1)}):"
        )
        lines.append(f"    Delta signs: ↓{n_dec}  ↑{n_inc}")
        lines.append(f"    Two-tailed: W={w:.1f}, p={p_two:.4f}")
        lines.append(f"    One-tailed (H1: c2<c1): p={p_less:.4f}  "
                     f"{'*' if p_less < 0.05 else 'ns'}")
        lines.append(f"    Cohen's d (paired): {d_val:.4f}")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
#  6. ANCOVA-style (residual approach)
# ──────────────────────────────────────────────────────────────
def section_ancova(W: pd.DataFrame) -> str:
    lines = [
        "",
        "=" * 70,
        "6. ANCOVA-STYLE: compare c2 controlling for c1",
        "   Method: regress c2 on c1 (all subjects), compare residuals",
        "=" * 70,
    ]
    x = W["c1_tbr_mean"].to_numpy()
    y = W["c2_tbr_mean"].to_numpy()
    groups = W["group"].to_numpy()

    slope, intercept, r_val, p_reg, se = stats.linregress(x, y)
    resid = y - (slope * x + intercept)

    r_sori = resid[groups == "Sori"]
    r_nt = resid[groups == "NeuroTune"]

    u, p_mw = stats.mannwhitneyu(r_sori, r_nt, alternative="two-sided")
    _, p_one = stats.mannwhitneyu(r_nt, r_sori, alternative="less")

    lines.append(f"  Regression c2~c1: slope={slope:.4f}, r={r_val:.4f}, p={p_reg:.4f}")
    lines.append(f"  Residual mean Sori:      {r_sori.mean():+.4f}")
    lines.append(f"  Residual mean NeuroTune: {r_nt.mean():+.4f}")
    lines.append(f"  Mann-Whitney on residuals (two-tailed): U={u:.1f}, p={p_mw:.4f}  "
                 f"{'*' if p_mw < 0.05 else 'ns'}")
    lines.append(f"  One-tailed (H1: NeuroTune residual < Sori): p={p_one:.4f}  "
                 f"{'*' if p_one < 0.05 else 'ns'}")

    if pg:
        df_anc = pd.DataFrame({
            "c1": x, "c2": y,
            "group": groups,
        })
        try:
            anc = pg.ancova(data=df_anc, dv="c2", covar="c1", between="group")
            lines.append(f"\n  Pingouin ANCOVA:")
            lines.append("  " + anc.to_string(index=False).replace("\n", "\n  "))
        except Exception as e:
            lines.append(f"\n  Pingouin ANCOVA failed: {e}")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
#  7. ARTool mixed ANOVA
# ──────────────────────────────────────────────────────────────
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


def section_art_anova(W: pd.DataFrame) -> str:
    lines = [
        "",
        "=" * 70,
        "7. ARTool MIXED ANOVA: group(between) × condition(within) on TBR mean",
        "=" * 70,
    ]
    if not pg:
        lines.append("  [SKIP] pingouin not installed")
        return "\n".join(lines)

    long_rows = []
    for _, r in W.iterrows():
        for c in (1, 2):
            long_rows.append({
                "subject": r["subject"], "group": r["group"],
                "condition": c, "tbr_mean": r[f"c{c}_tbr_mean"],
            })
    long = pd.DataFrame(long_rows)

    # Normality check
    norm_ok = True
    lines.append("\n  [Normality - Shapiro-Wilk]")
    for grp in ["Sori", "NeuroTune"]:
        for cond in [1, 2]:
            v = long[(long["group"] == grp) & (long["condition"] == cond)]["tbr_mean"].to_numpy()
            w, p = stats.shapiro(v)
            ok = p >= 0.05
            if not ok:
                norm_ok = False
            lines.append(f"    {grp} cond{cond}: W={w:.4f} p={p:.4f} {'OK' if ok else 'FAIL'}")

    # Homogeneity check
    homo_ok = True
    lines.append("\n  [Homogeneity - Levene]")
    for cond in [1, 2]:
        groups_data = [
            long[(long["group"] == grp) & (long["condition"] == cond)]["tbr_mean"].to_numpy()
            for grp in ["Sori", "NeuroTune"]
        ]
        f_val, p_lev = stats.levene(*groups_data)
        ok = p_lev >= 0.05
        if not ok:
            homo_ok = False
        lines.append(f"    cond{cond}: F={f_val:.4f} p={p_lev:.4f} {'OK' if ok else 'FAIL'}")

    if norm_ok and homo_ok:
        lines.append("\n  [Assumptions MET → parametric Mixed ANOVA]")
        aov = pg.mixed_anova(data=long, dv="tbr_mean", within="condition",
                             between="group", subject="subject")
        lines.append("  " + aov.to_string(index=False).replace("\n", "\n  "))
        inter_row = aov[aov["Source"] == "Interaction"]
        if not inter_row.empty:
            p_i = float(inter_row["p_unc"].iloc[0])
            lines.append(f"\n  Interaction p = {p_i:.4f}  {'*' if p_i < 0.05 else 'ns'}")
    else:
        lines.append(f"\n  [Assumptions VIOLATED → ARTool]")

    d = _art_transform(long, "tbr_mean")
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

    aov_i = pg.mixed_anova(data=d, dv="art_inter_rank", within="condition",
                           between="group", subject="subject")
    ri = aov_i[aov_i["Source"] == "Interaction"]
    if not ri.empty:
        p_art = float(ri["p_unc"].iloc[0])
        lines.append(f"\n  ARTool Interaction p = {p_art:.4f}  {'*' if p_art < 0.05 else 'ns'}")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
#  8. Bootstrap CI
# ──────────────────────────────────────────────────────────────
def section_bootstrap(W: pd.DataFrame) -> str:
    lines = [
        "",
        "=" * 70,
        f"8. BOOTSTRAP 95% CI FOR INTERACTION CONTRAST  (B={N_BOOT:,})",
        "   contrast = Δ_NeuroTune − Δ_Sori",
        "   (negative = NeuroTune decreased TBR more → favorable)",
        "=" * 70,
    ]
    W = W.copy()
    W["delta"] = W["c2_tbr_mean"] - W["c1_tbr_mean"]
    d_sori = W[W["group"] == "Sori"]["delta"].to_numpy()
    d_nt = W[W["group"] == "NeuroTune"]["delta"].to_numpy()

    observed = d_nt.mean() - d_sori.mean()
    rng = np.random.RandomState(42)
    boot = np.empty(N_BOOT)
    for i in range(N_BOOT):
        bs = rng.choice(d_sori, size=len(d_sori), replace=True)
        bn = rng.choice(d_nt, size=len(d_nt), replace=True)
        boot[i] = bn.mean() - bs.mean()

    lo, hi = np.percentile(boot, [2.5, 97.5])
    lines.append(f"  Δ_Sori mean      = {d_sori.mean():+.4f}")
    lines.append(f"  Δ_NeuroTune mean = {d_nt.mean():+.4f}")
    lines.append(f"  Observed contrast = {observed:+.4f}")
    lines.append(f"  Bootstrap 95% CI  = [{lo:+.4f}, {hi:+.4f}]")

    if hi < 0:
        lines.append("  → CI entirely below 0: NeuroTune reliably decreased TBR more (FAVORABLE)")
    elif lo > 0:
        lines.append("  → CI entirely above 0: Sori decreased TBR more")
    else:
        lines.append("  → CI spans 0: direction not conclusively determined")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
#  9. Permutation test
# ──────────────────────────────────────────────────────────────
def section_permutation(W: pd.DataFrame) -> str:
    lines = [
        "",
        "=" * 70,
        f"9. PERMUTATION TEST FOR INTERACTION  (n_perm={N_PERM:,})",
        "=" * 70,
    ]
    W = W.copy()
    W["delta"] = W["c2_tbr_mean"] - W["c1_tbr_mean"]
    d_sori = W[W["group"] == "Sori"]["delta"].to_numpy()
    d_nt = W[W["group"] == "NeuroTune"]["delta"].to_numpy()

    observed = d_nt.mean() - d_sori.mean()
    all_d = np.concatenate([d_sori, d_nt])
    n_s = len(d_sori)

    rng = np.random.RandomState(42)
    count_two = 0
    count_one = 0
    for _ in range(N_PERM):
        perm = rng.permutation(all_d)
        ps = perm[:n_s]
        pn = perm[n_s:]
        pc = pn.mean() - ps.mean()
        if abs(pc) >= abs(observed):
            count_two += 1
        if pc <= observed:
            count_one += 1

    p_two = count_two / N_PERM
    p_one = count_one / N_PERM

    lines.append(f"  Observed contrast = {observed:+.4f}")
    lines.append(f"  Two-tailed p = {p_two:.4f}")
    lines.append(f"  One-tailed p (H1: NeuroTune Δ < Sori Δ): {p_one:.4f}  "
                 f"{'*' if p_one < 0.05 else 'ns'}")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# 10. Cliff's delta effect size summary
# ──────────────────────────────────────────────────────────────
def section_effect_sizes(W: pd.DataFrame) -> str:
    lines = [
        "",
        "=" * 70,
        "10. EFFECT SIZE SUMMARY",
        "    Cliff's δ: >0 = Sori tends larger; <0 = NeuroTune tends larger",
        "    For TBR: Sori larger = Sori worse focus",
        "=" * 70,
    ]
    sori = W[W["group"] == "Sori"]
    nt = W[W["group"] == "NeuroTune"]

    for metric, label in [
        ("c1_tbr_mean", "Baseline (c1)"),
        ("c2_tbr_mean", "Post (c2)"),
    ]:
        cd = cliffs_delta(sori[metric].to_numpy(), nt[metric].to_numpy())
        lines.append(f"  {label:20s}: Cliff's δ = {cd:+.4f}")

    W = W.copy()
    W["delta"] = W["c2_tbr_mean"] - W["c1_tbr_mean"]
    cd_d = cliffs_delta(sori["c1_tbr_mean"].to_numpy() - sori["c2_tbr_mean"].to_numpy(),
                        nt["c1_tbr_mean"].to_numpy() - nt["c2_tbr_mean"].to_numpy())
    lines.append(f"  {'Delta (c2−c1)':20s}: Cliff's δ = {cliffs_delta(W[W['group']=='Sori']['delta'].to_numpy(), W[W['group']=='NeuroTune']['delta'].to_numpy()):+.4f}")

    def _interpret(d):
        ad = abs(d)
        if ad < 0.147:
            return "negligible"
        elif ad < 0.33:
            return "small"
        elif ad < 0.474:
            return "medium"
        else:
            return "large"

    cd_c2 = cliffs_delta(sori["c2_tbr_mean"].to_numpy(), nt["c2_tbr_mean"].to_numpy())
    lines.append(f"\n  Interpretation (Romano et al. 2006):")
    lines.append(f"    c2 Cliff's δ = {cd_c2:+.4f} → {_interpret(cd_c2)}")
    cd_delta = cliffs_delta(W[W['group']=='Sori']['delta'].to_numpy(), W[W['group']=='NeuroTune']['delta'].to_numpy())
    lines.append(f"    Δ  Cliff's δ = {cd_delta:+.4f} → {_interpret(cd_delta)}")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# 11. Individual-level change classification
# ──────────────────────────────────────────────────────────────
def section_individual_changes(W: pd.DataFrame) -> str:
    lines = [
        "",
        "=" * 70,
        "11. INDIVIDUAL-LEVEL CHANGE (per subject)",
        "    TBR decrease = improvement, increase = deterioration",
        "=" * 70,
    ]
    W = W.copy()
    W["delta"] = W["c2_tbr_mean"] - W["c1_tbr_mean"]
    W["pct"] = W["delta"] / W["c1_tbr_mean"] * 100

    for grp in ["Sori", "NeuroTune"]:
        g = W[W["group"] == grp].sort_values("delta")
        lines.append(f"\n  {grp}:")
        lines.append(f"    {'Subject':>8s}  {'c1':>8s}  {'c2':>8s}  {'Δ':>8s}  {'%chg':>8s}  dir")
        for _, r in g.iterrows():
            direction = "↓ better" if r["delta"] < 0 else "↑ worse"
            lines.append(
                f"    {r['subject']:>8s}  {r['c1_tbr_mean']:8.4f}  {r['c2_tbr_mean']:8.4f}  "
                f"{r['delta']:+8.4f}  {r['pct']:+8.1f}%  {direction}"
            )
        n_imp = (g["delta"] < 0).sum()
        lines.append(f"    → improved: {n_imp}/{len(g)} ({100*n_imp/len(g):.0f}%)")

    # Fisher's exact test on improvement proportion
    sori_imp = (W[W["group"] == "Sori"]["delta"] < 0).sum()
    sori_n = (W["group"] == "Sori").sum()
    nt_imp = (W[W["group"] == "NeuroTune"]["delta"] < 0).sum()
    nt_n = (W["group"] == "NeuroTune").sum()
    table = [[sori_imp, sori_n - sori_imp], [nt_imp, nt_n - nt_imp]]
    _, p_fisher = stats.fisher_exact(table)
    lines.append(f"\n  Fisher exact (improvement rate): p={p_fisher:.4f}  "
                 f"{'*' if p_fisher < 0.05 else 'ns'}")
    lines.append(f"    Sori: {sori_imp}/{sori_n}  NeuroTune: {nt_imp}/{nt_n}")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# 12. Log-transformed robustness
# ──────────────────────────────────────────────────────────────
def section_log_robustness(W: pd.DataFrame) -> str:
    lines = [
        "",
        "=" * 70,
        "12. LOG-TRANSFORMED TBR ROBUSTNESS CHECK",
        "=" * 70,
    ]
    W = W.copy()
    eps = 1e-6
    W["log_c1"] = np.log(np.maximum(W["c1_tbr_mean"], eps))
    W["log_c2"] = np.log(np.maximum(W["c2_tbr_mean"], eps))
    W["log_delta"] = W["log_c2"] - W["log_c1"]

    for grp in ["Sori", "NeuroTune"]:
        g = W[W["group"] == grp]
        d = g["log_delta"]
        direction = "↓ decreased" if d.mean() < 0 else "↑ increased"
        lines.append(
            f"  {grp:10s} log-Δ: M={d.mean():+.4f}  Mdn={d.median():+.4f}  SD={d.std():.4f}  {direction}"
        )

    d_sori = W[W["group"] == "Sori"]["log_delta"].to_numpy()
    d_nt = W[W["group"] == "NeuroTune"]["log_delta"].to_numpy()

    u, p_two = stats.mannwhitneyu(d_sori, d_nt, alternative="two-sided")
    _, p_one = stats.mannwhitneyu(d_nt, d_sori, alternative="less")
    lines.append(f"\n  Mann-Whitney on log-delta:")
    lines.append(f"    Two-tailed: U={u:.1f}, p={p_two:.4f}")
    lines.append(f"    One-tailed (H1: NeuroTune log-Δ < Sori): p={p_one:.4f}  "
                 f"{'*' if p_one < 0.05 else 'ns'}")

    contrast = d_nt.mean() - d_sori.mean()
    rng = np.random.RandomState(42)
    boot = np.empty(N_BOOT)
    for i in range(N_BOOT):
        bs = rng.choice(d_sori, size=len(d_sori), replace=True)
        bn = rng.choice(d_nt, size=len(d_nt), replace=True)
        boot[i] = bn.mean() - bs.mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    lines.append(f"\n  Log interaction contrast = {contrast:+.4f}")
    lines.append(f"  Bootstrap 95% CI (log) = [{lo:+.4f}, {hi:+.4f}]")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────
def main():
    W = load_tbr_wide()

    parts: List[str] = []
    parts.append("=" * 70)
    parts.append("TBR DELTA ANALYSIS (7ch MNE, warmup 10s, S23 excluded)")
    parts.append(f"  Sori (control): n={len(W[W['group']=='Sori'])}")
    parts.append(f"  NeuroTune:      n={len(W[W['group']=='NeuroTune'])}")
    parts.append("=" * 70)

    parts.append(section_cell_descriptives(W))
    parts.append(section_baseline(W))
    parts.append(section_delta_descriptives(W))
    parts.append(section_mw_delta(W))
    parts.append(section_wilcoxon(W))
    parts.append(section_ancova(W))
    parts.append(section_art_anova(W))
    parts.append(section_bootstrap(W))
    parts.append(section_permutation(W))
    parts.append(section_effect_sizes(W))
    parts.append(section_individual_changes(W))
    parts.append(section_log_robustness(W))

    report = "\n".join(parts)

    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    print(report)

    out_path = TBR_ROOT / "tbr_delta_analysis_report.txt"
    out_path.write_text(report, encoding="utf-8")
    print(f"\n[OK] saved: {out_path}")


if __name__ == "__main__":
    main()
