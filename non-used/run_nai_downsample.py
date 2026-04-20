"""
Downsample NAI (and paired TBR) from 0.25s step to 1.0s bins.

Input:
  analys/tbr_nai_calculate/Sxx.csv  (6 columns: time, tbr, nai per condition)

Output:
  - Per-subject full-range series (warmup NOT removed; same 6-column layout):
      analys/tbr_nai_calculate_ds1s/Sxx.csv
  - NAI-only summaries + group MWU (warmup 10s applied only for these stats):
      analys/tbr_nai_calculate_ds1s/subject_metrics_nai_ds1s.csv
      analys/tbr_nai_calculate_ds1s/group_comparison_nai_ds1s.csv
      analys/tbr_nai_calculate_ds1s/run_nai_downsample_report.txt

Episode metrics match run_analysis.py: sustain uses runs with length >= MIN_EPISODE_BINS;
n_episodes / eps / first_reach / gaps use all runs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

CONTROL_SUBJECTS = {4, 6, 8, 9, 15, 17, 18, 22, 24, 25, 30, 31, 32}
NEUROTUNE_SUBJECTS = {10, 11, 12, 13, 14, 16, 19, 20, 21, 27, 28, 29}

NAI_ROOT = Path(__file__).resolve().parent / "tbr_nai_calculate"
DS_ROOT = Path(__file__).resolve().parent / "tbr_nai_calculate_ds1s"
WARMUP_S = 10.0
DS_STEP_S = 1.0
NAI_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
MIN_EPISODE_BINS = 2

group_map: Dict[str, int] = {}
for _n in CONTROL_SUBJECTS:
    group_map[f"S{_n:02d}"] = 1
for _n in NEUROTUNE_SUBJECTS:
    group_map[f"S{_n:02d}"] = 2
SUBJECTS = sorted(group_map.keys(), key=lambda s: int(s[1:]))


def _warmup_cut(t: np.ndarray, *arrs: np.ndarray):
    if len(t) == 0:
        return (t, *arrs)
    t0 = t[0] + WARMUP_S
    m = t >= t0
    return (t[m] - t0, *(a[m] for a in arrs))


def downsample_triple(
    t: np.ndarray, tbr: np.ndarray, nai: np.ndarray, step_s: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same bin edges for TBR and NAI (one pass on time)."""
    if len(t) == 0:
        return t, tbr, nai
    b = np.floor(t / step_s).astype(int)
    uniq = np.unique(b)
    t_out = np.empty(len(uniq), float)
    tbr_out = np.empty(len(uniq), float)
    nai_out = np.empty(len(uniq), float)
    for i, bi in enumerate(uniq):
        m = b == bi
        t_out[i] = bi * step_s
        tbr_out[i] = float(np.mean(tbr[m]))
        nai_out[i] = float(np.mean(nai[m]))
    return t_out, tbr_out, nai_out


def contiguous_runs(mask: List[bool]) -> List[Tuple[int, int]]:
    runs = []
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs


def run_metrics(time: np.ndarray, value: np.ndarray, threshold: float, focus_if: str = "ge") -> Dict[str, float]:
    """Same logic as run_analysis.py (NAI: ge)."""
    if focus_if == "ge":
        mask = value >= threshold
    else:
        mask = value <= threshold

    total_time = time[-1] - time[0] if len(time) > 1 else np.nan
    prop_focus = float(np.mean(mask)) if len(mask) else np.nan

    runs_all = contiguous_runs(mask.tolist())
    runs_sustain = [(s, e) for s, e in runs_all if (e - s + 1) >= MIN_EPISODE_BINS]
    durations = [time[e] - time[s] if e > s else 0.0 for s, e in runs_sustain]
    n_episodes = len(runs_all)

    first_reach = time[runs_all[0][0]] - time[0] if n_episodes > 0 else np.nan
    gaps = [time[runs_all[k + 1][0]] - time[runs_all[k][1]] for k in range(len(runs_all) - 1)]
    eps_per_min = n_episodes / (total_time / 60.0) if (total_time and total_time > 0) else np.nan

    return {
        "prop_focus": prop_focus,
        "n_episodes": n_episodes,
        "eps_per_min": eps_per_min,
        "mean_sustain_s": float(np.mean(durations)) if durations else np.nan,
        "median_sustain_s": float(np.median(durations)) if durations else np.nan,
        "max_sustain_s": float(np.max(durations)) if durations else np.nan,
        "total_time_focus_s": float(np.sum(durations)) if durations else 0.0,
        "first_reach_s": first_reach,
        "mean_gap_s": float(np.mean(gaps)) if gaps else np.nan,
        "median_gap_s": float(np.median(gaps)) if gaps else np.nan,
    }


def summary_stats(time: np.ndarray, value: np.ndarray) -> Dict[str, float]:
    out = {
        "mean": float(np.mean(value)),
        "median": float(np.median(value)),
        "std": float(np.std(value)),
        "iqr": float(np.percentile(value, 75) - np.percentile(value, 25)),
        "min": float(np.min(value)),
        "max": float(np.max(value)),
        "p25": float(np.percentile(value, 25)),
        "p75": float(np.percentile(value, 75)),
    }
    if len(value) > 2:
        slope, intercept, r, p, se = stats.linregress(time, value)
        out["slope"] = float(slope)
        out["trend_r"] = float(r)
    else:
        out["slope"] = np.nan
        out["trend_r"] = np.nan
    return out


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return np.nan
    u, _ = stats.mannwhitneyu(a, b, alternative="two-sided")
    return 2 * (u / (n1 * n2)) - 1


def build_original_like_df(
    t1: np.ndarray,
    tbr1: np.ndarray,
    nai1: np.ndarray,
    t2: np.ndarray,
    tbr2: np.ndarray,
    nai2: np.ndarray,
) -> pd.DataFrame:
    n = max(len(t1), len(t2))
    out = pd.DataFrame(
        {
            "condition 1 time (s)": np.full(n, np.nan),
            "condition 1 tbr": np.full(n, np.nan),
            "condition 1 nai": np.full(n, np.nan),
            "condition 2 time (s)": np.full(n, np.nan),
            "condition 2 tbr": np.full(n, np.nan),
            "condition 2 nai": np.full(n, np.nan),
        }
    )
    out.loc[: len(t1) - 1, "condition 1 time (s)"] = t1
    out.loc[: len(tbr1) - 1, "condition 1 tbr"] = tbr1
    out.loc[: len(nai1) - 1, "condition 1 nai"] = nai1
    out.loc[: len(t2) - 1, "condition 2 time (s)"] = t2
    out.loc[: len(tbr2) - 1, "condition 2 tbr"] = tbr2
    out.loc[: len(nai2) - 1, "condition 2 nai"] = nai2
    return out


def main() -> None:
    DS_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []

    for s in SUBJECTS:
        path = NAI_ROOT / f"{s}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        g = group_map[s]
        ds_by_cond: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        for c in (1, 2):
            t_raw = df[f"condition {c} time (s)"].dropna().to_numpy()
            tbr_raw = df[f"condition {c} tbr"].dropna().to_numpy()
            nai_raw = df[f"condition {c} nai"].dropna().to_numpy()
            n = min(len(t_raw), len(tbr_raw), len(nai_raw))
            t_raw, tbr_raw, nai_raw = t_raw[:n], tbr_raw[:n], nai_raw[:n]

            t_save, tbr_save, nai_save = downsample_triple(t_raw, tbr_raw, nai_raw, DS_STEP_S)
            ds_by_cond[c] = (t_save, tbr_save, nai_save)

            t_stat, tbr_stat, nai_stat = _warmup_cut(t_save, tbr_save, nai_save)
            row = {"subject": s, "group": g, "condition": c, "n_samples": len(nai_stat)}

            for k, v in summary_stats(t_stat, nai_stat).items():
                row[f"nai_{k}"] = v
            for thr in NAI_THRESHOLDS:
                m = run_metrics(t_stat, nai_stat, thr, focus_if="ge")
                for k, v in m.items():
                    row[f"nai_ge_{thr}_{k}"] = v
            rows.append(row)

        if 1 in ds_by_cond and 2 in ds_by_cond:
            t1, tbr1, nai1 = ds_by_cond[1]
            t2, tbr2, nai2 = ds_by_cond[2]
            build_original_like_df(t1, tbr1, nai1, t2, tbr2, nai2).to_csv(DS_ROOT / f"{s}.csv", index=False)

    long_df = pd.DataFrame(rows)
    metric_cols = [c for c in long_df.columns if c not in ("subject", "group", "condition", "n_samples")]

    wide_rows = []
    for s, sub in long_df.groupby("subject"):
        r = {"subject": s, "group": int(sub["group"].iloc[0])}
        for c in (1, 2):
            sc = sub[sub["condition"] == c]
            if len(sc) == 0:
                continue
            for m in metric_cols:
                r[f"c{c}_{m}"] = sc[m].iloc[0]
        for m in metric_cols:
            c1k, c2k = f"c1_{m}", f"c2_{m}"
            if c1k in r and c2k in r:
                r[f"d_{m}"] = r[c2k] - r[c1k]
        wide_rows.append(r)

    W = pd.DataFrame(wide_rows)
    out_metrics = [c for c in W.columns if c not in ("subject", "group")]

    g1 = W[W["group"] == 1]
    g2 = W[W["group"] == 2]
    results = []
    for col in out_metrics:
        a = g1[col].dropna().to_numpy(float)
        b = g2[col].dropna().to_numpy(float)
        if len(a) < 3 or len(b) < 3:
            continue
        u, p_two = stats.mannwhitneyu(a, b, alternative="two-sided")
        _, p_g2_larger = stats.mannwhitneyu(a, b, alternative="less")
        _, p_g2_smaller = stats.mannwhitneyu(a, b, alternative="greater")
        p_fav = p_g2_larger
        results.append(
            {
                "metric": col,
                "favor_dir": "g2_larger",
                "g1_mean": float(np.mean(a)),
                "g1_median": float(np.median(a)),
                "g2_mean": float(np.mean(b)),
                "g2_median": float(np.median(b)),
                "diff_g2_minus_g1_mean": float(np.mean(b) - np.mean(a)),
                "p_two_sided": float(p_two),
                "p_g2_larger": float(p_g2_larger),
                "p_g2_smaller": float(p_g2_smaller),
                "p_favor_g2": float(p_fav),
                "cliffs_delta_g1_vs_g2": float(cliffs_delta(a, b)),
                "n1": int(len(a)),
                "n2": int(len(b)),
            }
        )

    R = pd.DataFrame(results).sort_values("p_favor_g2")
    W.to_csv(DS_ROOT / "subject_metrics_nai_ds1s.csv", index=False)
    R.to_csv(DS_ROOT / "group_comparison_nai_ds1s.csv", index=False)

    lines: List[str] = []
    lines.append("=" * 80)
    lines.append("NAI DOWNSAMPLED ANALYSIS (0.25s -> 1.0s bins)")
    lines.append(f"  per-subject CSV: full range (no warmup); stats: warmup {WARMUP_S:.0f}s then bins")
    lines.append(f"  ds step: {DS_STEP_S:.2f}s | MIN_EPISODE_BINS (sustain only): {MIN_EPISODE_BINS}")
    lines.append(f"  subjects: {len(W)} | groups: {W['group'].value_counts().to_dict()}")
    lines.append("=" * 80)
    top = R[R["p_two_sided"] < 0.1].head(30)
    if len(top):
        cols = ["metric", "g1_mean", "g2_mean", "diff_g2_minus_g1_mean", "p_two_sided", "p_favor_g2"]
        lines.append("\nTop signals (p_two_sided < 0.1):")
        lines.append(top[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    else:
        lines.append("\n(no signals at p<0.1)")
    report = "\n".join(lines)
    (DS_ROOT / "run_nai_downsample_report.txt").write_text(report + "\n", encoding="utf-8")

    print(report)
    print("\n[OK] saved:")
    print(f"  {DS_ROOT}/S*.csv (full-range downsampled)")
    print(f"  {DS_ROOT / 'subject_metrics_nai_ds1s.csv'}")
    print(f"  {DS_ROOT / 'group_comparison_nai_ds1s.csv'}")
    print(f"  {DS_ROOT / 'run_nai_downsample_report.txt'}")


if __name__ == "__main__":
    main()
