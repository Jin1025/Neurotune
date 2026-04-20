"""
Comprehensive TBR analysis comparing Group 1 (control) vs Group 2 (intervention).

Direction convention (per user):
  - TBR : lower  = more focused
  - condition 1 = first half (pre-intervention phase of the focus training)
  - condition 2 = second half (post-intervention phase)

For each subject we compute TBR metrics on condition 1 and condition 2,
plus deltas (c2-c1). Group comparisons use Mann-Whitney U (two-sided and one-sided)
and Cliff's delta. TBR thresholds are from per-subject condition 1 quantiles.
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
TBR_ROOT = BASE_DIR / "data" / "tbr_calculate"


def contiguous_runs(mask):
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


def run_metrics(time, value, threshold, focus_if='le'):
    """
    focus_if='le' means focus when value <= threshold (for TBR).
    """
    if focus_if == 'le':
        mask = value <= threshold
    else:
        mask = value >= threshold

    total_time = time[-1] - time[0] if len(time) > 1 else np.nan
    prop_focus = float(np.mean(mask)) if len(mask) else np.nan

    runs_all = contiguous_runs(mask.tolist())
    runs_sustain = [(s, e) for s, e in runs_all if (e - s + 1) >= MIN_EPISODE_BINS]
    durations = []
    for s, e in runs_sustain:
        durations.append(time[e] - time[s] if e > s else 0.0)
    n_episodes = len(runs_all)

    first_reach = time[runs_all[0][0]] - time[0] if n_episodes > 0 else np.nan

    gaps = [time[runs_all[k + 1][0]] - time[runs_all[k][1]] for k in range(len(runs_all) - 1)]

    eps_per_min = n_episodes / (total_time / 60.0) if (total_time and total_time > 0) else np.nan

    return {
        'prop_focus':         prop_focus,
        'n_episodes':         n_episodes,
        'eps_per_min':        eps_per_min,
        'mean_sustain_s':     float(np.mean(durations))    if durations else np.nan,
        'median_sustain_s':   float(np.median(durations))  if durations else np.nan,
        'max_sustain_s':      float(np.max(durations))     if durations else np.nan,
        'total_time_focus_s': float(np.sum(durations))     if durations else 0.0,
        'first_reach_s':      first_reach,
        'mean_gap_s':         float(np.mean(gaps))         if gaps else np.nan,
        'median_gap_s':       float(np.median(gaps))       if gaps else np.nan,
    }


def summary_stats(time, value):
    out = {
        'mean':   float(np.mean(value)),
        'median': float(np.median(value)),
        'std':    float(np.std(value)),
        'iqr':    float(np.percentile(value, 75) - np.percentile(value, 25)),
        'min':    float(np.min(value)),
        'max':    float(np.max(value)),
        'p25':    float(np.percentile(value, 25)),
        'p75':    float(np.percentile(value, 75)),
    }
    if len(value) > 2:
        slope, intercept, r, p, se = stats.linregress(time, value)
        out['slope']   = float(slope)
        out['trend_r'] = float(r)
    else:
        out['slope']   = np.nan
        out['trend_r'] = np.nan
    return out


# ------------- group definition (same as analyze_nai_threshold.py) -------------
CONTROL_SUBJECTS = {4, 6, 8, 9, 15, 17, 18, 22, 24, 25, 30, 31, 32}
NEUROTUNE_SUBJECTS = {10, 11, 12, 13, 14, 16, 19, 20, 21, 27, 28, 29}

group_map = {}
for _n in CONTROL_SUBJECTS:
    group_map[f'S{_n:02d}'] = 1
for _n in NEUROTUNE_SUBJECTS:
    group_map[f'S{_n:02d}'] = 2
SUBJECTS = sorted(group_map.keys(), key=lambda s: int(s[1:]))

TBR_Q          = [0.25, 0.33, 0.50]           # focus = TBR <= quantile threshold → tbr_leQ{pp}_*
TBR_Q_GE       = [0.67]                         # focus = TBR >= quantile threshold → tbr_geQ{pp}_*
WARMUP_S       = 10.0                          # discard first 10 s of each condition
MIN_EPISODE_BINS = 2                           # sustain definition: at least 2 bins


def _warmup_cut(t, *arrs):
    """Cut the first WARMUP_S seconds and re-zero the time axis."""
    if len(t) == 0:
        return (t, *arrs)
    t0 = t[0] + WARMUP_S
    m = t >= t0
    return (t[m] - t0, *(a[m] for a in arrs))


rows = []
for s in SUBJECTS:
    tbr_path = TBR_ROOT / f'{s}.csv'
    if not tbr_path.exists():
        continue
    df_tbr = pd.read_csv(tbr_path)
    g = group_map[s]

    # TBR quantile thresholds from BLOCK 1 / condition 1 only (after warmup).
    # Regardless of CSV format, thresholds are anchored to condition 1.
    if ('condition 1 time (s)' not in df_tbr.columns) or ('condition 1 tbr' not in df_tbr.columns):
        continue
    t_blk1 = df_tbr['condition 1 time (s)'].dropna().to_numpy()
    tb_blk1 = df_tbr['condition 1 tbr'].dropna().to_numpy()
    n_blk1 = min(len(t_blk1), len(tb_blk1))
    t_blk1, tb_blk1 = t_blk1[:n_blk1], tb_blk1[:n_blk1]
    t_blk1, tb_blk1 = _warmup_cut(t_blk1, tb_blk1)
    if len(tb_blk1) == 0:
        continue
    tbr_thr = {q: float(np.quantile(tb_blk1, q)) for q in (TBR_Q + TBR_Q_GE)}

    for c in (1, 2):
        t_tbr = df_tbr[f'condition {c} time (s)'].dropna().to_numpy()
        tbr   = df_tbr[f'condition {c} tbr'].dropna().to_numpy()
        n_tbr = min(len(t_tbr), len(tbr))
        t_tbr, tbr = t_tbr[:n_tbr], tbr[:n_tbr]
        t_tbr, tbr = _warmup_cut(t_tbr, tbr)
        if len(tbr) == 0:
            continue

        row = {'subject': s, 'group': g, 'condition': c, 'n_samples_tbr': len(tbr)}

        # TBR summary stats (7ch MNE time axis)
        for k, v in summary_stats(t_tbr, tbr).items():
            row[f'tbr_{k}'] = v

        # TBR episode metrics (7ch MNE time axis): lower-tail (≤) and upper-tail (≥)
        for q in TBR_Q:
            thr = tbr_thr[q]
            m = run_metrics(t_tbr, tbr, thr, focus_if='le')
            for k, v in m.items():
                row[f'tbr_leQ{int(q*100)}_{k}'] = v
        for q in TBR_Q_GE:
            thr = tbr_thr[q]
            m = run_metrics(t_tbr, tbr, thr, focus_if='ge')
            for k, v in m.items():
                row[f'tbr_geQ{int(q*100)}_{k}'] = v

        rows.append(row)

long_df = pd.DataFrame(rows)
if long_df.empty or 'subject' not in long_df.columns:
    raise SystemExit(
        f'No valid subject rows were built from {TBR_ROOT}. '
        'Check CSV files and required columns: condition 1/2 time,tbr.'
    )

metric_cols = [c for c in long_df.columns if c not in ('subject', 'group', 'condition', 'n_samples_tbr')]

wide_rows = []
for s, sub in long_df.groupby('subject'):
    r = {'subject': s, 'group': int(sub['group'].iloc[0])}
    for c in (1, 2):
        sub_c = sub[sub['condition'] == c]
        if len(sub_c) == 0:
            continue
        for m in metric_cols:
            r[f'c{c}_{m}'] = sub_c[m].iloc[0]
    for m in metric_cols:
        if f'c1_{m}' in r and f'c2_{m}' in r:
            r[f'd_{m}'] = r[f'c2_{m}'] - r[f'c1_{m}']     # c2 - c1
    wide_rows.append(r)

OUT_ROOT = TBR_ROOT
W = pd.DataFrame(wide_rows)
W.to_csv(OUT_ROOT / 'subject_metrics_full.csv', index=False)
print('subjects x metrics:', W.shape, '| groups:', W['group'].value_counts().to_dict())


# ------------- group comparisons -------------
g1 = W[W['group'] == 1]
g2 = W[W['group'] == 2]

results = []
for col in W.columns:
    if col in ('subject', 'group'):
        continue
    a = g1[col].dropna().to_numpy()
    b = g2[col].dropna().to_numpy()
    if len(a) < 3 or len(b) < 3:
        continue
    try:
        u_two, p_two = stats.mannwhitneyu(a, b, alternative='two-sided')
        _, p_g2_smaller = stats.mannwhitneyu(a, b, alternative='greater')
        _, p_g2_larger  = stats.mannwhitneyu(a, b, alternative='less')
    except ValueError:
        continue
    n1_, n2_ = len(a), len(b)
    p_a_gt_b = u_two / (n1_ * n2_)
    cliffs = 2 * p_a_gt_b - 1   # >0: group1 tends larger; <0: group2 tends larger

    # Determine which direction = "group 2 better" for readability (TBR-only).
    name = col.lower()
    is_tbr  = 'tbr' in name
    episode_like = any(k in name for k in (
        'prop_focus', 'n_episodes', 'eps_per_min',
        'mean_sustain_s', 'median_sustain_s', 'max_sustain_s',
        'total_time_focus_s',
    ))
    smaller_gap_or_first_reach_better = any(k in name for k in ('first_reach_s', 'mean_gap_s', 'median_gap_s'))

    if episode_like:
        favor = 'g2_larger'
    elif smaller_gap_or_first_reach_better:
        favor = 'g2_smaller'
    elif is_tbr:
        favor = 'g2_smaller'
    elif not is_tbr:
        favor = None

    # Overrides for deltas (improvement direction):
    if col.startswith('d_'):
        # improvement: TBR should decrease more in g2 => c2-c1 more negative => g2 smaller
        if 'tbr' in name and episode_like:
            favor = 'g2_larger'   # focus duration should grow
        elif 'tbr' in name and smaller_gap_or_first_reach_better:
            favor = 'g2_smaller'   # first-reach / gap should shrink

    p_fav = p_g2_smaller if favor == 'g2_smaller' else (p_g2_larger if favor == 'g2_larger' else min(p_g2_smaller, p_g2_larger))

    results.append({
        'metric': col,
        'favor_dir': favor,
        'g1_mean': float(np.mean(a)), 'g1_median': float(np.median(a)),
        'g2_mean': float(np.mean(b)), 'g2_median': float(np.median(b)),
        'diff_g2_minus_g1_mean': float(np.mean(b) - np.mean(a)),
        'p_two_sided': float(p_two),
        'p_g2_smaller': float(p_g2_smaller),
        'p_g2_larger': float(p_g2_larger),
        'p_favor_g2': float(p_fav),
        'cliffs_delta_g1_vs_g2': float(cliffs),
        'n1': n1_, 'n2': n2_,
    })

R = pd.DataFrame(results).sort_values('p_favor_g2')
R.to_csv(OUT_ROOT / 'group_comparison_all.csv', index=False)

# Save only TBR delta metrics (d_tbr_*) with p_two_sided < 0.1
delta_sig = R[R['metric'].str.startswith('d_tbr_') & (R['p_two_sided'] < 0.08)].sort_values('p_two_sided')
delta_sig.to_csv(OUT_ROOT / 'sig_main_delta.csv', index=False)

# Print readable reports
def fmt(df, n=30):
    cols = ['metric', 'favor_dir', 'g1_mean', 'g2_mean', 'diff_g2_minus_g1_mean',
            'p_two_sided', 'p_favor_g2', 'cliffs_delta_g1_vs_g2']
    return df[cols].head(n).to_string(index=False, float_format=lambda x: f'{x:.4f}')

report_lines = []
report_lines.append(f'subjects x metrics: {W.shape} | groups: {W["group"].value_counts().to_dict()}')

report_lines.append('\n========== Metrics where GROUP 2 IS SUPERIOR (one-sided p < 0.1) ==========')
report_lines.append(fmt(R[R['p_favor_g2'] < 0.1]))

report_lines.append('\n========== Top 25 by strongest group-2 advantage ==========')
report_lines.append(fmt(R.sort_values('p_favor_g2').head(25)))

report_lines.append('\n========== Two-sided p < 0.05 ==========')
report_lines.append(fmt(R.sort_values('p_two_sided').query('p_two_sided < 0.05')))

report_lines.append('\n========== Improvement (delta c2-c1) metrics, top 20 favoring group 2 ==========')
deltaR = R[R['metric'].str.startswith('d_tbr_')].sort_values('p_favor_g2')
report_lines.append(fmt(deltaR, n=20))

report_lines.append('\n========== Condition-1 (baseline) comparison, top 10 (sanity: groups should be similar) ==========')
c1R = R[R['metric'].str.startswith('c1_')].sort_values('p_two_sided')
report_lines.append(fmt(c1R, n=10))

report_text = '\n'.join(report_lines)
print(report_text)

report_path = OUT_ROOT / 'run_analysis_report.txt'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_text + '\n')
print(f'\n[OK] saved: {report_path}')
print(f'[OK] saved: {OUT_ROOT / "sig_main_delta.csv"}')

