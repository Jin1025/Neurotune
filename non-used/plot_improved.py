"""
Fig 3: TBR tri-band occupancy (low / mid / high) with thresholds from Block1 only.

Per subject:
  - q33, q67 = quantiles of condition-1 TBR (after same warmup as run_analysis.py)
  - Classify each sample in condition 1 and 2:
      low  : TBR <= q33
      mid  : q33 < TBR < q67
      high : TBR >= q67
  - Proportions sum to 1.0 for each subject × condition.

Outputs (under tbr_nai_calculate_7ch_mne/):
  - figures/fig3a_tbr_triband_stacked.png   (tri-band mean occupancy)
  - figures/fig3b_tbr_triband_improver.png  (improver % by band)
  - tbr_triband_block1ref.csv

Improvement vs Block1 (directional, lower TBR = more focused):
  - Low band:  Δ share > 0  (more time TBR≤Q33)
  - Mid band:  Δ share < 0  (less time in middle band)
  - High band: Δ share < 0  (less time TBR≥Q67)

Usage:
    python3 plot_fig3.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
TBR_ROOT = BASE_DIR / 'tbr_nai_calculate_7ch_mne'
OUT_DIR = TBR_ROOT / 'figures'
WARMUP_S = 10.0

CONTROL_SUBJECTS = {4, 6, 8, 9, 15, 17, 18, 22, 24, 25, 30, 31, 32}
NEUROTUNE_SUBJECTS = {10, 11, 12, 13, 14, 16, 19, 20, 21, 27, 28, 29}

group_map: dict[str, int] = {}
for _n in CONTROL_SUBJECTS:
    group_map[f'S{_n:02d}'] = 1
for _n in NEUROTUNE_SUBJECTS:
    group_map[f'S{_n:02d}'] = 2
SUBJECTS = sorted(group_map.keys(), key=lambda s: int(s[1:]))


def _warmup_cut(t: np.ndarray, *arrs: np.ndarray):
    if len(t) == 0:
        return (t, *arrs)
    t0 = t[0] + WARMUP_S
    m = t >= t0
    return (t[m] - t0, *(a[m] for a in arrs))


def triband_props(tbr: np.ndarray, q33: float, q67: float) -> tuple[float, float, float]:
    """Fraction of samples in low / mid / high (mutually exclusive, sum to 1)."""
    if len(tbr) == 0:
        return (np.nan, np.nan, np.nan)
    low = tbr <= q33
    high = tbr >= q67
    mid = (tbr > q33) & (tbr < q67)
    return float(low.mean()), float(mid.mean()), float(high.mean())


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    rows = []
    for s in SUBJECTS:
        path = TBR_ROOT / f'{s}.csv'
        if not path.exists():
            continue
        df = pd.read_csv(path)
        t1 = df['condition 1 time (s)'].dropna().to_numpy()
        b1 = df['condition 1 tbr'].dropna().to_numpy()
        n = min(len(t1), len(b1))
        t1, b1 = t1[:n], b1[:n]
        t1, b1 = _warmup_cut(t1, b1)
        if len(b1) < 3:
            continue
        q33 = float(np.quantile(b1, 0.33))
        q67 = float(np.quantile(b1, 0.67))
        if not (q33 < q67):
            continue

        rec = {
            'subject': s,
            'group': group_map[s],
            'b1_q33': q33,
            'b1_q67': q67,
        }
        for c in (1, 2):
            tt = df[f'condition {c} time (s)'].dropna().to_numpy()
            tb = df[f'condition {c} tbr'].dropna().to_numpy()
            nn = min(len(tt), len(tb))
            tt, tb = tt[:nn], tb[:nn]
            tt, tb = _warmup_cut(tt, tb)
            lo, mid, hi = triband_props(tb, q33, q67)
            rec[f'c{c}_tbr_low_prop'] = lo
            rec[f'c{c}_tbr_mid_prop'] = mid
            rec[f'c{c}_tbr_high_prop'] = hi
        rows.append(rec)

    W = pd.DataFrame(rows)
    if len(W) == 0:
        raise SystemExit('No subjects with valid Block1 TBR; check TBR_ROOT CSV paths.')

    csv_path = TBR_ROOT / 'tbr_triband_block1ref.csv'
    W.to_csv(csv_path, index=False)
    print('saved:', csv_path, '| N =', len(W))

    W = W.copy()
    for band in ('low', 'mid', 'high'):
        W[f'd_{band}'] = W[f'c2_tbr_{band}_prop'] - W[f'c1_tbr_{band}_prop']

    def _improver_prop(sub: pd.DataFrame, band: str) -> float:
        if len(sub) == 0:
            return np.nan
        if band == 'low':
            return float((sub['d_low'] > 0).mean())
        if band == 'mid':
            return float((sub['d_mid'] < 0).mean())
        return float((sub['d_high'] < 0).mean())

    COL_LOW, COL_MID, COL_HIGH = '#4c72b0', '#c4c4c4', '#dd8452'
    labels = ('TBR ≤ Q33\n(low)', 'Q33–Q67\n(mid)', 'TBR ≥ Q67\n(high)')

    def means(g: int, c: int):
        sub = W[W['group'] == g]
        lo = sub[f'c{c}_tbr_low_prop'].mean()
        mid = sub[f'c{c}_tbr_mid_prop'].mean()
        hi = sub[f'c{c}_tbr_high_prop'].mean()
        return np.array([lo, mid, hi])

    xlabels = ['Baseline\nBlock1', 'NeuroTune\nBlock1', 'Baseline\nBlock2', 'NeuroTune\nBlock2']
    stacks = [
        means(1, 1), means(2, 1), means(1, 2), means(2, 2),
    ]

    # --- Fig A: stacked tri-band occupancy
    fig_a, ax0 = plt.subplots(figsize=(7.5, 5))
    x = np.arange(len(xlabels))
    bottom = np.zeros(len(xlabels))
    for lab, col, arr in zip(labels, [COL_LOW, COL_MID, COL_HIGH], np.stack(stacks, axis=1)):
        ax0.bar(x, arr, bottom=bottom, label=lab, color=col, edgecolor='white', width=0.72)
        bottom = bottom + arr
    ax0.set_xticks(x)
    ax0.set_xticklabels(xlabels, fontsize=9)
    ax0.set_ylabel('Mean proportion of time (within condition)')
    ax0.set_ylim(0, 1.0)
    ax0.axhline(1.0, color='k', lw=0.5, alpha=0.2)
    ax0.set_title('Tri-band occupancy (thresholds: Block1 TBR quantiles)')
    ax0.legend(loc='upper right', fontsize=8, frameon=False)
    fig_a.tight_layout()
    out_a = OUT_DIR / 'fig3a_tbr_triband_stacked.png'
    fig_a.savefig(out_a, bbox_inches='tight', dpi=200)
    print('saved:', out_a)
    plt.close(fig_a)

    # --- Fig B: improver rate (old fig3-style bars)
    fig_b, ax1 = plt.subplots(figsize=(7, 5))
    bx = np.arange(3)
    w = 0.36
    C1, C2 = '#a1c9f4', '#ffb482'
    sub1 = W[W['group'] == 1]
    sub2 = W[W['group'] == 2]
    for i, band in enumerate(('low', 'mid', 'high')):
        m1 = _improver_prop(sub1, band)
        m2 = _improver_prop(sub2, band)
        ax1.bar(i - w / 2, m1 * 100, w, color=C1, label='Baseline' if i == 0 else None, edgecolor='white')
        ax1.bar(i + w / 2, m2 * 100, w, color=C2, label='NeuroTune' if i == 0 else None, edgecolor='white')
        ax1.text(
            i - w / 2, (m1 if not np.isnan(m1) else 0) * 100 + 1.5,
            'n/a' if np.isnan(m1) else f'{m1 * 100:.0f}%', ha='center', fontsize=9,
        )
        ax1.text(
            i + w / 2, (m2 if not np.isnan(m2) else 0) * 100 + 1.5,
            'n/a' if np.isnan(m2) else f'{m2 * 100:.0f}%', ha='center', fontsize=9,
        )

    ax1.axhline(50, color='gray', lw=0.8, ls='--', alpha=0.7)
    ax1.set_xticks(bx)
    ax1.set_xticklabels(['Low\n(TBR≤Q33)', 'Mid\n(Q33<TBR<Q67)', 'High\n(Q67≤TBR)'])
    ax1.set_ylabel('Improved Subject Rate (%)')
    ax1.set_ylim(0, 100)
    # ax1.set_title('Improver rate (low: Δshare>0; mid/high: Δshare<0)')
    ax1.legend(frameon=False)
    fig_b.tight_layout()
    out_b = OUT_DIR / 'fig3b_tbr_triband_improver.png'
    fig_b.savefig(out_b, bbox_inches='tight', dpi=200)
    print('saved:', out_b)
    plt.close(fig_b)


if __name__ == '__main__':
    main()
