"""
Figures for the metrics where Group 2 (intervention) is consistently superior
to Group 1 (control), based on condition-to-condition deltas (c2 - c1).

Outputs (saved to ./figures/):
  - delta_boxplots.png  : paired paired cond1→cond2 per subject, by group
                                for the headline metrics

Usage:
    python3 plot_tbr.py
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats

CONTROL_SUBJECTS = {4, 6, 8, 9, 15, 17, 18, 22, 24, 25, 30, 31, 32}
NEUROTUNE_SUBJECTS = {10, 11, 12, 13, 14, 16, 19, 20, 21, 27, 28, 29}

BASE_DIR = Path(__file__).resolve().parents[1]
ROOT = BASE_DIR / "data" / "tbr_calculate"
OUT = BASE_DIR / "figures"
OUT.mkdir(parents=True, exist_ok=True)

W = pd.read_csv(ROOT / 'subject_metrics_full.csv')

if 'group' not in W.columns:
    def _assign_grp(s):
        m = re.match(r'S(\d+)$', str(s).strip(), re.I)
        if not m:
            return None
        n = int(m.group(1))
        if n in CONTROL_SUBJECTS:
            return 1
        if n in NEUROTUNE_SUBJECTS:
            return 2
        return None
    W['group'] = W['subject'].apply(_assign_grp)
    W = W.dropna(subset=['group'])
    W['group'] = W['group'].astype(int)

# Color palette (colorblind-friendly)
C1 = '#a1c9f4'    # Group 1 - control
C2 = '#ffb482'   # Group 2 - intervention

plt.rcParams.update({
    'figure.dpi': 120,
    'savefig.dpi': 200,
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


# -------- helper --------
def mwu_annot(a, b, alternative='two-sided'):
    a = np.asarray(a); b = np.asarray(b)
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) < 3 or len(b) < 3:
        return np.nan
    _, p = stats.mannwhitneyu(a, b, alternative=alternative)
    return p


def stars(p):
    if np.isnan(p): return 'n/a'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    if p < 0.10: return '†'
    return 'ns'


# ============================================================
# Figure 1: paired before/after per subject, by group
# Metrics = headline metrics from the summary
# ============================================================
# (metric, title, better, y-axis unit label)
PAIRED_METRICS = [
    ('tbr_leQ33_n_episodes',      'TBR ≤ Q33 episodes',      'lower',  'Episodes (count)'),
    ('tbr_leQ33_eps_per_min',     'TBR ≤ Q33 eps per min',     'lower',  'Episodes / min'),
    ('tbr_leQ33_mean_sustain_s',  'TBR ≤ Q33 mean sustain',  'higher', 'Mean sustain (s)'),
]

fig, axes = plt.subplots(1, 3, figsize=(10, 4))
for ax_idx, (ax, (m, title, better, ylabel)) in enumerate(zip(axes.flat, PAIRED_METRICS)):
    c1 = W[f'c1_{m}']; c2 = W[f'c2_{m}']
    for grp, color, xoff in [(1, C1, -0.15), (2, C2, 0.15)]:
        sub = W[W['group'] == grp]
        for _, r in sub.iterrows():
            ax.plot([1 + xoff, 2 + xoff], [r[f'c1_{m}'], r[f'c2_{m}']],
                    color=color, alpha=0.35, lw=1.1)
            ax.scatter([1 + xoff, 2 + xoff], [r[f'c1_{m}'], r[f'c2_{m}']],
                       color=color, alpha=0.6, s=18, zorder=3)
        mean_label = 'Baseline mean' if grp == 1 else 'NeuroTune mean'
        ax.plot([1 + xoff, 2 + xoff], [sub[f'c1_{m}'].mean(), sub[f'c2_{m}'].mean()],
                color=color, lw=3.0, marker='o', markersize=7, zorder=4,
                markeredgecolor='white', markeredgewidth=1.2,
                label=mean_label)
    # annotate deltas
    d1 = (W[W['group']==1][f'c2_{m}'] - W[W['group']==1][f'c1_{m}']).dropna()
    d2 = (W[W['group']==2][f'c2_{m}'] - W[W['group']==2][f'c1_{m}']).dropna()
    p = mwu_annot(d1, d2, alternative='two-sided')
    ax.set_xticks([1, 2]); ax.set_xticklabels(['Block1', 'Block2'])
    ax.set_ylabel(ylabel)
    ax.set_title(f'{title}',
                 fontsize=10)
    ax.legend(fontsize=8, loc='best', frameon=False)
    ax.text(
        0.5, -0.09, f'({chr(ord("a") + ax_idx)})',
        transform=ax.transAxes, ha='center', va='top', fontsize=11,
    )

fig.tight_layout(rect=[0, 0.02, 1, 1])
out = OUT / 'tbr_delta_boxplots.png'
fig.savefig(out, bbox_inches='tight')
print('saved:', out)
plt.close(fig)


# ============================================================
# Figure 2: violin + strip of the delta (c2 - c1) by group
# ============================================================
# fig, axes = plt.subplots(2, 3, figsize=(14, 9))
# for ax, (m, title, better) in zip(axes.flat, PAIRED_METRICS):
#     d1 = (W[W['group']==1][f'c2_{m}'] - W[W['group']==1][f'c1_{m}']).dropna().to_numpy()
#     d2 = (W[W['group']==2][f'c2_{m}'] - W[W['group']==2][f'c1_{m}']).dropna().to_numpy()
#     parts = ax.violinplot([d1, d2], positions=[1, 2], widths=0.75, showmeans=False, showmedians=False, showextrema=False)
#     for body, color in zip(parts['bodies'], [C1, C2]):
#         body.set_facecolor(color); body.set_alpha(0.4); body.set_edgecolor(color)
#     # strip
#     for x, d, color in [(1, d1, C1), (2, d2, C2)]:
#         jitter = (np.random.RandomState(0).rand(len(d)) - 0.5) * 0.22
#         ax.scatter(np.full_like(d, x) + jitter, d, color=color, s=28, alpha=0.85, edgecolor='white', linewidth=0.6)
#         # mean bar
#         ax.plot([x-0.25, x+0.25], [d.mean(), d.mean()], color=color, lw=2.5)
#     ax.axhline(0, color='gray', lw=0.8, ls='--')
#     p_two = mwu_annot(d1, d2, alternative='two-sided')
#     # one-sided favoring group 2
#     if better == 'lower':   # delta should be more negative for group 2
#         p_fav = mwu_annot(d1, d2, alternative='greater')  # a > b
#     else:                   # delta should be more positive for group 2
#         p_fav = mwu_annot(d1, d2, alternative='less')     # a < b
#     ax.set_xticks([1, 2]); ax.set_xticklabels(['G1 (control)', 'G2 (intervention)'])
#     ax.set_ylabel('Δ (cond2 − cond1)')
#     ax.set_title(f'{title}\ntwo-sided p={p_two:.3f}   one-sided (G2 better) p={p_fav:.3f} {stars(p_fav)}',
#                  fontsize=10)
# fig.suptitle('Improvement magnitude (Δ c2−c1) by group', fontsize=14, fontweight='bold', y=1.00)
# fig.tight_layout()
# out = os.path.join(OUT, 'fig2_delta_violin.png')
# fig.savefig(out, bbox_inches='tight')
# print('saved:', out)
# plt.close(fig)


# ============================================================
# Figure 3: per-subject improver proportions across TBR quantile thresholds
# (same tbr_leQ* / tbr_geQ* as run_analysis.py; delta = c2−c1 on mean_sustain_s)
# Q < 50  → lower tail: tbr_leQ{thr}_*  (focus episodes: TBR ≤ Q_thr)
# Q > 50  → upper tail: tbr_geQ{thr}_*  (focus episodes: TBR ≥ Q_thr)
# Q == 50 → leQ50 (same convention as run_analysis)
# ============================================================
# def _fig3_sustain_metric(thr_pct: int) -> str:
#     if thr_pct > 50:
#         return f'tbr_geQ{thr_pct}_mean_sustain_s'
#     return f'tbr_leQ{thr_pct}_mean_sustain_s'


# def _fig3_xtick_label(thr_pct: int) -> str:
#     if thr_pct > 50:
#         return f'TBR ≥ Q{thr_pct}'
#     return f'TBR ≤ Q{thr_pct}'


# THRS = [33, 67]  # regenerate subject_metrics_full.csv after run_analysis adds geQ67
# prop_g1, prop_g2 = [], []
# for thr in THRS:
#     m = _fig3_sustain_metric(thr)
#     # Wide table uses c1_/c2_/d_ prefixes, not bare metric names
#     if f'c1_{m}' not in W.columns or f'c2_{m}' not in W.columns:
#         raise KeyError(
#             f'Missing columns c1_{m!r} / c2_{m!r}. Run analys/run_analysis.py to refresh '
#             f'subject_metrics_full.csv (needs tbr_leQ* and tbr_geQ* episode metrics).'
#         )
#     d1 = W[W['group'] == 1][f'c2_{m}'] - W[W['group'] == 1][f'c1_{m}']
#     d2 = W[W['group'] == 2][f'c2_{m}'] - W[W['group'] == 2][f'c1_{m}']
#     prop_g1.append((d1 > 0).mean())
#     prop_g2.append((d2 > 0).mean())

# fig, ax = plt.subplots(figsize=(8, 5))
# x = np.arange(len(THRS))
# w = 0.38
# ax.bar(x - w/2, np.array(prop_g1)*100, w, color=C1, label='Baseline', edgecolor='white')
# ax.bar(x + w/2, np.array(prop_g2)*100, w, color=C2, label='NeuroTune', edgecolor='white')
# ax.axhline(50, color='gray', lw=0.8, ls='--')
# for xi, (p1, p2) in enumerate(zip(prop_g1, prop_g2)):
#     ax.text(xi - w/2, p1*100 + 1.5, f'{p1*100:.0f}%', ha='center', fontsize=9)
#     ax.text(xi + w/2, p2*100 + 1.5, f'{p2*100:.0f}%', ha='center', fontsize=9)
# ax.set_xticks(x)
# ax.set_xticklabels([_fig3_xtick_label(t) for t in THRS])
# ax.set_ylabel('% of subjects whose focus-episode mean sustain\nincreased from cond1 to cond2')
# ax.set_title('Improver proportions across TBR quantile thresholds (≤ lower / ≥ upper tail)')
# ax.set_ylim(0, 80)
# ax.legend(frameon=False)
# fig.tight_layout()
# out = os.path.join(OUT, 'fig3_improver_bars.png')
# fig.savefig(out, bbox_inches='tight')
# print('saved:', out)
# plt.close(fig)


# # ============================================================
# # Figure 4: ANCOVA-like scatter: c2 vs c1, with group colors + fit line
# # ============================================================
# # fig, axes = plt.subplots(2, 3, figsize=(14, 9))
# # for ax, (m, title, better) in zip(axes.flat, PAIRED_METRICS):
# #     x = W[f'c1_{m}'].to_numpy()
# #     y = W[f'c2_{m}'].to_numpy()
# #     ok = ~np.isnan(x) & ~np.isnan(y)
# #     x, y = x[ok], y[ok]; grp = W.loc[ok, 'group'].to_numpy()
# #     for g, color, label in [(1, C1, 'G1'), (2, C2, 'G2')]:
# #         ax.scatter(x[grp==g], y[grp==g], color=color, s=48, alpha=0.8, edgecolor='white', linewidth=0.8, label=label)
# #     # y = x reference
# #     lims = [min(np.nanmin(x), np.nanmin(y)), max(np.nanmax(x), np.nanmax(y))]
# #     ax.plot(lims, lims, color='gray', ls='--', lw=0.9, label='y = x')
# #     # pooled regression
# #     slope, intercept, *_ = stats.linregress(x, y)
# #     xs = np.linspace(lims[0], lims[1], 50)
# #     ax.plot(xs, slope*xs + intercept, color='black', lw=1.2, label=f'fit (slope={slope:.2f})')
# #     # residuals by group
# #     resid = y - (slope*x + intercept)
# #     r1 = resid[grp==1]; r2 = resid[grp==2]
# #     p = mwu_annot(r1, r2, alternative='two-sided')
# #     ax.set_xlabel(f'cond 1 {m}'); ax.set_ylabel(f'cond 2 {m}')
# #     ax.set_title(f'{title}\nresid G1={r1.mean():+.2f}, G2={r2.mean():+.2f}  (p={p:.3f})', fontsize=10)
# #     ax.legend(fontsize=8, frameon=False)
# # fig.suptitle('ANCOVA-style: cond2 vs cond1 with pooled fit; group residual separation',
# #              fontsize=14, fontweight='bold', y=1.00)
# # fig.tight_layout()
# # out = os.path.join(OUT, 'fig4_ancova_residuals.png')
# # fig.savefig(out, bbox_inches='tight')
# # print('saved:', out)
# # plt.close(fig)


# # ============================================================
# # Figure 5: multi-threshold NAI sustain deltas (consistency across thresholds)
# # ============================================================
# # fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# # # Left: mean delta across thresholds
# # means_g1, means_g2, sem_g1, sem_g2 = [], [], [], []
# # for thr in THRS:
# #     m = f'nai_ge_{thr}_median_sustain_s'
# #     d1 = (W[W['group']==1][f'c2_{m}'] - W[W['group']==1][f'c1_{m}']).dropna()
# #     d2 = (W[W['group']==2][f'c2_{m}'] - W[W['group']==2][f'c1_{m}']).dropna()
# #     means_g1.append(d1.mean()); means_g2.append(d2.mean())
# #     sem_g1.append(d1.std(ddof=1)/np.sqrt(len(d1)))
# #     sem_g2.append(d2.std(ddof=1)/np.sqrt(len(d2)))

# # ax = axes[0]
# # ax.errorbar(THRS, means_g1, yerr=sem_g1, color=C1, marker='o', ms=8, lw=2, capsize=4, label='Group 1 (control)')
# # ax.errorbar(THRS, means_g2, yerr=sem_g2, color=C2, marker='s', ms=8, lw=2, capsize=4, label='Group 2 (intervention)')
# # ax.axhline(0, color='gray', lw=0.8, ls='--')
# # ax.set_xlabel('NAI threshold (deep-focus cutoff)')
# # ax.set_ylabel('Δ median sustain time (cond2 − cond1, s)')
# # ax.set_title('Deep-focus sustain improvement\nconsistent across thresholds')
# # ax.legend(frameon=False)

# # Right: p-values (one-sided favoring G2) across thresholds
# # pvals = []
# # for thr in THRS:
# #     m = f'nai_ge_{thr}_median_sustain_s'
# #     d1 = (W[W['group']==1][f'c2_{m}'] - W[W['group']==1][f'c1_{m}']).dropna().to_numpy()
# #     d2 = (W[W['group']==2][f'c2_{m}'] - W[W['group']==2][f'c1_{m}']).dropna().to_numpy()
# #     pvals.append(mwu_annot(d1, d2, alternative='less'))
# # ax = axes[1]
# # ax.plot(THRS, pvals, 'o-', color='#555', lw=2, ms=8)
# # ax.axhline(0.05, color='crimson', lw=1, ls='--', label='p = 0.05')
# # ax.axhline(0.10, color='orange', lw=1, ls=':', label='p = 0.10')
# # ax.set_xlabel('NAI threshold')
# # ax.set_ylabel('one-sided MWU p (G2 larger Δ)')
# # ax.set_title('Statistical support vs threshold')
# # ax.set_ylim(0, max(pvals)*1.2)
# # for xi, pv in zip(THRS, pvals):
# #     ax.text(xi, pv + 0.005, f'{pv:.3f}', ha='center', fontsize=8)
# # ax.legend(frameon=False)
# # fig.tight_layout()
# # out = os.path.join(OUT, 'fig5_multi_threshold.png')
# # fig.savefig(out, bbox_inches='tight')
# # print('saved:', out)
# # plt.close(fig)

# print('\nAll figures saved under', OUT)
