"""
Full statistical analysis of 7ch MNE TBR data.
Replicates colleague's analysis pipeline:
  1) Exclude S23
  2) Shapiro-Wilk normality test per group x block
  3) Levene homogeneity of variance test per block
  4) Mann-Whitney U test per block (sori vs neurotune)
  5) Mixed ANOVA (pingouin): within=block, between=group
  6) Post-hoc pairwise tests
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import shapiro, levene, mannwhitneyu
try:
    import pingouin as pg
except Exception:
    pg = None

ROOT = Path(__file__).resolve().parents[1] / 'data' / 'tbr_nai_calculate'

EXCLUDE = {23}

CONTROL = {4, 6, 8, 9, 15, 17, 18, 22, 24, 25, 30, 31, 32}  
NEUROTUNE = {10, 11, 12, 13, 14, 16, 19, 20, 21, 27, 28, 29}

group_map = {}
for n in CONTROL:
    group_map[f'S{n:02d}'] = 'sori'
for n in NEUROTUNE:
    group_map[f'S{n:02d}'] = 'neurotune'

SUBJECTS = sorted(group_map.keys(), key=lambda s: int(s[1:]))

WARMUP_S = 10.0


def _mean_median_after_warmup(df: pd.DataFrame, col_t: str, col_tbr: str, warmup_s: float = WARMUP_S):
    if col_t not in df.columns or col_tbr not in df.columns:
        return np.nan, np.nan
    t = df[col_t].to_numpy(dtype=float)
    v = df[col_tbr].to_numpy(dtype=float)
    n = min(len(t), len(v))
    t = t[:n]
    v = v[:n]
    ok = np.isfinite(t) & np.isfinite(v)
    t = t[ok]
    v = v[ok]
    if len(v) == 0:
        return np.nan, np.nan
    t0 = t[0] + warmup_s
    m = t >= t0
    if np.any(m):
        vv = v[m]
    else:
        vv = v
    return float(np.mean(vv)), float(np.median(vv))


# Build long-format dataframe: subject, condition, step(cali/1/2), mean_tbr
# All steps are computed after warmup 10s.
records = []
for s in SUBJECTS:
    p = ROOT / f'{s}.csv'
    if not p.exists():
        continue
    df = pd.read_csv(p)
    grp = group_map[s]

    step_specs = [
        ('cali', 'calibration time (s)', 'calibration tbr'),
        ('1', 'condition 1 time (s)', 'condition 1 tbr'),
        ('2', 'condition 2 time (s)', 'condition 2 tbr'),
    ]
    for step_label, col_t, col_tbr in step_specs:
        mean_tbr, median_tbr = _mean_median_after_warmup(df, col_t, col_tbr, warmup_s=WARMUP_S)
        if not np.isfinite(mean_tbr):
            continue
        records.append({
            'subject': s,
            'condition': grp,
            'step': step_label,
            'mean_tbr': mean_tbr,
            'median_tbr': median_tbr,
        })

df_main = pd.DataFrame(records)


def run_analysis(df_input, label, out_lines):
    out_lines.append('=' * 80)
    out_lines.append(f'  {label}')
    out_lines.append('=' * 80)

    n_sori = df_input[df_input['condition'] == 'sori']['subject'].nunique()
    n_nt = df_input[df_input['condition'] == 'neurotune']['subject'].nunique()
    out_lines.append(f'  Subjects: sori n={n_sori}, neurotune n={n_nt}, total={n_sori + n_nt}')
    out_lines.append('')

    # 1) Shapiro-Wilk normality
    out_lines.append('  --- 1. Normality Test (Shapiro-Wilk) ---')
    out_lines.append(f'  {"condition":>12}  {"step":>5}  {"W":>10}  {"p":>10}  {"normal?":>10}')
    all_normal = True
    for cond in ['sori', 'neurotune']:
        for step in sorted(df_input['step'].unique()):
            data = df_input[(df_input['condition'] == cond) & (df_input['step'] == step)]['mean_tbr']
            if len(data) < 3:
                out_lines.append(f'  {cond:>12}  {step:>5}  {"N/A":>10}  {"N/A":>10}  {"N/A":>10}')
                continue
            stat, p = shapiro(data)
            normal = 'Yes' if p >= 0.05 else 'No'
            if p < 0.05:
                all_normal = False
            out_lines.append(f'  {cond:>12}  {step:>5}  {stat:>10.6f}  {p:>10.6f}  {normal:>10}')
    out_lines.append(f'  => All groups normal? {"Yes" if all_normal else "No"}')
    out_lines.append('')

    # 2) Levene homogeneity of variance
    out_lines.append('  --- 2. Homogeneity of Variance (Levene) ---')
    out_lines.append(f'  {"step":>5}  {"Levene_F":>10}  {"p":>10}  {"equal var?":>12}')
    all_equal_var = True
    for step in sorted(df_input['step'].unique()):
        sori = df_input[(df_input['condition'] == 'sori') & (df_input['step'] == step)]['mean_tbr']
        nt = df_input[(df_input['condition'] == 'neurotune') & (df_input['step'] == step)]['mean_tbr']
        stat, p = levene(sori, nt)
        eq = 'Yes' if p >= 0.05 else 'No'
        if p < 0.05:
            all_equal_var = False
        out_lines.append(f'  {step:>5}  {stat:>10.6f}  {p:>10.6f}  {eq:>12}')
    out_lines.append(f'  => All equal variance? {"Yes" if all_equal_var else "No"}')
    out_lines.append('')

    # 3) Mann-Whitney U per step
    out_lines.append('  --- 3. Mann-Whitney U Test (two-sided) ---')
    out_lines.append(f'  {"step":>5}  {"sori_mean":>12}  {"nt_mean":>12}  {"U":>8}  {"p":>10}  {"sig":>5}')
    for step in sorted(df_input['step'].unique()):
        sori = df_input[(df_input['condition'] == 'sori') & (df_input['step'] == step)]['mean_tbr']
        nt = df_input[(df_input['condition'] == 'neurotune') & (df_input['step'] == step)]['mean_tbr']
        U, p = mannwhitneyu(sori, nt, alternative='two-sided')
        sig = '*' if p < 0.05 else ('(*)' if p < 0.1 else 'ns')
        out_lines.append(f'  {step:>5}  {sori.mean():>12.6f}  {nt.mean():>12.6f}  {U:>8.1f}  {p:>10.6f}  {sig:>5}')
    out_lines.append('')

    # 4) Mixed ANOVA (if 2 steps)
    steps = sorted(df_input['step'].unique())
    if len(steps) >= 2:
        out_lines.append('  --- 4. Mixed ANOVA (within=step, between=condition) ---')
        if pg is None:
            out_lines.append('  [SKIP] pingouin is not installed; Mixed ANOVA skipped.')
        else:
            try:
                aov = pg.mixed_anova(
                    dv='mean_tbr',
                    within='step',
                    between='condition',
                    subject='subject',
                    data=df_input
                )
                for _, row in aov.iterrows():
                    src = row['Source']
                    F = row['F']
                    p_unc = row['p-unc'] if 'p-unc' in row else row.get('p_unc', float('nan'))
                    np2 = row['np2']
                    df1 = row['DF1']
                    df2 = row['DF2']
                    sig = '*' if p_unc < 0.05 else ('(*)' if p_unc < 0.1 else 'ns')
                    out_lines.append(f'  {src:>15}  F({df1:.0f},{df2:.0f})={F:.4f}  p={p_unc:.6f}  np2={np2:.6f}  {sig}')

                    # sphericity info if available
                    if 'sphericity' in row and row['sphericity'] is not None:
                        sph = row['sphericity']
                        if sph == False:
                            p_gg = row.get('p-GG-corr', row.get('p_GG_corr', float('nan')))
                            eps = row.get('eps', float('nan'))
                            out_lines.append(f'    -> Sphericity violated. GG-corrected p={p_gg:.6f}, eps={eps:.6f}')
            except Exception as e:
                out_lines.append(f'  [ERROR] {e}')
        out_lines.append('')

        # 5) Post-hoc
        out_lines.append('  --- 5. Post-hoc Pairwise Tests ---')
        if pg is None:
            out_lines.append('  [SKIP] pingouin is not installed; post-hoc tests skipped.')
        else:
            try:
                posthoc = pg.pairwise_tests(
                    dv='mean_tbr',
                    within='step',
                    between='condition',
                    subject='subject',
                    data=df_input,
                    padjust='bonf'
                )
                for _, row in posthoc.iterrows():
                    contrast = row['Contrast']
                    A = row['A']
                    B = row['B']
                    T = row['T']
                    p_unc = row['p-unc'] if 'p-unc' in row else row.get('p_unc', float('nan'))
                    p_corr = row.get('p-corr', row.get('p_corr', float('nan')))
                    hedges = row.get('hedges', float('nan'))
                    sig = '*' if p_unc < 0.05 else ('(*)' if p_unc < 0.1 else 'ns')
                    step_level = row.get('step', None)
                    show_step = pd.notna(step_level) and str(step_level).strip() not in {'', '-'}
                    step_tag = f' [step={step_level}]' if show_step else ''
                    out_lines.append(
                        f'  {contrast:>12}{step_tag}  {A} vs {B}  '
                        f'T={T:.4f}  p_unc={p_unc:.6f}  p_corr={p_corr:.6f}  hedges={hedges:.4f}  {sig}'
                    )
            except Exception as e:
                out_lines.append(f'  [ERROR] {e}')
        out_lines.append('')

    out_lines.append('')


lines = []
lines.append('=' * 80)
lines.append('TBR Statistical Analysis (7ch MNE Welch PSD)')
lines.append('  Filtering: bandpass 0.1-50Hz + notch 60Hz')
lines.append('  TBR = mean(theta) / mean(beta)  (ratio-of-means)')
lines.append('  7 channels (EEG 1-4,6-8), MNE Welch PSD, No EMA')
lines.append('  Excluded: S23 only')
lines.append(f'  Warmup removed in all steps: first {WARMUP_S:.0f}s')
lines.append('  Steps included: calibration, block1, block2')
lines.append('=' * 80)
lines.append('')

run_analysis(df_main, 'WARMUP 10s REMOVED (Calibration vs Block 1 vs Block 2)', lines)

out_path = ROOT / 'tbr_full_stats.txt'
out_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
print('\n'.join(lines))
print(f'\nSaved -> {out_path}')
