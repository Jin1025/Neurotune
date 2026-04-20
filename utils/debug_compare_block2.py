"""
Compare Block 2 data for all neurotune subjects between CSV and FIF pipelines.
Check: duration, number of samples, first/last values, mean TBR.
"""
import sys, re
from pathlib import Path
import numpy as np
import pandas as pd
import mne

mne.set_log_level("ERROR")

EXPERIMENT_ROOT = Path(r"C:\Users\MARG\Desktop\NeuroTune\Experiment")
FIF_DIR = Path(r"C:\Users\MARG\Desktop\condition2_fif")

NEUROTUNE_SUBJECTS = [10, 11, 12, 13, 14, 16, 19, 20, 21, 27, 28, 29]
FOLDER_TO_REAL = {32: 31, 33: 32}

CH_INDICES_CSV = [0, 1, 2, 3, 5, 6, 7]
CH_NAMES_FIF = ['EEG 1', 'EEG 2', 'EEG 3', 'EEG 4', 'EEG 6', 'EEG 7', 'EEG 8']


def _parse_ts(stem):
    m = re.match(r"Subject_\d+_ProcessedData_(.+)$", stem, re.I)
    if not m:
        return (0,)
    parts = m.group(1).split("_")
    if len(parts) < 6:
        return (0,)
    try:
        return tuple(int(p) for p in parts[:6])
    except ValueError:
        return (0,)


def find_block2_csv(subj_num):
    """Find block2 CSV file for a subject (files[4] in sorted order)."""
    folder_num = subj_num
    for k, v in FOLDER_TO_REAL.items():
        if v == subj_num:
            folder_num = k
            break

    subj_dir = EXPERIMENT_ROOT / f"S{folder_num:02d}"
    if not subj_dir.is_dir():
        subj_dir = EXPERIMENT_ROOT / f"S{subj_num:02d}"
    if not subj_dir.is_dir():
        return None

    pat = re.compile(r"^Subject_(\d+)_ProcessedData_.+\.csv$", re.I)
    files = []
    for p in subj_dir.iterdir():
        m = pat.match(p.name)
        if m and p.is_file():
            fnum = int(m.group(1))
            if fnum == folder_num or fnum == subj_num:
                files.append(p)
    files.sort(key=lambda p: _parse_ts(p.stem))

    if len(files) >= 5:
        return files[4]
    return None


def find_block2_fif(subj_num):
    """Find block2 FIF file."""
    p = FIF_DIR / f"S{subj_num:02d}_neurotune_2.fif"
    return p if p.exists() else None


def main():
    sys.stdout.reconfigure(encoding='utf-8')

    print(f"{'Subj':>5} | {'CSV file':>55} | {'CSV rows':>9} | {'CSV dur(s)':>10} | {'FIF file':>25} | {'FIF samp':>9} | {'FIF dur(s)':>10} | {'dur diff':>9} | {'samp diff':>10}")
    print("-" * 170)

    for sn in NEUROTUNE_SUBJECTS:
        csv_path = find_block2_csv(sn)
        fif_path = find_block2_fif(sn)

        csv_rows = "-"
        csv_dur = "-"
        fif_samp = "-"
        fif_dur = "-"
        dur_diff = "-"
        samp_diff = "-"
        csv_name = "-"
        fif_name = "-"

        if csv_path:
            df = pd.read_csv(csv_path, dtype=np.float64, low_memory=False)
            csv_rows = str(df.shape[0])
            dt_ms = df.iloc[:, 10].to_numpy(dtype=np.float64)
            csv_dur_val = np.sum(dt_ms) / 1000.0
            csv_dur = f"{csv_dur_val:.2f}"
            csv_name = csv_path.name

        if fif_path:
            raw = mne.io.read_raw_fif(fif_path, preload=False, verbose=False)
            fif_samp_val = raw.n_times
            fif_dur_val = raw.n_times / raw.info['sfreq']
            fif_samp = str(fif_samp_val)
            fif_dur = f"{fif_dur_val:.2f}"
            fif_name = fif_path.name

        if csv_path and fif_path:
            dur_diff = f"{csv_dur_val - fif_dur_val:.2f}"
            samp_diff = f"{int(df.shape[0]) - fif_samp_val}"

        print(f"S{sn:02d}  | {csv_name:>55} | {csv_rows:>9} | {csv_dur:>10} | {fif_name:>25} | {fif_samp:>9} | {fif_dur:>10} | {dur_diff:>9} | {samp_diff:>10}")

    print("\n" + "=" * 170)
    print("\nDetailed EEG data comparison (first 10 samples, ch1) for each neurotune Block 2:")
    print("=" * 170)

    for sn in NEUROTUNE_SUBJECTS:
        csv_path = find_block2_csv(sn)
        fif_path = find_block2_fif(sn)
        if not csv_path or not fif_path:
            print(f"\nS{sn:02d}: MISSING {'CSV' if not csv_path else 'FIF'}")
            continue

        df = pd.read_csv(csv_path, dtype=np.float64, low_memory=False)
        csv_ch1 = df.iloc[:10, 0].to_numpy()

        raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
        picks = mne.pick_channels(raw.ch_names, include=['EEG 1'])
        fif_data = raw.get_data(picks=picks)[0]
        fif_ch1 = fif_data[:10]

        csv_n = df.shape[0]
        fif_n = raw.n_times
        dt_ms = df.iloc[:, 10].to_numpy(dtype=np.float64)
        csv_total_s = np.sum(dt_ms) / 1000.0
        fif_total_s = raw.n_times / raw.info['sfreq']

        print(f"\nS{sn:02d}: CSV={csv_n} samples ({csv_total_s:.2f}s) | FIF={fif_n} samples ({fif_total_s:.2f}s) | diff={csv_n - fif_n} samples ({csv_total_s - fif_total_s:.2f}s)")
        print(f"  CSV ch1 first 10: {csv_ch1}")
        print(f"  FIF ch1 first 10: {fif_ch1}")

        scale = 1e-6
        csv_scaled = csv_ch1 * scale
        if np.allclose(csv_scaled, fif_ch1, atol=1e-10):
            print(f"  -> MATCH (CSV * 1e-6 == FIF)")
        elif np.allclose(csv_ch1, fif_ch1 / scale, atol=0.1):
            print(f"  -> MATCH (CSV == FIF * 1e6, i.e. CSV in uV, FIF in V)")
        else:
            diff = np.abs(csv_ch1 * scale - fif_ch1)
            print(f"  -> MISMATCH! max_diff={np.max(diff):.2e}")

    print("\n\n" + "=" * 170)
    print("Checking if S19 block2 CSV might have been concatenated from split files:")
    print("=" * 170)

    s19_dir = EXPERIMENT_ROOT / "S19"
    pat = re.compile(r"^Subject_19_ProcessedData_.+\.csv$", re.I)
    all_csvs = sorted([p for p in s19_dir.iterdir() if pat.match(p.name)], key=lambda p: _parse_ts(p.stem))
    print(f"\nAll S19 ProcessedData CSVs ({len(all_csvs)}):")
    for i, p in enumerate(all_csvs):
        df = pd.read_csv(p, dtype=np.float64, low_memory=False)
        dt_ms = df.iloc[:, 10].to_numpy(dtype=np.float64)
        dur = np.sum(dt_ms) / 1000.0
        print(f"  [{i}] {p.name}  rows={df.shape[0]}  dur={dur:.2f}s")

    # Check S19 block1 situation
    print(f"\nS19 FIF files in condition2_fif:")
    for p in sorted(FIF_DIR.glob("S19_*")):
        raw = mne.io.read_raw_fif(p, preload=False, verbose=False)
        print(f"  {p.name}  n_times={raw.n_times}  dur={raw.n_times / raw.info['sfreq']:.2f}s")

    # Also check condition1_fif for S19
    fif1_dir = Path(r"C:\Users\MARG\Desktop\condition1_fif")
    print(f"\nS19 FIF files in condition1_fif:")
    for p in sorted(fif1_dir.glob("S19_*")):
        raw = mne.io.read_raw_fif(p, preload=False, verbose=False)
        print(f"  {p.name}  n_times={raw.n_times}  dur={raw.n_times / raw.info['sfreq']:.2f}s")


if __name__ == "__main__":
    main()
