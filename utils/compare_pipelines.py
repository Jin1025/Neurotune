"""
Compare TBR pipelines: user (CSV-based) vs colleague (FIF-based).

User pipeline (modified):
  - Read ProcessedData CSV → add 0.1-50Hz bandpass + 60Hz notch
  - 1024-sample FFT, Hanning, 0.25s step, 4 channels
  - TBR = ratio-of-means (mean_theta / mean_beta)
  - EMA α=0.2

Colleague pipeline (modified):
  - Read pre-filtered FIF (already 0.1-50Hz + 60Hz notch)
  - 1024-sample FFT, Hanning, 0.25s step, 4 channels (EEG 1-4)
  - TBR = ratio-of-means (to match)
  - EMA α=0.2

Usage:
  conda run -n EEG python compare_pipelines.py S06
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import signal

try:
    import mne
except ImportError:
    raise SystemExit("mne 필요: conda install -n EEG mne")

# ── Paths ──
EXPERIMENT_ROOT = Path(r"C:\Users\MARG\Desktop\NeuroTune\Experiment")
CONDITION1_FIF = Path(r"C:\Users\MARG\Desktop\condition1_fif")
CONDITION2_FIF = Path(r"C:\Users\MARG\Desktop\condition2_fif")

# ── Subject number remapping (user CSV → actual) ──
# In Experiment folder: S32 is actually S31, S33 is actually S32
CSV_TO_REAL = {"S32": "S31", "S33": "S32"}
REAL_TO_CSV = {v: k for k, v in CSV_TO_REAL.items()}

# ── Shared params ──
FFT_SIZE = 1024
STEP_SEC = 0.25
EMA_ALPHA = 0.2
NUM_CH = 4

THETA_LO, THETA_HI = 4.0, 8.0
BETA_LO, BETA_HI = 13.0, 30.0

# ── Group mapping ──
CONTROL_SUBJECTS = {4, 6, 8, 9, 15, 17, 18, 22, 23, 24, 25, 30, 31, 32}
NEUROTUNE_SUBJECTS = {10, 11, 12, 13, 14, 16, 19, 20, 21, 27, 28, 29}


def _subj_num(s: str) -> int:
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else -1


def _is_control(subj_num: int) -> bool:
    return subj_num in CONTROL_SUBJECTS


# ──────────────────────────────────────────────
#  Common: sliding FFT → TBR (ratio-of-means) + EMA
# ──────────────────────────────────────────────

def compute_tbr_series(
    eeg_data: np.ndarray,
    fs: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    eeg_data: (n_channels, n_samples) — first NUM_CH channels used.
    Returns (time_s, ema_tbr) arrays.
    """
    n_ch = min(NUM_CH, eeg_data.shape[0])
    n_samples = eeg_data.shape[1]
    step_samples = max(1, int(round(fs * STEP_SEC)))

    window = np.hanning(FFT_SIZE).astype(np.float64)
    window_power = float(np.sum(window ** 2))

    freq = np.fft.rfftfreq(FFT_SIZE, d=1.0 / fs)
    mask_theta = (freq >= THETA_LO) & (freq < THETA_HI)
    mask_beta = (freq >= BETA_LO) & (freq < BETA_HI)

    ring = np.zeros((n_ch, FFT_SIZE), dtype=np.float64)
    write_idx = 0
    total = 0
    since_update = 0

    times_out = []
    ema_out = []
    ema_val = None

    for i in range(n_samples):
        for ch in range(n_ch):
            ring[ch, write_idx] = eeg_data[ch, i]
        write_idx = (write_idx + 1) % FFT_SIZE
        if total < FFT_SIZE:
            total += 1
        else:
            since_update += 1

        if total >= FFT_SIZE and since_update >= step_samples:
            theta_powers = []
            beta_powers = []
            for ch in range(n_ch):
                if write_idx == 0:
                    ordered = ring[ch, :]
                else:
                    ordered = np.concatenate((ring[ch, write_idx:], ring[ch, :write_idx]))
                buf = ordered * window
                spec = np.fft.rfft(buf)
                power = (np.abs(spec) ** 2) / window_power
                theta_powers.append(float(np.sum(power[mask_theta])))
                beta_powers.append(float(np.sum(power[mask_beta])))

            mean_theta = np.mean(theta_powers)
            mean_beta = max(np.mean(beta_powers), 1e-12)
            tbr = mean_theta / mean_beta

            t = i / fs
            if ema_val is None:
                ema_val = tbr
            else:
                ema_val = EMA_ALPHA * tbr + (1 - EMA_ALPHA) * ema_val

            times_out.append(t)
            ema_out.append(ema_val)

            since_update -= step_samples
            if since_update < 0:
                since_update = 0

    return np.array(times_out), np.array(ema_out)


# ──────────────────────────────────────────────
#  Pipeline A: User's CSV + filtering
# ──────────────────────────────────────────────

def _apply_filters(data: np.ndarray, fs: float) -> np.ndarray:
    """Bandpass 0.1-50 Hz + 60 Hz notch (same as preprocess.ipynb)."""
    # bandpass
    sos_bp = signal.butter(5, [0.1, 50.0], btype="band", fs=fs, output="sos")
    filtered = signal.sosfiltfilt(sos_bp, data, axis=1)
    # notch 60Hz
    b_notch, a_notch = signal.iirnotch(60.0, Q=30.0, fs=fs)
    filtered = signal.filtfilt(b_notch, a_notch, filtered, axis=1)
    return filtered


def run_user_pipeline(csv_path: Path, force_fs: float = 250.0) -> Tuple[np.ndarray, np.ndarray, float]:
    """Read ProcessedData CSV, filter, compute TBR. Returns (time, ema_tbr, fs)."""
    df = pd.read_csv(csv_path, dtype=np.float64, low_memory=False)
    if df.shape[1] < 11:
        raise ValueError(f"Too few columns: {df.shape[1]}")

    eeg = df.iloc[:, :8].to_numpy(dtype=np.float64)
    fs = force_fs

    eeg_transposed = eeg[:, :NUM_CH].T  # (n_ch, n_samples)

    # nan → 0
    eeg_transposed = np.nan_to_num(eeg_transposed, nan=0.0)

    eeg_filtered = _apply_filters(eeg_transposed, fs)

    times, ema = compute_tbr_series(eeg_filtered, fs)
    return times, ema, fs


# ──────────────────────────────────────────────
#  Pipeline B: Colleague's FIF (already filtered)
# ──────────────────────────────────────────────

def run_colleague_pipeline(fif_path: Path) -> Tuple[np.ndarray, np.ndarray, float]:
    """Read FIF, pick ch1-4, compute TBR with same params. Returns (time, ema_tbr, fs)."""
    raw = mne.io.read_raw_fif(str(fif_path), preload=True, verbose=False)

    ch_pick = [f"EEG {i}" for i in range(1, NUM_CH + 1)]
    available = [c for c in ch_pick if c in raw.ch_names]
    if len(available) < NUM_CH:
        raise ValueError(f"Only {len(available)} of {NUM_CH} channels found: {available}")
    raw.pick(available)

    fs = raw.info["sfreq"]
    data = raw.get_data()  # (n_ch, n_samples) in Volts

    times, ema = compute_tbr_series(data, fs)
    return times, ema, fs


# ──────────────────────────────────────────────
#  File resolution
# ──────────────────────────────────────────────

def _parse_csv_timestamp(stem: str) -> tuple:
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


def find_csv_files(subj_id: str) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """Find calibration (1st), condition1 (3rd), condition2 (5th) CSVs."""
    csv_folder_name = REAL_TO_CSV.get(subj_id, subj_id)
    subj_dir = EXPERIMENT_ROOT / csv_folder_name
    if not subj_dir.is_dir():
        subj_dir = EXPERIMENT_ROOT / subj_id
    if not subj_dir.is_dir():
        return None, None, None

    subj_num = _subj_num(csv_folder_name)
    pat = re.compile(r"^Subject_(\d+)_ProcessedData_.+\.csv$", re.I)
    files = []
    for p in subj_dir.iterdir():
        if p.is_file():
            m = pat.match(p.name)
            if m and int(m.group(1)) == subj_num:
                files.append(p)
    files.sort(key=lambda p: _parse_csv_timestamp(p.stem))

    if len(files) < 5:
        return None, None, None
    return files[0], files[2], files[4]


def find_fif_files(subj_id: str) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """Find cali, block1, block2 FIF files."""
    num = _subj_num(subj_id)
    is_ctrl = _is_control(num)

    if is_ctrl:
        fif_dir = CONDITION1_FIF
        tag = "sori"
    else:
        fif_dir = CONDITION2_FIF
        tag = "neurotune"

    prefix = f"S{num:02d}"

    def _find(pattern_part: str) -> Optional[Path]:
        candidates = list(fif_dir.glob(f"{prefix}*{pattern_part}*.fif"))
        if not candidates:
            candidates = list(fif_dir.glob(f"S{num}*{pattern_part}*.fif"))
        return candidates[0] if candidates else None

    cali = _find("cali")
    b1 = _find(f"_{tag}_1") or _find("_1")
    b2 = _find(f"_{tag}_2") or _find("_2")

    return cali, b1, b2


# ──────────────────────────────────────────────
#  Compare
# ──────────────────────────────────────────────

def compare_subject(subj_id: str) -> None:
    print(f"\n{'='*70}")
    print(f"  SUBJECT: {subj_id}")
    print(f"{'='*70}")

    csv_cal, csv_b1, csv_b2 = find_csv_files(subj_id)
    fif_cal, fif_b1, fif_b2 = find_fif_files(subj_id)

    csv_folder_name = REAL_TO_CSV.get(subj_id, subj_id)
    print(f"  CSV folder: {csv_folder_name}")
    print(f"  CSV cal : {csv_cal.name if csv_cal else 'NOT FOUND'}")
    print(f"  CSV b1  : {csv_b1.name if csv_b1 else 'NOT FOUND'}")
    print(f"  CSV b2  : {csv_b2.name if csv_b2 else 'NOT FOUND'}")
    print(f"  FIF cal : {fif_cal.name if fif_cal else 'NOT FOUND'}")
    print(f"  FIF b1  : {fif_b1.name if fif_b1 else 'NOT FOUND'}")
    print(f"  FIF b2  : {fif_b2.name if fif_b2 else 'NOT FOUND'}")

    for label, csv_path, fif_path in [
        ("Block 1", csv_b1, fif_b1),
        ("Block 2", csv_b2, fif_b2),
    ]:
        print(f"\n  --- {label} ---")
        if csv_path is None:
            print(f"    [SKIP] CSV not found")
            continue
        if fif_path is None:
            print(f"    [SKIP] FIF not found")
            continue

        try:
            t_csv, ema_csv, fs_csv = run_user_pipeline(csv_path)
            print(f"    User pipeline  : {len(t_csv)} points, fs={fs_csv:.1f}Hz, "
                  f"duration={t_csv[-1]:.1f}s")
        except Exception as e:
            print(f"    [ERR] User pipeline: {e}")
            t_csv, ema_csv = None, None

        try:
            t_fif, ema_fif, fs_fif = run_colleague_pipeline(fif_path)
            print(f"    Colleague pipe : {len(t_fif)} points, fs={fs_fif:.1f}Hz, "
                  f"duration={t_fif[-1]:.1f}s")
        except Exception as e:
            print(f"    [ERR] Colleague pipeline: {e}")
            t_fif, ema_fif = None, None

        if ema_csv is not None and ema_fif is not None:
            # summary stats
            print(f"\n    User CSV   - mean TBR: {np.mean(ema_csv):.4f}  "
                  f"median: {np.median(ema_csv):.4f}  std: {np.std(ema_csv):.4f}")
            print(f"    Colleague  - mean TBR: {np.mean(ema_fif):.4f}  "
                  f"median: {np.median(ema_fif):.4f}  std: {np.std(ema_fif):.4f}")

            diff_mean = abs(np.mean(ema_csv) - np.mean(ema_fif))
            ratio = np.mean(ema_csv) / max(np.mean(ema_fif), 1e-12)
            print(f"    |delta mean|  : {diff_mean:.4f}")
            print(f"    ratio         : {ratio:.4f}")

            # time-aligned correlation (truncate to shorter)
            n_min = min(len(ema_csv), len(ema_fif))
            if n_min > 10:
                corr = np.corrcoef(ema_csv[:n_min], ema_fif[:n_min])[0, 1]
                print(f"    Pearson r  : {corr:.4f} (first {n_min} points)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subjects", nargs="*", default=["S06", "S10"],
                    help="Subject IDs to compare (default: S06 S10)")
    args = ap.parse_args()

    for s in args.subjects:
        compare_subject(s)


if __name__ == "__main__":
    main()
