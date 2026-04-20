"""
ProcessedData CSV -> TBR(ratio-of-means) + filtering + MNE Welch PSD + ECDF + NAI.

Same as tbr_calculate_7ch_noema.py but replaces manual FFT with
mne.time_frequency.psd_array_welch (matching colleague's final_tbr_calculate.ipynb).

Parameters (aligned with colleague):
  - win_sec  = 4 s  (sliding window length)
  - step_sec = 1 s  (sliding step)
  - n_fft    = 1024
  - n_overlap= 512
  - window   = 'hann'
  - fmin=1, fmax=40
  - TBR = mean(sum(PSD_theta)*df) / mean(sum(PSD_beta)*df)  ratio-of-means
  - No EMA smoothing

Channels: EEG 1-4, 6-8 (EEG 5 excluded as bad channel)
Output: analys/tbr_nai_calculate_7ch_mne/Sxx.csv + IQR.txt
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import signal
import mne

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from metrics_core import ECDF, SlidingSpectrum 

EXPERIMENT_ROOT = Path(r"C:\Users\MARG\Desktop\NeuroTune\Experiment")
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "tbr_nai_calculate"

# MNE Welch parameters (same as colleague)
WIN_SEC = 4.0
STEP_SEC = 1.0
N_FFT = 1024
N_OVERLAP = 512
WINDOW_TYPE = "hann"
FMIN, FMAX = 1.0, 40.0

CH_INDICES = [0, 1, 2, 3, 5, 6, 7]
NUM_CHANNELS = len(CH_INDICES)  # 7
SAMPLE_RATE_HZ = 250.0
CALIB_TIME_START_SEC = 120.0

THETA_LO, THETA_HI = 4.0, 8.0
BETA_LO, BETA_HI = 13.0, 30.0

FOLDER_TO_REAL = {"S32": "S31", "S33": "S32"}


def piecewise_linear_nai(x: np.ndarray) -> np.ndarray:
    a = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    out = np.empty_like(a)
    m0 = a <= 0.2
    m1 = (a > 0.2) & (a <= 0.8)
    m2 = a > 0.8
    out[m0] = (0.15 / 0.2) * a[m0]
    out[m1] = 0.15 + (0.7 / 0.6) * (a[m1] - 0.2)
    out[m2] = 0.85 + (0.15 / 0.2) * (a[m2] - 0.8)
    return out


def _apply_filters(data: np.ndarray, fs: float) -> np.ndarray:
    """Bandpass 0.1-50 Hz + 60 Hz notch, same as preprocess.ipynb."""
    sos_bp = signal.butter(5, [0.1, 50.0], btype="band", fs=fs, output="sos")
    filtered = signal.sosfiltfilt(sos_bp, data, axis=1)
    b_notch, a_notch = signal.iirnotch(60.0, Q=30.0, fs=fs)
    filtered = signal.filtfilt(b_notch, a_notch, filtered, axis=1)
    return filtered


def run_tbr_series_mne_welch(
    df: pd.DataFrame,
    fs: float = SAMPLE_RATE_HZ,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read CSV, apply filter, compute TBR with MNE Welch PSD (no EMA).
    Sliding window: WIN_SEC with STEP_SEC step.
    Returns (time_centers_s, raw_tbr).
    """
    if df.shape[1] < 11:
        raise ValueError(f"columns < 11: {df.shape[1]}")

    eeg = df.iloc[:, :8].to_numpy(dtype=np.float64)
    dt_ms = df.iloc[:, 10].to_numpy(dtype=np.float64)
    cum_s = np.cumsum(dt_ms) / 1000.0

    eeg_sel = eeg[:, CH_INDICES].T  # (7, n_samples)
    eeg_sel = np.nan_to_num(eeg_sel, nan=0.0)
    eeg_sel = _apply_filters(eeg_sel, fs)

    win_samples = int(round(fs * WIN_SEC))
    step_samples = int(round(fs * STEP_SEC))
    n_samples = eeg_sel.shape[1]

    times_out: List[float] = []
    tbr_out: List[float] = []

    start = 0
    while start + win_samples <= n_samples:
        stop = start + win_samples
        data_win = eeg_sel[:, start:stop]  # (n_ch, win_samples)

        n_fft_use = min(N_FFT, data_win.shape[1])

        psds, freqs = mne.time_frequency.psd_array_welch(
            data_win,
            sfreq=fs,
            fmin=FMIN,
            fmax=FMAX,
            n_fft=n_fft_use,
            n_overlap=N_OVERLAP,
            window=WINDOW_TYPE,
            average="mean",
            verbose=False,
        )
        # psds shape: (n_ch, n_freqs)

        df_freq = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0

        theta_idx = (freqs >= THETA_LO) & (freqs < THETA_HI)
        beta_idx = (freqs >= BETA_LO) & (freqs < BETA_HI)

        theta_power = np.sum(psds[:, theta_idx], axis=1) * df_freq  # per channel
        beta_power = np.sum(psds[:, beta_idx], axis=1) * df_freq

        theta_mean = float(np.mean(theta_power))
        beta_mean = max(float(np.mean(beta_power)), 1e-30)
        tbr = theta_mean / beta_mean

        center_idx = (start + stop) // 2
        times_out.append(float(cum_s[min(center_idx, n_samples - 1)]))
        tbr_out.append(tbr)

        start += step_samples

    return np.array(times_out), np.array(tbr_out)


# ── File helpers ──

def _folder_subject_num(name: str) -> Optional[int]:
    m = re.fullmatch(r"S(\d+)", name.strip(), re.I)
    return int(m.group(1)) if m else None


def _parse_ts(stem: str) -> tuple:
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


def _list_csvs(subj_dir: Path, subj_num: int) -> List[Path]:
    pat = re.compile(r"^Subject_(\d+)_ProcessedData_.+\.csv$", re.I)
    files = [p for p in subj_dir.iterdir()
             if p.is_file() and pat.match(p.name) and int(pat.match(p.name).group(1)) == subj_num]
    files.sort(key=lambda p: _parse_ts(p.stem))
    return files


def _pad(arr: np.ndarray, n: int) -> np.ndarray:
    out = np.full(n, np.nan, dtype=np.float64)
    m = min(arr.size, n)
    out[:m] = arr[:m]
    return out


def calibration_q1_q3_and_pool(
    times_s: np.ndarray, tbr: np.ndarray,
) -> Tuple[float, float, List[float]]:
    mask = times_s >= CALIB_TIME_START_SEC
    pool = tbr[mask]
    if pool.size == 0:
        raise ValueError("No calibration samples after 120s")
    q1 = float(np.percentile(pool, 25))
    q3 = float(np.percentile(pool, 75))
    return q1, q3, pool.tolist()


def load_iqr_txt(path: Path) -> Dict[str, Tuple[float, float]]:
    out: Dict[str, Tuple[float, float]] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) >= 3 and re.match(r"^S\d+$", parts[0], re.I):
            try:
                out[parts[0]] = (float(parts[1]), float(parts[2]))
            except ValueError:
                pass
    return out


def write_iqr_txt(path: Path, data: Dict[str, Tuple[float, float]]) -> None:
    def sk(n: str):
        m = re.match(r"S(\d+)", n, re.I)
        return int(m.group(1)) if m else 999
    lines = [f"{n} {data[n][0]} {data[n][1]}" for n in sorted(data, key=sk)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def process_subject(
    subj_dir: Path, out_dir: Path,
) -> Optional[Tuple[str, float, float]]:
    folder_name = subj_dir.name
    real_name = FOLDER_TO_REAL.get(folder_name, folder_name)
    subj_num = _folder_subject_num(folder_name)
    if subj_num is None:
        return None

    files = _list_csvs(subj_dir, subj_num)
    if len(files) < 5:
        print(f"[SKIP] {folder_name}: CSV < 5 ({len(files)})")
        return None

    # S29: rest2 파일 누락으로 5개만 존재 → block2 = files[3]
    if subj_num == 29 and len(files) == 5:
        cal_path, c1_path, c2_path = files[0], files[2], files[3]
    else:
        cal_path, c1_path, c2_path = files[0], files[2], files[4]
    print(f"\n=== {folder_name} (-> {real_name}) ===")
    print(f"  cal : {cal_path.name}")
    print(f"  b1  : {c1_path.name}")
    print(f"  b2  : {c2_path.name}")

    df_cal = pd.read_csv(cal_path, dtype=np.float64, low_memory=False)
    df_c1 = pd.read_csv(c1_path, dtype=np.float64, low_memory=False)
    df_c2 = pd.read_csv(c2_path, dtype=np.float64, low_memory=False)

    t_cal, tbr_cal = run_tbr_series_mne_welch(df_cal)
    q1, q3, pool = calibration_q1_q3_and_pool(t_cal, tbr_cal)

    ecdf = ECDF()
    ecdf.fit(pool, q1, q3)
    print(f"  Q1={q1:.6f}  Q3={q3:.6f}  ECDF n={len(ecdf.sorted_vals)}")

    tc1, tbr1 = run_tbr_series_mne_welch(df_c1)
    tc2, tbr2 = run_tbr_series_mne_welch(df_c2)

    ri1 = np.array([ecdf.transform(float(v)) for v in tbr1])
    nai1 = piecewise_linear_nai(ri1)
    ri2 = np.array([ecdf.transform(float(v)) for v in tbr2])
    nai2 = piecewise_linear_nai(ri2)

    n = max(tc1.size, tc2.size)
    out_df = pd.DataFrame({
        "condition 1 time (s)": _pad(tc1, n),
        "condition 1 tbr": _pad(tbr1, n),
        "condition 1 nai": _pad(nai1, n),
        "condition 2 time (s)": _pad(tc2, n),
        "condition 2 tbr": _pad(tbr2, n),
        "condition 2 nai": _pad(nai2, n),
    })

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_out = out_dir / f"{real_name}.csv"
    out_df.to_csv(csv_out, index=False, encoding="utf-8-sig")
    print(f"  -> {csv_out} ({len(out_df)} rows)")
    return (real_name, q1, q3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment_root", nargs="?", default=None)
    ap.add_argument("-o", "--output-dir", type=Path, default=None)
    args = ap.parse_args()

    root = Path(args.experiment_root or os.environ.get("NEUROTUNE_EXPERIMENT_ROOT", str(EXPERIMENT_ROOT)))
    if not root.is_dir():
        print(f"[ERR] {root} not found")
        sys.exit(1)

    out_dir = args.output_dir or OUTPUT_DIR
    iqr_path = out_dir / "IQR.txt"
    iqr_state = load_iqr_txt(iqr_path)

    subs = sorted(
        [p for p in root.iterdir() if p.is_dir() and _folder_subject_num(p.name) is not None],
        key=lambda p: _folder_subject_num(p.name) or 0,
    )

    for d in subs:
        try:
            ret = process_subject(d, out_dir)
            if ret:
                sname, q1, q3 = ret
                iqr_state[sname] = (q1, q3)
                write_iqr_txt(iqr_path, iqr_state)
        except Exception as e:
            print(f"[ERR] {d.name}: {e}")
            import traceback; traceback.print_exc()

    print(f"\nDone. {len(iqr_state)} subjects -> {out_dir}")


if __name__ == "__main__":
    main()
