import argparse
import math
import os
from typing import List, Optional, Tuple

import numpy as np


def try_read_numeric_first_line(path: str) -> Tuple[bool, int]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            first = f.readline()
    except UnicodeDecodeError:
        with open(path, "r", encoding="cp949") as f:
            first = f.readline()
    tokens = [t.strip() for t in first.replace("\t", ",").split(",") if t.strip() != ""]
    if len(tokens) == 0:
        return True, 0
    numeric = True
    for t in tokens:
        try:
            float(t)
        except Exception:
            numeric = False
            break
    return numeric, 0 if numeric else 1


def load_csv_as_float_matrix(path: str) -> np.ndarray:
    is_numeric, skip = try_read_numeric_first_line(path)
    try:
        arr = np.loadtxt(path, delimiter=",", dtype=float, ndmin=2, skiprows=skip)
    except UnicodeDecodeError:
        arr = np.loadtxt(path, delimiter=",", dtype=float, ndmin=2, skiprows=skip, encoding="cp949")
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def detect_time_column_and_fs(data: np.ndarray) -> Tuple[Optional[int], float]:
    if data.shape[1] == 0 or data.shape[0] < 3:
        return None, 0.0
    candidate = data[:, 0]
    diffs = np.diff(candidate.astype(float, copy=False))
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return None, 0.0
    median_dt = float(np.median(diffs))
    if median_dt <= 0:
        return None, 0.0
    # Allow reasonable dt (0.0005s ~ 0.5s) → fs in [2 Hz, 2000 Hz]
    if 0.0005 <= median_dt <= 0.5:
        fs = 1.0 / median_dt
        return 0, float(fs)
    return None, 0.0


def next_power_of_two(n: int) -> int:
    return 1 if n <= 1 else 1 << (int(n - 1).bit_length())


def compute_power_ratio_above(freq: np.ndarray, power: np.ndarray, cutoff_hz: float) -> Tuple[float, float]:
    if freq.size != power.size or freq.size == 0:
        return 0.0, 0.0
    mask_valid = np.isfinite(power)
    p_total = float(np.sum(power[mask_valid]))
    if p_total <= 0:
        return 0.0, 0.0
    mask_hi = (freq > cutoff_hz) & mask_valid
    p_hi = float(np.sum(power[mask_hi]))
    return p_hi, p_hi / p_total


def analyze_channels(
    data: np.ndarray,
    sample_rate_hz: float,
    time_col: Optional[int],
    cutoff_hz: float,
    max_channels: Optional[int] = 8,
) -> List[Tuple[int, float, float]]:
    if sample_rate_hz <= 0:
        raise ValueError("Sample rate must be positive to perform spectral analysis.")
    if time_col is None:
        channel_data = data
        channel_indices = list(range(data.shape[1]))
    else:
        channel_data = data[:, [i for i in range(data.shape[1]) if i != time_col]]
        channel_indices = [i for i in range(data.shape[1]) if i != time_col]
    # 앞에서부터 max_channels개만 사용 (1~8 컬럼 요구사항 대응; 시간 컬럼 제외 기준)
    if isinstance(max_channels, int) and max_channels > 0:
        use_k = min(max_channels, len(channel_indices))
        channel_indices = channel_indices[:use_k]
        channel_data = channel_data[:, :use_k]
    results: List[Tuple[int, float, float]] = []
    for idx_in_result, ch_idx in enumerate(channel_indices):
        x = channel_data[:, idx_in_result].astype(float, copy=False)
        x = x[np.isfinite(x)]
        if x.size < 8:
            results.append((ch_idx, 0.0, 0.0))
            continue
        x = x - float(np.mean(x))
        n = x.size
        n_fft = min(1 << 18, next_power_of_two(n))
        if n_fft > n:
            pad = n_fft - n
            xw = np.hanning(n_fft)
            xz = np.zeros(n_fft, dtype=float)
            xz[:n] = x
            x = xz * xw
        else:
            x = x[:n_fft]
            x = x * np.hanning(n_fft)
        spec = np.fft.rfft(x)
        power = (np.abs(spec) ** 2) / float(np.sum(np.hanning(n_fft) ** 2))
        freq = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate_hz)
        p_hi, ratio = compute_power_ratio_above(freq, power, cutoff_hz)
        results.append((ch_idx, p_hi, ratio))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Check for >cutoff Hz components in EEG CSV.")
    parser.add_argument(
        "--file",
        type=str,
        default=os.path.join("Sori", "input", "eeg", "clean.csv"),
        help="CSV 파일 경로. 기본: Sori/input/eeg/clean.csv",
    )
    parser.add_argument(
        "--fs",
        type=float,
        default=250.0,
        help="샘플레이트(Hz).",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=60.0,
        help="초과 주파수 컷오프(Hz). 기본 60",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-3,
        help="경고 기준(>cutoff 전력 / 총전력) 비율. 기본 1e-3",
    )
    parser.add_argument(
        "--first-n",
        type=int,
        default=8,
        help="시간 컬럼 제외 후 앞 N개 채널만 분석 (기본 8; 1~8 컬럼 요구사항)",
    )
    args = parser.parse_args()

    path = args.file
    if not os.path.exists(path):
        print(f"파일을 찾을 수 없습니다: {path}")
        return

    data = load_csv_as_float_matrix(path)
    # 시간 컬럼 없음 가정: 모든 컬럼을 채널로 사용
    time_col = None
    fs = float(args.fs) if args.fs and args.fs > 0 else 250.0
    print(f"fs = {fs:.3f} Hz 사용")

    results = analyze_channels(data, fs, time_col, args.cutoff, max_channels=args.first_n)
    any_warn = False
    print(f"> {args.cutoff:.1f} Hz 성분 비율 점검 (임계={args.threshold:g})")
    for ch_idx, p_hi, ratio in results:
        flag = "OK"
        if ratio > args.threshold:
            flag = "WARN"
            any_warn = True
        print(f"ch{ch_idx+1:02d}: ratio_above_{int(args.cutoff)}Hz = {ratio:.6g} ({flag})")

    if any_warn:
        print("결론: 일부 채널에서 컷오프 초과 성분이 임계치를 초과합니다.")
    else:
        print("결론: 컷오프 초과 성분이 임계치 이하입니다.")


if __name__ == "__main__":
    main()


