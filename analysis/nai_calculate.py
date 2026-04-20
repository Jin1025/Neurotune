"""
Experiment 폴더의 ProcessedData CSV를 neurotune/silence와 동일한 TBR 파이프라인으로 처리:
  TBRCalculator(0.25s step, fft 1024, ema 0.2, 4ch) → 캘리브는 누적 120s 이후 EMA TBR로 Q1/Q3
  → ECDF(metrics_core, [Q1,Q3] 안의 캘리브 샘플로 fit) → piecewise linear NAI
  → condition1/2 각각 누적 시간(s)·TBR·NAI를 같은 행 수로 나란히 저장(짧은 쪽은 NaN 패딩).
  → 결과 CSV는 tbr_nai_calculate/Sxx.csv (개별). Q1/Q3는 tbr_nai_calculate/IQR.txt 한 파일에
     피험자별 한 줄씩 누적(기존 행은 유지·같은 Sxx는 이번 실행 값으로 갱신 후 전체 덮어쓰기).

  TBR은 silence/neurotune과 동일하게 TBRCalculator(0.25s 스텝, fft 1024, ema 0.2, 4ch).
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

# 스크립트 위치 기준으로 metrics_core import
_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from metrics_core import ECDF, TBRCalculator  # noqa: E402

# --- 실험 루트 (입력 ProcessedData) ---
EXPERIMENT_ROOT = Path(r"C:\Users\MARG\Desktop\NeuroTune\Experiment")
# --- 결과 CSV / IQR.txt 저장 위치 ---
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "nai_calculate"

# neurotune.py / silence.py 와 동일
STEP_SECONDS = 0.25
FFT_SIZE = 1024
EMA_ALPHA = 0.2
NUM_CHANNELS = 4
CALIB_TIME_START_SEC = 120.0  # 2분 이후부터 Q1/Q3 및 ECDF용 캘리브 풀

def piecewise_linear_nai(x: np.ndarray | float) -> np.ndarray:
    """ECDF 출력(약 [0,1])에 대한 piecewise linear 매핑 (벡터화)."""
    a = np.asarray(x, dtype=np.float64)
    a = np.clip(a, 0.0, 1.0)
    out = np.empty_like(a, dtype=np.float64)
    m0 = a <= 0.2
    m1 = (a > 0.2) & (a <= 0.8)
    m2 = a > 0.8
    out[m0] = (0.15 / 0.2) * a[m0]
    out[m1] = 0.15 + (0.7 / 0.6) * (a[m1] - 0.2)
    out[m2] = 0.85 + (0.15 / 0.2) * (a[m2] - 0.8)
    return out


def _folder_subject_num(folder_name: str) -> Optional[int]:
    m = re.fullmatch(r"S(\d+)", folder_name.strip(), flags=re.IGNORECASE)
    if not m:
        return None
    return int(m.group(1))


def _parse_processed_timestamp(stem: str) -> Tuple:
    """
    Subject_04_ProcessedData_24_11_2025_19_12_540 → 정렬용 튜플.
    접미사: d_m_y_h_min_frac (frac은 ms로 해석해 microsecond에 반영)
    """
    m = re.match(r"Subject_\d+_ProcessedData_(.+)$", stem, flags=re.IGNORECASE)
    if not m:
        return (0,)
    parts = m.group(1).split("_")
    if len(parts) < 6:
        return (0,)
    try:
        d, mo, y, h, mi = (int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]))
        frac = int(parts[5])
    except ValueError:
        return (0,)
    # 예: ..._19_12_540 → 19:12:00.540 (540은 ms)
    if frac < 1000:
        micro = int(min(max(frac, 0), 999)) * 1000
    else:
        micro = int(min(max(frac, 0), 999999))
    return (y, mo, d, h, mi, micro)


def _list_subject_csvs(subject_dir: Path, subj_num: int) -> List[Path]:
    """파일명 Subject_<id>_ProcessedData_*.csv — id는 숫자만 일치하면 됨 (04 vs 4)."""
    pat = re.compile(r"^Subject_(\d+)_ProcessedData_.+\.csv$", re.IGNORECASE)
    files: List[Path] = []
    for p in subject_dir.iterdir():
        if not p.is_file():
            continue
        m = pat.match(p.name)
        if m and int(m.group(1)) == subj_num:
            files.append(p)
    files.sort(key=lambda p: _parse_processed_timestamp(p.stem))
    return files


def _read_processed_csv(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path, dtype=np.float64, low_memory=False)


def _effective_sample_rate_hz(dt_ms: np.ndarray) -> float:
    d = np.asarray(dt_ms, dtype=np.float64)
    d = d[np.isfinite(d) & (d > 0)]
    if d.size == 0:
        return 250.0
    return float(1000.0 / np.median(d))

def run_tbr_series(
    df: pd.DataFrame,
    sample_rate_hz: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    행 단위 EEG → TBRCalculator 스트리밍.
    반환: (누적시간_s 각 업데이트 시점, 해당 시점 EMA TBR)

    sample_rate_hz: None이면 DT 중앙값으로 fs 추정. 실험 LSL과 맞추려면 250.0 등 고정 권장.
    """
    if df.shape[1] < 11:
        raise ValueError(f"컬럼 수 부족 (DT=11번째 필요): {df.shape[1]}")

    eeg = df.iloc[:, :8].to_numpy(dtype=np.float64)
    dt_ms = df.iloc[:, 10].to_numpy(dtype=np.float64)

    cum_s = np.cumsum(dt_ms) / 1000.0
    if sample_rate_hz is not None and sample_rate_hz > 0:
        fs = float(sample_rate_hz)
    else:
        fs = _effective_sample_rate_hz(dt_ms)

    calc = TBRCalculator(
        sample_rate_hz=fs,
        step_seconds=STEP_SECONDS,
        fft_size=FFT_SIZE,
        ema_alpha=EMA_ALPHA,
        num_channels=NUM_CHANNELS,
    )

    times_out: List[float] = []
    ema_out: List[float] = []

    n = eeg.shape[0]
    for i in range(n):
        row = eeg[i, :NUM_CHANNELS]
        if not np.all(np.isfinite(row)):
            continue

        sample = [float(row[j]) for j in range(NUM_CHANNELS)]
        ts = float(cum_s[i])
        updated = calc.add_sample(sample, ts)
        if updated and calc.has_value and calc.latest_ema is not None:
            times_out.append(ts)
            ema_out.append(float(calc.latest_ema))

    return np.asarray(times_out, dtype=np.float64), np.asarray(ema_out, dtype=np.float64)


def calibration_q1_q3_and_pool(
    times_s: np.ndarray, ema_tbr: np.ndarray
) -> Tuple[float, float, List[float]]:
    """누적 120s 이후 EMA TBR만 모아 Q1, Q3 및 ECDF용 값 리스트."""
    mask = times_s >= CALIB_TIME_START_SEC
    pool = ema_tbr[mask]
    if pool.size == 0:
        raise ValueError("캘리브레이션에서 120s 이후 샘플이 없습니다.")
    q1 = float(np.percentile(pool, 25))
    q3 = float(np.percentile(pool, 75))
    return q1, q3, pool.astype(float).tolist()


def condition_tbr_nai(
    times_s: np.ndarray,
    ema_tbr: np.ndarray,
    ecdf: ECDF,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """전 구간 EMA TBR에 ECDF(re-index) + piecewise NAI."""
    reidx = np.array([ecdf.transform(float(v)) for v in ema_tbr], dtype=np.float64)
    nai = piecewise_linear_nai(reidx)
    return times_s, ema_tbr, nai


def _pad_to_length(arr: np.ndarray, n: int) -> np.ndarray:
    out = np.full(n, np.nan, dtype=np.float64)
    if arr.size:
        m = min(arr.size, n)
        out[:m] = arr[:m]
    return out


def load_iqr_txt(path: Path) -> Dict[str, Tuple[float, float]]:
    """IQR.txt → { 'S04': (q1, q3), ... }"""
    out: Dict[str, Tuple[float, float]] = {}
    if not path.is_file():
        return out
    tag = re.compile(r"^S\d+$", re.IGNORECASE)
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        if not tag.match(parts[0]):
            continue
        try:
            out[parts[0]] = (float(parts[1]), float(parts[2]))
        except ValueError:
            continue
    return out


def write_iqr_txt(path: Path, data: Dict[str, Tuple[float, float]]) -> None:
    """피험자 번호 순으로 한 줄씩 덮어쓰기."""
    def sort_key(name: str) -> Tuple[int, str]:
        m = re.match(r"S(\d+)", name, re.I)
        return (int(m.group(1)) if m else 999, name)

    lines: List[str] = []
    for name in sorted(data.keys(), key=sort_key):
        q1, q3 = data[name]
        lines.append(f"{name} {q1} {q3}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def build_condition_csv(
    t1: np.ndarray,
    tbr1: np.ndarray,
    nai1: np.ndarray,
    t2: np.ndarray,
    tbr2: np.ndarray,
    nai2: np.ndarray,
) -> pd.DataFrame:
    """조건별로 실제 누적 시간(s)과 TBR·NAI를 따로 열에 기록. 행 수는 긴 쪽에 맞추고 짧은 쪽은 NaN."""
    n = max(int(t1.size), int(t2.size))
    return pd.DataFrame(
        {
            "condition 1 time (s)": _pad_to_length(t1, n),
            "condition 1 tbr": _pad_to_length(tbr1, n),
            "condition 1 nai": _pad_to_length(nai1, n),
            "condition 2 time (s)": _pad_to_length(t2, n),
            "condition 2 tbr": _pad_to_length(tbr2, n),
            "condition 2 nai": _pad_to_length(nai2, n),
        }
    )


def process_subject_folder(
    subject_dir: Path,
    output_dir: Path,
    sample_rate_hz: Optional[float] = None,
) -> Optional[Tuple[str, float, float]]:
    """성공 시 (폴더명, q1, q3) 반환 — IQR.txt 병합용. 스킵/실패 시 None."""
    name = subject_dir.name
    subj_num = _folder_subject_num(name)
    if subj_num is None:
        print(f"[SKIP] 폴더명이 S숫자 형식이 아님: {name}")
        return None

    files = _list_subject_csvs(subject_dir, subj_num)
    if len(files) < 5:
        print(f"[SKIP] {name}: Subject_{subj_num:02d}_ProcessedData CSV가 5개 미만 ({len(files)}).")
        return None

    cal_path = files[0]
    c1_path = files[2]
    # S29: rest2 파일 누락으로 5개만 존재 → block2 = files[3]
    if subj_num == 29 and len(files) == 5:
        c2_path = files[3]
    else:
        c2_path = files[4]

    print(f"\n=== {name} ===")
    print(f"  cal (1st):  {cal_path.name}")
    print(f"  cond1 (3rd): {c1_path.name}")
    print(f"  cond2 (5th): {c2_path.name}")

    df_cal = _read_processed_csv(cal_path)
    df_c1 = _read_processed_csv(c1_path)
    df_c2 = _read_processed_csv(c2_path)

    t_cal, ema_cal = run_tbr_series(df_cal, sample_rate_hz=sample_rate_hz)
    q1, q3, pool = calibration_q1_q3_and_pool(t_cal, ema_cal)

    ecdf = ECDF()
    ecdf.fit(pool, q1, q3)
    fs_note = f"{sample_rate_hz:g} Hz (고정)" if sample_rate_hz else "DT→median"
    print(f"  fs={fs_note} | Q1={q1:.6f} Q3={q3:.6f} | ECDF n={len(ecdf.sorted_vals) if ecdf.sorted_vals is not None else 0}")

    tc1, ema1 = run_tbr_series(df_c1, sample_rate_hz=sample_rate_hz)
    tc2, ema2 = run_tbr_series(df_c2, sample_rate_hz=sample_rate_hz)
    t1, tbr1, nai1 = condition_tbr_nai(tc1, ema1, ecdf)
    t2, tbr2, nai2 = condition_tbr_nai(tc2, ema2, ecdf)

    out_df = build_condition_csv(t1, tbr1, nai1, t2, tbr2, nai2)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{name}.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  → 저장: {out_path} ({len(out_df)} rows)")
    return (name, q1, q3)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="ProcessedData CSV → TBR / ECDF / NAI → tbr_nai_calculate/Sxx.csv + IQR.txt"
    )
    ap.add_argument(
        "experiment_root",
        nargs="?",
        default=None,
        help=f"Experiment 폴더 (기본: 환경변수 NEUROTUNE_EXPERIMENT_ROOT 또는 {EXPERIMENT_ROOT})",
    )
    ap.add_argument(
        "--sample-rate",
        "-r",
        type=float,
        default=250.0,
        metavar="HZ",
        help="TBRCalculator용 샘플링레이트(Hz). 생략 시 CSV의 DT 중앙값으로 추정. LSL이 250이면 250 권장.",
    )
    ap.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help=f"CSV 및 IQR.txt 저장 폴더 (기본: {OUTPUT_DIR})",
    )
    args = ap.parse_args()
    root = Path(
        args.experiment_root
        or os.environ.get("NEUROTUNE_EXPERIMENT_ROOT", str(EXPERIMENT_ROOT))
    )
    if not root.is_dir():
        print(f"[ERR] 실험 루트가 없습니다: {root}")
        sys.exit(1)

    subs = sorted(
        [p for p in root.iterdir() if p.is_dir() and _folder_subject_num(p.name) is not None],
        key=lambda p: _folder_subject_num(p.name) or 0,
    )
    if not subs:
        print(f"[WARN] {root} 아래 S숫자 폴더가 없습니다.")
        return

    out_dir = Path(args.output_dir) if args.output_dir is not None else OUTPUT_DIR
    iqr_path = out_dir / "IQR.txt"
    iqr_state = load_iqr_txt(iqr_path)

    for d in subs:
        try:
            ret = process_subject_folder(d, out_dir, sample_rate_hz=args.sample_rate)
            if ret is not None:
                sname, q1, q3 = ret
                iqr_state[sname] = (q1, q3)
                write_iqr_txt(iqr_path, iqr_state)
                print(f"  → IQR.txt 갱신 ({len(iqr_state)}명)")
        except Exception as e:
            print(f"[ERR] {d.name}: {e}")


if __name__ == "__main__":
    main()
