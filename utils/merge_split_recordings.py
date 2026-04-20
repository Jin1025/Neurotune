"""
끊긴 녹음 두 개를 시간 순으로 이어 하나의 ProcessedData CSV로 만든다.
- 첫 번째 파일명으로 병합본을 저장하고, 원본 두 파일은 <Sxx>/split_segments/ 로 이동한다.
- DT(ms)는 그대로 이어 붙이므로 누적 시간(cumsum)은 자연스럽게 이어진다.
- CNT(9번째 열)가 두 번째 파일에서 리셋된 경우 이어 붙인다.

이후 같은 폴더에서 tbr_calculate.py 를 돌리면 8개가 아닌 6개 세션 기준으로 1·3·5번째가 맞는다.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

EXPERIMENT_ROOT = Path(r"C:\Users\MARG\Desktop\NeuroTune\Experiment")

# (폴더명, 첫 CSV, 둘째 CSV) — 첫 파일이 시간상 앞
MERGE_PAIRS: List[Tuple[str, str, str]] = [
    ("S10", "Subject_10_ProcessedData_27_11_2025_18_04_490.csv", "Subject_10_ProcessedData_27_11_2025_18_07_050.csv"),
    ("S11", "Subject_11_ProcessedData_28_11_2025_11_33_280.csv", "Subject_11_ProcessedData_28_11_2025_11_41_080.csv"),
    ("S13", "Subject_13_ProcessedData_28_11_2025_15_35_590.csv", "Subject_13_ProcessedData_28_11_2025_15_37_010.csv"),
    ("S18", "Subject_18_ProcessedData_29_11_2025_13_06_480.csv", "Subject_18_ProcessedData_29_11_2025_13_13_230.csv"),
    ("S19", "Subject_19_ProcessedData_30_11_2025_13_22_060.csv", "Subject_19_ProcessedData_30_11_2025_13_24_070.csv"),
    ("S25", "Subject_25_ProcessedData_01_12_2025_19_05_570.csv", "Subject_25_ProcessedData_01_12_2025_19_09_410.csv"),
    ("S29", "Subject_29_ProcessedData_03_12_2025_15_21_460.csv", "Subject_29_ProcessedData_03_12_2025_15_23_290.csv"),
    ("S30", "Subject_30_ProcessedData_04_12_2025_11_17_490.csv", "Subject_30_ProcessedData_04_12_2025_11_28_180.csv"),
]


def _continue_cnt(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """CNT가 열 인덱스 8이라고 가정하고, df2가 리셋된 경우 오프셋 적용."""
    if df1.shape[1] < 9 or df2.shape[1] < 9:
        return df2
    out = df2.copy()
    try:
        c1 = df1.iloc[:, 8].to_numpy(dtype=np.float64)
        c2 = out.iloc[:, 8].to_numpy(dtype=np.float64)
        if c1.size == 0 or c2.size == 0:
            return out
        last1 = float(c1[-1])
        first2 = float(c2[0])
        if first2 <= last1 + 0.5:
            offset = last1 + 1.0 - first2
            out.iloc[:, 8] = c2 + offset
    except Exception:
        pass
    return out


def merge_pair(
    subject_dir: Path,
    first_name: str,
    second_name: str,
    dry_run: bool,
) -> bool:
    p1 = subject_dir / first_name
    p2 = subject_dir / second_name
    if not p1.is_file():
        print(f"[SKIP] 없음: {p1}")
        return False
    if not p2.is_file():
        print(f"[SKIP] 없음: {p2}")
        return False

    df1 = pd.read_csv(p1, low_memory=False)
    df2 = pd.read_csv(p2, low_memory=False)
    if df1.columns.tolist() != df2.columns.tolist():
        print(f"[ERR] 헤더 불일치: {p1.name} vs {p2.name}")
        return False

    df2_adj = _continue_cnt(df1, df2)
    merged = pd.concat([df1, df2_adj], ignore_index=True)

    archive = subject_dir / "split_segments"
    if dry_run:
        print(f"[DRY] {subject_dir.name}: {len(df1)} + {len(df2)} → {len(merged)} rows → {p1.name}")
        return True

    archive.mkdir(parents=True, exist_ok=True)
    shutil.move(str(p1), str(archive / p1.name))
    shutil.move(str(p2), str(archive / p2.name))
    merged.to_csv(p1, index=False, encoding="utf-8-sig")
    print(f"[OK] {subject_dir.name}: {len(merged)} rows → {p1.name} (원본 → {archive.name}/)")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="끊긴 ProcessedData CSV 두 개를 하나로 병합")
    ap.add_argument(
        "experiment_root",
        nargs="?",
        default=None,
        help=f"Experiment 루트 (기본: {EXPERIMENT_ROOT})",
    )
    ap.add_argument("--dry-run", action="store_true", help="이동/쓰기 없이 계획만 출력")
    args = ap.parse_args()
    root = Path(args.experiment_root or EXPERIMENT_ROOT)
    if not root.is_dir():
        print(f"[ERR] 폴더 없음: {root}")
        sys.exit(1)

    for folder, f1, f2 in MERGE_PAIRS:
        d = root / folder
        if not d.is_dir():
            print(f"[SKIP] 폴더 없음: {d}")
            continue
        merge_pair(d, f1, f2, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
