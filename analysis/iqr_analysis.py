"""
캘리브레이션 기반 개인별 IQR(Q1, Q3)을 로드해 personalization 가시화:
  - 피험자별 IQR 폭(Q3−Q1) 요약 표(CSV) + 집단 통계(txt)
  - 막대 그림(IQR 폭), 범위 플롯(Q1–Q3 구간), IQR 폭 분포 히스토그램

실행:
  python analyze_iqr_personalization.py
  python analyze_iqr_personalization.py --iqr path/to/IQR.txt --out path/to/output_dir
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
except ImportError as e:
    raise SystemExit("matplotlib 필요: pip install matplotlib") from e


_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
_FIGURES_ROOT = _ROOT / "figures"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from metrics_core import TBRCalculator  # type: ignore
except Exception as e:
    raise SystemExit("metrics_core import 실패: 경로를 확인하세요 (neurotune_ver1/analys 또는 상위 폴더에서 실행 권장).") from e


STEP_SECONDS = 0.25
FFT_SIZE = 1024
EMA_ALPHA = 0.2
NUM_CHANNELS = 4
CALIB_TIME_START_SEC = 120.0


def load_iqr_txt(path: Path) -> pd.DataFrame:
    rows: List[Tuple[str, float, float]] = []
    tag = re.compile(r"^S\d+$", re.IGNORECASE)
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 3 or not tag.match(parts[0]):
            continue
        try:
            rows.append((parts[0], float(parts[1]), float(parts[2])))
        except ValueError:
            continue
    if not rows:
        raise ValueError(f"유효한 행이 없음: {path}")

    norm: List[Tuple[str, float, float]] = []
    for name, q1, q3 in rows:
        m = re.match(r"S(\d+)", name, re.I)
        if not m:
            continue
        norm.append((f"S{int(m.group(1))}", q1, q3))

    df = pd.DataFrame(norm, columns=["subject", "Q1", "Q3"])
    df["IQR_width"] = df["Q3"] - df["Q1"]
    df["midpoint"] = (df["Q1"] + df["Q3"]) / 2.0
    df["subj_num"] = df["subject"].str.extract(r"S(\d+)").astype(int)
    df = df.sort_values("subj_num").reset_index(drop=True)
    return df


def _parse_processed_timestamp(stem: str) -> Tuple:
    """
    Subject_04_ProcessedData_24_11_2025_19_12_540 → 정렬용 튜플.
    접미사: d_m_y_h_min_frac (frac은 ms로 해석)
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
    micro = frac * 1000 if frac < 1000 else min(max(frac, 0), 999999)
    return (y, mo, d, h, mi, micro)


def _effective_sample_rate_hz(dt_ms: np.ndarray) -> float:
    d = np.asarray(dt_ms, dtype=np.float64)
    d = d[np.isfinite(d) & (d > 0)]
    if d.size == 0:
        return 250.0
    return float(1000.0 / np.median(d))


def _run_calibration_pool(df: pd.DataFrame, sample_rate_hz: Optional[float] = None) -> np.ndarray:
    """
    캘리브 세션 CSV에서 silence.py와 동일한 TBRCalculator로 EMA TBR을 구하고
    누적 120s 이후 업데이트 값만 반환.
    """
    if df.shape[1] < 11:
        raise ValueError(f"컬럼 수 부족 (DT=11번째 필요): {df.shape[1]}")
    eeg = df.iloc[:, :8].to_numpy(dtype=np.float64)
    dt_ms = df.iloc[:, 10].to_numpy(dtype=np.float64)
    cum_s = np.cumsum(dt_ms) / 1000.0
    fs = float(sample_rate_hz) if (sample_rate_hz is not None and sample_rate_hz > 0) else _effective_sample_rate_hz(dt_ms)

    calc = TBRCalculator(
        sample_rate_hz=fs,
        step_seconds=STEP_SECONDS,
        fft_size=FFT_SIZE,
        ema_alpha=EMA_ALPHA,
        num_channels=NUM_CHANNELS,
    )
    t_list: List[float] = []
    v_list: List[float] = []
    for i in range(eeg.shape[0]):
        row = eeg[i, :NUM_CHANNELS]
        if not np.all(np.isfinite(row)):
            continue
        ts = float(cum_s[i])
        updated = calc.add_sample([float(x) for x in row], ts)
        if updated and calc.has_value and calc.latest_ema is not None:
            t_list.append(ts)
            v_list.append(float(calc.latest_ema))
    if not v_list:
        return np.zeros(0, dtype=np.float64)
    t = np.asarray(t_list, dtype=np.float64)
    v = np.asarray(v_list, dtype=np.float64)
    return v[t >= CALIB_TIME_START_SEC]


def compute_pooled_global_iqr(
    experiment_root: Path,
    subjects: List[str],
    sample_rate_hz: Optional[float] = None,
) -> Tuple[float, float, int]:
    """
    pooled global IQR:
      - 각 subject 폴더의 ProcessedData 중 가장 이른 파일(캘리브)을 선택
      - 그 파일에서 EMA TBR 업데이트 값 중 누적 120s 이후를 풀링
      - 풀링된 모든 값에서 25/75 percentile
    """
    pooled: List[float] = []
    for s in subjects:
        m = re.match(r"S(\d+)", s, re.I)
        if not m:
            continue
        subj_num = int(m.group(1))
        # 폴더명이 S4 또는 S04일 수 있어 둘 다 허용
        d = experiment_root / s
        if not d.is_dir():
            d_alt = experiment_root / f"S{subj_num:02d}"
            if d_alt.is_dir():
                d = d_alt
        if not d.is_dir():
            continue
        pat = re.compile(r"^Subject_(\d+)_ProcessedData_.+\.csv$", re.IGNORECASE)
        files: List[Path] = []
        for p in d.iterdir():
            mm = pat.match(p.name) if p.is_file() else None
            if mm and int(mm.group(1)) == subj_num:
                files.append(p)
        if not files:
            continue
        files.sort(key=lambda p: _parse_processed_timestamp(p.stem))
        cal_path = files[0]
        df_cal = pd.read_csv(cal_path, dtype=np.float64, low_memory=False)
        pool = _run_calibration_pool(df_cal, sample_rate_hz=sample_rate_hz)
        if pool.size:
            pooled.extend(pool.tolist())

    if not pooled:
        raise ValueError("pooled global IQR 계산 실패: 풀링된 calibration 값이 없음")
    arr = np.asarray(pooled, dtype=np.float64)
    return float(np.percentile(arr, 25)), float(np.percentile(arr, 75)), int(arr.size)


def setup_pub_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def write_key_stats(
    df: pd.DataFrame,
    out_dir: Path,
    pooled_global_q1: Optional[float] = None,
    pooled_global_q3: Optional[float] = None,
    pooled_n: Optional[int] = None,
) -> None:
    """
    핵심 수치 저장.
      - mean_iqr, std_iqr, cv_iqr
      - pooled_global_q1, pooled_global_q3 (all calibration values pooled; 첫 120s 제외)
    """
    iqr_values = df["IQR_width"].to_numpy(dtype=np.float64)
    mean_iqr = float(np.mean(iqr_values))
    std_iqr = float(np.std(iqr_values, ddof=0))
    cv_iqr = float(std_iqr / mean_iqr) if mean_iqr != 0 else float("nan")

    stats = pd.DataFrame(
        [
            {
                "n_subjects": int(len(df)),
                "mean_iqr": mean_iqr,
                "std_iqr": std_iqr,
                "cv_iqr": cv_iqr,
                "pooled_global_q1": pooled_global_q1,
                "pooled_global_q3": pooled_global_q3,
                "pooled_n_values": pooled_n,
            }
        ]
    )
    stats.to_csv(out_dir / "key_stats.csv", index=False, encoding="utf-8-sig")

def plot_q1_q3_range(
    df: pd.DataFrame,
    out_base: Path,
    pooled_global_q1: Optional[float] = None,
    pooled_global_q3: Optional[float] = None,
) -> None:
    """각 피험자별 [Q1, Q3] 구간을 가로선으로 (같은 TBR 축 위에서 비교)."""
    fig, ax = plt.subplots(figsize=(8, max(4.0, len(df) * 0.22)))
    y = np.arange(len(df))
    print(pooled_global_q1, pooled_global_q3)
    if pooled_global_q1 is not None and pooled_global_q3 is not None:
        ax.axvspan(
            pooled_global_q1,
            pooled_global_q3,
            color="#a1c9f4",
            alpha=0.08,
            zorder=0,
            label="Global IQR",
        )
        ax.axvline(pooled_global_q1, linestyle="--", color="#a1c9f4", lw=1)
        ax.axvline(pooled_global_q3, linestyle="--", color="#a1c9f4", lw=1)

    for i, row in df.iterrows():
        ax.plot([row["Q1"], row["Q3"]], [i, i], color="#000000", lw=2, solid_capstyle="round")
        ax.scatter([row["Q1"], row["Q3"]], [i, i], color="#000000", s=10, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(df["subject"])
    ax.set_xlabel("Interquartile Range (IQR) of Calibration TBR")
    ax.set_ylabel("Subject ID")
    if pooled_global_q1 is not None and pooled_global_q3 is not None:
        ax.legend(loc="lower right", frameon=False)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=10))
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="IQR.txt 기반 personalization 요약 표·그림")
    ap.add_argument(
        "--iqr",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "nai_calculate" / "IQR.txt",
        help="IQR.txt 경로",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "nai_calculate" / "iqr_report",
        help="출력 폴더 (CSV, 통계, 그림)",
    )
    ap.add_argument(
        "--experiment-root",
        type=Path,
        default=Path(r"C:\Users\MARG\Desktop\m\marg\NeuroTune_data\Experiment"),
        help="pooled global IQR 계산용 Experiment 루트 (ProcessedData 원본)",
    )
    ap.add_argument(
        "--sample-rate",
        "-r",
        type=float,
        default=250.0,
        metavar="HZ",
        help="pooled 계산 시 TBRCalculator 샘플링레이트(Hz).",
    )
    args = ap.parse_args()

    if not args.iqr.is_file():
        raise SystemExit(f"파일 없음: {args.iqr}")

    df = load_iqr_txt(args.iqr)
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_pub_style()

    pooled_q1, pooled_q3, pooled_n = compute_pooled_global_iqr(
        args.experiment_root, subjects=df["subject"].tolist(), sample_rate_hz=args.sample_rate
    )

    table_csv = out_dir / "iqr_per_subject.csv"
    df[["subject", "Q1", "Q3", "IQR_width", "midpoint"]].to_csv(table_csv, index=False, encoding="utf-8-sig")

    write_key_stats(df, out_dir, pooled_global_q1=pooled_q1, pooled_global_q3=pooled_q3, pooled_n=pooled_n)

    fig_path = _FIGURES_ROOT / "iqr"
    plot_q1_q3_range(df, fig_path, pooled_global_q1=pooled_q1, pooled_global_q3=pooled_q3)

    print(f"저장: {out_dir}")
    print(f"  - {table_csv.name}")
    print("  - key_stats.csv")
    print(f"그림 저장: {fig_path.with_suffix('.png')}")


if __name__ == "__main__":
    main()
