"""
NeuroTune 그룹(12명) NAI 시계열: block1 → block2 이어붙임, 각 블록 warmup 10s 제거.

EMA(이전 스무딩에 decay 가중, 기본 0.99) 기준으로 선형 y trajectory를 그림.

데이터: analys/tbr_nai_calculate/Sxx.csv (기본)
출력:
  - analys/tbr_nai_calculate/figures/nai_trajectory_neurotune.png
  - analys/tbr_nai_calculate/figures/nai_trajectory_neurotune_no_raw.png
  - analys/tbr_nai_calculate/figures/nai_trajectory_neurotune_all12.png

Usage:
    python plot_nai_trajectory.py
    python plot_nai_trajectory.py --ema-decay 0.99 --out /path/to/out.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

NEUROTUNE_SUBJECTS = (10, 11, 12, 13, 14, 16, 19, 20, 21, 27, 28, 29)
WARMUP_S = 10.0

ROOT = Path(__file__).resolve().parent / "tbr_nai_calculate"
DEFAULT_OUT_GRID = ROOT / "figures" / "nai_trajectory_neurotune.png"
DEFAULT_OUT_GRID_NO_RAW = ROOT / "figures" / "nai_trajectory_neurotune_no_raw.png"
DEFAULT_OUT_ALL = ROOT / "figures" / "nai_trajectory_neurotune_all12.png"

# EMA: smoothed[t] = decay * smoothed[t-1] + (1 - decay) * x[t]
DEFAULT_EMA_DECAY = 0.99


def _warmup_trim(t: np.ndarray, y: np.ndarray, warmup_s: float) -> tuple[np.ndarray, np.ndarray]:
    """첫 warmup_s 초 제거 후 시간을 블록 시작 0으로 맞춤."""
    if len(t) == 0:
        return t, y
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    if len(t) == 0:
        return t, y
    t0 = t[0] + warmup_s
    m = t >= t0
    t, y = t[m], y[m]
    if len(t) == 0:
        return t, y
    t = t - t[0]
    return t, y


def load_concat_nai(csv_path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """
    condition 1, 2 NAI를 warmup 제거 후 x축으로 이어붙임.
    Returns: x_sec, nai, boundary_sec (block1 끝 = block2 시작 직전 시각)
    """
    df = pd.read_csv(csv_path)
    t1 = df["condition 1 time (s)"].to_numpy(dtype=float)
    n1 = df["condition 1 nai"].to_numpy(dtype=float)
    t2 = df["condition 2 time (s)"].to_numpy(dtype=float)
    n2 = df["condition 2 nai"].to_numpy(dtype=float)

    t1, n1 = _warmup_trim(t1, n1, WARMUP_S)
    t2, n2 = _warmup_trim(t2, n2, WARMUP_S)
    if len(t1) == 0 or len(t2) == 0:
        return np.array([]), np.array([]), np.nan

    end1 = float(t1[-1])
    x = np.concatenate([t1, end1 + t2])
    nai = np.concatenate([n1, n2])
    return x, nai, end1


def ema_series(x: np.ndarray, decay: float) -> np.ndarray:
    """지수 이동 평균. decay→1일수록 더 부드럽게."""
    if len(x) == 0:
        return x
    out = np.empty_like(x, dtype=float)
    out[0] = float(x[0])
    d = float(decay)
    for i in range(1, len(x)):
        out[i] = d * out[i - 1] + (1.0 - d) * float(x[i])
    return out


def y_for_axis(nai: np.ndarray) -> np.ndarray:
    """y축 고정: 선형 NAI (0~1 clip)."""
    return np.clip(nai.astype(float), 0.0, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-grid", type=Path, default=DEFAULT_OUT_GRID, help="4x3(raw+EMA) 저장 경로")
    ap.add_argument(
        "--out-grid-no-raw",
        type=Path,
        default=DEFAULT_OUT_GRID_NO_RAW,
        help="4x3(EMA only) 저장 경로",
    )
    ap.add_argument("--out-all", type=Path, default=DEFAULT_OUT_ALL, help="12명 합본 저장 경로")
    ap.add_argument(
        "--root",
        type=Path,
        default=None,
        help="CSV 폴더 (기본: analys/tbr_nai_calculate)",
    )
    ap.add_argument(
        "--ema-decay",
        type=float,
        default=DEFAULT_EMA_DECAY,
        help="EMA에서 이전 스무딩 가중 (기본 0.99, 클수록 더 매끈)",
    )
    args = ap.parse_args()

    data_root = args.root if args.root is not None else ROOT
    data_root.mkdir(parents=True, exist_ok=True)
    args.out_grid.parent.mkdir(parents=True, exist_ok=True)
    args.out_grid_no_raw.parent.mkdir(parents=True, exist_ok=True)
    args.out_all.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    boundary_style = dict(color="0.35", ls="--", lw=1.0, zorder=1)

    y_min_g, y_max_g = np.inf, -np.inf
    traces = []

    for sid in NEUROTUNE_SUBJECTS:
        label = f"S{sid:02d}"
        path = data_root / f"{label}.csv"
        if not path.exists():
            continue

        x_sec, nai, b_sec = load_concat_nai(path)
        if len(x_sec) == 0:
            continue

        smooth = ema_series(nai, args.ema_decay)
        y_raw = y_for_axis(nai)
        y_smt = y_for_axis(smooth)
        y_min_g = min(y_min_g, float(np.nanmin(y_raw)), float(np.nanmin(y_smt)))
        y_max_g = max(y_max_g, float(np.nanmax(y_raw)), float(np.nanmax(y_smt)))

        x_min = x_sec / 60.0
        b_min = b_sec / 60.0
        traces.append((label, x_min, y_raw, y_smt, b_min))

    # Global y-range (same across all figures)
    pad = 0.0
    def _apply_common_axis(ax):
        ax.set_xlim(0.0, 20.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Time (min), block1 -> block2 (warmup 10 s removed per block)")
        ax.set_ylabel("NAI (0~1)")

    # 1) grid 4x3: raw + EMA (기존 logy 스타일)
    fig, axes = plt.subplots(4, 3, figsize=(12, 10), sharex=True, sharey=True)
    axes = np.ravel(axes)
    for i, (label, x_min, y_raw, y_smt, b_min) in enumerate(traces):
        ax = axes[i]
        color = "#DD8452"
        ax.plot(x_min, y_raw, color=color, lw=0.9, alpha=0.20, zorder=2)
        ax.plot(x_min, y_smt, color=color, lw=2.0, alpha=0.98, zorder=3)
        ax.axvline(b_min, **boundary_style)
        ax.set_title(label, fontsize=10)
    for j in range(len(traces), len(axes)):
        axes[j].set_visible(False)
    for ax in axes[: len(traces)]:
        ax.set_xlim(0.0, 20.0)
        if np.isfinite(y_min_g) and np.isfinite(y_max_g):
            ax.set_ylim(y_min_g - pad, y_max_g + pad)
    fig.supxlabel("Time (min), block1 -> block2 (warmup 10 s removed per block)", fontsize=11)
    fig.supylabel("NAI (0~1)", fontsize=11)
    fig.suptitle(f"NeuroTune trajectories, raw+EMA | EMA decay={args.ema_decay:.2f}", fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(args.out_grid, bbox_inches="tight")
    print("saved:", args.out_grid)
    plt.close(fig)

    # 2) grid 4x3: EMA only (raw 제거)
    fig, axes = plt.subplots(4, 3, figsize=(12, 10), sharex=True, sharey=True)
    axes = np.ravel(axes)
    for i, (label, x_min, _, y_smt, b_min) in enumerate(traces):
        ax = axes[i]
        color = "#DD8452"
        ax.plot(x_min, y_smt, color=color, lw=2.0, alpha=0.98, zorder=3)
        ax.axvline(b_min, **boundary_style)
        ax.set_title(label, fontsize=10)
    for j in range(len(traces), len(axes)):
        axes[j].set_visible(False)
    for ax in axes[: len(traces)]:
        ax.set_xlim(0.0, 20.0)
        if np.isfinite(y_min_g) and np.isfinite(y_max_g):
            ax.set_ylim(y_min_g - pad, y_max_g + pad)
    fig.supxlabel("Time (min), block1 -> block2 (warmup 10 s removed per block)", fontsize=11)
    fig.supylabel("NAI (0~1)", fontsize=11)
    fig.suptitle(f"NeuroTune trajectories, EMA only | EMA decay={args.ema_decay:.2f}", fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(args.out_grid_no_raw, bbox_inches="tight")
    print("saved:", args.out_grid_no_raw)
    plt.close(fig)

    # 3) all12 one-panel (EMA only)
    fig, ax = plt.subplots(figsize=(12, 6.5))
    cmap = plt.get_cmap("tab20")
    for i, (label, x_min, _, y_smt, b_min) in enumerate(traces):
        color = cmap(i % 20)
        ax.plot(x_min, y_smt, color=color, lw=2.0, alpha=0.95, zorder=3, label=label)
        ax.axvline(b_min, color=color, lw=0.7, alpha=0.25, ls="--", zorder=1)
    _apply_common_axis(ax)
    ax.set_title(f"NeuroTune all subjects on one panel (EMA decay={args.ema_decay:.2f})")
    ax.legend(ncol=4, frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(args.out_all, bbox_inches="tight")
    print("saved:", args.out_all)


if __name__ == "__main__":
    main()
