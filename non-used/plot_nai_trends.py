from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    raise SystemExit("matplotlib 필요: pip install matplotlib") from e


MEAN_VERSION_METRICS = [
    ("proportion_ge_thr", "Proportion ≥ 0.8"),
    ("mean_sustain_s", "Mean sustain (s)"),
    ("mean_reach_again_s", "Mean reach-again (s)"),
    ("first_reach_time_s", "First reach time (s)"),
]

MEDIAN_VERSION_METRICS = [
    ("proportion_ge_thr", "Proportion ≥ thr"),
    ("median_sustain_s", "Median sustain (s)"),
    ("median_reach_again_s", "Median reach-again (s)"),
    ("first_reach_time_s", "First reach time (s)"),
]


def _group_color(group: str) -> str:
    g = str(group).lower()
    if g == "control":
        return "#a1c9f4"  # seaborn pastel[0] (연한 파랑)
    return "#ffb482"  # seaborn pastel[1] (연한 주황)


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _prepare_pairs(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    sub = df[["subject", "group", "condition", metric]].copy()
    p = sub.pivot_table(index=["subject", "group"], columns="condition", values=metric, aggfunc="first").reset_index()
    p.columns.name = None
    p = p.rename(columns={1: "c1", 2: "c2"})
    p = p[np.isfinite(p["c1"]) & np.isfinite(p["c2"])].copy()
    return p


def _plot_slope_grid(df: pd.DataFrame, metrics: List[Tuple[str, str]], out_path: Path) -> None:
    n = len(metrics)
    if n == 4:
        fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.0))
        axes = list(np.ravel(axes))
    else:
        fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.2))
        if n == 1:
            axes = [axes]
        else:
            axes = list(np.ravel(axes))
    fig.patch.set_facecolor("white")

    for ax, (metric, label) in zip(axes, metrics):
        p = _prepare_pairs(df, metric)
        if p.empty:
            ax.set_title(label)
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_xticks([0, 1], ["Block 1", "Block 2"])
            continue

        # group mean line + SEM error bars (paper-friendly, no cluttered background lines)
        for group in ("control", "neurotune"):
            g = p[p["group"].str.lower() == group]
            if g.empty:
                continue
            c1 = g["c1"].to_numpy(dtype=np.float64)
            c2 = g["c2"].to_numpy(dtype=np.float64)
            m1 = float(np.nanmean(c1))
            m2 = float(np.nanmean(c2))
            se1 = float(np.nanstd(c1, ddof=1) / np.sqrt(len(c1))) if len(c1) > 1 else 0.0
            se2 = float(np.nanstd(c2, ddof=1) / np.sqrt(len(c2))) if len(c2) > 1 else 0.0
            color = _group_color(group)
            ax.plot([0, 1], [m1, m2], color=color, linewidth=2.0, marker="o", markersize=4.2)
            ax.errorbar([0, 1], [m1, m2], yerr=[se1, se2], fmt="none", ecolor=color, elinewidth=1.1, capsize=3)

        ax.set_title(label)
        ax.set_xticks([0, 1], ["Block 1", "Block 2"])
        ax.grid(alpha=0.12, axis="y")

    handles = [
        plt.Line2D([0], [0], color="#a1c9f4", lw=1.8, label="Sori Group"),
        plt.Line2D([0], [0], color="#ffb482", lw=1.8, label="NeuroTune Group"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.98))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot NAI trend figures from subject_metrics.csv")
    ap.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent / "tbr_nai_calculate" / "nai_threshold_report" / "subject_metrics.csv",
        help="analyze_nai_threshold.py 산출 subject_metrics.csv",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "tbr_nai_calculate" / "nai_threshold_report",
        help="그림 저장 폴더",
    )
    args = ap.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"[ERR] input not found: {args.input}")

    df = pd.read_csv(args.input, low_memory=False)
    required = {"subject", "group", "condition"} | {
        m for m, _ in (MEAN_VERSION_METRICS + MEDIAN_VERSION_METRICS)
    }
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise SystemExit(f"[ERR] required columns missing: {miss}")

    _setup_style()
    out_dir = args.out_dir
    _plot_slope_grid(
        df,
        MEAN_VERSION_METRICS,
        out_path=out_dir / "fig_nai_trend_mean.png",
    )
    _plot_slope_grid(
        df,
        MEDIAN_VERSION_METRICS,
        out_path=out_dir / "fig_nai_trend_median.png",
    )
    print(f"[OK] saved: {out_dir / 'fig_nai_trend_mean.png'}")
    print(f"[OK] saved: {out_dir / 'fig_nai_trend_median.png'}")


if __name__ == "__main__":
    main()

