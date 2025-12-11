import argparse
import csv
import os
from typing import List, Optional, Tuple

from metrics_core import TBRCalculator, ABRCalculator


def get_input_dir() -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(base_dir, "..", "input"))


def detect_columns(header: List[str]) -> Tuple[List[int], Optional[int]]:
    """
    헤더에서 EEG 채널 컬럼들의 인덱스와, 'DT' 컬럼 인덱스를 찾는다.
    - EEG: 'EEG '로 시작하는 컬럼들(정렬된 순서)
    - DT: 정확히 'DT'
    """
    eeg_idx: List[int] = []
    dt_idx: Optional[int] = None
    for i, name in enumerate(header):
        n = (name or "").strip()
        if n.upper().startswith("EEG "):
            eeg_idx.append(i)
        elif n.upper() == "DT":
            dt_idx = i
    return eeg_idx, dt_idx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NeuroTune index calculator (offline CSV → calibration)")
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        required=True,
        help="입력 CSV 경로. 예: input/eeg.csv",
    )
    parser.add_argument(
        "--start-sec",
        type=float,
        default=120.0,
        help="이 시점(초) 이후 구간만 캘리브레이션에 반영 (기본 120초)",
    )
    parser.add_argument(
        "--fs",
        type=float,
        default=250.0,
        help="DT 컬럼 없을 때 사용할 샘플링 레이트(Hz), 기본 250.0",
    )
    return parser.parse_args()


def main() -> None:
    """
    - 입력: input/eeg.csv
    - 처리: DT(ms) 누적합이 120초가 되는 시점 이후부터 끝까지 사용
    - 계산: TBR/ABR (fs=250, step=0.25s, fft=1024, ema_alpha=0.2, ch 1~4)
    - 출력:
        - 'input/callibration_tbr'        (헤더 ['tbr'], 각 행에 TBR EMA 값)
        - 'input/callibration_abr'    (헤더 ['abr'], 각 행에 ABR EMA 값)
    """
    args = parse_args()

    input_dir = get_input_dir()
    try:
        os.makedirs(input_dir, exist_ok=True)
    except Exception:
        pass

    csv_arg = args.file
    if os.path.isabs(csv_arg):
        in_path = os.path.abspath(csv_arg)
    else:
        # 1) 현재 작업 디렉터리 기준
        if os.path.exists(csv_arg):
            in_path = csv_arg
        else:
            project_root = os.path.dirname(input_dir)
            candidate_project = os.path.join(project_root, csv_arg)
            starts_with_input = (
                csv_arg.startswith("input/") or
                csv_arg.startswith("input" + os.sep)
            )
            # 2) 프로젝트 루트 기준 일반 상대경로
            if os.path.exists(candidate_project):
                in_path = candidate_project
            # 3) 프로젝트 루트 기준 'input/...'로 주어진 경우
            elif starts_with_input:
                in_path = os.path.join(project_root, csv_arg)
            # 4) 마지막으로 input 폴더 내부에서 탐색
            else:
                in_path = os.path.join(input_dir, csv_arg)

    if not os.path.exists(in_path):
        print(f"[ERROR] input CSV not found: {in_path}")
        return

    # 파라미터 (metrics_core / neurotune와 동일)
    fs_hz = float(args.fs)
    step_seconds = 0.25
    fft_size = 1024
    ema_alpha = 0.2
    num_channels = 4

    # 캘리브레이션 출력 준비
    out_tbr = os.path.join(input_dir, "callibration_tbr")
    out_abr = os.path.join(input_dir, "callibration_abr")
    tbr_fh = None
    abr_fh = None
    tbr_writer = None
    abr_writer = None
    try:
        tbr_fh = open(out_tbr, "w", newline="")
        tbr_writer = csv.writer(tbr_fh)
        tbr_writer.writerow(["tbr"])
        tbr_fh.flush()
    except Exception as e:
        print(f"[WARN] failed to open '{out_tbr}': {e}")
        tbr_fh = None
        tbr_writer = None
    try:
        abr_fh = open(out_abr, "w", newline="")
        abr_writer = csv.writer(abr_fh)
        abr_writer.writerow(["abr"])
        abr_fh.flush()
    except Exception as e:
        print(f"[WARN] failed to open '{out_abr}': {e}")
        abr_fh = None
        abr_writer = None

    # 계산기 생성
    tbr_calc = TBRCalculator(
        sample_rate_hz=fs_hz,
        step_seconds=step_seconds,
        fft_size=fft_size,
        ema_alpha=ema_alpha,
        num_channels=num_channels,
    )
    abr_calc = ABRCalculator(
        sample_rate_hz=fs_hz,
        step_seconds=step_seconds,
        fft_size=fft_size,
        ema_alpha=ema_alpha,
        num_channels=num_channels,
    )

    print(f"[INFO] Reading CSV: {in_path}")
    wrote_any = False
    # CSV 읽기 및 처리
    with open(in_path, "r", newline="") as f:
        rdr = csv.reader(f)
        header = next(rdr, None)
        if header is None:
            print("[ERROR] empty CSV")
            return
        eeg_idx, dt_idx = detect_columns(header)
        if not eeg_idx:
            # 헤더가 없거나, 바로 데이터인 경우: 첫 8개 컬럼을 EEG로 가정
            f.seek(0)
            rdr = csv.reader(f)
            eeg_idx = list(range(8))
            dt_idx = None

        # 채널 1~4만 사용
        eeg_idx = eeg_idx[:num_channels]

        # 임계 도달 전까지는 누적만, 이후부터 계산
        elapsed_sec = 0.0
        reached_threshold = False

        for row in rdr:
            if not row:
                continue
            # dt 파싱 (ms)
            if dt_idx is not None and dt_idx < len(row):
                try:
                    dt_ms = float(row[dt_idx])
                    dt_sec = max(0.0, dt_ms / 1000.0)
                except Exception:
                    dt_sec = 1.0 / fs_hz
            else:
                dt_sec = 1.0 / fs_hz

            elapsed_sec += dt_sec
            if not reached_threshold:
                if elapsed_sec >= float(args.start_sec):
                    reached_threshold = True
                else:
                    continue

            # EEG 샘플 파싱
            sample_vals: List[float] = []
            try:
                for i in eeg_idx:
                    v = float(row[i])
                    sample_vals.append(v)
            except Exception:
                # 잘못된 행은 스킵
                continue

            # 업데이트
            updated_tbr = tbr_calc.add_sample(sample_vals, elapsed_sec)
            updated_abr = abr_calc.add_sample(sample_vals, elapsed_sec)

            # TBR EMA 저장
            if (
                tbr_writer is not None
                and updated_tbr
                and tbr_calc.has_value
                and tbr_calc.latest_ema is not None
            ):
                try:
                    tbr_writer.writerow([tbr_calc.latest_ema])
                    if tbr_fh is not None:
                        tbr_fh.flush()
                    wrote_any = True
                except Exception:
                    pass

            # ABR EMA 저장
            if (
                abr_writer is not None
                and updated_abr
                and abr_calc.has_value
                and abr_calc.latest_ema is not None
            ):
                try:
                    abr_writer.writerow([abr_calc.latest_ema])
                    if abr_fh is not None:
                        abr_fh.flush()
                    wrote_any = True
                except Exception:
                    pass

    # 마무리
    try:
        if tbr_fh is not None:
            tbr_fh.flush()
            tbr_fh.close()
        if abr_fh is not None:
            abr_fh.flush()
            abr_fh.close()
    except Exception:
        pass

    if not wrote_any:
        print(f"[WARN] '{in_path}'에서 시작 임계({args.start_sec:.1f}s) 이후 데이터가 없어 저장되지 않았습니다.")
    else:
        print(f"[INFO] Calibration written to:\n  - {out_tbr}\n  - {out_abr}")


if __name__ == "__main__":
    main()

