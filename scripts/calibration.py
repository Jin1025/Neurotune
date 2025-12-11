import argparse
import csv
import os
import time
from typing import List

from pylsl import StreamInlet, resolve_streams

from metrics_core import TBRCalculator, ABRCalculator


def get_input_dir() -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(base_dir, "..", "input"))


def list_streams():
    results = None
    print("Scanning for LSL streams...")
    while results is None or len(results) == 0:
        results = resolve_streams()
    print()
    print("[id] streamname")
    print("---------------")
    for i, sinfo in enumerate(results):
        try:
            name = sinfo.name()
        except Exception:
            name = "<unknown>"
        print(f"[{i}] {name}")
    print()
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NeuroTune calibration (silence session)")
    parser.add_argument(
        "--duration-min",
        "-d",
        type=float,
        default=10.0,
        help="캘리브레이션 전체 길이(분 단위). 기본 10분.",
    )
    parser.add_argument(
        "--start-calib-min",
        type=float,
        default=2.0,
        help="캘리브레이션에 포함할 시작 시점(분). 기본 2분 이후부터 저장.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # LSL 스트림 선택
    results = list_streams()
    selected = None

    while True:
        try:
            ids = input("Select available LSL stream by the [id] and press enter: ").strip()
            idx = int(ids)
            if 0 <= idx < len(results):
                selected = results[idx]
                break
        except KeyboardInterrupt:
            print("\nInterrupted.")
            return
        except Exception:
            pass
        print("Invalid selection. Try again.")

    inlet = StreamInlet(selected)
    info = inlet.info()
    try:
        num_channels = info.channel_count()
    except Exception:
        num_channels = 1
    try:
        fs = info.nominal_srate()
    except Exception:
        fs = 0.0
    if fs is None or fs <= 0:
        fs = 250.0

    print(f"Connected to stream: {info.name()} | channels={num_channels} | fs={fs} Hz")

    # 지표 계산
    tbr_calc = TBRCalculator(sample_rate_hz=fs, step_seconds=0.25, fft_size=1024, ema_alpha=0.2, num_channels=4)
    abr_calc = ABRCalculator(sample_rate_hz=fs, step_seconds=0.25, fft_size=1024, ema_alpha=0.2, num_channels=4)

    # 출력 CSV 준비: input/callibration_tbr (tbr), input/callibration_abr (abr) - 매 세션마다 덮어쓰기
    input_dir = get_input_dir()
    try:
        os.makedirs(input_dir, exist_ok=True)
    except Exception:
        pass
    calib_path = os.path.join(input_dir, "callibration_tbr")
    calib_path_abr = os.path.join(input_dir, "callibration_abr")
    try:
        calib_fh = open(calib_path, "w", newline="")
        calib_writer = csv.writer(calib_fh)
        calib_writer.writerow(["tbr"])
        calib_fh.flush()
    except Exception as e:
        print(f"[WARN] Failed to open '{calib_path}' for write: {e}")
        calib_fh = None
        calib_writer = None
    try:
        calib_fh_abr = open(calib_path_abr, "w", newline="")
        calib_writer_abr = csv.writer(calib_fh_abr)
        calib_writer_abr.writerow(["abr"])
        calib_fh_abr.flush()
    except Exception as e:
        print(f"[WARN] Failed to open '{calib_path_abr}' for write: {e}")
        calib_fh_abr = None
        calib_writer_abr = None

    print(f"Calibration session started for {args.duration_min} minutes.")
    print(f"  - output TBR -> '{calib_path}'")
    print(f"  - output ABR -> '{calib_path_abr}'")
    print("Silence session: no OSC output, only CSV logging.")

    duration_sec = max(0.0, float(args.duration_min) * 60.0)
    calib_start_sec = max(0.0, float(args.start_calib_min) * 60.0)

    start_time = time.time()
    try:
        while True:
            sample, ts = inlet.pull_sample()
            if sample is None:
                continue

            elapsed = time.time() - start_time
            if elapsed >= duration_sec:
                print("Calibration duration reached. Stopping.")
                break

            # 지표 업데이트 (TBR, ABR)
            updated_tbr = tbr_calc.add_sample(sample, ts)
            updated_abr = abr_calc.add_sample(sample, ts)

            if elapsed >= calib_start_sec:
                if (
                    calib_writer is not None
                    and updated_tbr
                    and tbr_calc.has_value
                    and tbr_calc.latest_ema is not None
                ):
                    # EMA 값만 사용해서 한 행에 TBR 저장
                    tbr_val = tbr_calc.latest_ema
                    try:
                        calib_writer.writerow([tbr_val])
                        # 터미널에도 현재 TBR 값을 계속 출력
                        print(f"TBR: {tbr_val}")
                        if calib_fh is not None:
                            calib_fh.flush()
                    except Exception:
                        pass
                if (
                    calib_writer_abr is not None
                    and updated_abr
                    and abr_calc.has_value
                    and abr_calc.latest_ema is not None
                ):
                    # EMA 값만 사용해서 한 행에 ABR 저장
                    abr_val = abr_calc.latest_ema
                    try:
                        calib_writer_abr.writerow([abr_val])
                        # 터미널에도 현재 ABR 값을 계속 출력
                        print(f"ABR: {abr_val}")
                        if calib_fh_abr is not None:
                            calib_fh_abr.flush()
                    except Exception:
                        pass

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        try:
            if calib_fh is not None:
                try:
                    calib_fh.flush()
                    calib_fh.close()
                except Exception:
                    pass
            if calib_fh_abr is not None:
                try:
                    calib_fh_abr.flush()
                    calib_fh_abr.close()
                except Exception:
                    pass
        except Exception:
            pass

    print("Calibration files written.")


if __name__ == "__main__":
    main()


