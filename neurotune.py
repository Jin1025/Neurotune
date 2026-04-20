import sys
import time
import math
from typing import List
import os
import csv
import argparse

import numpy as np
from pylsl import StreamInlet, resolve_streams
from pythonosc import udp_client

from metrics_core import create_metric_calculator, ECDF


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
    parser = argparse.ArgumentParser(description="NeuroTune")
    parser.add_argument(
        "--save-filename",
        "-o",
        type=str,
        default=None,
        help="저장할 CSV 파일의 기본 이름/폴더명 (예: S01). 지정 안 하면 날짜시간으로 자동 저장.",
    )
    return parser.parse_args()


def load_calibration():
    """
    silence.py에서 매 세션마다 덮어쓰는 'callibration' 파일을 읽어서
    TBR 값들만 모아 ECDF 기반 개인화 객체를 만든다.

    파일 형식(고정):
        헤더:  tbr (두 번째 컬럼은 존재하더라도 무시)
        데이터: 각 행에 tbr 값
    """
    csv_path = "callibration"

    if not os.path.exists(csv_path):
        print("[WARN] Calibration file not found for TBR. Re-index will stay at 0.0.")
        return None, None, None

    loaded = []
    try:
        with open(csv_path, "r", newline="") as f:
            rdr = csv.reader(f)
            header = next(rdr, None)
            # silence.py에서 항상 첫 번째 컬럼(tbr)만 사용한다.
            idx_col = 0
            for row in rdr:
                if not row:
                    continue
                if len(row) <= idx_col:
                    continue
                try:
                    loaded.append(float(row[idx_col]))
                except Exception:
                    pass
    except Exception as e:
        print(f"[WARN] Failed to read calibration file '{csv_path}': {e}")
        return None, None, None

    if not loaded:
        print(f"[WARN] No valid samples in calibration file '{csv_path}'.")
        return None, None, None

    arr = np.asarray(loaded, dtype=np.float64)
    q1 = float(np.percentile(arr, 25)) 
    q3 = float(np.percentile(arr, 75))
    ecdf = ECDF()
    ecdf.fit(loaded, q1, q3)
    print(f"[INFO] Loaded calibration from '{csv_path}' | n={len(loaded)}, Q1={q1:.4f}, Q3={q3:.4f}")
    return ecdf, q1, q3


def main() -> None:
    args = parse_args()

    # 스트림 선택
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

    # 사용자 입력: 저장 파일명
    # if args.save_filename is not None:
    #     save_base = args.save_filename.strip()
    #     if not save_base:
    #         save_base = "log"
    # else:
    #     try:
    #         save_base = input("Enter save filename (e.g., S01): ").strip()
    #         if not save_base:
    #             save_base = "log"
    #     except Exception:
    #         save_base = "log"
    
    # 저장 이름: 입력을 묻지 않고 자동(필요 시 CLI로만 지정)
    save_base = (args.save_filename or "").strip()
    # 지표: TBR만 사용
    metric_sel = "tbr"

    # TBR 계산기, 0.25초 스텝
    metric_calc = create_metric_calculator(
        metric_sel,
        sample_rate_hz=fs,
        step_seconds=0.25,
        fft_size=1024,
        ema_alpha=0.2,
        num_channels=4,
    )

    # OSC 설정 (Max/MSP: udpreceive 12000 → route /time, route /index, route /re-index)
    osc_host = "127.0.0.1"
    osc_port = 12000
    osc_client = udp_client.SimpleUDPClient(osc_host, osc_port)

    # CSV 지속 append 준비
    # save_base를 지정하면 metric/<save_base>/ 아래에 저장,
    # 지정하지 않으면 metric/ 아래에 바로 저장한다.
    # 예) metric/S01/20251123_153045_tbr.csv 또는 metric/20251123_153045_tbr.csv
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    folder_path = os.path.join("metric", save_base)
    folder_path = os.path.join("metric", save_base) if save_base else "metric"
    out_name = os.path.join(folder_path, f"{timestamp_str}_{metric_sel}.csv")

    csv_file = None
    csv_writer = None
    try:
        # 폴더가 없다면 생성
        os.makedirs(folder_path, exist_ok=True)

        need_header = (not os.path.exists(out_name)) or (os.path.getsize(out_name) == 0)
        csv_file = open(out_name, "a", newline="")
        csv_writer = csv.writer(csv_file)
        if need_header:
            csv_writer.writerow(["time_sec", "tbr", "re-tbr"])
            csv_file.flush()
    except Exception as e:
        print(f"Failed to open CSV for append: {e}")
        csv_file = None
        csv_writer = None

    print("Receiving... Press Ctrl+C to stop.")

    # 사전 캘리브레이션 로드 (필수 전제) - TBR 기준
    personalizer, q1, q3 = load_calibration()
    last_reindex = 0.0  # OSC 송신용 최근 re-index 값
    if personalizer is not None and q1 is not None and q3 is not None:
                try:
                    osc_client.send_message("/IQR", [float(q1), float(q3)])
                except Exception:
                    pass

    # 시작 시각 기록
    start_time = time.time()
    try:
        while True:
            sample, ts = inlet.pull_sample()
            if sample is None:
                continue

            # 경과 시간(전송 기준)
            elapsed = time.time() - start_time

            updated = metric_calc.add_sample(sample, ts)

            # 콘솔 출력: 샘플 값 , 로 나열 + 마지막에 TBR/NA + reTBR/NA
            try:
                line = ",".join(f"{float(v)}" for v in sample)
            except Exception:
                line = ",".join(str(v) for v in sample)
            disp = None
            if metric_calc.has_value:
                disp = metric_calc.latest_ema if metric_calc.latest_ema is not None else metric_calc.latest_value
                line = f"{line},{disp:.4f}"
            else:
                line = f"{line},NA"

            # re-index 계산 및 출력(캘리브레이션 파일이 있을 때만)
            if disp is not None and personalizer is not None:
                try:
                    cur_reindex = float(personalizer.transform(disp))
                except Exception:
                    cur_reindex = float("nan")
                last_reindex = cur_reindex
                line = f"{line},{cur_reindex:.4f}"
            else:
                line = f"{line},NA"
            print(line)

            # 매 샘플마다 time 및 index 송신 
            try:
                osc_client.send_message("/time", elapsed)
                osc_client.send_message("/index", (metric_calc.latest_ema if metric_calc.latest_ema is not None else 0.0))
                # re-index(개인화 변환값)는 캘리브레이션 파일이 있을 때만 의미 있음, 없으면 0.0
                osc_client.send_message("/re-index", (last_reindex if personalizer is not None else 0.0))
            except Exception:
                pass

            # 업데이트 시점에만 저장 및 캘리브레이션 로직 수행
            if updated and metric_calc.has_value and disp is not None:
                # CSV 즉시 append
                try:
                    row_reindex = float(personalizer.transform(disp)) if (personalizer is not None) else float("nan")
                except Exception:
                    row_reindex = float("nan")
                if csv_writer is not None:
                    try:
                        csv_writer.writerow([float(elapsed), float(disp), row_reindex])
                        if csv_file is not None:
                            csv_file.flush()
                    except Exception:
                        pass

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        try:
            if csv_file is not None:
                csv_file.flush()
                csv_file.close()
                print(f"CSV appended to '{out_name}'.")
        except Exception:
            pass


if __name__ == "__main__":
    main()