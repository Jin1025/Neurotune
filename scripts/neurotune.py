
import sys
import time
import math
from typing import List, Optional, Tuple
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
        help="저장할 CSV 파일의 기본 이름 (예: S01).",
    )
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        default=None,
        help="지정 시 CSV 파일을 실시간처럼 재생(오프라인 모드). (예:input/eeg.csv).",
    )
    parser.add_argument(
        "--fs",
        type=float,
        default=250.0,
        help="오프라인 모드에서 사용할 샘플링 레이트(Hz). --use-dt 미사용 시 적용 (기본 250.0)",
    )
    parser.add_argument(
        "--use-dt",
        action="store_true",
        help="오프라인 모드에서 CSV의 DT(ms) 컬럼 간격으로 페이싱",
    )
    parser.add_argument(
        "--num-channels",
        type=int,
        default=4,
        help="지표 계산 시 사용할 채널 수 (앞에서부터, 기본 4)",
    )
    return parser.parse_args()


def load_calibration(csv_path: str):
    """
    silence.py에서 매 세션마다 덮어쓰는 'callibration' 파일을 읽어서
    TBR 값들만 모아 ECDF 기반 개인화 객체를 만든다.

    파일 형식(고정):
        헤더:  tbr or abr(두 번째 컬럼은 존재하더라도 무시)
        데이터: 각 행에 tbr or abr 값
    """

    if not os.path.exists(csv_path):
        print("[WARN] Calibration file not found. Re-index will stay at 0.0.")
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

def get_input_dir() -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(base_dir, "..", "input"))


def detect_columns(header: List[str]) -> Tuple[List[int], Optional[int]]:
    """
    헤더에서 EEG 채널 컬럼들의 인덱스와 'DT' 컬럼 인덱스를 찾는다.
    - EEG 컬럼: 'EEG '로 시작하는 컬럼명들(대소문자 무시)
    - DT 컬럼: 정확히 'DT'
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


def iter_csv_samples(
    file_path: str,
    use_dt: bool,
    fallback_fs_hz: float,
):
    """
    CSV를 열고 한 줄씩 EEG 채널 값 리스트와 해당 샘플 간격(sec)을 생성한다.
    - use_dt=True이면 DT(ms) 컬럼을 간격으로 사용(없거나 파싱 실패 시 fs로 대체)
    - use_dt=False이면 고정 간격 1/fs 사용
    """
    with open(file_path, "r", newline="") as f:
        rdr = csv.reader(f)
        header = next(rdr, None)
        if header is None:
            return
        eeg_idx, dt_idx = detect_columns(header)
        if not eeg_idx:
            # 헤더가 없거나, 바로 데이터인 경우: 첫 8개 컬럼을 EEG로 가정
            # 이 경우 header는 실제 데이터였을 수 있으므로, 다시부터 읽기
            f.seek(0)
            rdr = csv.reader(f)
            eeg_idx = list(range(8))
            dt_idx = None
        fixed_dt = 1.0 / max(1.0, float(fallback_fs_hz))
        for row in rdr:
            if not row:
                continue
            # 채널 값 파싱
            sample_vals: List[float] = []
            try:
                for i in eeg_idx:
                    sample_vals.append(float(row[i]))
            except Exception:
                # 잘못된 행은 스킵
                continue
            # 간격 결정
            if use_dt and dt_idx is not None and dt_idx < len(row):
                try:
                    dt_ms = float(row[dt_idx])
                    dt_sec = max(0.0, dt_ms / 1000.0)
                except Exception:
                    dt_sec = fixed_dt
            else:
                dt_sec = fixed_dt
            yield sample_vals, dt_sec


def main() -> None:
    args = parse_args()

    # 오프라인 모드: --file 지정 시 CSV 재생
    if args.file:
        # 파일 경로 정책:
        csv_arg = args.file
        input_dir = get_input_dir()
        if os.path.isabs(csv_arg):
            csv_path = os.path.abspath(csv_arg)
            try:
                common = os.path.commonpath([csv_path, input_dir])
            except Exception:
                common = ""
        else:
            starts_with_input = (
                csv_arg.startswith("input/") or
                csv_arg.startswith("input" + os.sep)
            )
            project_root = os.path.dirname(input_dir)
            csv_path = os.path.join(project_root, csv_arg)
        if not os.path.exists(csv_path):
            print(f"[ERROR] CSV 파일을 찾을 수 없습니다: {csv_arg}")
            return

        # 지표 계산기: TBR, ABR (오프라인 파라미터 사용)
        tbr_calc = create_metric_calculator(
            "tbr",
            sample_rate_hz=args.fs,
            step_seconds=0.25,
            fft_size=1024,
            ema_alpha=0.2,
            num_channels=args.num_channels,
        )
        abr_calc = create_metric_calculator(
            "abr",
            sample_rate_hz=args.fs,
            step_seconds=0.25,
            fft_size=1024,
            ema_alpha=0.2,
            num_channels=args.num_channels,
        )

        # OSC 설정 (Max/MSP: udpreceive 12000 → route /time, /re-tbr, /re-abr)
        osc_host = "127.0.0.1"
        osc_port = 12000
        osc_client = udp_client.SimpleUDPClient(osc_host, osc_port)

        # 캘리브레이션 로드
        tbr_personalizer, tbr_q1, tbr_q3 = load_calibration(os.path.join(get_input_dir(), "callibration_tbr"))
        abr_personalizer, abr_q1, abr_q3 = load_calibration(os.path.join(get_input_dir(), "callibration_abr"))
        last_reindex_tbr = 0.0
        last_reindex_abr = 0.0
        if tbr_personalizer is not None and tbr_q1 is not None and tbr_q3 is not None:
            try:
                osc_client.send_message("/IQR", [float(tbr_q1), float(tbr_q3)])
            except Exception:
                pass
        if abr_personalizer is not None and abr_q1 is not None and abr_q3 is not None:
            try:
                osc_client.send_message("/IQR-abr", [float(abr_q1), float(abr_q3)])
            except Exception:
                pass

        # 실시간 페이싱: 목표 시각 누적 방식으로 드리프트 최소화
        start_time = time.time()
        scheduled_time = start_time
        try:
            for sample, dt_sec in iter_csv_samples(csv_path, args.use_dt, args.fs):
                scheduled_time += dt_sec
                now = time.time()
                sleep_sec = scheduled_time - now
                if sleep_sec > 0:
                    try:
                        time.sleep(sleep_sec)
                    except Exception:
                        pass

                elapsed = time.time() - start_time

                # 지표 업데이트 (TBR, ABR)
                _ = tbr_calc.add_sample(sample, elapsed)
                _ = abr_calc.add_sample(sample, elapsed)

                # 콘솔 출력 구성
                try:
                    line = ",".join(f"{float(v)}" for v in sample)
                except Exception:
                    line = ",".join(str(v) for v in sample)
                # TBR 값 및 re-index
                tbr_disp = None
                if tbr_calc.has_value:
                    tbr_disp = tbr_calc.latest_ema if tbr_calc.latest_ema is not None else tbr_calc.latest_value
                    line = f"{line},{tbr_disp:.4f}"
                else:
                    line = f"{line},NA"
                if tbr_disp is not None and tbr_personalizer is not None:
                    try:
                        last_reindex_tbr = float(tbr_personalizer.transform(tbr_disp))
                    except Exception:
                        last_reindex_tbr = float("nan")
                    line = f"{line},{last_reindex_tbr:.4f}"
                else:
                    line = f"{line},NA"
                # ABR 값 및 re-index
                abr_disp = None
                if abr_calc.has_value:
                    abr_disp = abr_calc.latest_ema if abr_calc.latest_ema is not None else abr_calc.latest_value
                    line = f"{line},{abr_disp:.4f}"
                else:
                    line = f"{line},NA"
                if abr_disp is not None and abr_personalizer is not None:
                    try:
                        # ABR은 낮을수록 안 좋음 → ECDF 누적확률(p)을 그대로 사용 (1 - transform)
                        last_reindex_abr = float(1.0 - abr_personalizer.transform(abr_disp))
                    except Exception:
                        last_reindex_abr = float("nan")
                    line = f"{line},{last_reindex_abr:.4f}"
                else:
                    line = f"{line},NA"
                print(line)

                # OSC 송신
                try:
                    osc_client.send_message("/time", elapsed)
                    osc_client.send_message("/re-tbr", (last_reindex_tbr if tbr_personalizer is not None else 0.0))
                    osc_client.send_message("/re-abr", (last_reindex_abr if abr_personalizer is not None else 0.0))
                except Exception:
                    pass
        except KeyboardInterrupt:
            print("\nStopped.")
        return

    # 온라인 모드: LSL 스트림 연결(기존 동작)
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

    # 지표 계산기: TBR, ABR
    tbr_calc = create_metric_calculator(
        "tbr",
        sample_rate_hz=fs,
        step_seconds=0.25,
        fft_size=1024,
        ema_alpha=0.2,
        num_channels=4,
    )
    abr_calc = create_metric_calculator(
        "abr",
        sample_rate_hz=fs,
        step_seconds=0.25,
        fft_size=1024,
        ema_alpha=0.2,
        num_channels=4,
    )

    # OSC 설정 (Max/MSP: udpreceive 12000 → route /time, route /re-tbr, route /re-abr)
    osc_host = "127.0.0.1"
    mac_host = "127.0.0.1"
    osc_port = 12000
    mac_port = 8001
    mac_port2 = 8002
    osc_client = udp_client.SimpleUDPClient(osc_host, osc_port)
    mac_client = udp_client.SimpleUDPClient(mac_host, mac_port)
    mac_client2 = udp_client.SimpleUDPClient(mac_host, mac_port2)

    # 사전 캘리브레이션 로드 (TBR, ABR) - 항상 input 폴더
    tbr_personalizer, tbr_q1, tbr_q3 = load_calibration(os.path.join(get_input_dir(), "callibration_tbr"))
    abr_personalizer, abr_q1, abr_q3 = load_calibration(os.path.join(get_input_dir(), "callibration_abr"))
    last_reindex_tbr = 0.0
    last_reindex_abr = 0.0
    if tbr_personalizer is not None and tbr_q1 is not None and tbr_q3 is not None:
        try:
            osc_client.send_message("/IQR", [float(tbr_q1), float(tbr_q3)])
        except Exception:
            pass
    if abr_personalizer is not None and abr_q1 is not None and abr_q3 is not None:
        try:
            osc_client.send_message("/IQR-abr", [float(abr_q1), float(abr_q3)])
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

            _ = tbr_calc.add_sample(sample, ts)
            _ = abr_calc.add_sample(sample, ts)

            # 콘솔 출력: 샘플 값 , 로 나열 + TBR/NA + reTBR/NA + ABR/NA + reABR/NA
            try:
                line = ",".join(f"{float(v)}" for v in sample)
            except Exception:
                line = ",".join(str(v) for v in sample)
            tbr_disp = None
            if tbr_calc.has_value:
                tbr_disp = tbr_calc.latest_ema if tbr_calc.latest_ema is not None else tbr_calc.latest_value
                line = f"{line},{tbr_disp:.4f}"
            else:
                line = f"{line},NA"
            if tbr_disp is not None and tbr_personalizer is not None:
                try:
                    last_reindex_tbr = float(tbr_personalizer.transform(tbr_disp))
                except Exception:
                    last_reindex_tbr = float("nan")
                line = f"{line},{last_reindex_tbr:.4f}"
            else:
                line = f"{line},NA"
            abr_disp = None
            if abr_calc.has_value:
                abr_disp = abr_calc.latest_ema if abr_calc.latest_ema is not None else abr_calc.latest_value
                line = f"{line},{abr_disp:.4f}"
            else:
                line = f"{line},NA"
            if abr_disp is not None and abr_personalizer is not None:
                try:
                    # ABR은 낮을수록 안 좋음 → ECDF 누적확률(p)을 그대로 사용 (1 - transform)
                    last_reindex_abr = float(1.0 - abr_personalizer.transform(abr_disp))
                except Exception:
                    last_reindex_abr = float("nan")
                line = f"{line},{last_reindex_abr:.4f}"
            else:
                line = f"{line},NA"
            print(line)

            # 매 샘플마다 time 및 re-지표 송신 
            try:
                osc_client.send_message("/time", elapsed)
                osc_client.send_message("/re-tbr", (last_reindex_tbr if tbr_personalizer is not None else 0.0))
                osc_client.send_message("/re-abr", (last_reindex_abr if abr_personalizer is not None else 0.0))
                mac_client.send_message("/re-tbr", (last_reindex_tbr if tbr_personalizer is not None else 0.0))
                mac_client2.send_message("/re-abr", (last_reindex_abr if abr_personalizer is not None else 0.0))
            except Exception:
                pass

    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()