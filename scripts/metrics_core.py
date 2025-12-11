import math
from typing import List

import numpy as np


class SlidingSpectrum:
    """
    공통 FFT/밴드 전처리: N채널 링버퍼, 해닝 윈도우, 파워 스펙트럼, 밴드 파워 합산
    """

    THETA_LO_HZ = 4.0
    THETA_HI_HZ = 8.0
    ALPHA_LO_HZ = 8.0
    ALPHA_HI_HZ = 13.0
    BETA_LO_HZ = 13.0
    BETA_HI_HZ = 30.0

    def __init__(
        self,
        sample_rate_hz: float,
        step_seconds: float = 0.25,
        fft_size: int = 1024,
        num_channels: int = 8,
    ) -> None:
        self.sample_rate_hz = float(sample_rate_hz) if sample_rate_hz and sample_rate_hz > 0 else 250.0
        self.step_seconds = float(step_seconds)
        self.fft_size = int(fft_size)
        self.num_channels = int(num_channels)

        self.step_samples = max(1, int(round(self.sample_rate_hz * self.step_seconds)))
        self.window = np.hanning(self.fft_size).astype(np.float64)
        self.window_power_scale = float(np.sum(self.window * self.window))

        self.ring = np.zeros((self.num_channels, self.fft_size), dtype=np.float64)
        self.write_index = 0
        self.total_samples = 0
        self.samples_since_update = 0

        self.freq = np.fft.rfftfreq(self.fft_size, d=1.0 / self.sample_rate_hz)
        self.mask_theta = (self.freq >= self.THETA_LO_HZ) & (self.freq < self.THETA_HI_HZ)
        self.mask_alpha = (self.freq >= self.ALPHA_LO_HZ) & (self.freq < self.ALPHA_HI_HZ)
        self.mask_beta = (self.freq >= self.BETA_LO_HZ) & (self.freq < self.BETA_HI_HZ)

        self.latest_power_spectrum = None  # shape: (n_use, len(freq))
        self.n_use = 0
        self.latest_timestamp = 0.0

    def add_sample(self, sample: List[float], timestamp: float) -> bool:
        """
        새 샘플을 추가하고, step_seconds마다 한 번씩 파워 스펙트럼을 업데이트.
        업데이트가 일어났으면 True, 아니면 False를 반환.
        """
        n_use = min(self.num_channels, len(sample))
        for ch in range(n_use):
            self.ring[ch, self.write_index] = float(sample[ch])

        self.write_index = (self.write_index + 1) % self.fft_size
        if self.total_samples < self.fft_size:
            self.total_samples += 1
        else:
            self.samples_since_update += 1

        if self.total_samples >= self.fft_size and self.samples_since_update >= self.step_samples:
            if n_use <= 0:
                self.samples_since_update = 0
                return False

            ps_list = []
            for ch in range(n_use):
                if self.write_index == 0:
                    ts_ordered = self.ring[ch, :]
                else:
                    ts_ordered = np.concatenate(
                        (self.ring[ch, self.write_index :], self.ring[ch, : self.write_index])
                    )
                buf = ts_ordered * self.window
                spec = np.fft.rfft(buf)
                power = (np.abs(spec) ** 2) / self.window_power_scale
                ps_list.append(power)
            self.latest_power_spectrum = np.stack(ps_list, axis=0)  # (n_use, F)
            self.n_use = n_use
            self.latest_timestamp = float(timestamp)

            self.samples_since_update -= self.step_samples
            if self.samples_since_update < 0:
                self.samples_since_update = 0
            return True
        return False

    def band_power_per_channel(self, band: str) -> np.ndarray:
        """
        밴드 이름(theta/alpha/beta)에 해당하는 파워를 채널별로 합산해서 반환.
        """
        if self.latest_power_spectrum is None or self.n_use <= 0:
            return np.zeros(0, dtype=np.float64)
        if band == "theta":
            mask = self.mask_theta
        elif band == "alpha":
            mask = self.mask_alpha
        elif band == "beta":
            mask = self.mask_beta
        else:
            raise ValueError("unknown band")
        return np.sum(self.latest_power_spectrum[:, mask], axis=1)


class BaseMetricCalculator:
    """
    공통 지표 계산 베이스: SlidingSpectrum + EMA
    """

    def __init__(
        self,
        sample_rate_hz: float,
        step_seconds: float = 0.25,
        fft_size: int = 1024,
        ema_alpha: float = 0.2,
        num_channels: int = 8,
    ) -> None:
        self.spectrum = SlidingSpectrum(sample_rate_hz, step_seconds, fft_size, num_channels)
        self.ema_alpha = float(ema_alpha)
        self.has_value = False
        self.latest_value = 0.0
        self.latest_ema = None
        self.latest_timestamp = 0.0

    def add_sample(self, sample: List[float], timestamp: float) -> bool:
        updated = self.spectrum.add_sample(sample, timestamp)
        if not updated:
            return False
        val = self.compute_metric()
        self.latest_value = float(val)
        self.latest_timestamp = self.spectrum.latest_timestamp
        self.has_value = True
        if self.latest_ema is None:
            self.latest_ema = self.latest_value
        else:
            a = self.ema_alpha
            self.latest_ema = a * self.latest_value + (1.0 - a) * self.latest_ema
        return True

    def compute_metric(self) -> float:
        raise NotImplementedError


class TBRCalculator(BaseMetricCalculator):
    """
    Theta/Beta Ratio
    """

    def compute_metric(self) -> float:
        theta = self.spectrum.band_power_per_channel("theta")
        beta = self.spectrum.band_power_per_channel("beta")
        if theta.size == 0 or beta.size == 0:
            return 0.0
        beta = np.maximum(beta, 1e-12)
        ratios = theta / beta
        return float(np.mean(ratios))


class ABRCalculator(BaseMetricCalculator):
    """
    Alpha/Beta Ratio
    """

    def compute_metric(self) -> float:
        alpha = self.spectrum.band_power_per_channel("alpha")
        beta = self.spectrum.band_power_per_channel("beta")
        if alpha.size == 0 or beta.size == 0:
            return 0.0
        beta = np.maximum(beta, 1e-12)
        ratios = alpha / beta
        return float(np.mean(ratios))


class ECDF:
    """
    캘리브레이션 값의 터키 펜스 범위 내 샘플들로 ECDF를 만들고,
    focus = 1 - P(metric' ≤ t) 로 반환. (값이 낮을수록 더 큰 focus)
    """

    def __init__(self) -> None:
        self.low = None
        self.high = None
        self.sorted_vals = None

    def fit(self, calib_values: List[float], low: float, high: float) -> None:
        self.low = float(low)
        self.high = float(high)
        if calib_values is None or len(calib_values) == 0:
            self.sorted_vals = np.asarray([0.0], dtype=np.float64)
            return
        arr = np.asarray(calib_values, dtype=np.float64)
        mask = (arr >= self.low) & (arr <= self.high)
        in_range = arr[mask]
        if in_range.size == 0:
            in_range = arr
        self.sorted_vals = np.sort(in_range)

    def transform(self, value: float) -> float:
        if self.sorted_vals is None or self.sorted_vals.size == 0:
            return 0.0
        v = float(value)
        if self.low is not None and self.high is not None:
            v = float(min(max(v, self.low), self.high))
        idx = int(np.searchsorted(self.sorted_vals, v, side="right"))
        p = idx / float(self.sorted_vals.size)
        return 1.0 - float(p)


def create_metric_calculator(
    name: str,
    sample_rate_hz: float,
    step_seconds: float = 0.25,
    fft_size: int = 1024,
    ema_alpha: float = 0.2,
    num_channels: int = 8,
) -> BaseMetricCalculator:
    """
    이름으로 지표 계산기 생성: "tbr" 또는 "atr"
    """
    name = (name or "").strip().lower()
    if name == "tbr":
        return TBRCalculator(sample_rate_hz, step_seconds, fft_size, ema_alpha, num_channels)
    elif name == "abr":
        return ABRCalculator(sample_rate_hz, step_seconds, fft_size, ema_alpha, num_channels)
    else:
        raise ValueError(f"unknown metric: {name}")


