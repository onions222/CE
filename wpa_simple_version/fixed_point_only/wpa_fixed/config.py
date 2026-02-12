from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FixedWPAConfig:
    q_bits: int = 12

    WA_EN: bool = True
    WA_SEL: int = 64

    luma_nodes_12: tuple[int, ...] = (15, 31, 47, 63, 95, 127, 159, 191, 223, 239, 247, 255)
    bin_interp: bool = True

    sat_s0_255: int = 51   # ~= 0.20 * 255
    sat_s1_255: int = 510  # ~= 2.00 * 255

    warm_gain_global: tuple[float, float, float] = (1.60, 1.00, 0.40)
    cool_gain_global: tuple[float, float, float] = (0.40, 1.00, 1.60)

    # strengths in current Q domain: qone == (1 << q_bits)
    warm_strength_q: int | None = None
    cool_strength_q: int | None = None

    warm_bins_q: np.ndarray | None = None
    cool_bins_q: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not (4 <= self.q_bits <= 16):
            raise ValueError("q_bits must be in [4,16].")
        if not (0 <= self.WA_SEL <= 127):
            raise ValueError("WA_SEL must be in [0,127].")
        if self.sat_s1_255 <= self.sat_s0_255:
            raise ValueError("sat_s1_255 must be > sat_s0_255.")
        if len(self.luma_nodes_12) != 12:
            raise ValueError("luma_nodes_12 must have 12 nodes.")
        if any(self.luma_nodes_12[i] > self.luma_nodes_12[i + 1] for i in range(11)):
            raise ValueError("luma_nodes_12 must be monotonic.")

        if self.warm_strength_q is None:
            self.warm_strength_q = self.qone
        if self.cool_strength_q is None:
            self.cool_strength_q = self.qone
        if self.warm_strength_q < 0 or self.cool_strength_q < 0:
            raise ValueError("strength_q must be >= 0.")

        if self.warm_bins_q is None:
            self.warm_bins_q = self._generate_default_bins_q(np.asarray(self.warm_gain_global, dtype=np.float32))
        if self.cool_bins_q is None:
            self.cool_bins_q = self._generate_default_bins_q(np.asarray(self.cool_gain_global, dtype=np.float32))

        if self.warm_bins_q.shape != (12, 3):
            raise ValueError("warm_bins_q shape must be (12,3)")
        if self.cool_bins_q.shape != (12, 3):
            raise ValueError("cool_bins_q shape must be (12,3)")

    @property
    def qone(self) -> int:
        return 1 << self.q_bits

    def _atten_q(self, y: np.ndarray) -> np.ndarray:
        """Piecewise attenuation in current Q domain.

        Control points:
        - y <= 31  : 0.55
        - y == 127 : 1.00
        - y >= 239 : 0.65
        """
        y = y.astype(np.int32)
        out = np.empty_like(y, dtype=np.int32)
        q = self.qone
        a_low = int(round(0.55 * q))
        a_mid = q
        a_hi = int(round(0.65 * q))

        for i, yi in enumerate(y):
            if yi <= 31:
                out[i] = a_low
            elif yi < 127:
                out[i] = a_low + ((yi - 31) * (a_mid - a_low) + 48) // 96
            elif yi < 239:
                out[i] = a_mid + ((yi - 127) * (a_hi - a_mid) + 56) // 112
            else:
                out[i] = a_hi
        return out

    def _generate_default_bins_q(self, gain_global: np.ndarray) -> np.ndarray:
        y = np.asarray(self.luma_nodes_12, dtype=np.int32)
        q = self.qone
        gain_global_q = np.rint(gain_global * q).astype(np.int32)
        atten = self._atten_q(y)
        # gain = 1 + atten*(gain_global-1)
        out = np.empty((12, 3), dtype=np.int32)
        for i in range(12):
            delta = gain_global_q - q
            out[i] = q + ((atten[i] * delta + (q // 2)) >> self.q_bits)
        return out
