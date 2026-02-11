"""Configuration for White Point Adjustment (WPA)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


ArrayLikeF32 = np.ndarray


@dataclass
class WPAConfig:
    """WPA configuration with DDIC-like controls and defaults.

    Key semantics:
    - `WA_SEL` in [0, 127], where 64 is identity.
    - 12-bin luma nodes are fixed by default and used for luma-binned gains.
    - Per-bin warm/cool gain tables are auto-generated when not provided.
    """

    WA_EN: bool = True
    WA_SEL: int = 64

    # Processing model:
    # - diag_rgb: per-channel diagonal gains (fast, conservative)
    # - kelvin_ycocg: Bradford CAT-derived Kelvin shift in YCoCg on linear RGB
    wa_mode: Literal["diag_rgb", "kelvin_ycocg"] = "diag_rgb"

    # 12-bin fixed luma nodes (8-bit Y samples), monotonic increasing.
    luma_nodes_12: tuple[int, ...] = (
        15,
        31,
        47,
        63,
        95,
        127,
        159,
        191,
        223,
        239,
        247,
        255,
    )
    bin_interp: bool = True

    # Domain selection for luma proxy and saturation weight.
    luma_domain: Literal["gamma", "linear"] = "gamma"
    sat_weight_domain: Literal["gamma", "linear"] = "gamma"

    # Gain application domain conversion.
    gamma_mode: Literal["srgb", "power", "none"] = "srgb"
    use_gamma_lut: bool = False
    power_gamma: float = 2.2

    # Low-saturation protection thresholds in normalized gamma domain.
    # s = |R-G| + |G-B| + |B-R| over normalized RGB channels (range [0, 2]).
    #
    # Practical note: for real photos, s is often >= 0.8 for a large fraction of pixels.
    # If sat_s1 is too small, w becomes ~0 for most pixels and warm/cool looks ineffective.
    sat_s0: float = 0.20
    sat_s1: float = 1.80

    # Global endpoints used only for default per-bin table generation.
    warm_gain_global: tuple[float, float, float] = (1.00, 0.98, 0.82)
    cool_gain_global: tuple[float, float, float] = (0.82, 0.98, 1.00)

    # Side-specific strength for gain attenuation. 1.0 keeps default table behavior.
    warm_strength: float = 1.0
    cool_strength: float = 1.0

    # Kelvin/YCoCg model parameters (only used when wa_mode == "kelvin_ycocg").
    kelvin_warm_end: float = 3500.0
    kelvin_cool_end: float = 9000.0
    kelvin_w_shadow: float = 0.60
    kelvin_w_mid: float = 1.00
    kelvin_w_highlight: float = 0.70
    kelvin_bin_mid: int = 6

    kelvin_bright_minfac: float = 0.25
    kelvin_y_dark2: float = 8.0
    kelvin_y_dark1: float = 16.0
    kelvin_y_bright1: float = 240.0
    kelvin_y_bright2: float = 248.0

    # Extra global scaling for Kelvin/YCoCg transform strength.
    # Keeps the Kelvin-based model from being overly aggressive at alpha=1.
    kelvin_strength: float = 0.20

    warm_gains_bins: ArrayLikeF32 | None = None
    cool_gains_bins: ArrayLikeF32 | None = None

    def __post_init__(self) -> None:
        if not (0 <= self.WA_SEL <= 127):
            raise ValueError("WA_SEL must be in [0, 127].")
        if self.sat_s1 <= self.sat_s0:
            raise ValueError("sat_s1 must be greater than sat_s0.")
        if self.warm_strength < 0.0:
            raise ValueError("warm_strength must be >= 0.")
        if self.cool_strength < 0.0:
            raise ValueError("cool_strength must be >= 0.")
        if self.wa_mode not in ("diag_rgb", "kelvin_ycocg"):
            raise ValueError("wa_mode must be 'diag_rgb' or 'kelvin_ycocg'.")
        if not (1667.0 <= self.kelvin_warm_end <= 25000.0):
            raise ValueError("kelvin_warm_end must be in [1667,25000].")
        if not (1667.0 <= self.kelvin_cool_end <= 25000.0):
            raise ValueError("kelvin_cool_end must be in [1667,25000].")
        if not (1 <= int(self.kelvin_bin_mid) <= 12):
            raise ValueError("kelvin_bin_mid must be in [1,12].")
        if self.kelvin_strength < 0.0:
            raise ValueError("kelvin_strength must be >= 0.")

        nodes = np.asarray(self.luma_nodes_12, dtype=np.float32)
        if nodes.shape != (12,):
            raise ValueError("luma_nodes_12 must contain exactly 12 nodes.")
        if not np.all(nodes[1:] >= nodes[:-1]):
            raise ValueError("luma_nodes_12 must be monotonic non-decreasing.")

        self.warm_gains_bins = self._ensure_gain_bins(
            self.warm_gains_bins,
            np.asarray(self.warm_gain_global, dtype=np.float32),
        )
        self.cool_gains_bins = self._ensure_gain_bins(
            self.cool_gains_bins,
            np.asarray(self.cool_gain_global, dtype=np.float32),
        )

    def _ensure_gain_bins(
        self,
        bins: ArrayLikeF32 | None,
        gain_global: ArrayLikeF32,
    ) -> ArrayLikeF32:
        if bins is None:
            return self._generate_default_bins(gain_global)

        out = np.asarray(bins, dtype=np.float32)
        if out.shape != (12, 3):
            raise ValueError("gain bins must have shape (12, 3).")
        return out

    def _generate_default_bins(self, gain_global: ArrayLikeF32) -> ArrayLikeF32:
        nodes = np.asarray(self.luma_nodes_12, dtype=np.float32)
        atten = _atten_curve(nodes)
        # gain_bin = 1 + atten(y_i) * (gain_global - 1)
        return 1.0 + atten[:, None] * (gain_global[None, :] - 1.0)



def _atten_curve(y: ArrayLikeF32) -> ArrayLikeF32:
    """Piecewise-linear attenuation for default bin table generation.

    Constraint points:
    - y <= 31:  atten = 0.55
    - y == 127: atten = 1.00
    - y >= 239: atten = 0.65
    """

    y = np.asarray(y, dtype=np.float32)
    out = np.empty_like(y, dtype=np.float32)

    mask_low = y <= 31.0
    mask_mid = (y > 31.0) & (y < 127.0)
    mask_hi_mid = (y >= 127.0) & (y < 239.0)
    mask_high = y >= 239.0

    out[mask_low] = 0.55
    out[mask_mid] = 0.55 + (y[mask_mid] - 31.0) * (1.0 - 0.55) / (127.0 - 31.0)
    out[mask_hi_mid] = 1.0 + (y[mask_hi_mid] - 127.0) * (0.65 - 1.0) / (239.0 - 127.0)
    out[mask_high] = 0.65
    return out
