"""Fixed-point WPA configuration and offline LUT generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from wpa.config import (
    CCT_COOL_DEFAULT,
    CCT_NEUTRAL_DEFAULT,
    CCT_WARM_DEFAULT,
    CCT_XY_BLEND_HALF_WIDTH_DEFAULT,
    CCT_XY_SPLIT_DEFAULT,
    build_cct_gain_lut,
)

# ---------------------------------------------------------------------------
# Fixed 12-bin luma nodes (same as float version)
# ---------------------------------------------------------------------------
LUMA_NODES_12: list[int] = [15, 31, 47, 63, 95, 127, 159, 191, 223, 239, 247, 255]

def _atten_curve(y: float) -> float:
    """3-segment piecewise-linear attenuation (same as float version).

        y <= 31  : atten = 0.55
        y == 127 : atten = 1.00
        y >= 239 : atten = 0.35
    """
    if y <= 31:
        return 0.55
    elif y <= 127:
        return 0.55 + 0.45 * (y - 31) / (127 - 31)
    elif y <= 239:
        return 1.00 - 0.65 * (y - 127) / (239 - 127)
    else:
        return 0.35


def _build_wa_base_gain_lut_fixed(
    coeff_frac_bits: int,
    warm_k: float,
    neutral_k: float,
    cool_k: float,
    xy_split_k: float,
    xy_blend_half_width_k: float,
) -> np.ndarray:
    """Build fixed-point anchor gain LUT: (3, 3) int32.

    Anchors are [warm(wa=0), neutral(wa=64), cool(wa=127)].
    """
    one = 1 << coeff_frac_bits
    g_base = build_cct_gain_lut(
        warm_k=warm_k,
        neutral_k=neutral_k,
        cool_k=cool_k,
        xy_split_k=xy_split_k,
        xy_blend_half_width_k=xy_blend_half_width_k,
    )  # (128, 3), float

    table = np.empty((3, 3), dtype=np.int32)
    anchors = g_base[[0, 64, 127]].copy()
    gain_q = np.round(anchors * one)
    table[:, :] = np.clip(gain_q, 0, (1 << 16) - 1).astype(np.int32)
    table[1, :] = one
    return table


@dataclass
class FixedWPAConfig:
    """Fixed-point White Point Adjustment configuration.

    Parameters
    ----------
    coeff_frac_bits : int
        Fractional bits for gain coefficients.
        Default 8 (UQ1.8, encoded scale 256, requires >=9 storage bits).
        Optional 10 (UQ1.10, encoded scale 1024).

    frac_bits : int
        Fractional bits for internal linear pixel values and saturation weights.
        Default 10.
    """

    # --- precision -------------------------------------------------------
    frac_bits: int = 10
    coeff_frac_bits: int = 8

    # --- control ---------------------------------------------------------
    wa_en: bool = True
    wa_sel: int = 64

    # --- gamma (float allowed per user spec) -----------------------------
    gamma_mode: str = "srgb"
    gamma_power: float = 2.2

    # --- luma binning ----------------------------------------------------
    luma_nodes: list[int] = field(default_factory=lambda: list(LUMA_NODES_12))
    bin_interp: bool = True
    luma_domain: str = "gamma"

    # --- saturation protection -------------------------------------------
    sat_en: bool = False
    sat_s0: int = 100           # threshold in 0-255 uint8 scale
    sat_s1: int = 500           # threshold in 0-255 uint8 scale

    # --- gain tables ------------------------------------------------------
    cct_warm_k: float = CCT_WARM_DEFAULT
    cct_neutral_k: float = CCT_NEUTRAL_DEFAULT
    cct_cool_k: float = CCT_COOL_DEFAULT
    cct_xy_split_k: float = CCT_XY_SPLIT_DEFAULT
    cct_xy_blend_half_width_k: float = CCT_XY_BLEND_HALF_WIDTH_DEFAULT
    wa_base_gain_lut_fixed: Optional[np.ndarray] = None   # (3,3) int32: warm/neutral/cool
    atten_q_lut_fixed: Optional[np.ndarray] = None        # (12,) int32 in Q0.F

    def __post_init__(self) -> None:
        """Auto-generate fixed-point base gain LUT if not provided."""
        if self.coeff_frac_bits not in (8, 10):
            raise ValueError("coeff_frac_bits must be 8 or 10")
        if self.wa_base_gain_lut_fixed is None:
            self.wa_base_gain_lut_fixed = _build_wa_base_gain_lut_fixed(
                self.coeff_frac_bits,
                warm_k=self.cct_warm_k,
                neutral_k=self.cct_neutral_k,
                cool_k=self.cct_cool_k,
                xy_split_k=self.cct_xy_split_k,
                xy_blend_half_width_k=self.cct_xy_blend_half_width_k,
            )
        self.wa_base_gain_lut_fixed = np.asarray(
            self.wa_base_gain_lut_fixed, dtype=np.int32
        )
        if self.wa_base_gain_lut_fixed.shape != (3, 3):
            raise ValueError(
                "wa_base_gain_lut_fixed shape must be (3, 3), "
                f"got {self.wa_base_gain_lut_fixed.shape}"
            )
        if self.atten_q_lut_fixed is None:
            one = self.COEFF_ONE
            atten = np.array([_atten_curve(y) for y in self.luma_nodes], dtype=np.float64)
            self.atten_q_lut_fixed = np.round(atten * one).astype(np.int32)
        self.atten_q_lut_fixed = np.asarray(self.atten_q_lut_fixed, dtype=np.int32)
        if self.atten_q_lut_fixed.shape != (len(self.luma_nodes),):
            raise ValueError(
                "atten_q_lut_fixed shape must be (len(luma_nodes),), "
                f"got {self.atten_q_lut_fixed.shape}"
            )

    @property
    def ONE(self) -> int:
        """Fixed-point representation of 1.0."""
        return 1 << self.frac_bits

    @property
    def HALF(self) -> int:
        """Rounding bias = ONE / 2."""
        return 1 << (self.frac_bits - 1)

    @property
    def COEFF_ONE(self) -> int:
        """Fixed-point representation of gain 1.0 in coefficient format."""
        return 1 << self.coeff_frac_bits

    @property
    def COEFF_HALF(self) -> int:
        """Rounding bias for coefficient-domain shifts."""
        return 1 << (self.coeff_frac_bits - 1)

    def runtime_base_gain_fixed(self, wa_sel: int) -> np.ndarray:
        """Compute runtime base gain (3,) by piecewise interpolation."""
        wa = int(np.clip(wa_sel, 0, 127))
        warm = self.wa_base_gain_lut_fixed[0].astype(np.int32)
        neutral = self.wa_base_gain_lut_fixed[1].astype(np.int32)
        cool = self.wa_base_gain_lut_fixed[2].astype(np.int32)

        if wa <= 64:
            # warm -> neutral over 64 steps (/64 -> shift-friendly)
            num = wa
            delta = neutral - warm
            base = warm + ((num * delta + 32) >> 6)
            return base.astype(np.int32)

        # neutral -> cool over 64 steps (weaker cool end, shift-friendly)
        num = wa - 64
        delta = cool - neutral
        base = neutral + ((num * delta + 32) >> 6)
        return base.astype(np.int32)

    def runtime_bin_gains_fixed(self, wa_sel: int) -> np.ndarray:
        """Compute per-WA runtime 12x3 gains in UQ1.F (int32).

        This is the RAM-saving runtime path: store only 3 anchor gains
        and expand to (12,3) for current WA_SEL when needed.
        """
        wa = int(np.clip(wa_sel, 0, 127))
        one = self.COEFF_ONE
        base = self.runtime_base_gain_fixed(wa)  # (3,)
        atten_q = self.atten_q_lut_fixed
        delta = base - one                                        # (3,)
        gains = one + (
            (atten_q[:, np.newaxis].astype(np.int64) * delta[np.newaxis, :].astype(np.int64)
             + self.COEFF_HALF) >> self.coeff_frac_bits
        ).astype(np.int32)
        if wa > 64:
            green_cap = one - int(round(((wa - 64) / 64.0) * 0.02 * one))
            high_idx = np.asarray(self.luma_nodes, dtype=np.int32) >= 223
            gains[high_idx, 1] = np.minimum(gains[high_idx, 1], green_cap)
        if wa == 64:
            gains[:, :] = one
        return gains
