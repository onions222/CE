"""
Fixed-Point WPA Configuration.

All gain-related values are stored as fixed-point integers.
Degamma/engamma are allowed to use float (per user spec).

Key parameter: ``frac_bits`` controls the precision of internal
fixed-point arithmetic.  ``ONE = 1 << frac_bits`` represents 1.0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Fixed 12-bin luma nodes (same as float version)
# ---------------------------------------------------------------------------
LUMA_NODES_12: list[int] = [15, 31, 47, 63, 95, 127, 159, 191, 223, 239, 247, 255]

# ---------------------------------------------------------------------------
# Global warm / cool gain endpoints (float, used for table generation only)
# ---------------------------------------------------------------------------
WARM_GAIN_GLOBAL: tuple[float, float, float] = (1.60, 1.00, 0.40)
COOL_GAIN_GLOBAL: tuple[float, float, float] = (0.60, 1.00, 1.40)


def _atten_curve(y: float) -> float:
    """3-segment piecewise-linear attenuation (same as float version).

        y <= 31  : atten = 0.55
        y == 127 : atten = 1.00
        y >= 239 : atten = 0.65
    """
    if y <= 31:
        return 0.55
    elif y <= 127:
        return 0.55 + 0.45 * (y - 31) / (127 - 31)
    elif y <= 239:
        return 1.00 - 0.35 * (y - 127) / (239 - 127)
    else:
        return 0.65


def _generate_fixed_bin_gains(
    gain_global: tuple[float, float, float],
    luma_nodes: list[int],
    frac_bits: int,
) -> np.ndarray:
    """Generate a (12, 3) fixed-point gain table.

    Each entry = round(gain_float * ONE), where ONE = 1 << frac_bits.

    Returns
    -------
    np.ndarray of shape (N, 3), dtype int32.
    """
    one = 1 << frac_bits
    g = np.array(gain_global, dtype=np.float64)
    table = np.empty((len(luma_nodes), 3), dtype=np.int32)
    for i, y in enumerate(luma_nodes):
        a = _atten_curve(y)
        gain_f = 1.0 + a * (g - 1.0)
        table[i] = np.round(gain_f * one).astype(np.int32)
    return table


@dataclass
class FixedWPAConfig:
    """Fixed-point White Point Adjustment configuration.

    Parameters
    ----------
    frac_bits : int
        Number of fractional bits for fixed-point representation.
        Higher = more precise but wider intermediate values.
        Default 10 → ONE = 1024 (Q6.10 format).
    """

    # --- precision -------------------------------------------------------
    frac_bits: int = 10

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

    # --- gain tables (auto-generated if None) ----------------------------
    warm_gains_fixed: Optional[np.ndarray] = None   # (12,3) int32
    cool_gains_fixed: Optional[np.ndarray] = None   # (12,3) int32

    def __post_init__(self) -> None:
        """Auto-generate fixed-point gain tables if not provided."""
        if self.warm_gains_fixed is None:
            self.warm_gains_fixed = _generate_fixed_bin_gains(
                WARM_GAIN_GLOBAL, self.luma_nodes, self.frac_bits
            )
        if self.cool_gains_fixed is None:
            self.cool_gains_fixed = _generate_fixed_bin_gains(
                COOL_GAIN_GLOBAL, self.luma_nodes, self.frac_bits
            )
        self.warm_gains_fixed = np.asarray(self.warm_gains_fixed, dtype=np.int32)
        self.cool_gains_fixed = np.asarray(self.cool_gains_fixed, dtype=np.int32)

    @property
    def ONE(self) -> int:
        """Fixed-point representation of 1.0."""
        return 1 << self.frac_bits

    @property
    def HALF(self) -> int:
        """Rounding bias = ONE / 2."""
        return 1 << (self.frac_bits - 1)
