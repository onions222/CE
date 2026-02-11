"""
WPA Configuration — WPAConfig dataclass with all parameters.

Defaults:
- 12-bin luma nodes (fixed)
- Automatic per-bin gain generation via 3-segment attenuation curve
- sRGB gamma mode, saturation protection enabled
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Fixed 12-bin luma nodes (8-bit Y sample points, monotonically increasing)
# ---------------------------------------------------------------------------
LUMA_NODES_12: list[int] = [15, 31, 47, 63, 95, 127, 159, 191, 223, 239, 247, 255]

# ---------------------------------------------------------------------------
# Global warm / cool gain endpoints (per-channel RGB)
# ---------------------------------------------------------------------------
WARM_GAIN_GLOBAL: tuple[float, float, float] = (1.40, 1.00, 0.60)
COOL_GAIN_GLOBAL: tuple[float, float, float] = (0.60, 1.00, 1.40)


def _atten_curve(y: float) -> float:
    """3-segment piecewise-linear attenuation curve for default gain table.

    Designed so that dark and bright regions have reduced gain (fewer
    artefacts), while mid-tones have full strength.

        y <= 31  : atten = 0.55
        y == 127 : atten = 1.00
        y >= 239 : atten = 0.65
        linear interpolation between breakpoints.
    """
    if y <= 31:
        return 0.55
    elif y <= 127:
        # linear from 0.55 @ y=31  to  1.00 @ y=127
        return 0.55 + (1.00 - 0.55) * (y - 31) / (127 - 31)
    elif y <= 239:
        # linear from 1.00 @ y=127  to  0.65 @ y=239
        return 1.00 + (0.65 - 1.00) * (y - 127) / (239 - 127)
    else:
        return 0.65


def generate_default_bin_gains(
    gain_global: tuple[float, float, float],
    luma_nodes: list[int] | None = None,
) -> np.ndarray:
    """Generate a (12, 3) gain table from a global gain endpoint.

    For each luma node *y_i*:
        gain_bin[i] = 1 + atten(y_i) * (gain_global - 1)

    Parameters
    ----------
    gain_global : (R, G, B) global gain endpoint (warm or cool).
    luma_nodes  : list of luma sample points (default: LUMA_NODES_12).

    Returns
    -------
    np.ndarray of shape (N, 3), dtype float64.
    """
    if luma_nodes is None:
        luma_nodes = LUMA_NODES_12
    g = np.array(gain_global, dtype=np.float64)
    table = np.empty((len(luma_nodes), 3), dtype=np.float64)
    for i, y in enumerate(luma_nodes):
        a = _atten_curve(y)
        table[i] = 1.0 + a * (g - 1.0)
    return table


@dataclass
class WPAConfig:
    """White Point Adjustment configuration.

    Control
    -------
    wa_en   : master enable.
    wa_sel  : 0..127 adjustment selector.
              64 = identity (no change).
              0..63  = warmer (smaller → stronger warm).
              65..127 = cooler (larger → stronger cool).

    Gamma
    -----
    gamma_mode     : ``"srgb"`` | ``"power"`` | ``"none"``.
    gamma_power    : exponent when *gamma_mode* = ``"power"`` (default 2.2).
    use_gamma_lut  : if True, use 256-entry LUT for sRGB degamma/engamma.

    Luma Binning
    -------------
    luma_nodes     : 12 monotone 8-bit luma sample points.
    bin_interp     : linearly interpolate gains between nodes.
    luma_domain    : ``"gamma"`` or ``"linear"`` — domain for luma proxy.

    Saturation Protection
    ---------------------
    sat_en              : enable/disable saturation weight.
    sat_s0, sat_s1      : ramp thresholds (s ≤ s0 → w=1, s ≥ s1 → w=0).
    sat_weight_domain   : ``"gamma"`` or ``"linear"`` — domain for weight calc.

    Gain Tables
    -----------
    warm_gains_bins : (12,3) per-bin warm gains; auto-generated if None.
    cool_gains_bins : (12,3) per-bin cool gains; auto-generated if None.
    """

    # --- control ---------------------------------------------------------
    wa_en: bool = True
    wa_sel: int = 64

    # --- gamma -----------------------------------------------------------
    gamma_mode: str = "srgb"        # "srgb" | "power" | "none"
    gamma_power: float = 2.2
    use_gamma_lut: bool = False

    # --- luma binning ----------------------------------------------------
    luma_nodes: list[int] = field(default_factory=lambda: list(LUMA_NODES_12))
    bin_interp: bool = True
    luma_domain: str = "gamma"      # "gamma" | "linear"

    # --- saturation protection -------------------------------------------
    sat_en: bool = False
    sat_s0: float = 100.0           # s <= s0 → w = 1  (grey → full effect)
    sat_s1: float = 500.0           # s >= s1 → w = 0  (saturated → no effect)
    sat_weight_domain: str = "gamma"  # "gamma" | "linear"

    # --- gain tables (user-overridable, auto-generated if None) ----------
    warm_gains_bins: Optional[np.ndarray] = None
    cool_gains_bins: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        """Auto-generate default per-bin gain tables when not supplied."""
        if self.warm_gains_bins is None:
            self.warm_gains_bins = generate_default_bin_gains(
                WARM_GAIN_GLOBAL, self.luma_nodes
            )
        if self.cool_gains_bins is None:
            self.cool_gains_bins = generate_default_bin_gains(
                COOL_GAIN_GLOBAL, self.luma_nodes
            )
        # Ensure numpy arrays
        self.warm_gains_bins = np.asarray(self.warm_gains_bins, dtype=np.float64)
        self.cool_gains_bins = np.asarray(self.cool_gains_bins, dtype=np.float64)
