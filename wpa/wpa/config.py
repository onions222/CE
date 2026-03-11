"""
WPA Configuration — WPAConfig dataclass with all parameters.

Defaults:
- 12-bin luma nodes (fixed)
- Automatic per-bin gain generation via 3-segment attenuation curve
- sRGB gamma mode, saturation protection enabled
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from .gamma import srgb_degamma


# ---------------------------------------------------------------------------
# Fixed 12-bin luma nodes (8-bit Y sample points, monotonically increasing)
# ---------------------------------------------------------------------------
LUMA_NODES_12: list[int] = [15, 31, 47, 63, 95, 127, 159, 191, 223, 239, 247, 255]

# ---------------------------------------------------------------------------
# Default CCT anchors (Kelvin)
# ---------------------------------------------------------------------------
CCT_WARM_DEFAULT: float = 3000.0
CCT_NEUTRAL_DEFAULT: float = 6500.0
CCT_COOL_DEFAULT: float = 9300.0
CCT_XY_SPLIT_DEFAULT: float = 4000.0
CCT_XY_BLEND_HALF_WIDTH_DEFAULT: float = 100.0


def wa_sel_to_cct(
    wa_sel: int,
    warm_k: float = CCT_WARM_DEFAULT,
    neutral_k: float = CCT_NEUTRAL_DEFAULT,
    cool_k: float = CCT_COOL_DEFAULT,
) -> float:
    """Map WA_SEL (0..127) to CCT (Kelvin) using piecewise linear mapping."""
    w = int(np.clip(wa_sel, 0, 127))
    if w <= 64:
        return neutral_k - (64 - w) * (neutral_k - warm_k) / 64.0
    return neutral_k + (w - 64) * (cool_k - neutral_k) / 63.0


def map_sat_threshold_gamma_to_linear(s_gamma: float) -> float:
    """Map saturation threshold from gamma-domain scale [0,510] to linear [0,2].

    We model saturation with:
        s = |R-G| + |G-B| + |B-R|
    For a one-axis case (R=v,G=0,B=0), s = 2v.  We map the normalized
    channel delta with sRGB degamma, then scale back to s-range [0,2].
    """
    s_clamped = float(np.clip(s_gamma, 0.0, 510.0))
    d_gamma = s_clamped / 510.0  # normalized channel delta
    d_linear = float(srgb_degamma(np.array(d_gamma, dtype=np.float64)))
    return 2.0 * d_linear


def cct_to_xy_approx(
    cct: float,
    split_k: float = CCT_XY_SPLIT_DEFAULT,
    blend_half_width_k: float = CCT_XY_BLEND_HALF_WIDTH_DEFAULT,
) -> tuple[float, float]:
    """Approximate CIE 1931 xy from CCT using standard polynomial fits.

    Parameters
    ----------
    cct : float
        Correlated color temperature in Kelvin.
    split_k : float
        Branch split center (default 4000K).
    blend_half_width_k : float
        Smooth blending half-width around ``split_k``.
    """
    t = float(np.clip(cct, 1667.0, 25000.0))

    x_low = (
        -0.2661239e9 / (t ** 3)
        - 0.2343580e6 / (t ** 2)
        + 0.8776956e3 / t
        + 0.179910
    )
    x_high = (
        -3.0258469e9 / (t ** 3)
        + 2.1070379e6 / (t ** 2)
        + 0.2226347e3 / t
        + 0.240390
    )

    split = float(split_k)
    half = max(float(blend_half_width_k), 0.0)
    blend_lo = split - half
    blend_hi = split + half
    if half <= 0.0:
        x = x_low if t <= split else x_high
        u = 0.0
    elif t <= blend_lo:
        x = x_low
        u = 0.0
    elif t >= blend_hi:
        x = x_high
        u = 1.0
    else:
        u = (t - blend_lo) / (blend_hi - blend_lo)
        u = u * u * (3.0 - 2.0 * u)  # smoothstep
        x = (1.0 - u) * x_low + u * x_high

    if t <= 2222.0:
        y = -1.1063814 * (x ** 3) - 1.34811020 * (x ** 2) + 2.18555832 * x - 0.20219683
    elif t < blend_lo:
        y = -0.9549476 * (x ** 3) - 1.37418593 * (x ** 2) + 2.09137015 * x - 0.16748867
    elif t > blend_hi:
        y = 3.0817580 * (x ** 3) - 5.87338670 * (x ** 2) + 3.75112997 * x - 0.37001483
    else:
        y_mid = -0.9549476 * (x ** 3) - 1.37418593 * (x ** 2) + 2.09137015 * x - 0.16748867
        y_high = 3.0817580 * (x ** 3) - 5.87338670 * (x ** 2) + 3.75112997 * x - 0.37001483
        y = (1.0 - u) * y_mid + u * y_high

    return float(x), float(y)


def _xy_to_linear_srgb_white(x: float, y: float) -> np.ndarray:
    """Convert xy white point to linear sRGB white vector."""
    y_safe = max(y, 1e-8)
    xyz = np.array([x / y_safe, 1.0, (1.0 - x - y_safe) / y_safe], dtype=np.float64)
    m_xyz_to_srgb = np.array(
        [
            [3.2406, -1.5372, -0.4986],
            [-0.9689, 1.8758, 0.0415],
            [0.0557, -0.2040, 1.0570],
        ],
        dtype=np.float64,
    )
    rgb = m_xyz_to_srgb @ xyz
    return np.clip(rgb, 1e-6, None)


def build_cct_gain_lut(
    warm_k: float = CCT_WARM_DEFAULT,
    neutral_k: float = CCT_NEUTRAL_DEFAULT,
    cool_k: float = CCT_COOL_DEFAULT,
    xy_split_k: float = CCT_XY_SPLIT_DEFAULT,
    xy_blend_half_width_k: float = CCT_XY_BLEND_HALF_WIDTH_DEFAULT,
    gain_min: float = 0.5,
    gain_max: float = 1.8,
) -> np.ndarray:
    """Build (128,3) per-WA_SEL RGB gain LUT from CCT anchors."""
    x_n, y_n = cct_to_xy_approx(
        neutral_k,
        split_k=xy_split_k,
        blend_half_width_k=xy_blend_half_width_k,
    )
    neutral_rgb = _xy_to_linear_srgb_white(x_n, y_n)

    lut = np.empty((128, 3), dtype=np.float64)
    for wa_sel in range(128):
        cct = wa_sel_to_cct(
            wa_sel, warm_k=warm_k, neutral_k=neutral_k, cool_k=cool_k
        )
        x_t, y_t = cct_to_xy_approx(
            cct,
            split_k=xy_split_k,
            blend_half_width_k=xy_blend_half_width_k,
        )
        target_rgb = _xy_to_linear_srgb_white(x_t, y_t)

        gain = target_rgb / neutral_rgb
        # Keep luma stable to avoid global brightness drift.
        y_gain = 0.2126 * gain[0] + 0.7152 * gain[1] + 0.0722 * gain[2]
        gain = gain / max(y_gain, 1e-6)
        # For cool white points, avoid lifting G above identity. Otherwise
        # near-white highlights can drift toward cyan/green around high-luma nodes.
        if wa_sel > 64:
            gain[1] = min(gain[1], 1.0)
        lut[wa_sel] = np.clip(gain, gain_min, gain_max)

    # Guarantee identity at WA_SEL=64 exactly.
    lut[64] = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    return lut


_DEFAULT_CCT_GAIN_LUT = build_cct_gain_lut()
WARM_GAIN_GLOBAL: tuple[float, float, float] = tuple(_DEFAULT_CCT_GAIN_LUT[0].tolist())
COOL_GAIN_GLOBAL: tuple[float, float, float] = tuple(_DEFAULT_CCT_GAIN_LUT[127].tolist())


def _atten_curve(y: float) -> float:
    """3-segment piecewise-linear attenuation curve for default gain table.

    Designed so that dark and bright regions have reduced gain (fewer
    artefacts), while mid-tones have full strength.

        y <= 31  : atten = 0.55
        y == 127 : atten = 1.00
        y >= 239 : atten = 0.35
        linear interpolation between breakpoints.
    """
    if y <= 31:
        return 0.55
    elif y <= 127:
        # linear from 0.55 @ y=31  to  1.00 @ y=127
        return 0.55 + (1.00 - 0.55) * (y - 31) / (127 - 31)
    elif y <= 239:
        # linear from 1.00 @ y=127  to  0.35 @ y=239
        return 1.00 + (0.35 - 1.00) * (y - 127) / (239 - 127)
    else:
        return 0.35


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
    nodes = np.asarray(luma_nodes, dtype=np.int32)
    is_cool = g[2] > g[0]
    is_warm = g[0] > g[2]
    warm_red_caps = {223: 1.30, 239: 1.22, 247: 1.16, 255: 1.10}
    warm_green_caps = {223: 0.94, 239: 0.94, 247: 0.94, 255: 0.94}
    warm_blue_floors = {223: 0.80, 239: 0.84, 247: 0.87, 255: 0.90}
    cool_green_caps = {223: 0.98, 239: 0.975, 247: 0.97, 255: 0.965}
    cool_blue_caps = {223: 1.15, 239: 1.12, 247: 1.08, 255: 1.06}
    for i, y in enumerate(luma_nodes):
        a = _atten_curve(y)
        table[i] = 1.0 + a * (g - 1.0)
        if is_warm and y >= 223:
            node = int(y)
            table[i, 0] = min(table[i, 0], warm_red_caps.get(node, warm_red_caps[int(nodes[-1])]))
            table[i, 1] = min(table[i, 1], warm_green_caps.get(node, warm_green_caps[int(nodes[-1])]))
            table[i, 2] = max(table[i, 2], warm_blue_floors.get(node, warm_blue_floors[int(nodes[-1])]))
        if is_cool and y >= 223:
            table[i, 1] = min(table[i, 1], cool_green_caps.get(int(y), 0.98))
        if is_cool and y >= 223:
            table[i, 2] = min(table[i, 2], cool_blue_caps.get(int(y), cool_blue_caps[int(nodes[-1])]))
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
    use_cct_gains: bool = True
    cct_warm_k: float = CCT_WARM_DEFAULT
    cct_neutral_k: float = CCT_NEUTRAL_DEFAULT
    cct_cool_k: float = CCT_COOL_DEFAULT
    cct_xy_split_k: float = CCT_XY_SPLIT_DEFAULT
    cct_xy_blend_half_width_k: float = CCT_XY_BLEND_HALF_WIDTH_DEFAULT
    warm_gains_bins: Optional[np.ndarray] = None
    cool_gains_bins: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        """Auto-generate default per-bin gain tables when not supplied."""
        if self.use_cct_gains:
            lut = build_cct_gain_lut(
                warm_k=self.cct_warm_k,
                neutral_k=self.cct_neutral_k,
                cool_k=self.cct_cool_k,
                xy_split_k=self.cct_xy_split_k,
                xy_blend_half_width_k=self.cct_xy_blend_half_width_k,
            )
            warm_gain_global = tuple(lut[0].tolist())
            cool_gain_global = tuple(lut[127].tolist())
        else:
            warm_gain_global = WARM_GAIN_GLOBAL
            cool_gain_global = COOL_GAIN_GLOBAL

        if self.warm_gains_bins is None:
            self.warm_gains_bins = generate_default_bin_gains(
                warm_gain_global, self.luma_nodes
            )
        if self.cool_gains_bins is None:
            self.cool_gains_bins = generate_default_bin_gains(
                cool_gain_global, self.luma_nodes
            )
        # Ensure numpy arrays
        self.warm_gains_bins = np.asarray(self.warm_gains_bins, dtype=np.float64)
        self.cool_gains_bins = np.asarray(self.cool_gains_bins, dtype=np.float64)
