"""
Fixed-point helper functions — luma proxy, gain interpolation, saturation weight.

All runtime arithmetic is pure integer (int32/int64).
No floating-point operations.
"""

from __future__ import annotations

import numpy as np


# ========================================================================
# Luma proxy  —  Y = (R + 2*G + B) >> 2   (pure uint8 arithmetic)
# ========================================================================

def compute_luma_proxy_u8(rgb_u8: np.ndarray) -> np.ndarray:
    """Compute luma proxy from gamma-domain uint8 RGB.

    Y = (R + 2*G + B) >> 2

    Parameters
    ----------
    rgb_u8 : (H, W, 3) uint8

    Returns
    -------
    (H, W) uint8 luma proxy [0, 255].
    """
    # Promote to int16 to avoid uint8 overflow: max = 255+510+255 = 1020
    r = rgb_u8[..., 0].astype(np.int16)
    g = rgb_u8[..., 1].astype(np.int16)
    b = rgb_u8[..., 2].astype(np.int16)
    y = (r + 2 * g + b) >> 2
    return y.astype(np.uint8)


# ========================================================================
# 12-bin gain interpolation (integer arithmetic)
# ========================================================================

def interpolate_gains_fixed(
    luma_u8: np.ndarray,
    gains_table: np.ndarray,
    luma_nodes: list[int] | np.ndarray,
    interp: bool = True,
    interp_bits: int = 10,
) -> np.ndarray:
    """Look up per-pixel gains from 12-bin table using integer interpolation.

    All arithmetic is integer.  Interpolation parameter t is represented
    with ``interp_bits`` fractional bits.

    Parameters
    ----------
    luma_u8     : (H, W) uint8 luma proxy [0, 255].
    gains_table : (N, 3) int32 fixed-point gains.
    luma_nodes  : length-N nodes (uint8 scale).
    interp      : if True, linear interpolation; else floor snap.
    interp_bits : fractional bits for the interpolation parameter t.

    Returns
    -------
    (H, W, 3) int32 fixed-point gains.
    """
    nodes = np.asarray(luma_nodes, dtype=np.int32)
    n_nodes = len(nodes)
    gains = np.asarray(gains_table, dtype=np.int32)
    y = luma_u8.astype(np.int32)  # (H, W) in [0, 255]

    if not interp:
        idx = np.searchsorted(nodes, y, side="right").astype(np.int32) - 1
        idx = np.clip(idx, 0, n_nodes - 1)
        return gains[idx]

    # --- Integer linear interpolation ---
    idx_hi = np.searchsorted(nodes, y, side="right").astype(np.int32)
    idx_hi = np.clip(idx_hi, 1, n_nodes - 1)
    idx_lo = idx_hi - 1

    node_lo = nodes[idx_lo]  # (H, W)
    node_hi = nodes[idx_hi]  # (H, W)

    span = (node_hi - node_lo).astype(np.int32)
    safe_span = np.where(span > 0, span, 1)
    span_all = np.diff(nodes).astype(np.int32)
    spans_are_pow2 = (
        np.all(span_all > 0)
        and np.all((span_all & (span_all - 1)) == 0)
    )

    # t_fixed = ((y - node_lo) << interp_bits) / span
    # Use rounded right-shift when span is power-of-two, otherwise
    # fall back to integer division.
    numer = ((y - node_lo) << interp_bits) + (safe_span >> 1)
    if spans_are_pow2:
        shift_lut = np.zeros(n_nodes, dtype=np.int32)
        shift_lut[:-1] = np.array(
            [int(s).bit_length() - 1 for s in span_all],
            dtype=np.int32,
        )
        t = np.clip(numer >> shift_lut[idx_lo], 0, 1 << interp_bits).astype(np.int32)
    else:
        t = np.clip(numer // safe_span, 0, 1 << interp_bits).astype(np.int32)

    gain_lo = gains[idx_lo]  # (H, W, 3)
    gain_hi = gains[idx_hi]  # (H, W, 3)

    t_3d = t[..., np.newaxis]  # (H, W, 1)
    # gain = gain_lo + (t * (gain_hi - gain_lo) + HALF_INTERP) >> interp_bits
    delta = gain_hi - gain_lo
    half_interp = 1 << (interp_bits - 1)
    result = gain_lo + ((t_3d * delta + half_interp) >> interp_bits)
    return result.astype(np.int32)


# ========================================================================
# Saturation weight (integer arithmetic)
# ========================================================================

def compute_sat_weight_fixed(
    rgb_u8: np.ndarray,
    s0: int,
    s1: int,
    frac_bits: int,
) -> np.ndarray:
    """Compute saturation weight in fixed-point.

    s = |R-G| + |G-B| + |B-R|   (uint8 arithmetic, range 0-510)

    w_fixed = clip((s1 - s) << frac_bits / (s1 - s0), 0, ONE)

    Parameters
    ----------
    rgb_u8    : (H, W, 3) uint8 gamma-domain pixels.
    s0, s1    : integer thresholds (0-510 scale).
    frac_bits : fractional bits for output weight.

    Returns
    -------
    (H, W) int32 weight in [0, ONE].
    """
    r = rgb_u8[..., 0].astype(np.int32)
    g = rgb_u8[..., 1].astype(np.int32)
    b = rgb_u8[..., 2].astype(np.int32)

    s = np.abs(r - g) + np.abs(g - b) + np.abs(b - r)

    one = 1 << frac_bits
    denom = max(s1 - s0, 1)

    # w = ((s1 - s) << frac_bits + denom // 2) // denom
    numer = ((s1 - s) << frac_bits) + (denom >> 1)
    w = numer // denom
    return np.clip(w, 0, one).astype(np.int32)
