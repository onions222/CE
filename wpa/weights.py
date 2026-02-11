"""
WPA helper functions — luma proxy, 12-bin gain interpolation, saturation weight.

All functions operate on numpy arrays for vectorised per-pixel processing.
"""

from __future__ import annotations

import numpy as np


# ========================================================================
# Luma proxy  —  Y ≈ (R + 2*G + B) / 4
# ========================================================================

def compute_luma_proxy(rgb: np.ndarray) -> np.ndarray:
    """Compute a cheap luma approximation per pixel.

    Parameters
    ----------
    rgb : (H, W, 3) float32 — either gamma-encoded [0,1] or linear [0,1]
          depending on ``cfg.luma_domain``.

    Returns
    -------
    (H, W) float32 luma proxy in the same domain as *rgb*.

    DDIC/WPA note: ``Y = (R + 2G + B) >> 2`` in fixed-point HW.
    """
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (r + 2.0 * g + b) * 0.25


# ========================================================================
# 12-bin gain interpolation
# ========================================================================

def interpolate_gains_12bin(
    luma: np.ndarray,
    gains_table: np.ndarray,
    luma_nodes: list[int] | np.ndarray,
    interp: bool = True,
) -> np.ndarray:
    """Look up per-pixel gains from the 12-bin table with linear interpolation.

    Parameters
    ----------
    luma        : (H, W) float32 luma proxy, range [0, 1].
    gains_table : (N, 3) float64/32 per-bin gains (one row per node).
    luma_nodes  : length-N monotone nodes, 8-bit scale (0..255).
    interp      : if True, linearly interpolate between adjacent nodes;
                  otherwise snap to the nearest-lower node (floor).

    Returns
    -------
    (H, W, 3) float32 gains for each pixel.

    Implementation
    --------------
    * Scale luma from [0,1] to [0,255] for node matching.
    * For pixels below the first node → clamp to first node gain.
    * For pixels above the last node  → clamp to last node gain.
    * Between nodes: ``gain = (1-t)*gain[lo] + t*gain[hi]``.
    """
    nodes = np.asarray(luma_nodes, dtype=np.float32)
    n_nodes = len(nodes)
    gains = np.asarray(gains_table, dtype=np.float32)  # (N, 3)

    # Map luma [0,1] → [0,255]
    y = luma * 255.0  # (H, W)

    if not interp:
        # Snap to nearest-lower node index
        idx = np.searchsorted(nodes, y, side="right").astype(np.int32) - 1
        idx = np.clip(idx, 0, n_nodes - 1)
        return gains[idx]  # (H, W, 3)

    # --- Linear interpolation between nodes ---
    # searchsorted(side='right') gives the insertion index;
    # the "lower" node is idx_hi - 1.
    idx_hi = np.searchsorted(nodes, y, side="right").astype(np.int32)
    idx_hi = np.clip(idx_hi, 1, n_nodes - 1)       # at least 1
    idx_lo = idx_hi - 1                               # always >= 0

    node_lo = nodes[idx_lo]  # (H, W)
    node_hi = nodes[idx_hi]  # (H, W)

    span = node_hi - node_lo
    # Avoid division by zero when y is exactly at a node (span == 0)
    safe_span = np.where(span > 0, span, 1.0)
    t = np.clip((y - node_lo) / safe_span, 0.0, 1.0)  # (H, W)

    gain_lo = gains[idx_lo]  # (H, W, 3)
    gain_hi = gains[idx_hi]  # (H, W, 3)

    t_3d = t[..., np.newaxis]  # broadcast to (H, W, 1)
    return (1.0 - t_3d) * gain_lo + t_3d * gain_hi  # (H, W, 3)


# ========================================================================
# Saturation weight  —  low-saturation protection
# ========================================================================

def compute_sat_weight(
    rgb: np.ndarray,
    s0: float = 20.0,
    s1: float = 110.0,
) -> np.ndarray:
    """Compute a [0,1] weight that attenuates the gain for saturated pixels.

    Cheap saturation proxy:
        ``s = |R-G| + |G-B| + |B-R|``

    Weight ramp:
        * s ≤ s0 → w = 1  (achromatic / grey — full WPA effect)
        * s ≥ s1 → w = 0  (saturated — suppress WPA to avoid hue shift)
        * linear between s0 and s1

    Parameters
    ----------
    rgb : (H, W, 3) float32.  The domain (gamma or linear) is determined
          by ``cfg.sat_weight_domain``; the caller is responsible for
          providing the correct representation.
    s0, s1 : ramp thresholds.  These should be in the *same units* as the
             pixel values (e.g., 0–255 for gamma-domain uint8-scaled data,
             or 0–1 for linear).

    Returns
    -------
    (H, W) float32 weight in [0, 1].
    """
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    s = np.abs(r - g) + np.abs(g - b) + np.abs(b - r)
    w = np.clip((s1 - s) / (s1 - s0 + 1e-12), 0.0, 1.0)
    return w
