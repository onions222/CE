"""Weighting and interpolation helpers for WPA."""

from __future__ import annotations

import numpy as np


def luma_proxy_y8(rgb01: np.ndarray) -> np.ndarray:
    """Compute 8-bit luma proxy Y ~= (R + 2G + B) / 4 from normalized RGB."""
    rgb = np.asarray(rgb01, dtype=np.float32)
    y01 = (rgb[..., 0] + 2.0 * rgb[..., 1] + rgb[..., 2]) * 0.25
    return np.clip(y01 * 255.0, 0.0, 255.0).astype(np.float32)


def saturation_proxy(rgb01: np.ndarray) -> np.ndarray:
    """Cheap saturation proxy: |R-G| + |G-B| + |B-R| on normalized RGB."""
    rgb = np.asarray(rgb01, dtype=np.float32)
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    return (np.abs(r - g) + np.abs(g - b) + np.abs(b - r)).astype(np.float32)


def saturation_weight(s: np.ndarray, s0: float, s1: float) -> np.ndarray:
    """Piecewise linear weight: s<=s0 => 1, s>=s1 => 0."""
    s = np.asarray(s, dtype=np.float32)
    w = np.empty_like(s, dtype=np.float32)
    w[s <= s0] = 1.0
    w[s >= s1] = 0.0
    mid = (s > s0) & (s < s1)
    w[mid] = (s1 - s[mid]) / (s1 - s0)
    return w


def interpolate_bins(
    y8: np.ndarray,
    nodes: np.ndarray,
    bin_values: np.ndarray,
    *,
    interp: bool,
) -> np.ndarray:
    """Interpolate per-luma bin values over fixed Y nodes.

    Args:
        y8: Y in [0,255], arbitrary shape.
        nodes: shape (N,), monotonic increasing.
        bin_values: shape (N, C) or (N,).
        interp: if False, use nearest lower bin (step lookup).
    """

    y = np.asarray(y8, dtype=np.float32)
    nd = np.asarray(nodes, dtype=np.float32)
    bv = np.asarray(bin_values, dtype=np.float32)

    if nd.ndim != 1:
        raise ValueError("nodes must be 1-D")
    if bv.shape[0] != nd.shape[0]:
        raise ValueError("bin_values first dimension must match nodes length")

    y_clamped = np.clip(y, nd[0], nd[-1])

    if not interp:
        idx = np.searchsorted(nd, y_clamped, side="right") - 1
        idx = np.clip(idx, 0, nd.shape[0] - 1)
        return bv[idx]

    idx_hi = np.searchsorted(nd, y_clamped, side="left")
    idx_hi = np.clip(idx_hi, 0, nd.shape[0] - 1)
    idx_lo = np.clip(idx_hi - 1, 0, nd.shape[0] - 1)

    n_lo = nd[idx_lo]
    n_hi = nd[idx_hi]
    denom = np.where(n_hi > n_lo, n_hi - n_lo, 1.0)
    t = ((y_clamped - n_lo) / denom).astype(np.float32)
    t = np.where(idx_hi == idx_lo, 0.0, t)

    v_lo = bv[idx_lo]
    v_hi = bv[idx_hi]

    if bv.ndim == 1:
        return (v_lo + t * (v_hi - v_lo)).astype(np.float32)
    return (v_lo + t[..., None] * (v_hi - v_lo)).astype(np.float32)
