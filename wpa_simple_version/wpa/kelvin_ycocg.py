"""Kelvin-based white-point adjustment in YCoCg domain.

This is a conservative port of the previous implementation (wpa_python_port),
adapted to this repo's pipeline:
- Build 12-bin warm/cool endpoint transforms via Bradford CAT.
- Apply a per-bin interpolated delta transform in YCoCg on *linear-light* RGB.
- Use headroom-limited strength to avoid clipping.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

Array = np.ndarray


def rgb_to_ycocg(rgb255: Array) -> tuple[Array, Array, Array]:
    rgb = np.asarray(rgb255, dtype=np.float32)
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    y = 0.25 * r + 0.50 * g + 0.25 * b
    co = 0.50 * (r - b)
    cg = -0.25 * r + 0.50 * g - 0.25 * b
    return y, co, cg


def ycocg_to_rgb(y: Array, co: Array, cg: Array) -> Array:
    y = np.asarray(y, dtype=np.float32)
    co = np.asarray(co, dtype=np.float32)
    cg = np.asarray(cg, dtype=np.float32)
    r = y + co - cg
    g = y + cg
    b = y - co - cg
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def _brightness_factor_smooth(y: Array, *, y_dark2: float, y_dark1: float, y_bright1: float, y_bright2: float, minfac: float) -> Array:
    y = np.asarray(y, dtype=np.float32)
    minfac = float(minfac)
    out = np.ones_like(y, dtype=np.float32)

    if y_dark1 > y_dark2:
        out[y <= y_dark2] = minfac
        m = (y > y_dark2) & (y < y_dark1)
        out[m] = minfac + (y[m] - y_dark2) * (1.0 - minfac) / (y_dark1 - y_dark2)

    if y_bright2 > y_bright1:
        out[y >= y_bright2] = minfac
        m = (y > y_bright1) & (y < y_bright2)
        out[m] = 1.0 - (y[m] - y_bright1) * (1.0 - minfac) / (y_bright2 - y_bright1)

    return np.clip(out, minfac, 1.0).astype(np.float32)


def _bound_channel(c0: Array, dc: Array, rgb_max: float) -> Array:
    c0 = np.asarray(c0, dtype=np.float32).reshape(-1)
    dc = np.asarray(dc, dtype=np.float32).reshape(-1)
    rgb_max = float(rgb_max)

    bnd = np.full_like(c0, np.inf, dtype=np.float32)
    eps = 1e-12
    pos = dc > eps
    neg = dc < -eps
    bnd[pos] = (rgb_max - c0[pos]) / dc[pos]
    bnd[neg] = (0.0 - c0[neg]) / dc[neg]
    bad0 = (c0 < 0.0) | (c0 > rgb_max)
    bnd[bad0] = 0.0
    return np.maximum(bnd, 0.0).astype(np.float32)


def _limit_strength_by_rgb_headroom(y0_3xn: Array, dy_3xn: Array, s: Array, rgb_max: float) -> Array:
    s = np.asarray(s, dtype=np.float32).reshape(-1)
    r0, g0, b0 = ycocg_to_rgb(y0_3xn[0, :], y0_3xn[1, :], y0_3xn[2, :]).T
    dr, dg, db = ycocg_to_rgb(dy_3xn[0, :], dy_3xn[1, :], dy_3xn[2, :]).T

    smax = np.full_like(s, np.inf, dtype=np.float32)
    smax = np.minimum(smax, _bound_channel(r0, dr, rgb_max))
    smax = np.minimum(smax, _bound_channel(g0, dg, rgb_max))
    smax = np.minimum(smax, _bound_channel(b0, db, rgb_max))
    return np.maximum(np.minimum(s, smax), 0.0).astype(np.float32)


@dataclass(frozen=True)
class KelvinTableOptions:
    t_warm_end: float = 3500.0
    t_cool_end: float = 9000.0
    xy_base: tuple[float, float] = (0.3127, 0.3290)  # D65

    w_shadow: float = 0.60
    w_mid: float = 1.00
    w_highlight: float = 0.70
    bin_mid: int = 6


@dataclass(frozen=True)
class KelvinTables12Bin:
    opts: KelvinTableOptions
    w: Array  # (12,)
    d_warm: Array  # (12,3,3)
    d_cool: Array  # (12,3,3)


def _build_weights_12(w_shadow: float, w_mid: float, w_highlight: float, bin_mid: int) -> Array:
    w = np.zeros(12, dtype=np.float32)
    for i in range(12):
        ii = i + 1  # 1-based
        if ii <= bin_mid:
            t = (ii - 1) / max(1, (bin_mid - 1))
            w[i] = w_shadow + t * (w_mid - w_shadow)
        else:
            t = (ii - bin_mid) / max(1, (12 - bin_mid))
            w[i] = w_mid + t * (w_highlight - w_mid)
    return np.clip(w, 0.0, 1.0).astype(np.float32)


def _interp_cct_mired(t_base: float, t_end: float, w: Array) -> Array:
    m_base = 1e6 / float(t_base)
    m_end = 1e6 / float(t_end)
    m = (1.0 - w) * m_base + w * m_end
    return (1e6 / m).astype(np.float32)


def _xy_to_xyz_y1(x: float, y: float) -> Array:
    yv = 1.0
    X = (x / y) * yv
    Z = ((1.0 - x - y) / y) * yv
    return np.array([X, yv, Z], dtype=np.float32)


def _xy_to_cct_mccamy(x: float, y: float) -> float:
    n = (x - 0.3320) / (0.1858 - y)
    return float(449.0 * n**3 + 3525.0 * n**2 + 6823.3 * n + 5520.33)


def _cct_to_xy_kang2002(t: float) -> tuple[float, float]:
    if t < 1667 or t > 25000:
        raise ValueError("Kang2002 valid range is 1667..25000K.")
    if t <= 4000:
        x = -0.2661239e9 / t**3 - 0.2343589e6 / t**2 + 0.8776956e3 / t + 0.179910
    else:
        x = -3.0258469e9 / t**3 + 2.1070379e6 / t**2 + 0.2226347e3 / t + 0.240390

    if t <= 2222:
        y = -1.1063814 * x**3 - 1.34811020 * x**2 + 2.18555832 * x - 0.20219683
    elif t <= 4000:
        y = -0.9549476 * x**3 - 1.37418593 * x**2 + 2.09137015 * x - 0.16748867
    else:
        y = 3.0817580 * x**3 - 5.8733867 * x**2 + 3.75112997 * x - 0.37001483
    return float(x), float(y)


@lru_cache(maxsize=8)
def build_tables_kelvin_12bin(opts: KelvinTableOptions = KelvinTableOptions()) -> KelvinTables12Bin:
    # YCoCg matrices (matches rgb_to_ycocg/ycocg_to_rgb)
    M_R2Y = np.array([[0.25, 0.50, 0.25], [0.50, 0.00, -0.50], [-0.25, 0.50, -0.25]], dtype=np.float32)
    M_Y2R = np.array([[1, 1, -1], [1, 0, 1], [1, -1, -1]], dtype=np.float32)

    # linear sRGB / Rec.709 RGB <-> XYZ
    M_R2X = np.array(
        [[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]],
        dtype=np.float32,
    )
    M_X2R = np.linalg.inv(M_R2X).astype(np.float32)

    # Bradford CAT XYZ <-> LMS
    M_X2L = np.array([[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]], dtype=np.float32)
    M_L2X = np.linalg.inv(M_X2L).astype(np.float32)

    x0, y0 = opts.xy_base
    xyz_base = _xy_to_xyz_y1(x0, y0)
    lms_base = M_X2L @ xyz_base

    w = _build_weights_12(opts.w_shadow, opts.w_mid, opts.w_highlight, opts.bin_mid)
    t_base = _xy_to_cct_mccamy(x0, y0)
    tw = _interp_cct_mired(t_base, opts.t_warm_end, w)
    tc = _interp_cct_mired(t_base, opts.t_cool_end, w)

    d_warm = np.zeros((12, 3, 3), dtype=np.float32)
    d_cool = np.zeros((12, 3, 3), dtype=np.float32)
    eye = np.eye(3, dtype=np.float32)

    for i in range(12):
        xy_w = _cct_to_xy_kang2002(float(tw[i]))
        xy_c = _cct_to_xy_kang2002(float(tc[i]))
        xyz_w = _xy_to_xyz_y1(xy_w[0], xy_w[1])
        xyz_c = _xy_to_xyz_y1(xy_c[0], xy_c[1])

        lms_w = M_X2L @ xyz_w
        lms_c = M_X2L @ xyz_c

        r_w = lms_w / lms_base
        r_c = lms_c / lms_base

        A_w = M_L2X @ np.diag(r_w) @ M_X2L
        A_c = M_L2X @ np.diag(r_c) @ M_X2L

        # YCoCg-domain endpoint transforms:
        T_w = M_R2Y @ M_X2R @ A_w @ M_R2X @ M_Y2R
        T_c = M_R2Y @ M_X2R @ A_c @ M_R2X @ M_Y2R
        d_warm[i] = (T_w - eye).astype(np.float32)
        d_cool[i] = (T_c - eye).astype(np.float32)

    return KelvinTables12Bin(opts=opts, w=w, d_warm=d_warm, d_cool=d_cool)


def apply_kelvin_ycocg_linear_rgb255(
    rgb255_linear: Array,
    y8: Array,
    nodes: Array,
    *,
    side: str,
    alpha_global: float,
    strength: float,
    tbl: KelvinTables12Bin,
    kelvin_strength: float = 0.20,
    rgb_max: float = 255.0,
    bright_minfac: float = 0.25,
    y_dark2: float = 8.0,
    y_dark1: float = 16.0,
    y_bright1: float = 240.0,
    y_bright2: float = 248.0,
) -> Array:
    """Apply Kelvin/YCoCg transform to linear-light RGB in [0,255] float."""
    rgb = np.asarray(rgb255_linear, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb255_linear must have shape (H,W,3)")

    y, co, cg = rgb_to_ycocg(rgb)
    y0 = np.stack([y, co, cg], axis=0).reshape(3, -1)  # 3xN

    y_clamped = np.clip(np.asarray(y8, dtype=np.float32).reshape(-1), float(nodes[0]), float(nodes[-1]))
    nd = np.asarray(nodes, dtype=np.float32).reshape(-1)

    idx_hi = np.searchsorted(nd, y_clamped, side="left").astype(np.int32)
    idx_hi = np.clip(idx_hi, 0, nd.shape[0] - 1)
    idx_lo = np.clip(idx_hi - 1, 0, nd.shape[0] - 1)

    n_lo = nd[idx_lo]
    n_hi = nd[idx_hi]
    denom = np.where(n_hi > n_lo, n_hi - n_lo, 1.0).astype(np.float32)
    t = ((y_clamped - n_lo) / denom).astype(np.float32)
    t = np.where(idx_hi == idx_lo, 0.0, t).astype(np.float32)

    if side == "warm":
        d = tbl.d_warm
    else:
        d = tbl.d_cool

    w = tbl.w
    yv = y0[0, :]
    bfac = _brightness_factor_smooth(
        yv,
        y_dark2=y_dark2,
        y_dark1=y_dark1,
        y_bright1=y_bright1,
        y_bright2=y_bright2,
        minfac=bright_minfac,
    ).reshape(-1)

    out = y0.copy()
    high_start = float(nd[-4])  # nodes >= 223
    for k in range(12):
        m = idx_hi == k
        if not np.any(m):
            continue
        lo = idx_lo[m]
        tm = t[m]

        # Per-pixel interpolated delta matrix application:
        # delta = D_lo @ y0 + (D_hi-D_lo) @ y0 * t
        # (where hi==k, lo is either k or k-1)
        yk = y0[:, m]  # 3xM
        d_lo = d[lo]  # (M,3,3) via fancy indexing but M is per-bin chunk
        d_hi = d[k]  # (3,3)

        # Compute delta1 = D_lo @ yk and delta2 = D_hi @ yk
        delta_lo = np.einsum("mij,jm->im", d_lo, yk, optimize=True).astype(np.float32)
        delta_hi = (d_hi @ yk).astype(np.float32)
        delta = delta_lo + (delta_hi - delta_lo) * tm[None, :]

        # High-light safety: in very bright regions, use one-sided adjustments only.
        # This avoids hard falloff near 255 and prevents highlight tint spikes.
        hm = yk[0, :] >= high_start
        if np.any(hm):
            rgb_delta = ycocg_to_rgb(delta[0, :], delta[1, :], delta[2, :])  # (M,3)
            if side == "warm":
                # Warm highlights: primarily reduce B (optionally could reduce G, but keep it stable by default).
                rgb_delta[hm, 0] = 0.0
                rgb_delta[hm, 1] = 0.0
                rgb_delta[hm, 2] = np.minimum(rgb_delta[hm, 2], 0.0)
            else:
                # Cool highlights: primarily reduce R.
                rgb_delta[hm, 0] = np.minimum(rgb_delta[hm, 0], 0.0)
                rgb_delta[hm, 1] = 0.0
                rgb_delta[hm, 2] = 0.0
            dy, dco, dcg = rgb_to_ycocg(rgb_delta)
            delta[:, hm] = np.stack([dy[hm], dco[hm], dcg[hm]], axis=0).astype(np.float32)

        w_lo = w[lo].astype(np.float32)
        w_hi = float(w[k])
        wbin = w_lo + tm * (w_hi - w_lo)

        s = (float(alpha_global) * float(strength) * float(kelvin_strength)) * wbin
        s = s * bfac[m]

        s = _limit_strength_by_rgb_headroom(yk, delta, s, rgb_max)
        s = np.clip(s, 0.0, 1.0).astype(np.float32)
        out[:, m] = yk + delta * s[None, :]

    rgb_out = ycocg_to_rgb(out[0, :], out[1, :], out[2, :]).reshape(rgb.shape)
    return np.clip(rgb_out, 0.0, float(rgb_max)).astype(np.float32)
