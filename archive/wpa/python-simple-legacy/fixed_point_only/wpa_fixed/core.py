from __future__ import annotations

import numpy as np

from .config import FixedWPAConfig


def _round_shift(x: np.ndarray | np.int64, shift: int) -> np.ndarray:
    half = 1 << (shift - 1)
    return (x + half) >> shift


def _wa_sel_to_side_alpha_q(wa_sel: int, qone: int) -> tuple[str, int]:
    if wa_sel < 64:
        return "warm", ((64 - wa_sel) * qone + 32) // 64
    if wa_sel > 64:
        return "cool", ((wa_sel - 64) * qone + 31) // 63
    return "identity", 0


def _luma_proxy_y8(rgb_u8: np.ndarray) -> np.ndarray:
    r = rgb_u8[..., 0].astype(np.int32)
    g = rgb_u8[..., 1].astype(np.int32)
    b = rgb_u8[..., 2].astype(np.int32)
    return np.clip((r + (g << 1) + b + 2) >> 2, 0, 255).astype(np.int32)


def _sat_proxy_255(rgb_u8: np.ndarray) -> np.ndarray:
    r = rgb_u8[..., 0].astype(np.int32)
    g = rgb_u8[..., 1].astype(np.int32)
    b = rgb_u8[..., 2].astype(np.int32)
    return (np.abs(r - g) + np.abs(g - b) + np.abs(b - r)).astype(np.int32)


def _sat_weight_q(s255: np.ndarray, s0_255: int, s1_255: int, q_bits: int, qone: int) -> np.ndarray:
    out = np.empty_like(s255, dtype=np.int32)
    out[s255 <= s0_255] = qone
    out[s255 >= s1_255] = 0
    mid = (s255 > s0_255) & (s255 < s1_255)
    num = (s1_255 - s255[mid]).astype(np.int64) << q_bits
    out[mid] = (num // (s1_255 - s0_255)).astype(np.int32)
    return np.clip(out, 0, qone)


def _interpolate_bins_q(y8: np.ndarray, nodes: np.ndarray, bins_q: np.ndarray, interp: bool, q_bits: int) -> np.ndarray:
    y = np.clip(y8.astype(np.int32), int(nodes[0]), int(nodes[-1]))
    if not interp:
        idx = np.searchsorted(nodes, y, side="right") - 1
        idx = np.clip(idx, 0, nodes.shape[0] - 1)
        return bins_q[idx]

    idx_hi = np.searchsorted(nodes, y, side="left")
    idx_hi = np.clip(idx_hi, 0, nodes.shape[0] - 1)
    idx_lo = np.clip(idx_hi - 1, 0, nodes.shape[0] - 1)

    n_lo = nodes[idx_lo]
    n_hi = nodes[idx_hi]
    den = np.where(n_hi > n_lo, n_hi - n_lo, 1).astype(np.int32)
    t_q = (((y - n_lo).astype(np.int64) << q_bits) // den.astype(np.int64)).astype(np.int32)
    t_q = np.where(idx_hi == idx_lo, 0, t_q).astype(np.int32)

    v_lo = bins_q[idx_lo].astype(np.int32)
    v_hi = bins_q[idx_hi].astype(np.int32)
    dv = (v_hi - v_lo).astype(np.int64)
    return (v_lo + _round_shift(t_q[..., None].astype(np.int64) * dv, q_bits)).astype(np.int32)


def _apply_gain_safety_and_strength(
    gain_q: np.ndarray,
    y8: np.ndarray,
    nodes: np.ndarray,
    side: str,
    strength_q: int,
    q_bits: int,
    qone: int,
) -> np.ndarray:
    highlight_side_min_q = (192 * qone + 127) // 255
    highlight_rolloff_q = int(round(0.15 * qone))

    out = gain_q.astype(np.int32).copy()
    high_start = int(nodes[-4])  # nominal high-light boundary (223)
    transition_start = max(int(nodes[-5]), high_start - 24)

    safe = np.minimum(out, qone)
    if side == "warm":
        safe[..., 2] = np.maximum(safe[..., 2], highlight_side_min_q)
    else:
        safe[..., 0] = np.maximum(safe[..., 0], highlight_side_min_q)

    if transition_start < high_start:
        t_safe = (((y8.astype(np.int64) - transition_start) << q_bits) // (high_start - transition_start)).astype(np.int32)
        t_safe = np.clip(t_safe, 0, qone)
        out = (
            out.astype(np.int64)
            + _round_shift(t_safe[..., None].astype(np.int64) * (safe.astype(np.int64) - out.astype(np.int64)), q_bits)
        ).astype(np.int32)
        high_mask = y8 >= high_start
        if np.any(high_mask):
            out[high_mask] = safe[high_mask]

    high_mask = y8 >= high_start
    if np.any(high_mask):
        t = (((y8[high_mask].astype(np.int64) - high_start) << q_bits) // (255 - high_start)).astype(np.int32)
        t = np.clip(t, 0, qone)
        roll = _round_shift(t.astype(np.int64) * highlight_rolloff_q, q_bits).astype(np.int32)
        tmp = out[high_mask].astype(np.int64)
        tmp = tmp + _round_shift(roll[:, None].astype(np.int64) * (qone - tmp), q_bits)
        out[high_mask] = np.clip(tmp, 0, qone).astype(np.int32)

    if strength_q != qone:
        tmp = out.astype(np.int64)
        out = (qone - _round_shift(strength_q * (qone - tmp), q_bits)).astype(np.int32)
        out = np.minimum(out, qone)
        if side == "warm":
            out[..., 2] = np.maximum(out[..., 2], highlight_side_min_q)
        else:
            out[..., 0] = np.maximum(out[..., 0], highlight_side_min_q)
    return out


def wpa_process_rgb_uint8_fixed(img: np.ndarray, cfg: FixedWPAConfig | None = None) -> np.ndarray:
    if cfg is None:
        cfg = FixedWPAConfig()

    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        raise TypeError("img must be uint8")
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("img must have shape (H,W,3)")

    if (not cfg.WA_EN) or cfg.WA_SEL == 64:
        return arr.copy()

    q_bits = cfg.q_bits
    qone = cfg.qone
    side, alpha_q = _wa_sel_to_side_alpha_q(cfg.WA_SEL, qone)
    if side == "identity" or alpha_q <= 0:
        return arr.copy()

    in_u8 = arr.astype(np.int32)
    y8 = _luma_proxy_y8(in_u8)
    nodes = np.asarray(cfg.luma_nodes_12, dtype=np.int32)

    warm_max = _interpolate_bins_q(y8, nodes, cfg.warm_bins_q, cfg.bin_interp, q_bits)
    cool_max = _interpolate_bins_q(y8, nodes, cfg.cool_bins_q, cfg.bin_interp, q_bits)
    base = warm_max if side == "warm" else cool_max
    strength_q = cfg.warm_strength_q if side == "warm" else cfg.cool_strength_q

    gain_q = (qone + _round_shift(alpha_q * (base.astype(np.int64) - qone), q_bits)).astype(np.int32)
    gain_q = _apply_gain_safety_and_strength(gain_q, y8, nodes, side, strength_q, q_bits, qone)

    gained = _round_shift(in_u8.astype(np.int64) * gain_q.astype(np.int64), q_bits).astype(np.int32)
    gained = np.clip(gained, 0, 255)

    s255 = _sat_proxy_255(in_u8)
    w_q = _sat_weight_q(s255, cfg.sat_s0_255, cfg.sat_s1_255, q_bits, qone)

    delta = gained.astype(np.int64) - in_u8.astype(np.int64)
    out = in_u8.astype(np.int64) + _round_shift(w_q[..., None].astype(np.int64) * delta, q_bits)
    return np.clip(out, 0, 255).astype(np.uint8)
