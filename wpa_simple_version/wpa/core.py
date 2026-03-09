"""Core WPA processing pipeline."""

from __future__ import annotations

import numpy as np
import warnings

try:
    from .config import WPAConfig
    from .gamma import from_linear, to_linear
    from .kelvin_ycocg import KelvinTableOptions, apply_kelvin_ycocg_linear_rgb255, build_tables_kelvin_12bin
    from .weights import interpolate_bins, luma_proxy_y8, saturation_proxy, saturation_weight
except ImportError:
    from config import WPAConfig
    from gamma import from_linear, to_linear
    from kelvin_ycocg import KelvinTableOptions, apply_kelvin_ycocg_linear_rgb255, build_tables_kelvin_12bin
    from weights import interpolate_bins, luma_proxy_y8, saturation_proxy, saturation_weight

_HIGHLIGHT_SIDE_MIN_GAIN = np.float32(192.0 / 255.0)
_HIGHLIGHT_ROLLOFF = np.float32(0.15)


def _wa_sel_to_side_alpha(wa_sel: int) -> tuple[str, float]:
    """Map WA_SEL to DDIC-style side + interpolation alpha."""
    if wa_sel < 64:
        return "warm", (64 - wa_sel) / 64.0
    if wa_sel > 64:
        return "cool", (wa_sel - 64) / 63.0
    return "identity", 0.0


def _apply_gain_safety_and_strength(
    gain: np.ndarray,
    y8: np.ndarray,
    nodes: np.ndarray,
    *,
    side: str,
    strength: float,
) -> tuple[np.ndarray, bool]:
    """Apply one-sided high-light safety clamp and strength scaling."""
    out = np.asarray(gain, dtype=np.float32).copy()
    high_start = float(nodes[-4])  # nominal high-light boundary (223)
    transition_start = float(max(int(nodes[-5]), int(high_start) - 24))

    touched = False
    if transition_start < high_start:
        safe = np.minimum(out, 1.0)
        if side == "warm":
            safe[..., 2] = np.maximum(safe[..., 2], _HIGHLIGHT_SIDE_MIN_GAIN)
        elif side == "cool":
            safe[..., 0] = np.maximum(safe[..., 0], _HIGHLIGHT_SIDE_MIN_GAIN)

        # Blend into safety constraints before the high-light boundary, then fully clamp above it.
        t_safe = np.clip((y8 - transition_start) / (high_start - transition_start), 0.0, 1.0).astype(np.float32)
        before = out.copy()
        out = out + t_safe[..., None] * (safe - out)
        high_mask = y8 >= high_start
        if np.any(high_mask):
            out[high_mask] = safe[high_mask]
        touched = not np.allclose(before, out)

        # Keep a soft rolloff near peak white to avoid color spikes.
        t_roll = np.clip((y8 - high_start) / (255.0 - high_start), 0.0, 1.0).astype(np.float32)
        out = out + (t_roll[..., None] * _HIGHLIGHT_ROLLOFF) * (1.0 - out)

    if strength != 1.0:
        # Scale attenuation magnitude around identity: 1 - s*(1-gain).
        out = 1.0 - float(strength) * (1.0 - out)
        out = np.minimum(out, 1.0)
        if side == "warm":
            out[..., 2] = np.maximum(out[..., 2], _HIGHLIGHT_SIDE_MIN_GAIN)
        else:
            out[..., 0] = np.maximum(out[..., 0], _HIGHLIGHT_SIDE_MIN_GAIN)

    return out, touched


def wpa_process_rgb_uint8(img: np.ndarray, cfg: WPAConfig | None = None) -> np.ndarray:
    """Apply White Point Adjustment on uint8 RGB image.

    Args:
        img: Input image, shape (H, W, 3), dtype uint8, gamma-domain by default.
        cfg: WPAConfig. If None, default config is used.

    Returns:
        Output uint8 image of same shape.
    """

    if cfg is None:
        cfg = WPAConfig()

    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        raise TypeError("img must be uint8")
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("img must have shape (H, W, 3)")

    # Hardware semantic: bypass path when disabled or centered selector.
    if (not cfg.WA_EN) or cfg.WA_SEL == 64:
        return arr.copy()

    side, alpha = _wa_sel_to_side_alpha(cfg.WA_SEL)
    if side == "identity" or alpha <= 0.0:
        return arr.copy()

    in_gamma = arr.astype(np.float32) / 255.0
    in_linear = to_linear(
        in_gamma,
        mode=cfg.gamma_mode,
        use_lut=cfg.use_gamma_lut,
        power_gamma=cfg.power_gamma,
    )

    # 12-bin luma lookup: Y may be computed in gamma or linear domain.
    luma_src = in_gamma if cfg.luma_domain == "gamma" else in_linear
    y8 = luma_proxy_y8(luma_src)

    nodes = np.asarray(cfg.luma_nodes_12, dtype=np.float32)

    if cfg.wa_mode == "diag_rgb":
        warm_bins = np.asarray(cfg.warm_gains_bins, dtype=np.float32)
        cool_bins = np.asarray(cfg.cool_gains_bins, dtype=np.float32)

        warm_max = interpolate_bins(y8, nodes, warm_bins, interp=cfg.bin_interp)
        cool_max = interpolate_bins(y8, nodes, cool_bins, interp=cfg.bin_interp)

        if side == "warm":
            gain = 1.0 + alpha * (warm_max - 1.0)
            gain, touched = _apply_gain_safety_and_strength(
                gain,
                y8,
                nodes,
                side="warm",
                strength=cfg.warm_strength,
            )
            if touched:
                warnings.warn(
                    "Applied warm high-light safety clamp (nodes >=223) to keep gains <= 1.0.",
                    stacklevel=2,
                )
        else:
            gain = 1.0 + alpha * (cool_max - 1.0)
            gain, touched = _apply_gain_safety_and_strength(
                gain,
                y8,
                nodes,
                side="cool",
                strength=cfg.cool_strength,
            )
            if touched:
                warnings.warn(
                    "Applied cool high-light safety clamp (nodes >=223) to keep gains <= 1.0.",
                    stacklevel=2,
                )

        gained_linear = in_linear * gain
    else:
        # Kelvin/YCoCg model runs on linear-light RGB and limits strength by RGB headroom.
        opts = KelvinTableOptions(
            t_warm_end=float(cfg.kelvin_warm_end),
            t_cool_end=float(cfg.kelvin_cool_end),
            w_shadow=float(cfg.kelvin_w_shadow),
            w_mid=float(cfg.kelvin_w_mid),
            w_highlight=float(cfg.kelvin_w_highlight),
            bin_mid=int(cfg.kelvin_bin_mid),
        )
        tbl = build_tables_kelvin_12bin(opts)
        side_strength = float(cfg.warm_strength if side == "warm" else cfg.cool_strength)
        side_scale = float(cfg.kelvin_warm_side_scale if side == "warm" else cfg.kelvin_cool_side_scale)
        strength = side_strength * side_scale
        rgb255 = (in_linear * 255.0).astype(np.float32)
        out255 = apply_kelvin_ycocg_linear_rgb255(
            rgb255,
            y8,
            nodes,
            side=side,
            alpha_global=float(alpha),
            strength=strength,
            tbl=tbl,
            kelvin_strength=float(cfg.kelvin_strength),
            rgb_max=255.0,
            bright_minfac=float(cfg.kelvin_bright_minfac),
            y_dark2=float(cfg.kelvin_y_dark2),
            y_dark1=float(cfg.kelvin_y_dark1),
            y_bright1=float(cfg.kelvin_y_bright1),
            y_bright2=float(cfg.kelvin_y_bright2),
        )
        gained_linear = (out255 / 255.0).astype(np.float32)

    # Low-saturation protection w: blend toward original for high saturation colors.
    sat_src = in_gamma if cfg.sat_weight_domain == "gamma" else in_linear
    s = saturation_proxy(sat_src)
    w = saturation_weight(s, cfg.sat_s0, cfg.sat_s1)

    out_linear = in_linear + w[..., None] * (gained_linear - in_linear)
    out_linear = np.clip(out_linear, 0.0, 1.0)

    out_gamma = from_linear(
        out_linear,
        mode=cfg.gamma_mode,
        use_lut=cfg.use_gamma_lut,
        power_gamma=cfg.power_gamma,
    )
    out_u8 = np.clip(np.rint(out_gamma * 255.0), 0.0, 255.0).astype(np.uint8)
    return out_u8
