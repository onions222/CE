"""
Fixed-Point WPA Core — main processing pipeline.

Entry point: ``wpa_fixed_process(img, cfg)``

Pipeline:
  1. Degamma (float, reuses wpa.gamma)
  2. Convert linear float → fixed-point int32
  3. Compute luma proxy Y (uint8 integer)
  4. Look up per-bin gains (fixed-point integer)
  5. Blend gains with identity via WA_SEL (integer)
  6. Apply gain (integer multiply + shift)
  7. Saturation protection (integer, optional)
  8. Convert fixed-point → float linear
  9. Engamma (float, reuses wpa.gamma)
  10. Quantise to uint8
"""

from __future__ import annotations

import numpy as np

from wpa.gamma import degamma, engamma             # float — allowed per user spec
from .config import FixedWPAConfig
from .weights import (
    compute_luma_proxy_u8,
    compute_sat_weight_fixed,
    interpolate_gains_fixed,
)


def _wa_sel_to_alpha_fixed(wa_sel: int) -> tuple[str, int, int]:
    """Convert WA_SEL to integer alpha and its denominator shift.

    Returns (side, alpha_num, alpha_den_shift) where:
        effective_alpha = alpha_num / (1 << alpha_den_shift)

    DDIC mapping (natural 6-bit precision):
        WA_SEL < 64  → warm,  alpha_num = 64 - WA_SEL,  den_shift = 6
        WA_SEL = 64  → identity
        WA_SEL > 64  → cool,  alpha_num = WA_SEL - 64,  den_shift = 6
                       (denominator is 63, but we use 64 for power-of-2 shift;
                        max alpha = 63/64 ≈ 0.984 — close enough for HW)
    """
    if wa_sel < 64:
        return "warm", 64 - wa_sel, 6
    elif wa_sel == 64:
        return "identity", 0, 6
    else:
        return "cool", wa_sel - 64, 6


def wpa_fixed_process(
    img: np.ndarray,
    cfg: FixedWPAConfig | None = None,
) -> np.ndarray:
    """Apply White Point Adjustment using fixed-point arithmetic.

    Degamma and engamma use float (allowed by spec).
    All gain computation is pure integer with configurable ``frac_bits``.

    Parameters
    ----------
    img : (H, W, 3) uint8 — input image (sRGB gamma-encoded).
    cfg : FixedWPAConfig instance (uses defaults if None).

    Returns
    -------
    (H, W, 3) uint8 — adjusted image.
    """
    if cfg is None:
        cfg = FixedWPAConfig()

    if not cfg.wa_en or cfg.wa_sel == 64:
        return img.copy()

    assert img.dtype == np.uint8 and img.ndim == 3 and img.shape[2] == 3

    side, alpha_num, alpha_shift = _wa_sel_to_alpha_fixed(cfg.wa_sel)
    if side == "identity":
        return img.copy()

    ONE = cfg.ONE           # 1 << frac_bits
    HALF = cfg.HALF         # rounding bias
    frac_bits = cfg.frac_bits

    # ------------------------------------------------------------------
    # Step 1: Degamma (float — allowed by spec)
    # ------------------------------------------------------------------
    linear_f = degamma(img, cfg.gamma_mode, power=cfg.gamma_power,
                       use_lut=False, is_uint8=True)  # float32 [0,1]

    # ------------------------------------------------------------------
    # Step 2: Float linear → fixed-point int32
    #   pixel_fix = round(linear_f * ONE)
    # ------------------------------------------------------------------
    pixel_fix = np.clip(
        np.round(linear_f * ONE).astype(np.int32), 0, ONE
    )  # (H, W, 3) int32 in [0, ONE]

    # ------------------------------------------------------------------
    # Step 3: Luma proxy (uint8 integer arithmetic)
    # ------------------------------------------------------------------
    if cfg.luma_domain == "gamma":
        luma_u8 = compute_luma_proxy_u8(img)          # (H, W) uint8
    else:
        # Linear domain luma: compute from fixed-point, scale to [0, 255]
        r_fix = pixel_fix[..., 0]
        g_fix = pixel_fix[..., 1]
        b_fix = pixel_fix[..., 2]
        luma_fix = (r_fix + 2 * g_fix + b_fix) >> 2   # [0, ONE]
        # Scale to [0, 255]: luma_u8 = (luma_fix * 255 + HALF) >> frac_bits
        luma_u8 = np.clip(
            (luma_fix * 255 + HALF) >> frac_bits, 0, 255
        ).astype(np.uint8)

    # ------------------------------------------------------------------
    # Step 4: Look up per-bin gains (fixed-point integer interpolation)
    # ------------------------------------------------------------------
    if side == "warm":
        gains_max = interpolate_gains_fixed(
            luma_u8, cfg.warm_gains_fixed, cfg.luma_nodes,
            interp=cfg.bin_interp, interp_bits=frac_bits,
        )
    else:
        gains_max = interpolate_gains_fixed(
            luma_u8, cfg.cool_gains_fixed, cfg.luma_nodes,
            interp=cfg.bin_interp, interp_bits=frac_bits,
        )
    # gains_max: (H, W, 3) int32, representing gain in Q-format

    # ------------------------------------------------------------------
    # Step 5: Blend with identity via WA_SEL (integer)
    #   gain = ONE + (alpha_num * (gains_max - ONE) + (1<<(alpha_shift-1)))
    #              >> alpha_shift
    # ------------------------------------------------------------------
    alpha_half = 1 << (alpha_shift - 1)
    gain = ONE + (
        (alpha_num * (gains_max.astype(np.int64) - ONE) + alpha_half)
        >> alpha_shift
    ).astype(np.int32)
    # gain: (H, W, 3) int32

    # ------------------------------------------------------------------
    # Step 6: Apply gain (integer multiply + shift)
    #   result = (pixel_fix * gain + HALF) >> frac_bits
    # ------------------------------------------------------------------
    adjusted = (
        (pixel_fix.astype(np.int64) * gain.astype(np.int64) + HALF)
        >> frac_bits
    ).astype(np.int32)

    # ------------------------------------------------------------------
    # Step 7: Saturation protection (integer, optional)
    # ------------------------------------------------------------------
    if cfg.sat_en:
        w = compute_sat_weight_fixed(img, cfg.sat_s0, cfg.sat_s1, frac_bits)
        w_3d = w[..., np.newaxis]  # (H, W, 1)
        # out = pixel_fix + (w * (adjusted - pixel_fix) + HALF) >> frac_bits
        delta = adjusted - pixel_fix
        adjusted = pixel_fix + (
            (w_3d.astype(np.int64) * delta.astype(np.int64) + HALF)
            >> frac_bits
        ).astype(np.int32)

    # ------------------------------------------------------------------
    # Step 8: Clip to valid fixed-point range [0, ONE]
    # ------------------------------------------------------------------
    adjusted = np.clip(adjusted, 0, ONE)

    # ------------------------------------------------------------------
    # Step 9: Convert fixed-point → float linear, then engamma (float)
    # ------------------------------------------------------------------
    linear_out = adjusted.astype(np.float32) / float(ONE)  # [0, 1]
    encoded = engamma(linear_out, cfg.gamma_mode, power=cfg.gamma_power)

    # ------------------------------------------------------------------
    # Step 10: Quantise to uint8
    # ------------------------------------------------------------------
    out = np.clip(np.round(encoded * 255.0), 0, 255).astype(np.uint8)
    return out
