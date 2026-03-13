"""
Fixed-Point WPA Core — main processing pipeline.

Entry point: ``wpa_fixed_process(img, cfg)``

Pipeline:
  1. Degamma (float, reuses wpa.gamma)
  2. Convert linear float → fixed-point int32
  3. Compute luma proxy Y (uint8 integer)
  4. Look up per-bin gains from offline WA LUT (fixed-point integer)
  5. Apply gain (integer multiply + shift)
  6. Saturation protection (integer, optional)
  7. Convert fixed-point → float linear
  8. Engamma (float, reuses wpa.gamma)
  9. Quantise to uint8
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

    ONE = cfg.ONE           # 1 << frac_bits
    HALF = cfg.HALF         # rounding bias
    COEFF_HALF = cfg.COEFF_HALF
    frac_bits = cfg.frac_bits
    coeff_frac_bits = cfg.coeff_frac_bits

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
    # Step 4: Look up per-bin gains from offline WA LUT (fixed-point)
    # ------------------------------------------------------------------
    wa = int(np.clip(cfg.wa_sel, 0, 127))
    gains_wa = cfg.runtime_bin_gains_fixed(wa)  # (12,3) int32 in UQ1.F
    gain = interpolate_gains_fixed(
        luma_u8, gains_wa, cfg.luma_nodes,
        interp=cfg.bin_interp, interp_bits=frac_bits,
    )
    # gain: (H, W, 3) int32 in coefficient Q-format (UQ1.F)

    # ------------------------------------------------------------------
    # Step 5: Apply gain (integer multiply + shift)
    #   pixel_fix is Q(frac_bits), gain is Q(coeff_frac_bits)
    #   result = (pixel_fix * gain + coeff_half) >> coeff_frac_bits
    #   => back to Q(frac_bits)
    # ------------------------------------------------------------------
    adjusted = (
        (pixel_fix.astype(np.int64) * gain.astype(np.int64) + COEFF_HALF)
        >> coeff_frac_bits
    ).astype(np.int32)

    # ------------------------------------------------------------------
    # Step 6: Saturation protection (integer, optional)
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
    # Step 7: Clip to valid fixed-point range [0, ONE]
    # ------------------------------------------------------------------
    adjusted = np.clip(adjusted, 0, ONE)

    # ------------------------------------------------------------------
    # Step 8: Convert fixed-point → float linear, then engamma (float)
    # ------------------------------------------------------------------
    linear_out = adjusted.astype(np.float32) / float(ONE)  # [0, 1]
    encoded = engamma(linear_out, cfg.gamma_mode, power=cfg.gamma_power)

    # ------------------------------------------------------------------
    # Step 9: Quantise to uint8
    # ------------------------------------------------------------------
    out = np.clip(np.round(encoded * 255.0), 0, 255).astype(np.uint8)
    return out
