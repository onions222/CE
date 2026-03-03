"""
WPA Core — main processing pipeline.

Entry point: ``wpa_process_rgb_uint8(img, cfg)``

Pipeline (per-pixel):
  1. Degamma (sRGB → linear)
  2. Compute luma proxy Y
  3. Look up per-bin warm/cool gains via 12-bin table
  4. Blend gains with identity based on WA_SEL
  5. Apply saturation-protection weight
  6. Apply gain in linear domain
  7. Engamma (linear → sRGB)
  8. Clip & quantise to uint8
"""

from __future__ import annotations

import numpy as np

from .config import WPAConfig, map_sat_threshold_gamma_to_linear
from .gamma import degamma, engamma
from .weights import compute_luma_proxy, compute_sat_weight, interpolate_gains_12bin


def _wa_sel_to_alpha(wa_sel: int) -> tuple[str, float]:
    """Convert WA_SEL (0..127) to interpolation side and alpha.

    DDIC/WPA mapping:
        WA_SEL < 64  →  side="warm",  a = (64 - WA_SEL) / 64
        WA_SEL = 64  →  identity (a = 0)
        WA_SEL > 64  →  side="cool",  a = (WA_SEL - 64) / 63

    Returns
    -------
    (side, alpha) where side ∈ {"warm", "cool", "identity"}, alpha ∈ [0, 1].
    """
    # The warm/cool denominator asymmetry is intentional:
    # warm side has 64 codes (0..63), cool side has 63 codes (65..127),
    # with 64 reserved as neutral identity.
    if wa_sel < 64:
        return "warm", (64 - wa_sel) / 64.0
    elif wa_sel == 64:
        return "identity", 0.0
    else:
        return "cool", (wa_sel - 64) / 63.0


def wpa_process_rgb_uint8(
    img: np.ndarray,
    cfg: WPAConfig | None = None,
) -> np.ndarray:
    """Apply White Point Adjustment to an 8-bit RGB image.

    Parameters
    ----------
    img : (H, W, 3) uint8  — input image, assumed sRGB gamma-encoded.
    cfg : WPAConfig instance (uses defaults if None).

    Returns
    -------
    (H, W, 3) uint8  — adjusted image.
    """
    if cfg is None:
        cfg = WPAConfig()

    # --- Bypass if disabled or identity ---
    if not cfg.wa_en or cfg.wa_sel == 64:
        return img.copy()

    assert img.dtype == np.uint8 and img.ndim == 3 and img.shape[2] == 3, \
        "Input must be (H, W, 3) uint8"

    side, alpha = _wa_sel_to_alpha(cfg.wa_sel)
    if side == "identity":
        return img.copy()

    # ------------------------------------------------------------------
    # Step 1: Degamma — gamma domain → linear domain
    # ------------------------------------------------------------------
    linear = degamma(
        img, cfg.gamma_mode,
        power=cfg.gamma_power,
        use_lut=cfg.use_gamma_lut,
        is_uint8=True,
    )  # (H, W, 3) float32, [0, 1]

    # Also keep a gamma-domain float image for optional weight/luma calc
    gamma_f32 = img.astype(np.float32) / 255.0  # [0, 1]

    # ------------------------------------------------------------------
    # Step 2: Compute luma proxy Y
    # ------------------------------------------------------------------
    if cfg.luma_domain == "linear":
        luma = compute_luma_proxy(linear)        # [0, 1]
    else:
        luma = compute_luma_proxy(gamma_f32)     # [0, 1]

    # ------------------------------------------------------------------
    # Step 3: Lookup per-bin gains via 12-bin table + interpolation
    # ------------------------------------------------------------------
    if side == "warm":
        gains_max = interpolate_gains_12bin(
            luma, cfg.warm_gains_bins, cfg.luma_nodes, interp=cfg.bin_interp
        )  # (H, W, 3) — warm_max gains at each pixel
    else:
        gains_max = interpolate_gains_12bin(
            luma, cfg.cool_gains_bins, cfg.luma_nodes, interp=cfg.bin_interp
        )  # (H, W, 3) — cool_max gains at each pixel

    # ------------------------------------------------------------------
    # Step 4: Blend gains with identity (1,1,1) using alpha from WA_SEL
    #   gain = (1 - alpha) * 1 + alpha * gains_max
    # ------------------------------------------------------------------
    gain = 1.0 + alpha * (gains_max - 1.0)  # (H, W, 3)

    # ------------------------------------------------------------------
    # Step 5: Apply gain in linear domain
    # ------------------------------------------------------------------
    adjusted = linear * gain  # (H, W, 3)

    # ------------------------------------------------------------------
    # Step 6: Saturation protection — blend back toward input for
    #         saturated pixels to avoid hue shifts.
    #   out = in + w * (gain(in) - in)
    # ------------------------------------------------------------------
    if cfg.sat_en:
        if cfg.sat_weight_domain == "linear":
            # Map gamma-domain thresholds (0..510 scale) into linear-domain
            # saturation metric scale [0,2]. This preserves threshold intent
            # much better than naive division by 255.
            s0_linear = map_sat_threshold_gamma_to_linear(cfg.sat_s0)
            s1_linear = map_sat_threshold_gamma_to_linear(cfg.sat_s1)
            w = compute_sat_weight(
                linear, s0=s0_linear, s1=s1_linear,
            )
        else:
            # Gamma-domain weight — thresholds in 0-255 scale
            w = compute_sat_weight(
                gamma_f32 * 255.0, s0=cfg.sat_s0, s1=cfg.sat_s1,
            )
        w = w[..., np.newaxis]  # (H, W, 1)
        adjusted = linear + w * (adjusted - linear)

    # ------------------------------------------------------------------
    # Step 7: Clip to valid linear range
    # ------------------------------------------------------------------
    adjusted = np.clip(adjusted, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Step 8: Engamma — linear → gamma domain
    # ------------------------------------------------------------------
    if cfg.use_gamma_lut and cfg.gamma_mode == "srgb":
        # LUT path returns uint8 directly
        out_uint8 = engamma(adjusted, cfg.gamma_mode, use_lut=True)
        return out_uint8

    encoded = engamma(adjusted, cfg.gamma_mode, power=cfg.gamma_power)
    # encoded is float32 [0,1]; convert to uint8
    out_uint8 = np.clip(np.round(encoded * 255.0), 0, 255).astype(np.uint8)
    return out_uint8
