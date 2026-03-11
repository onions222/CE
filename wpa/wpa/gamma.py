"""
Gamma transfer functions for WPA.

Supports three modes controlled by ``WPAConfig.gamma_mode``:
- ``"srgb"``  — standard sRGB piecewise formula (IEC 61966-2-1)
- ``"power"`` — simple power-law  γ / 1/γ
- ``"none"``  — passthrough (data already linear or no conversion desired)

Each mode provides *degamma* (encoded → linear) and *engamma* (linear → encoded).

An optional 256-entry LUT path (``cfg.use_gamma_lut=True``) is provided for the
sRGB case to trade accuracy for speed on large images.
"""

from __future__ import annotations

import numpy as np

# ========================================================================
# sRGB formula (exact, vectorised)
# ========================================================================

def srgb_degamma(x: np.ndarray) -> np.ndarray:
    """sRGB encoded [0,1] → linear [0,1].

    IEC 61966-2-1:
        linear = x / 12.92                   if x <= 0.04045
        linear = ((x + 0.055) / 1.055)^2.4   otherwise
    """
    return np.where(
        x <= 0.04045,
        x / 12.92,
        np.power((x + 0.055) / 1.055, 2.4),
    )


def srgb_engamma(x: np.ndarray) -> np.ndarray:
    """Linear [0,1] → sRGB encoded [0,1].

    IEC 61966-2-1:
        encoded = 12.92 * x                          if x <= 0.0031308
        encoded = 1.055 * x^(1/2.4) - 0.055          otherwise
    """
    return np.where(
        x <= 0.0031308,
        12.92 * x,
        1.055 * np.power(np.maximum(x, 0.0), 1.0 / 2.4) - 0.055,
    )


# ========================================================================
# sRGB LUT (256-entry, built once then cached)
# ========================================================================

_SRGB_DEGAMMA_LUT: np.ndarray | None = None
_SRGB_ENGAMMA_LUT: np.ndarray | None = None


def _build_srgb_degamma_lut() -> np.ndarray:
    """Build a 256-entry degamma LUT: code 0..255 → linear float32."""
    global _SRGB_DEGAMMA_LUT
    if _SRGB_DEGAMMA_LUT is None:
        codes = np.arange(256, dtype=np.float64) / 255.0
        _SRGB_DEGAMMA_LUT = srgb_degamma(codes).astype(np.float32)
    return _SRGB_DEGAMMA_LUT


def _build_srgb_engamma_lut() -> np.ndarray:
    """Build a 4096-entry engamma LUT: linear index → 8-bit encoded uint8.

    We quantise [0,1] into 4096 steps for reasonable precision.
    """
    global _SRGB_ENGAMMA_LUT
    if _SRGB_ENGAMMA_LUT is None:
        lin = np.linspace(0.0, 1.0, 4096, dtype=np.float64)
        enc = srgb_engamma(lin)
        _SRGB_ENGAMMA_LUT = np.clip(np.round(enc * 255.0), 0, 255).astype(np.uint8)
    return _SRGB_ENGAMMA_LUT


def srgb_degamma_lut(img_uint8: np.ndarray) -> np.ndarray:
    """Degamma using pre-built 256-entry LUT.  uint8 → float32 linear."""
    lut = _build_srgb_degamma_lut()
    return lut[img_uint8]  # fancy indexing


def srgb_engamma_lut(linear: np.ndarray) -> np.ndarray:
    """Engamma using pre-built 4096-entry LUT.  float linear → uint8.

    The caller still needs to convert types if needed; we return uint8 here.
    """
    lut = _build_srgb_engamma_lut()
    idx = np.clip(np.round(linear * 4095.0).astype(np.int32), 0, 4095)
    return lut[idx]


# ========================================================================
# Power-law gamma
# ========================================================================

def power_degamma(x: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    """Simple power-law degamma:  linear = x^gamma."""
    return np.power(np.clip(x, 0.0, 1.0), gamma)


def power_engamma(x: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    """Simple power-law engamma:  encoded = x^(1/gamma)."""
    return np.power(np.clip(x, 0.0, 1.0), 1.0 / gamma)


# ========================================================================
# Dispatcher — called by core.py
# ========================================================================

def degamma(img: np.ndarray, mode: str, *,
            power: float = 2.2,
            use_lut: bool = False,
            is_uint8: bool = False) -> np.ndarray:
    """Convert from gamma-encoded domain to linear domain.

    Parameters
    ----------
    img      : input image, float32 [0,1] or uint8 [0,255].
    mode     : ``"srgb"`` | ``"power"`` | ``"none"``.
    power    : exponent for ``"power"`` mode.
    use_lut  : use LUT path for ``"srgb"`` mode (requires uint8 input).
    is_uint8 : hint that *img* is uint8 (enables LUT fast-path).
    """
    if mode == "none":
        if is_uint8:
            return img.astype(np.float32) / 255.0
        return img.astype(np.float32)

    if mode == "srgb":
        if use_lut and is_uint8:
            return srgb_degamma_lut(img)  # uint8 in, float32 out
        # formula path: normalise to [0,1] first
        x = img.astype(np.float32) / 255.0 if is_uint8 else img.astype(np.float32)
        return srgb_degamma(x).astype(np.float32)

    if mode == "power":
        x = img.astype(np.float32) / 255.0 if is_uint8 else img.astype(np.float32)
        return power_degamma(x, power).astype(np.float32)

    raise ValueError(f"Unknown gamma mode: {mode!r}")


def engamma(img_linear: np.ndarray, mode: str, *,
            power: float = 2.2,
            use_lut: bool = False) -> np.ndarray:
    """Convert from linear domain to gamma-encoded [0,1] float or uint8.

    Parameters
    ----------
    img_linear : float32, [0,1] linear-light image.
    mode       : ``"srgb"`` | ``"power"`` | ``"none"``.
    power      : exponent for ``"power"`` mode.
    use_lut    : use LUT path for ``"srgb"`` mode (returns uint8 directly).

    Returns
    -------
    np.ndarray : float32 [0,1] encoded (or uint8 if LUT path).
    """
    if mode == "none":
        return img_linear

    if mode == "srgb":
        if use_lut:
            return srgb_engamma_lut(img_linear)  # returns uint8
        return srgb_engamma(img_linear).astype(np.float32)

    if mode == "power":
        return power_engamma(img_linear, power).astype(np.float32)

    raise ValueError(f"Unknown gamma mode: {mode!r}")
