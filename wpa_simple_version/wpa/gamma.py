"""Gamma-domain conversions for WPA."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

import numpy as np


def srgb_to_linear_formula(x: np.ndarray) -> np.ndarray:
    """Convert sRGB gamma-coded values in [0,1] to linear light."""
    x = np.asarray(x, dtype=np.float32)
    return np.where(
        x <= 0.04045,
        x / 12.92,
        ((x + 0.055) / 1.055) ** 2.4,
    ).astype(np.float32)


def linear_to_srgb_formula(x: np.ndarray) -> np.ndarray:
    """Convert linear-light values in [0,1] to sRGB gamma-coded."""
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, 0.0, 1.0)
    return np.where(
        x <= 0.0031308,
        x * 12.92,
        1.055 * (x ** (1.0 / 2.4)) - 0.055,
    ).astype(np.float32)


@lru_cache(maxsize=1)
def _srgb_luts() -> tuple[np.ndarray, np.ndarray]:
    """Build 256-entry LUTs for sRGB <-> linear conversions."""
    x = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    degamma = srgb_to_linear_formula(x)
    engamma = linear_to_srgb_formula(x)
    return degamma.astype(np.float32), engamma.astype(np.float32)


def srgb_to_linear_lut(x: np.ndarray) -> np.ndarray:
    """LUT-based sRGB->linear using nearest 8-bit code value."""
    lut, _ = _srgb_luts()
    idx = np.clip(np.rint(np.asarray(x, dtype=np.float32) * 255.0), 0, 255).astype(np.int32)
    return lut[idx]


def linear_to_srgb_lut(x: np.ndarray) -> np.ndarray:
    """LUT-based linear->sRGB using nearest 8-bit linear sample."""
    _, lut = _srgb_luts()
    idx = np.clip(np.rint(np.asarray(x, dtype=np.float32) * 255.0), 0, 255).astype(np.int32)
    return lut[idx]


def to_linear(
    x: np.ndarray,
    *,
    mode: Literal["srgb", "power", "none"],
    use_lut: bool,
    power_gamma: float,
) -> np.ndarray:
    """Convert gamma-domain normalized RGB to linear RGB."""
    x = np.asarray(x, dtype=np.float32)
    if mode == "none":
        return x
    if mode == "power":
        x = np.clip(x, 0.0, 1.0)
        return np.power(x, power_gamma, dtype=np.float32)

    if use_lut:
        return srgb_to_linear_lut(x)
    return srgb_to_linear_formula(x)


def from_linear(
    x: np.ndarray,
    *,
    mode: Literal["srgb", "power", "none"],
    use_lut: bool,
    power_gamma: float,
) -> np.ndarray:
    """Convert linear RGB to gamma-domain normalized RGB."""
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, 0.0, 1.0)

    if mode == "none":
        return x
    if mode == "power":
        return np.power(x, 1.0 / power_gamma, dtype=np.float32)

    if use_lut:
        return linear_to_srgb_lut(x)
    return linear_to_srgb_formula(x)
