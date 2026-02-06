from __future__ import annotations
from dataclasses import dataclass
import numpy as np

Array = np.ndarray

def srgb_to_linear(srgb: Array) -> Array:
    """
    Convert sRGB (gamma-compressed) to linear RGB.

    Parameters
    ----------
    srgb : ndarray
        Float array in [0, 1].

    Returns
    -------
    lin : ndarray
        Float array in [0, 1].
    """
    srgb = np.asarray(srgb, dtype=np.float64)
    a = 0.055
    out = np.empty_like(srgb, dtype=np.float64)
    m = srgb <= 0.04045
    out[m] = srgb[m] / 12.92
    out[~m] = ((srgb[~m] + a) / (1 + a)) ** 2.4
    return out

def linear_to_srgb(lin: Array) -> Array:
    """
    Convert linear RGB to sRGB (gamma-compressed).

    Parameters
    ----------
    lin : ndarray
        Float array in [0, 1].

    Returns
    -------
    srgb : ndarray
        Float array in [0, 1].
    """
    lin = np.asarray(lin, dtype=np.float64)
    a = 0.055
    out = np.empty_like(lin, dtype=np.float64)
    m = lin <= 0.0031308
    out[m] = 12.92 * lin[m]
    out[~m] = (1 + a) * (lin[~m] ** (1 / 2.4)) - a
    return out

def rgb_to_ycocg(rgb_255: Array) -> tuple[Array, Array, Array]:
    """
    RGB -> YCoCg (same definition as your MATLAB rgb2ycocg.m).

    Parameters
    ----------
    rgb_255 : ndarray
        HxWx3 array, expected RGB in [0,255] (uint8 or float).

    Returns
    -------
    Y, Co, Cg : ndarray
        Each HxW float64.
    """
    rgb = np.asarray(rgb_255, dtype=np.float64)
    R = rgb[..., 0]
    G = rgb[..., 1]
    B = rgb[..., 2]
    Y  = 0.25 * R + 0.50 * G + 0.25 * B
    Co = 0.50 * (R - B)
    Cg = -0.25 * R + 0.50 * G - 0.25 * B
    return Y, Co, Cg

def ycocg_to_rgb(Y: Array, Co: Array, Cg: Array) -> Array:
    """
    YCoCg -> RGB (inverse of rgb_to_ycocg), matches your MATLAB ycocg2rgb.m.

    Returns float RGB (not clipped).
    """
    Y = np.asarray(Y, dtype=np.float64)
    Co = np.asarray(Co, dtype=np.float64)
    Cg = np.asarray(Cg, dtype=np.float64)
    R = Y + Co - Cg
    G = Y + Cg
    B = Y - Co - Cg
    return np.stack([R, G, B], axis=-1)

def clip_u8(rgb_255: Array) -> Array:
    """Clip to [0,255] and cast to uint8."""
    x = np.clip(rgb_255, 0.0, 255.0)
    return np.rint(x).astype(np.uint8)
