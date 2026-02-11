"""
WPA — White Point Adjustment for mobile display.

Per-channel RGB gain with 12-bin luma-segmented gains, saturation
protection, and configurable gamma handling.

Quick start::

    import numpy as np
    from wpa import wpa_process_rgb_uint8, WPAConfig

    img = ...  # (H, W, 3) uint8 sRGB
    cfg = WPAConfig(wa_sel=20)  # warm
    out = wpa_process_rgb_uint8(img, cfg)
"""

from .config import WPAConfig
from .core import wpa_process_rgb_uint8

__all__ = ["WPAConfig", "wpa_process_rgb_uint8"]
