"""
WPA Fixed-Point — integer-arithmetic White Point Adjustment.

Degamma/engamma use float; all gain computation is pure integer
with configurable precision (``frac_bits``).

Quick start::

    from wpa_fixed import wpa_fixed_process, FixedWPAConfig

    cfg = FixedWPAConfig(wa_sel=20, frac_bits=10)
    out = wpa_fixed_process(img, cfg)
"""

from .config import FixedWPAConfig
from .core import wpa_fixed_process

__all__ = ["FixedWPAConfig", "wpa_fixed_process"]
