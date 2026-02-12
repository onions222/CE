"""
Fixed-Point WPA CLI.

Usage::

    python -m wpa_fixed.cli --in input.jpg --out output.jpg --wa-sel 20 --frac-bits 12
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def _ensure_pillow():
    try:
        from PIL import Image
        return Image
    except ImportError:
        print("Error: Pillow is required for CLI.\n"
              "Install: pip install Pillow", file=sys.stderr)
        sys.exit(1)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="wpa_fixed",
        description="Fixed-point White Point Adjustment with configurable precision.",
    )
    parser.add_argument("--in", dest="input", required=True)
    parser.add_argument("--out", dest="output", required=True)
    parser.add_argument("--wa-sel", type=int, default=64,
                        help="0=warmest, 64=identity, 127=coolest")
    parser.add_argument("--frac-bits", type=int, default=10,
                        help="Fractional bits for fixed-point precision (default 10)")
    parser.add_argument("--wa-en", type=int, default=1, choices=[0, 1])
    parser.add_argument("--gamma", default="srgb", choices=["srgb", "power", "none"])
    parser.add_argument("--no-sat", action="store_true")

    args = parser.parse_args(argv)
    Image = _ensure_pillow()

    pil_img = Image.open(args.input).convert("RGB")
    img = np.array(pil_img, dtype=np.uint8)

    from .config import FixedWPAConfig
    from .core import wpa_fixed_process

    cfg = FixedWPAConfig(
        frac_bits=args.frac_bits,
        wa_en=bool(args.wa_en),
        wa_sel=args.wa_sel,
        gamma_mode=args.gamma,
        sat_en=not args.no_sat,
    )
    out = wpa_fixed_process(img, cfg)
    Image.fromarray(out).save(args.output)
    print(f"Saved: {args.output}  (wa_sel={cfg.wa_sel}, frac_bits={cfg.frac_bits})")


if __name__ == "__main__":
    main()
