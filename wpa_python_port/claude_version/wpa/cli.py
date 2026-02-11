"""
WPA CLI — command-line interface for White Point Adjustment.

Usage::

    python -m wpa.cli --in input.jpg --out output.jpg --wa-sel 20

Requires Pillow for image I/O.  If Pillow is not installed the CLI will
print a friendly error and exit.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def _ensure_pillow():
    """Import and return PIL.Image, or exit with a helpful message."""
    try:
        from PIL import Image
        return Image
    except ImportError:
        print(
            "Error: Pillow is required for CLI image I/O.\n"
            "Install it with:  pip install Pillow",
            file=sys.stderr,
        )
        sys.exit(1)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="wpa",
        description="White Point Adjustment — per-channel RGB gain with "
                    "12-bin luma-segmented gains.",
    )
    parser.add_argument("--in", dest="input", required=True,
                        help="Input image path (8-bit RGB, e.g. .jpg/.png)")
    parser.add_argument("--out", dest="output", required=True,
                        help="Output image path")
    parser.add_argument("--wa-sel", type=int, default=64,
                        help="WA_SEL 0..127 (64=identity, <64 warm, >64 cool)")
    parser.add_argument("--wa-en", type=int, default=1, choices=[0, 1],
                        help="WA_EN enable flag (0=off, 1=on, default 1)")
    parser.add_argument("--gamma", default="srgb",
                        choices=["srgb", "power", "none"],
                        help="Gamma mode (default: srgb)")
    parser.add_argument("--gamma-lut", action="store_true",
                        help="Use LUT for sRGB gamma (faster, slightly less precise)")
    parser.add_argument("--no-sat", action="store_true",
                        help="Disable saturation protection")

    args = parser.parse_args(argv)

    Image = _ensure_pillow()

    # --- Load image ---
    pil_img = Image.open(args.input).convert("RGB")
    img = np.array(pil_img, dtype=np.uint8)

    # --- Configure ---
    from .config import WPAConfig
    cfg = WPAConfig(
        wa_en=bool(args.wa_en),
        wa_sel=args.wa_sel,
        gamma_mode=args.gamma,
        use_gamma_lut=args.gamma_lut,
        sat_en=not args.no_sat,
    )

    # --- Process ---
    from .core import wpa_process_rgb_uint8
    out = wpa_process_rgb_uint8(img, cfg)

    # --- Save ---
    Image.fromarray(out).save(args.output)
    print(f"Saved: {args.output}  (wa_sel={cfg.wa_sel}, gamma={cfg.gamma_mode})")


if __name__ == "__main__":
    main()
