"""CLI for WPA image processing."""

from __future__ import annotations

import argparse
import sys

import numpy as np

try:
    from .config import WPAConfig
    from .core import wpa_process_rgb_uint8
except ImportError:
    # Fallback for running this file directly: python wpa/cli.py ...
    from config import WPAConfig
    from core import wpa_process_rgb_uint8


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="White Point Adjustment (WPA) CLI")
    parser.add_argument("--in", dest="in_path", required=True, help="Input image path")
    parser.add_argument("--out", dest="out_path", required=True, help="Output image path")
    parser.add_argument("--wa-en", action="store_true", default=False, help="Enable WPA")
    parser.add_argument("--wa-sel", type=int, default=64, help="WA_SEL in [0,127], 64 is identity")
    parser.add_argument(
        "--wa-mode",
        choices=["diag_rgb", "kelvin_ycocg"],
        default="diag_rgb",
        help="WPA model: diag gains or Kelvin/YCoCg",
    )
    parser.add_argument(
        "--luma-domain",
        choices=["gamma", "linear"],
        default="gamma",
        help="Domain for luma proxy",
    )
    parser.add_argument(
        "--sat-domain",
        choices=["gamma", "linear"],
        default="gamma",
        help="Domain for saturation weight",
    )
    parser.add_argument(
        "--gamma-mode",
        choices=["srgb", "power", "none"],
        default="srgb",
        help="Gamma conversion mode",
    )
    parser.add_argument("--use-gamma-lut", action="store_true", help="Use LUT gamma conversion")
    parser.add_argument("--power-gamma", type=float, default=2.2, help="Power gamma value")
    parser.add_argument("--sat-s0", type=float, default=0.20, help="Saturation weight low threshold")
    parser.add_argument("--sat-s1", type=float, default=1.80, help="Saturation weight high threshold")
    parser.add_argument("--warm-strength", type=float, default=1.0, help="Warm-side attenuation strength")
    parser.add_argument("--cool-strength", type=float, default=1.0, help="Cool-side attenuation strength")
    parser.add_argument("--kelvin-strength", type=float, default=0.20, help="Extra Kelvin/YCoCg strength scale")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        from PIL import Image
    except ImportError:
        print(
            "Pillow is required for CLI image I/O. Install with: pip install pillow",
            file=sys.stderr,
        )
        return 2

    if not (0 <= args.wa_sel <= 127):
        print("--wa-sel must be in [0,127]", file=sys.stderr)
        return 2

    img = Image.open(args.in_path).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)

    cfg = WPAConfig(
        WA_EN=args.wa_en,
        WA_SEL=args.wa_sel,
        wa_mode=args.wa_mode,
        luma_domain=args.luma_domain,
        sat_weight_domain=args.sat_domain,
        gamma_mode=args.gamma_mode,
        use_gamma_lut=args.use_gamma_lut,
        power_gamma=args.power_gamma,
        sat_s0=args.sat_s0,
        sat_s1=args.sat_s1,
        warm_strength=args.warm_strength,
        cool_strength=args.cool_strength,
        kelvin_strength=args.kelvin_strength,
    )

    out = wpa_process_rgb_uint8(arr, cfg)
    Image.fromarray(out, mode="RGB").save(args.out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
