from __future__ import annotations

import argparse
import sys

import numpy as np

from .config import FixedWPAConfig
from .core import wpa_process_rgb_uint8_fixed


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fixed-point-only WPA CLI")
    p.add_argument("--in", dest="in_path", required=True, help="Input image path")
    p.add_argument("--out", dest="out_path", required=True, help="Output image path")
    p.add_argument("--q-bits", type=int, default=12, help="Fixed-point fractional bits, e.g. 8 or 12")
    p.add_argument("--wa-en", dest="wa_en", action="store_true", default=True, help="Enable WPA (default: on)")
    p.add_argument("--no-wa-en", dest="wa_en", action="store_false", help="Disable WPA")
    p.add_argument("--wa-sel", type=int, default=64, help="WA_SEL in [0,127], 64 identity")
    p.add_argument("--sat-s0-255", type=int, default=51, help="Saturation low threshold in [0,510]")
    p.add_argument("--sat-s1-255", type=int, default=510, help="Saturation high threshold in [0,510]")
    p.add_argument("--warm-strength-q", type=int, default=None, help="Warm strength in current Q domain (qone=1<<q_bits)")
    p.add_argument("--cool-strength-q", type=int, default=None, help="Cool strength in current Q domain (qone=1<<q_bits)")
    # Backward-compatible aliases for previous interface
    p.add_argument("--warm-strength-q12", dest="warm_strength_q", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--cool-strength-q12", dest="cool_strength_q", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--report", action="store_true", help="Print simple integer-path metrics")
    return p


def _print_report(src: np.ndarray, out: np.ndarray, wa_sel: int) -> None:
    d = out.astype(np.int16) - src.astype(np.int16)
    l1 = np.abs(d).sum(axis=-1).astype(np.float32)
    mean_delta = d.mean(axis=(0, 1))
    print(
        "Fixed WPA Report: "
        f"WA_SEL={wa_sel} "
        f"mean_l1={float(np.mean(l1)):.3f} "
        f"p50/p90/p99={float(np.percentile(l1,50)):.1f}/{float(np.percentile(l1,90)):.1f}/{float(np.percentile(l1,99)):.1f} "
        f"mean_delta_rgb=[{mean_delta[0]:.3f},{mean_delta[1]:.3f},{mean_delta[2]:.3f}]"
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from PIL import Image
    except ImportError:
        print("Pillow is required. Install with: pip install pillow", file=sys.stderr)
        return 2

    if not (0 <= args.wa_sel <= 127):
        print("--wa-sel must be in [0,127]", file=sys.stderr)
        return 2
    if not (4 <= args.q_bits <= 16):
        print("--q-bits must be in [4,16]", file=sys.stderr)
        return 2
    if not (0 <= args.sat_s0_255 <= 510 and 0 <= args.sat_s1_255 <= 510):
        print("--sat-s0-255 and --sat-s1-255 must be in [0,510]", file=sys.stderr)
        return 2

    qone = 1 << args.q_bits
    img = np.asarray(Image.open(args.in_path).convert("RGB"), dtype=np.uint8)
    cfg = FixedWPAConfig(
        q_bits=args.q_bits,
        WA_EN=args.wa_en,
        WA_SEL=args.wa_sel,
        sat_s0_255=args.sat_s0_255,
        sat_s1_255=args.sat_s1_255,
        warm_strength_q=(qone if args.warm_strength_q is None else args.warm_strength_q),
        cool_strength_q=(qone if args.cool_strength_q is None else args.cool_strength_q),
    )
    out = wpa_process_rgb_uint8_fixed(img, cfg)
    Image.fromarray(out, mode="RGB").save(args.out_path)
    if args.report:
        _print_report(img, out, args.wa_sel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
