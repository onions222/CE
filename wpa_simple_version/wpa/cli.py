"""CLI for WPA image processing."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
    parser.add_argument("--wa-en", dest="wa_en", action="store_true", default=True, help="Enable WPA (default: on)")
    parser.add_argument("--no-wa-en", dest="wa_en", action="store_false", help="Disable WPA")
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
    parser.add_argument("--sat-s1", type=float, default=2.00, help="Saturation weight high threshold")
    parser.add_argument("--warm-strength", type=float, default=1.0, help="Warm-side attenuation strength")
    parser.add_argument("--cool-strength", type=float, default=1.0, help="Cool-side attenuation strength")
    parser.add_argument("--kelvin-strength", type=float, default=1.20, help="Extra Kelvin/YCoCg strength scale")
    parser.add_argument("--kelvin-warm-side-scale", type=float, default=0.2, help="Warm-side scale for Kelvin/YCoCg")
    parser.add_argument("--kelvin-cool-side-scale", type=float, default=1.0, help="Cool-side scale for Kelvin/YCoCg")
    parser.add_argument("--report", action="store_true", help="Print quantitative WPA metrics for this run")
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality when output extension is .jpg/.jpeg (1-100, default: 95)",
    )
    parser.add_argument(
        "--jpeg-subsampling",
        type=int,
        choices=[0, 1, 2],
        default=0,
        help="JPEG chroma subsampling for .jpg/.jpeg output (0:4:4:4, 1:4:2:2, 2:4:2:0)",
    )
    return parser


def _gray_ramp_spike_metric(cfg: WPAConfig) -> float:
    ys = [0, 32, 64, 96, 128, 160, 192, 224, 255]
    size = 64
    ramp = np.concatenate([np.full((size, size, 3), y, dtype=np.uint8) for y in ys], axis=1)
    out = wpa_process_rgb_uint8(ramp, cfg)

    hi_idx = [i for i, y in enumerate(ys) if y >= 223]
    rg: list[float] = []
    gb: list[float] = []
    for i in range(len(ys)):
        p = out[:, i * size : (i + 1) * size, :].astype(np.int32)
        rg.append(float(np.mean(p[..., 0] - p[..., 1])))
        gb.append(float(np.mean(p[..., 1] - p[..., 2])))

    rg_arr = np.asarray(rg, dtype=np.float32)
    gb_arr = np.asarray(gb, dtype=np.float32)
    if len(hi_idx) <= 1:
        return 0.0
    drg = np.diff(rg_arr[hi_idx])
    dgb = np.diff(gb_arr[hi_idx])
    return float(max(np.max(np.abs(drg)), np.max(np.abs(dgb))))


def _print_report(src: np.ndarray, out: np.ndarray, cfg: WPAConfig) -> None:
    src_i = src.astype(np.int16)
    out_i = out.astype(np.int16)
    d = out_i - src_i
    l1 = np.abs(d).sum(axis=-1).astype(np.float32)

    mean_delta = d.mean(axis=(0, 1))
    mean_l1 = float(np.mean(l1))
    p50 = float(np.percentile(l1, 50))
    p90 = float(np.percentile(l1, 90))
    p99 = float(np.percentile(l1, 99))
    spike = _gray_ramp_spike_metric(cfg)

    print(
        "WPA Report: "
        f"wa_mode={cfg.wa_mode} WA_SEL={cfg.WA_SEL} "
        f"mean_l1={mean_l1:.3f} p50/p90/p99={p50:.1f}/{p90:.1f}/{p99:.1f} "
        f"mean_delta_rgb=[{mean_delta[0]:.3f},{mean_delta[1]:.3f},{mean_delta[2]:.3f}] "
        f"highlight_spike={spike:.3f}"
    )


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
    if not (1 <= args.jpeg_quality <= 100):
        print("--jpeg-quality must be in [1,100]", file=sys.stderr)
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
        kelvin_warm_side_scale=args.kelvin_warm_side_scale,
        kelvin_cool_side_scale=args.kelvin_cool_side_scale,
    )

    out = wpa_process_rgb_uint8(arr, cfg)
    out_img = Image.fromarray(out, mode="RGB")
    ext = Path(args.out_path).suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        out_img.save(
            args.out_path,
            quality=args.jpeg_quality,
            subsampling=args.jpeg_subsampling,
            optimize=True,
        )
    else:
        out_img.save(args.out_path)
    if args.report:
        _print_report(arr, out, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
