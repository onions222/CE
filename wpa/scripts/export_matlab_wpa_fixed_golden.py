#!/usr/bin/env python3
"""Export Python fixed-WPA outputs as golden cases for MATLAB validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wpa_fixed import FixedWPAConfig, wpa_fixed_process


def _default_images() -> list[Path]:
    candidates = [
        PROJECT_ROOT / "test_images" / "synthetic" / "01_grey_ramp.png",
        PROJECT_ROOT / "test_images" / "synthetic" / "near_white_steps.png",
        PROJECT_ROOT / "test_images" / "synthetic" / "11_ui_text_contrast.png",
        PROJECT_ROOT / "test_images" / "real" / "real_sanity" / "public_hdr_window" / "gfp_sunroom.jpg",
    ]
    return [p for p in candidates if p.exists()]


def export_cases(
    images: list[Path],
    output_dir: Path,
    wa_values: list[int],
    *,
    frac_bits: int,
    coeff_frac_bits: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_cases: list[dict] = []

    for image_path in images:
        img = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        for wa_sel in wa_values:
            cfg = FixedWPAConfig(
                wa_sel=wa_sel,
                frac_bits=frac_bits,
                coeff_frac_bits=coeff_frac_bits,
            )
            out = wpa_fixed_process(img, cfg)
            stem = image_path.stem
            out_name = f"{stem}_wa{wa_sel}.png"
            out_path = output_dir / out_name
            Image.fromarray(out).save(out_path)

            manifest_cases.append(
                {
                    "source_image": str(image_path),
                    "output_image": out_name,
                    "wa_sel": wa_sel,
                    "shape": list(out.shape),
                    "dtype": str(out.dtype),
                    "mean_rgb": np.round(out.reshape(-1, 3).mean(axis=0), 4).tolist(),
                    "max_rgb": out.reshape(-1, 3).max(axis=0).tolist(),
                }
            )

    manifest = {
        "generator": "scripts/export_matlab_wpa_fixed_golden.py",
        "frac_bits": frac_bits,
        "coeff_frac_bits": coeff_frac_bits,
        "cases": manifest_cases,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export Python fixed-WPA golden cases for MATLAB.")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "matlab" / "golden_cases"),
        help="Directory for exported golden outputs.",
    )
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        help="Input image path. Can be provided multiple times.",
    )
    parser.add_argument("--frac-bits", type=int, default=10)
    parser.add_argument("--coeff-frac-bits", type=int, default=8, choices=[8, 10])
    parser.add_argument("--wa-sel", action="append", type=int, default=[])
    args = parser.parse_args(argv)

    image_paths = [Path(p).resolve() for p in args.image] if args.image else _default_images()
    if not image_paths:
        raise SystemExit("No input images found for golden export.")

    wa_values = args.wa_sel if args.wa_sel else [0, 64, 127]
    manifest = export_cases(
        image_paths,
        Path(args.output_dir),
        wa_values,
        frac_bits=args.frac_bits,
        coeff_frac_bits=args.coeff_frac_bits,
    )
    print(f"Exported {len(manifest['cases'])} cases to {args.output_dir}")


if __name__ == "__main__":
    main()
