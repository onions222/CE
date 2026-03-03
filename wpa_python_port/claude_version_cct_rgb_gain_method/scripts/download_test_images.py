#!/usr/bin/env python3
"""
Download real-world test images for WPA algorithm verification.

Sources:
  - Kodak image set (24 images, public domain, 768×512 PNG)
    http://r0k.us/graphics/kodak/

Selected images cover:
  - Portraits / skin tones
  - Outdoor landscapes
  - Indoor scenes
  - High dynamic range / high contrast
  - Saturated colors / flowers
  - Low-key / dark scenes
  - Architecture / fine detail

Usage:
    python scripts/download_test_images.py [--output-dir test_images/real]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError

# Kodak image set — curated subset with descriptions
# Full set: http://r0k.us/graphics/kodak/kodak/
KODAK_IMAGES = {
    # Portraits & skin tones
    "kodim04.png": "Outdoor portrait — warm skin tone, natural lighting",
    "kodim05.png": "Indoor portrait — artificial lighting, skin tone",
    "kodim15.png": "Portrait with hat — diverse skin tones, backlight",

    # Landscapes & outdoor
    "kodim06.png": "River landscape — greens, blues, natural scene",
    "kodim07.png": "Gardens — rich greens, flowers, mixed lighting",
    "kodim13.png": "Snowy village — cool tones, white balance stress test",

    # High saturation / color-rich
    "kodim01.png": "Lighthouse — saturated sky, warm/cool regions",
    "kodim09.png": "Flowers — highly saturated reds/yellows/greens",
    "kodim22.png": "Sailboat — vivid colors, sky and ocean",

    # Architecture / detail
    "kodim08.png": "Building — fine details, neutral tones",
    "kodim21.png": "Architecture — structured patterns, neutral grey",

    # Indoor / mixed lighting
    "kodim03.png": "Indoor with plants — mixed green and neutral",
    "kodim19.png": "Fence and garden — mixed lighting, textures",

    # Low-key / dark / high-contrast
    "kodim10.png": "Stone building — dark regions, high contrast",
    "kodim24.png": "Parrot — saturated color on mid-dark background",
}

KODAK_BASE_URL = "http://r0k.us/graphics/kodak/kodak/"


def download_kodak(output_dir: Path, names: dict[str, str]) -> list[Path]:
    """Download Kodak images."""
    downloaded = []
    kodak_dir = output_dir / "kodak"
    kodak_dir.mkdir(parents=True, exist_ok=True)

    for filename, desc in names.items():
        url = KODAK_BASE_URL + filename
        dest = kodak_dir / filename
        if dest.exists():
            print(f"  ⏭ {filename:18s}  (already exists)")
            downloaded.append(dest)
            continue
        try:
            print(f"  ⬇ {filename:18s}  {desc}")
            urlretrieve(url, str(dest))
            downloaded.append(dest)
        except (URLError, OSError) as e:
            print(f"  ✗ {filename:18s}  FAILED: {e}")

    return downloaded


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download real-world test images for WPA testing"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="test_images/real",
        help="Directory to save downloaded images (default: test_images/real)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Download ALL 24 Kodak images instead of curated subset"
    )
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Downloading test images → {out}/\n")

    # --- Kodak images ---
    if args.all:
        all_kodak = {f"kodim{i:02d}.png": f"Kodak #{i}" for i in range(1, 25)}
        print(f"📦 Kodak image set (all 24 images):\n")
        downloaded = download_kodak(out, all_kodak)
    else:
        print(f"📦 Kodak image set (curated {len(KODAK_IMAGES)} images):\n")
        downloaded = download_kodak(out, KODAK_IMAGES)

    print(f"\n✅ Downloaded {len(downloaded)} images to {out}/")

    print("\n" + "=" * 60)
    print("Image selection rationale for WPA testing:")
    print("=" * 60)
    print("""
  Category              Images          Why
  ─────────────────     ──────────      ──────────────────────────
  Skin tones            04, 05, 15      WPA 最不能搞砸的场景
  Landscapes            06, 07, 13      自然场景色温适应
  High saturation       01, 09, 22      测试 gain 对饱和色的色相偏移
  Architecture          08, 21          中性灰、细节保持能力
  Mixed lighting        03, 19          混合光源挑战
  Dark / high contrast  10, 24          暗部保护和 clipping 测试
""")


if __name__ == "__main__":
    main()
