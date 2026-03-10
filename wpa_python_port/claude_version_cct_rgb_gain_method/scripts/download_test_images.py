#!/usr/bin/env python3
"""
Download legacy Kodak baseline images for WPA algorithm verification.

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
from urllib.request import Request, urlopen, urlretrieve
from urllib.error import URLError

# Kodak image set — retained as a legacy baseline subset
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

COMMONS_FILEPATH = "https://commons.wikimedia.org/wiki/Special:FilePath/"

REAL_WORLD_IMAGE_GROUPS = {
    "public_portrait": [
        {
            "filename": "portrait_of_woman.jpg",
            "url": f"{COMMONS_FILEPATH}Portrait_of_woman.jpg",
            "description": "Daylight portrait for skin-tone plausibility sanity checks",
            "source_group": "public_portrait",
        },
    ],
    "public_hdr_window": [
        {
            "filename": "gfp_sunroom.jpg",
            "url": f"{COMMONS_FILEPATH}Gfp-sunroom.jpg",
            "description": "Bright-window interior for HDR and neutral-wall inspection",
            "source_group": "public_hdr_window",
        },
    ],
    "public_night_neon": [
        {
            "filename": "led_and_neon_signs_on_portland_street_at_night.jpg",
            "url": f"{COMMONS_FILEPATH}LED_and_neon_signs_on_Portland_Street_at_night.jpg",
            "description": "Night neon scene for saturated highlights and mixed-color light",
            "source_group": "public_night_neon",
        },
    ],
    "public_ui_workspace": [
        {
            "filename": "desk_setup_unsplash.jpg",
            "url": f"{COMMONS_FILEPATH}Desk_Setup_%28Unsplash%29.jpg",
            "description": "Workspace scene with monitor and peripherals for screen-like sanity checks",
            "source_group": "public_ui_workspace",
        },
    ],
    "research_mixed_light": [
        {
            "filename": "lsmi_mixed_light_sample.png",
            "url": "https://user-images.githubusercontent.com/24367643/130312876-5b2955c2-0176-4e87-ba90-7c466fa3961b.png",
            "description": "Official LSMI sample figure representing mixed-illuminant content",
            "source_group": "research_mixed_light",
        },
    ],
    "research_outdoor_sanity": [
        {
            "filename": "cubepp_examples.jpg",
            "url": "https://github.com/Visillect/CubePlusPlus/raw/master/description/examples.jpg",
            "description": "Official Cube++ example sheet for outdoor and illumination sanity coverage",
            "source_group": "research_outdoor_sanity",
        },
    ],
}


def download_kodak(output_dir: Path, names: dict[str, str]) -> list[Path]:
    """Download legacy Kodak baseline images."""
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


def download_real_world_images(
    output_dir: Path,
    groups: list[str],
    root: Path | None = None,
) -> dict[str, str]:
    """Download a curated real-world sanity subset grouped by scene type."""
    downloaded: dict[str, str] = {}
    sanity_dir = output_dir / "real_sanity"
    sanity_dir.mkdir(parents=True, exist_ok=True)

    for group in groups:
        entries = REAL_WORLD_IMAGE_GROUPS.get(group)
        if not entries:
            raise ValueError(f"Unknown real-world image group: {group}")
        group_dir = sanity_dir / group
        group_dir.mkdir(parents=True, exist_ok=True)
        for entry in entries:
            dest = group_dir / entry["filename"]
            if dest.exists():
                print(f"  ⏭ {entry['filename']:36s}  ({group})")
            else:
                try:
                    print(f"  ⬇ {entry['filename']:36s}  {entry['description']}")
                    _download_with_browser_headers(entry["url"], dest)
                except (URLError, OSError) as exc:
                    print(f"  ✗ {entry['filename']:36s}  FAILED: {exc}")
                    continue

            rel = dest.relative_to(root) if root is not None else dest
            downloaded[entry["filename"]] = str(rel)

    return downloaded


def _download_with_browser_headers(url: str, dest: Path) -> None:
    """Fetch assets from hosts that reject urllib's default user agent."""
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req) as src, dest.open("wb") as dst:
        dst.write(src.read())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download legacy Kodak baseline images for WPA testing"
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
        print(f"📦 Kodak legacy baseline (all 24 images):\n")
        downloaded = download_kodak(out, all_kodak)
    else:
        print(f"📦 Kodak legacy baseline (curated {len(KODAK_IMAGES)} images):\n")
        downloaded = download_kodak(out, KODAK_IMAGES)

    print(f"\n✅ Downloaded {len(downloaded)} images to {out}/")

    print(f"\n📦 Real-world sanity subset ({len(REAL_WORLD_IMAGE_GROUPS)} groups):\n")
    download_real_world_images(out, list(REAL_WORLD_IMAGE_GROUPS.keys()))

    print("\n" + "=" * 60)
    print("Legacy baseline selection rationale for WPA testing:")
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
