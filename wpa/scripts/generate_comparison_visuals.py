#!/usr/bin/env python3
"""Generate side-by-side WPA comparison panels for test images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    raise SystemExit("Pillow is required: pip install Pillow")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wpa import WPAConfig, wpa_process_rgb_uint8


LABELS = ["original", "wa=0", "wa=64", "wa=127"]
WA_VALUES = [0, 64, 127]
BG_COLOR = (248, 248, 244)
TEXT_COLOR = (32, 32, 32)
BORDER_COLOR = (205, 205, 198)
PADDING = 16
GAP = 14
TITLE_HEIGHT = 30
LABEL_HEIGHT = 24


def choose_layout(width: int, height: int) -> str:
    """Use vertical stacking for extra-wide strip images."""
    return "vertical" if width >= height * 3 else "horizontal"


def _load_font() -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        return ImageFont.load_default()


def _fit_image(img: Image.Image, *, max_width: int, max_height: int) -> Image.Image:
    scale = min(max_width / img.width, max_height / img.height, 1.0)
    new_size = (max(1, int(round(img.width * scale))), max(1, int(round(img.height * scale))))
    if new_size == img.size:
        return img.copy()
    return img.resize(new_size, Image.Resampling.LANCZOS)


def _render_variant(image: Image.Image, wa_sel: int) -> Image.Image:
    arr = np.array(image.convert("RGB"), dtype=np.uint8)
    out = wpa_process_rgb_uint8(arr, WPAConfig(wa_sel=wa_sel))
    return Image.fromarray(out)


def _draw_text_center(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, font) -> None:
    left, top, right, bottom = box
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    x = left + (right - left - width) // 2
    y = top + (bottom - top - height) // 2
    draw.text((x, y), text, fill=TEXT_COLOR, font=font)


def render_comparison_panel(image: Image.Image, image_name: str) -> tuple[Image.Image, str]:
    """Render a 4-up comparison panel and return the image plus chosen layout."""
    layout = choose_layout(image.width, image.height)
    font = _load_font()

    variants = [image.convert("RGB")]
    variants.extend(_render_variant(image, wa_sel) for wa_sel in WA_VALUES)

    if layout == "vertical":
        fitted = [_fit_image(v, max_width=1200, max_height=220) for v in variants]
        tile_width = max(v.width for v in fitted)
        content_width = tile_width
        content_height = sum(v.height + LABEL_HEIGHT for v in fitted) + GAP * (len(fitted) - 1)
        panel = Image.new(
            "RGB",
            (content_width + PADDING * 2, TITLE_HEIGHT + content_height + PADDING * 2),
            BG_COLOR,
        )
        draw = ImageDraw.Draw(panel)
        _draw_text_center(draw, (PADDING, 0, panel.width - PADDING, TITLE_HEIGHT), image_name, font)

        y = TITLE_HEIGHT + PADDING
        for variant, label in zip(fitted, LABELS):
            draw.rectangle((PADDING, y, PADDING + tile_width, y + LABEL_HEIGHT), outline=BORDER_COLOR, width=1)
            _draw_text_center(draw, (PADDING, y, PADDING + tile_width, y + LABEL_HEIGHT), label, font)
            y += LABEL_HEIGHT
            x = PADDING + (tile_width - variant.width) // 2
            panel.paste(variant, (x, y))
            draw.rectangle((x - 1, y - 1, x + variant.width, y + variant.height), outline=BORDER_COLOR, width=1)
            y += variant.height + GAP
        return panel, layout

    fitted = [_fit_image(v, max_width=360, max_height=280) for v in variants]
    tile_width = max(v.width for v in fitted)
    tile_height = max(v.height for v in fitted)
    content_width = len(fitted) * tile_width + GAP * (len(fitted) - 1)
    content_height = LABEL_HEIGHT + tile_height
    panel = Image.new(
        "RGB",
        (content_width + PADDING * 2, TITLE_HEIGHT + content_height + PADDING * 2),
        BG_COLOR,
    )
    draw = ImageDraw.Draw(panel)
    _draw_text_center(draw, (PADDING, 0, panel.width - PADDING, TITLE_HEIGHT), image_name, font)

    x = PADDING
    y = TITLE_HEIGHT + PADDING
    for variant, label in zip(fitted, LABELS):
        draw.rectangle((x, y, x + tile_width, y + LABEL_HEIGHT), outline=BORDER_COLOR, width=1)
        _draw_text_center(draw, (x, y, x + tile_width, y + LABEL_HEIGHT), label, font)
        paste_x = x + (tile_width - variant.width) // 2
        paste_y = y + LABEL_HEIGHT + (tile_height - variant.height) // 2
        panel.paste(variant, (paste_x, paste_y))
        draw.rectangle(
            (paste_x - 1, paste_y - 1, paste_x + variant.width, paste_y + variant.height),
            outline=BORDER_COLOR,
            width=1,
        )
        x += tile_width + GAP
    return panel, layout


def _iter_input_images(input_root: Path) -> list[Path]:
    return sorted(
        p for p in input_root.rglob("*")
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )


def build_panels(input_roots: list[Path], output_root: Path) -> list[Path]:
    created: list[Path] = []
    for root in input_roots:
        for path in _iter_input_images(root):
            rel = path.relative_to(root)
            out_dir = output_root / root.name / rel.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{path.stem}_comparison.png"
            image = Image.open(path).convert("RGB")
            panel, layout = render_comparison_panel(image, image_name=path.name)
            panel.save(out_path)
            print(f"  ✓ {out_path.relative_to(output_root)}  ({layout})")
            created.append(out_path)
    return created


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate WPA comparison visuals for test images.")
    parser.add_argument(
        "--output-dir",
        default="test_images/visualizations",
        help="Directory for generated comparison panels.",
    )
    args = parser.parse_args(argv)

    synthetic_root = PROJECT_ROOT / "test_images" / "synthetic"
    real_sanity_root = PROJECT_ROOT / "test_images" / "real" / "real_sanity"
    output_root = PROJECT_ROOT / args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)

    inputs = [p for p in [synthetic_root, real_sanity_root] if p.exists()]
    if not inputs:
        raise SystemExit("No input images found under test_images/synthetic or test_images/real/real_sanity")

    print(f"Generating comparison panels -> {output_root}")
    created = build_panels(inputs, output_root)
    print(f"\nCreated {len(created)} comparison panels.")


if __name__ == "__main__":
    main()
