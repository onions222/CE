#!/usr/bin/env python3
"""
Generate synthetic test images for WPA algorithm verification.

Covers:
  1. Grey ramp (0–255 horizontal gradient, R=G=B)
  2. Color checker (Macbeth-style 24 patches)
  3. Pure channel sweeps (R, G, B individual gradients)
  4. Saturation gradient (grey→saturated for each hue)
  5. High-key / Low-key patches (near-white / near-black)
  6. Skin tone patches (common Fitzpatrick scale approximations)
  7. Smooth gradient (large-area for banding detection)
  8. Random noise image (stress test)

Usage:
    python scripts/generate_test_images.py [--output-dir tests/images]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except ImportError:
    raise SystemExit("Pillow is required: pip install Pillow")


def save(img: np.ndarray, path: Path, name: str) -> Path:
    """Save uint8 image and print path."""
    fp = path / f"{name}.png"
    Image.fromarray(img).save(fp)
    print(f"  ✓ {fp.name:40s}  {img.shape[1]}×{img.shape[0]}")
    return fp


# ── 1. Grey ramp ────────────────────────────────────────────────────────

def gen_grey_ramp(h: int = 64, w: int = 256) -> np.ndarray:
    """Horizontal 0–255 grey gradient."""
    row = np.linspace(0, 255, w, dtype=np.uint8)
    plane = np.tile(row, (h, 1))
    return np.stack([plane, plane, plane], axis=-1)


# ── 2. Color checker (Macbeth-like 24 patches) ──────────────────────────

# sRGB values for Macbeth ColorChecker (approximate, from BabelColor)
MACBETH_SRGB = np.array([
    [115,  82,  68],  # 1  Dark Skin
    [194, 150, 130],  # 2  Light Skin
    [ 98, 122, 157],  # 3  Blue Sky
    [ 87, 108,  67],  # 4  Foliage
    [133, 128, 177],  # 5  Blue Flower
    [103, 189, 170],  # 6  Bluish Green
    [214, 126,  44],  # 7  Orange
    [ 80,  91, 166],  # 8  Purplish Blue
    [193,  90,  99],  # 9  Moderate Red
    [ 94,  60, 108],  # 10 Purple
    [157, 188,  64],  # 11 Yellow Green
    [224, 163,  46],  # 12 Orange Yellow
    [ 56,  61, 150],  # 13 Blue
    [ 70, 148,  73],  # 14 Green
    [175,  54,  60],  # 15 Red
    [231, 199,  31],  # 16 Yellow
    [187,  86, 149],  # 17 Magenta
    [  8, 133, 161],  # 18 Cyan
    [243, 243, 242],  # 19 White
    [200, 200, 200],  # 20 Neutral 8
    [160, 160, 160],  # 21 Neutral 6.5
    [122, 122, 121],  # 22 Neutral 5
    [ 85,  85,  85],  # 23 Neutral 3.5
    [ 52,  52,  52],  # 24 Dark Grey
], dtype=np.uint8)


def gen_color_checker(patch_size: int = 64) -> np.ndarray:
    """4×6 Macbeth ColorChecker layout."""
    rows, cols = 4, 6
    h = rows * patch_size
    w = cols * patch_size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, color in enumerate(MACBETH_SRGB):
        r = idx // cols
        c = idx % cols
        img[r*patch_size:(r+1)*patch_size, c*patch_size:(c+1)*patch_size] = color
    return img


# ── 3. Pure channel sweeps ──────────────────────────────────────────────

def gen_channel_sweeps(h: int = 48, w: int = 256) -> np.ndarray:
    """Three horizontal bands: R, G, B channel sweeps 0–255."""
    ramp = np.linspace(0, 255, w, dtype=np.uint8)
    img = np.zeros((h * 3, w, 3), dtype=np.uint8)
    # Red
    img[:h, :, 0] = np.tile(ramp, (h, 1))
    # Green
    img[h:2*h, :, 1] = np.tile(ramp, (h, 1))
    # Blue
    img[2*h:3*h, :, 2] = np.tile(ramp, (h, 1))
    return img


# ── 4. Saturation gradient ─────────────────────────────────────────────

def gen_saturation_gradient(h_per_hue: int = 48, w: int = 256) -> np.ndarray:
    """For 6 primary/secondary hues, gradient from grey to full saturation."""
    hues = [
        (255, 0, 0),    # Red
        (255, 255, 0),  # Yellow
        (0, 255, 0),    # Green
        (0, 255, 255),  # Cyan
        (0, 0, 255),    # Blue
        (255, 0, 255),  # Magenta
    ]
    strips = []
    for r, g, b in hues:
        strip = np.zeros((h_per_hue, w, 3), dtype=np.uint8)
        for x in range(w):
            t = x / max(w - 1, 1)
            # lerp from mid-grey (128,128,128) to (r,g,b)
            strip[:, x, 0] = int(128 * (1 - t) + r * t)
            strip[:, x, 1] = int(128 * (1 - t) + g * t)
            strip[:, x, 2] = int(128 * (1 - t) + b * t)
        strips.append(strip)
    return np.concatenate(strips, axis=0)


# ── 5. High-key / Low-key patches ──────────────────────────────────────

def gen_highlight_lowlight(patch_size: int = 64) -> np.ndarray:
    """Near-white and near-black patches to test clipping behavior."""
    values = [0, 1, 2, 3, 4, 8, 16, 240, 248, 251, 252, 253, 254, 255]
    cols = len(values)
    img = np.zeros((patch_size, cols * patch_size, 3), dtype=np.uint8)
    for i, v in enumerate(values):
        img[:, i*patch_size:(i+1)*patch_size] = v
    return img


# ── 6. Skin tone patches ───────────────────────────────────────────────

def gen_skin_tones(patch_size: int = 80) -> np.ndarray:
    """Approximate Fitzpatrick I–VI skin tone patches in sRGB."""
    skin_tones = [
        (255, 224, 196),  # Type I  — Very fair
        (240, 200, 166),  # Type II — Fair
        (220, 175, 140),  # Type III — Medium
        (185, 140, 105),  # Type IV — Olive
        (140, 100,  70),  # Type V  — Brown
        ( 90,  60,  40),  # Type VI — Dark
    ]
    n = len(skin_tones)
    img = np.zeros((patch_size, n * patch_size, 3), dtype=np.uint8)
    for i, (r, g, b) in enumerate(skin_tones):
        img[:, i*patch_size:(i+1)*patch_size] = [r, g, b]
    return img


# ── 7. Smooth gradient (banding detection) ──────────────────────────────

def gen_smooth_gradient(h: int = 256, w: int = 512) -> np.ndarray:
    """2D smooth gradient: horizontal = R channel, vertical = B channel.
    Large area makes banding easily visible.
    """
    r_row = np.linspace(0, 255, w, dtype=np.float32)
    b_col = np.linspace(0, 255, h, dtype=np.float32)
    r_plane = np.tile(r_row, (h, 1))
    b_plane = np.tile(b_col[:, np.newaxis], (1, w))
    g_plane = np.full((h, w), 128, dtype=np.float32)
    img = np.stack([r_plane, g_plane, b_plane], axis=-1)
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


# ── 8. Random noise image (stress test) ─────────────────────────────────

def gen_random_noise(h: int = 256, w: int = 256, seed: int = 42) -> np.ndarray:
    """Uniform random RGB for stress-testing edge cases."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, (h, w, 3), dtype=np.uint8)


# ── 9. Grey step chart ──────────────────────────────────────────────────

def gen_grey_steps(patch_size: int = 64) -> np.ndarray:
    """16 uniform grey steps from 0 to 255."""
    steps = np.linspace(0, 255, 16, dtype=np.uint8)
    img = np.zeros((patch_size, len(steps) * patch_size, 3), dtype=np.uint8)
    for i, v in enumerate(steps):
        img[:, i*patch_size:(i+1)*patch_size] = v
    return img


# ── 10. Luma-node aligned chart ─────────────────────────────────────────

def gen_luma_node_chart(patch_size: int = 64) -> np.ndarray:
    """Grey patches at each of the 12 luma nodes used by the algorithm.
    Useful for verifying gain table lookup at exact node boundaries.
    """
    from wpa.config import LUMA_NODES_12
    nodes = LUMA_NODES_12
    img = np.zeros((patch_size, len(nodes) * patch_size, 3), dtype=np.uint8)
    for i, v in enumerate(nodes):
        img[:, i*patch_size:(i+1)*patch_size] = v
    return img


# ── main ────────────────────────────────────────────────────────────────

GENERATORS = {
    "01_grey_ramp":          gen_grey_ramp,
    "02_color_checker":      gen_color_checker,
    "03_channel_sweeps":     gen_channel_sweeps,
    "04_saturation_gradient": gen_saturation_gradient,
    "05_highlight_lowlight": gen_highlight_lowlight,
    "06_skin_tones":         gen_skin_tones,
    "07_smooth_gradient":    gen_smooth_gradient,
    "08_random_noise":       gen_random_noise,
    "09_grey_steps":         gen_grey_steps,
    "10_luma_node_chart":    gen_luma_node_chart,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate WPA test images")
    parser.add_argument(
        "--output-dir", "-o",
        default="tests/images",
        help="Directory to save generated images (default: tests/images)",
    )
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Generating test images → {out}/\n")
    for name, gen_fn in GENERATORS.items():
        img = gen_fn()
        save(img, out, name)

    print(f"\n✅ {len(GENERATORS)} test images generated in {out}/")
    print("\nImage descriptions:")
    print("  01  Grey ramp           — 亮度单调递增灰阶，验证 gain 表查表连续性")
    print("  02  Color checker       — Macbeth 24 色卡，验证色相保持与饱和度影响")
    print("  03  Channel sweeps      — R/G/B 通道独立递增，验证各通道 gain 方向")
    print("  04  Saturation gradient — 灰→饱和渐变，验证饱和度保护过渡")
    print("  05  Highlight/Lowlight  — 极暗/极亮色块，验证 clipping 与暗部保护")
    print("  06  Skin tones          — 6 级肤色，验证 WPA 对人类肤色的影响")
    print("  07  Smooth gradient     — 大面积渐变，检测 banding 伪影")
    print("  08  Random noise        — 随机噪声，stress test 极端值")
    print("  09  Grey steps          — 16 级均匀灰阶，验证亮度分段一致性")
    print("  10  Luma node chart     — 12 节点精确灰度，验证 gain 节点精确匹配")


if __name__ == "__main__":
    main()
