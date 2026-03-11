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
  9. Grey step chart
 10. Luma-node aligned chart
 11. UI text contrast chart
 12. Specular clip chart
 13. Mixed-illumination chart

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


def gen_near_black_steps(patch_size: int = 64) -> np.ndarray:
    """Dense neutral steps in the dark region for shadow stability checks."""
    values = list(range(0, 34, 2))
    img = np.zeros((patch_size, len(values) * patch_size, 3), dtype=np.uint8)
    for i, v in enumerate(values):
        img[:, i*patch_size:(i+1)*patch_size] = v
    return img


def gen_near_white_steps(patch_size: int = 64) -> np.ndarray:
    """Dense neutral steps in the highlight region for near-white stability checks."""
    values = list(range(223, 256, 2))
    img = np.zeros((patch_size, len(values) * patch_size, 3), dtype=np.uint8)
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
    nodes = [15, 31, 47, 63, 95, 127, 159, 191, 223, 239, 247, 255]
    img = np.zeros((patch_size, len(nodes) * patch_size, 3), dtype=np.uint8)
    for i, v in enumerate(nodes):
        img[:, i*patch_size:(i+1)*patch_size] = v
    return img


# ── 11. UI text contrast chart ──────────────────────────────────────────

def gen_ui_text_contrast(h: int = 256, w: int = 512) -> np.ndarray:
    """High-contrast UI-like chart with sharp edges and neutral text blocks."""
    img = np.full((h, w, 3), 245, dtype=np.uint8)
    # Header band
    img[0:48, :, :] = 32
    # Two body cards
    img[64:164, 24:244, :] = 230
    img[64:164, 268:488, :] = 230
    # Dark text bars
    for y in [78, 92, 106, 120, 134]:
        img[y:y + 4, 36:232, :] = 25
        img[y:y + 4, 280:476, :] = 25
    # Accent icon colors
    img[190:228, 36:72, :] = [220, 50, 50]
    img[190:228, 92:128, :] = [50, 160, 240]
    img[190:228, 148:184, :] = [70, 180, 70]
    return img


def gen_ui_dark_theme_chart(h: int = 256, w: int = 512) -> np.ndarray:
    """Dark-theme UI chart with neutral layers and bright text bars."""
    img = np.full((h, w, 3), 18, dtype=np.uint8)
    img[0:52, :, :] = 28
    img[72:172, 24:244, :] = 36
    img[72:172, 268:488, :] = 36
    for y in [88, 104, 120, 136, 152]:
        img[y:y + 4, 36:232, :] = 215
        img[y:y + 4, 280:476, :] = 215
    img[192:228, 36:72, :] = [210, 85, 85]
    img[192:228, 92:128, :] = [90, 145, 220]
    img[192:228, 148:184, :] = [95, 175, 105]
    return img


def gen_rgb_cmy_color_bars(h: int = 96, w_per_bar: int = 64) -> np.ndarray:
    """Primary and secondary color bars for hue-shift checks."""
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 0),
    ]
    img = np.zeros((h, len(colors) * w_per_bar, 3), dtype=np.uint8)
    for i, color in enumerate(colors):
        img[:, i*w_per_bar:(i+1)*w_per_bar] = color
    return img


# ── 12. Specular clip chart ─────────────────────────────────────────────

def gen_specular_clip_chart(h: int = 256, w: int = 512) -> np.ndarray:
    """Dark background with specular-like highlights for clipping checks."""
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.full((h, w, 3), [14, 16, 20], dtype=np.float32)

    # Three gaussian highlight blobs
    centers = [(128, 128), (180, 280), (90, 410)]
    colors = np.array([[255, 240, 220], [220, 235, 255], [255, 255, 255]], dtype=np.float32)
    sigmas = [28.0, 22.0, 18.0]
    for (cy, cx), col, s in zip(centers, colors, sigmas):
        g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * s * s))
        base += g[..., None] * col * 0.95

    # Add near-white edge strip to test high-end compression
    base[:, -40:, :] = np.maximum(base[:, -40:, :], np.array([248, 248, 248], dtype=np.float32))
    return np.clip(np.round(base), 0, 255).astype(np.uint8)


# ── 13. Mixed-illumination chart ───────────────────────────────────────

def gen_mixed_illumination_chart(h: int = 256, w: int = 512) -> np.ndarray:
    """Single frame with warm-left and cool-right illumination gradient."""
    x = np.linspace(0.0, 1.0, w, dtype=np.float32)
    warm = np.array([255, 214, 170], dtype=np.float32) / 255.0
    cool = np.array([185, 215, 255], dtype=np.float32) / 255.0
    illum = warm[None, None, :] * (1.0 - x[None, :, None]) + cool[None, None, :] * x[None, :, None]

    scene = np.full((h, w, 3), 0.55, dtype=np.float32)
    # neutral object
    scene[56:200, 180:332, :] = 0.78
    # skin-like patch
    scene[88:188, 66:156, :] = np.array([0.80, 0.64, 0.54], dtype=np.float32)
    # foliage-like patch
    scene[88:188, 356:446, :] = np.array([0.36, 0.55, 0.30], dtype=np.float32)

    out = scene * illum
    return np.clip(np.round(out * 255.0), 0, 255).astype(np.uint8)


def gen_two_axis_neutral_gradient(h: int = 256, w: int = 512) -> np.ndarray:
    """Large-area neutral gradient for banding and tint checks."""
    x = np.linspace(0.0, 1.0, w, dtype=np.float32)
    y = np.linspace(0.0, 1.0, h, dtype=np.float32)
    plane = 255.0 * (0.1 + 0.75 * x[None, :] + 0.15 * y[:, None])
    plane = np.clip(plane, 0, 255).astype(np.uint8)
    return np.stack([plane, plane, plane], axis=-1)


def gen_bin_boundary_triplet_chart(patch_size: int = 36, gap: int = 10) -> np.ndarray:
    """Triplets around each luma node to expose interpolation jumps."""
    nodes = [15, 31, 47, 63, 95, 127, 159, 191, 223, 239, 247, 255]
    values: list[int] = []
    for idx, node in enumerate(nodes):
        values.extend([max(node - 1, 0), node, min(node + 1, 255)])
        if idx != len(nodes) - 1:
            values.append(-1)

    width = len(values) * patch_size
    img = np.zeros((patch_size, width, 3), dtype=np.uint8)
    for i, value in enumerate(values):
        x0 = i * patch_size
        x1 = x0 + patch_size
        if value < 0:
            img[:, x0:x1] = 18
        else:
            img[:, x0:x1] = value

    if gap > 0:
        for i, value in enumerate(values):
            if value < 0:
                x0 = i * patch_size
                img[:, max(x0 - gap // 2, 0):min(x0 + gap // 2, width)] = 12
    return img


def gen_near_node_patch_grid(
    patch_w: int = 28,
    patch_h: int = 36,
    radius: int = 4,
) -> np.ndarray:
    """Compact 12x9 grey patch grid centered on each luma node.

    Each row corresponds to one luma node. Within a row, the 9 patches are
    ``node-4 .. node+4`` with uint8 clipping, matching the compact reference
    chart provided under ``test_images/synthetic/1773196575672.png``.
    """
    nodes = [15, 31, 47, 63, 95, 127, 159, 191, 223, 239, 247, 255]
    offsets = list(range(-radius, radius + 1))
    width = len(offsets) * patch_w
    height = len(nodes) * patch_h
    img = np.zeros((height, width, 3), dtype=np.uint8)

    for row_idx, node in enumerate(nodes):
        y0 = row_idx * patch_h
        for col_idx, offset in enumerate(offsets):
            x0 = col_idx * patch_w
            value = int(np.clip(node + offset, 0, 255))
            img[y0:y0 + patch_h, x0:x0 + patch_w] = value
    return img


def gen_near_node_ramp_chart(patch_size: int = 20, span: int = 8) -> np.ndarray:
    """Local grey ramps centered on each luma node to expose discontinuities."""
    nodes = [15, 31, 47, 63, 95, 127, 159, 191, 223, 239, 247, 255]
    samples = list(range(-span, span + 1))
    step_count = len(samples)
    gap = patch_size
    width = len(nodes) * step_count * patch_size + (len(nodes) - 1) * gap
    height = len(nodes) * patch_size
    img = np.full((height, width, 3), 18, dtype=np.uint8)

    for row, node in enumerate(nodes):
        y0 = row * patch_size
        x = 0
        for group_idx, group_node in enumerate(nodes):
            if group_idx == row:
                for offset in samples:
                    value = int(np.clip(group_node + offset, 0, 255))
                    img[y0:y0 + patch_size, x:x + patch_size] = value
                    x += patch_size
            else:
                neutral = int(np.clip(group_node, 0, 255))
                img[y0:y0 + patch_size, x:x + step_count * patch_size] = neutral
                x += step_count * patch_size
            if group_idx != len(nodes) - 1:
                img[y0:y0 + patch_size, x:x + gap] = 12
                x += gap
    return img


def gen_iso_gray_18_70_pair(h: int = 128, w: int = 512) -> np.ndarray:
    """Paired low/mid and high neutral blocks for luminance-dependent drift checks."""
    img = np.full((h, w, 3), 90, dtype=np.uint8)
    img[:, :w // 4] = 46
    img[:, w // 4:w // 2] = 46
    img[:, w // 2:3 * w // 4] = 179
    img[:, 3 * w // 4:] = 179
    img[:, w // 4 - 6:w // 4 + 6] = 110
    img[:, w // 2 - 6:w // 2 + 6] = 110
    img[:, 3 * w // 4 - 6:3 * w // 4 + 6] = 110
    return img


def gen_midtone_neutral_texture(h: int = 256, w: int = 512) -> np.ndarray:
    """Mid-grey texture field with fine neutral details."""
    img = np.full((h, w, 3), 124, dtype=np.uint8)
    for x in range(0, w, 24):
        img[:, x:x + 2] = 132
    for y in range(0, h, 24):
        img[y:y + 2, :] = 132
    img[48:208, 80:432] = 118
    for y in range(68, 196, 14):
        img[y:y + 2, 112:400] = 160
    for x in range(120, 392, 20):
        img[80:184, x:x + 1] = 145
    return img


def gen_saturation_threshold_ladder(h_per_hue: int = 40, w_per_step: int = 28) -> np.ndarray:
    """Hue strips with gradually increasing saturation near protection thresholds."""
    base_grey = np.array([128, 128, 128], dtype=np.float32)
    hues = np.array(
        [
            [220, 70, 70],
            [220, 180, 70],
            [70, 180, 70],
            [70, 170, 220],
            [90, 90, 220],
            [190, 90, 200],
        ],
        dtype=np.float32,
    )
    steps = np.linspace(0.0, 1.0, 10, dtype=np.float32)
    img = np.zeros((len(hues) * h_per_hue, len(steps) * w_per_step, 3), dtype=np.uint8)
    for row, hue in enumerate(hues):
        for col, t in enumerate(steps):
            patch = np.round(base_grey * (1.0 - t) + hue * t).astype(np.uint8)
            y0 = row * h_per_hue
            x0 = col * w_per_step
            img[y0:y0 + h_per_hue, x0:x0 + w_per_step] = patch
    return img


def gen_skin_tone_luma_strip(patch_size: int = 40) -> np.ndarray:
    """Skin-tone families repeated across luminance levels."""
    bases = np.array(
        [
            [240, 200, 166],
            [220, 175, 140],
            [185, 140, 105],
            [140, 100, 70],
        ],
        dtype=np.float32,
    )
    scales = np.array([0.55, 0.75, 0.95, 1.0], dtype=np.float32)
    img = np.zeros((len(bases) * patch_size, len(scales) * patch_size, 3), dtype=np.uint8)
    for row, base in enumerate(bases):
        for col, scale in enumerate(scales):
            patch = np.clip(np.round(base * scale), 0, 255).astype(np.uint8)
            y0 = row * patch_size
            x0 = col * patch_size
            img[y0:y0 + patch_size, x0:x0 + patch_size] = patch
    return img


def gen_warm_cool_split_field(h: int = 256, w: int = 512) -> np.ndarray:
    """Warm-left, cool-right field with neutral reference blocks."""
    x = np.linspace(0.0, 1.0, w, dtype=np.float32)
    warm = np.array([255, 220, 180], dtype=np.float32) / 255.0
    cool = np.array([190, 220, 255], dtype=np.float32) / 255.0
    illum = warm[None, None, :] * (1.0 - x[None, :, None]) + cool[None, None, :] * x[None, :, None]
    scene = np.full((h, w, 3), 0.62, dtype=np.float32)
    scene[48:208, 96:192] = 0.76
    scene[48:208, 224:320] = 0.76
    scene[48:208, 352:448] = 0.76
    scene[96:160, 224:320] = np.array([0.72, 0.58, 0.48], dtype=np.float32)
    out = scene * illum
    return np.clip(np.round(out * 255.0), 0, 255).astype(np.uint8)


def gen_shadow_with_colored_highlight(h: int = 256, w: int = 512) -> np.ndarray:
    """Dark field with neutral and colored highlights near clip."""
    img = np.full((h, w, 3), [20, 20, 22], dtype=np.uint8)
    img[48:208, 60:150] = [250, 248, 246]
    img[48:208, 210:300] = [252, 235, 214]
    img[48:208, 360:450] = [226, 240, 252]
    img[86:170, 88:122] = [255, 255, 255]
    img[86:170, 238:272] = [255, 245, 228]
    img[86:170, 388:422] = [240, 248, 255]
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
    "11_ui_text_contrast":   gen_ui_text_contrast,
    "12_specular_clip_chart": gen_specular_clip_chart,
    "13_mixed_illumination_chart": gen_mixed_illumination_chart,
    "near_black_steps":      gen_near_black_steps,
    "near_white_steps":      gen_near_white_steps,
    "ui_dark_theme_chart":   gen_ui_dark_theme_chart,
    "rgb_cmy_color_bars":    gen_rgb_cmy_color_bars,
    "two_axis_neutral_gradient": gen_two_axis_neutral_gradient,
    "bin_boundary_triplet_chart": gen_bin_boundary_triplet_chart,
    "near_node_patch_grid":  gen_near_node_patch_grid,
    "near_node_ramp_chart": gen_near_node_ramp_chart,
    "iso_gray_18_70_pair":   gen_iso_gray_18_70_pair,
    "midtone_neutral_texture": gen_midtone_neutral_texture,
    "saturation_threshold_ladder": gen_saturation_threshold_ladder,
    "skin_tone_luma_strip":  gen_skin_tone_luma_strip,
    "warm_cool_split_field": gen_warm_cool_split_field,
    "shadow_with_colored_highlight": gen_shadow_with_colored_highlight,
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
    print("  11  UI text contrast    — 文本/细线/高对比，检查边缘与中性色偏")
    print("  12  Specular clip chart — 高光斑与近白区域，检查 clipping 附近色偏")
    print("  13  Mixed illumination  — 左暖右冷混光，检查白平衡过渡与肤色稳定")


if __name__ == "__main__":
    main()
