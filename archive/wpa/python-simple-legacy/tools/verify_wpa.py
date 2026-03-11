#!/usr/bin/env python3
"""Reproducible WPA verification with quantitative diagnostics."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from wpa.config import WPAConfig
from wpa.core import _wa_sel_to_side_alpha, wpa_process_rgb_uint8
from wpa.weights import saturation_proxy, saturation_weight


WA_SET = [0, 20, 40, 63, 64, 65, 90, 110, 127]
GRAY_Y = [0, 32, 64, 96, 128, 160, 192, 224, 255]
PURE_COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
]
NEAR_NEUTRAL = [(200, 195, 190), (180, 180, 170), (230, 225, 220)]


@dataclass
class Patch:
    kind: str
    label: str
    rgb: tuple[int, int, int]
    x0: int
    x1: int


def make_synthetic(patch_size: int = 64) -> tuple[np.ndarray, list[Patch]]:
    patches: list[np.ndarray] = []
    meta: list[Patch] = []

    x = 0
    for y in GRAY_Y:
        p = np.full((patch_size, patch_size, 3), y, dtype=np.uint8)
        patches.append(p)
        meta.append(Patch("gray", f"gray_{y}", (y, y, y), x, x + patch_size))
        x += patch_size

    for rgb in PURE_COLORS:
        p = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
        p[:] = rgb
        patches.append(p)
        meta.append(Patch("pure", f"pure_{rgb[0]}_{rgb[1]}_{rgb[2]}", rgb, x, x + patch_size))
        x += patch_size

    for rgb in NEAR_NEUTRAL:
        p = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
        p[:] = rgb
        patches.append(p)
        meta.append(Patch("near", f"near_{rgb[0]}_{rgb[1]}_{rgb[2]}", rgb, x, x + patch_size))
        x += patch_size

    return np.concatenate(patches, axis=1), meta


def _base_cfg_for_preset(preset: str, wa_sel: int) -> WPAConfig:
    if preset == "legacy":
        return WPAConfig(
            WA_EN=True,
            WA_SEL=wa_sel,
            sat_s0=0.05,
            sat_s1=0.75,
            warm_gain_global=(1.06, 1.00, 0.92),
            cool_gain_global=(0.94, 1.00, 1.06),
            warm_strength=1.0,
            cool_strength=1.0,
        )
    return WPAConfig(WA_EN=True, WA_SEL=wa_sel)


def _with_optional_adaptive_sat(img: np.ndarray, cfg: WPAConfig, adaptive_sat: bool) -> WPAConfig:
    if not adaptive_sat:
        return cfg
    gamma = img.astype(np.float32) / 255.0
    s = saturation_proxy(gamma)
    s0 = float(np.percentile(s, 10.0))
    s1 = float(np.percentile(s, 60.0))
    if s1 <= s0:
        s1 = s0 + 1e-4
    cfg.sat_s0 = s0
    cfg.sat_s1 = s1
    return cfg


def _clip_ratio(arr_u8: np.ndarray) -> tuple[float, float, float]:
    clip = (arr_u8 == 0) | (arr_u8 == 255)
    return (
        float(np.mean(clip[..., 0])),
        float(np.mean(clip[..., 1])),
        float(np.mean(clip[..., 2])),
    )


def _w_stats(img_u8: np.ndarray, cfg: WPAConfig) -> dict[str, float]:
    src = img_u8.astype(np.float32) / 255.0
    s = saturation_proxy(src)
    w = saturation_weight(s, cfg.sat_s0, cfg.sat_s1)
    return {
        "mean": float(np.mean(w)),
        "p10": float(np.percentile(w, 10)),
        "p50": float(np.percentile(w, 50)),
        "p90": float(np.percentile(w, 90)),
    }


def _gray_metrics(out: np.ndarray, meta: Iterable[Patch]) -> tuple[list[dict[str, float]], float]:
    rows: list[dict[str, float]] = []
    for p in meta:
        if p.kind != "gray":
            continue
        patch = out[:, p.x0 : p.x1, :].astype(np.int32)
        rg = float(np.mean(patch[..., 0] - patch[..., 1]))
        gb = float(np.mean(patch[..., 1] - patch[..., 2]))
        rb = float(np.mean(patch[..., 0] - patch[..., 2]))
        rows.append({"y": p.rgb[0], "rg": rg, "gb": gb, "rb": rb})

    rg = np.array([r["rg"] for r in rows], dtype=np.float32)
    gb = np.array([r["gb"] for r in rows], dtype=np.float32)
    rb = np.array([r["rb"] for r in rows], dtype=np.float32)
    spike = 0.0
    if rg.size > 1:
        spike = float(max(np.max(np.abs(np.diff(rg))), np.max(np.abs(np.diff(gb))), np.max(np.abs(np.diff(rb)))))
    return rows, spike


def _pure_l1(img: np.ndarray, out: np.ndarray, meta: Iterable[Patch]) -> dict[str, float]:
    vals: list[float] = []
    for p in meta:
        if p.kind != "pure":
            continue
        src = img[:, p.x0 : p.x1, :].astype(np.int32)
        dst = out[:, p.x0 : p.x1, :].astype(np.int32)
        vals.append(float(np.mean(np.abs(dst - src).sum(axis=-1))))
    return {
        "mean": float(np.mean(vals)),
        "max": float(np.max(vals)) if vals else 0.0,
    }


def _gray_delta_at(img: np.ndarray, out: np.ndarray, meta: Iterable[Patch], y: int) -> float:
    for p in meta:
        if p.kind == "gray" and p.rgb[0] == y:
            src = img[:, p.x0 : p.x1, :].astype(np.int32)
            dst = out[:, p.x0 : p.x1, :].astype(np.int32)
            return float(np.mean(np.abs(dst - src).sum(axis=-1)))
    return 0.0


def _high_gray_clip_delta(
    out_identity: np.ndarray,
    out_side: np.ndarray,
    meta: Iterable[Patch],
    y_values: set[int],
) -> tuple[float, float, float]:
    idx = [(p.x0, p.x1) for p in meta if p.kind == "gray" and p.rgb[0] in y_values]
    if not idx:
        return (0.0, 0.0, 0.0)
    src = np.concatenate([out_identity[:, x0:x1, :] for x0, x1 in idx], axis=1)
    dst = np.concatenate([out_side[:, x0:x1, :] for x0, x1 in idx], axis=1)
    c0 = np.array(_clip_ratio(src))
    c1 = np.array(_clip_ratio(dst))
    d = c1 - c0
    return float(d[0]), float(d[1]), float(d[2])


def run_report(preset: str, adaptive_sat: bool, wa_mode: str) -> None:
    img, meta = make_synthetic(64)
    print("=" * 80)
    print(f"Preset: {preset} | adaptive_sat={adaptive_sat} | wa_mode={wa_mode}")

    cfg_id = _base_cfg_for_preset(preset, 64)
    if preset == "current":
        cfg_id.wa_mode = wa_mode
    out_identity = wpa_process_rgb_uint8(img, _with_optional_adaptive_sat(img, cfg_id, adaptive_sat))
    baseline_w_cfg = _with_optional_adaptive_sat(img, cfg_id, adaptive_sat)

    for wa_sel in WA_SET:
        side, a = _wa_sel_to_side_alpha(wa_sel)
        cfg = _base_cfg_for_preset(preset, wa_sel)
        if preset == "current":
            cfg.wa_mode = wa_mode
        cfg = _with_optional_adaptive_sat(img, cfg, adaptive_sat)
        out = wpa_process_rgb_uint8(img, cfg)

        wstats = _w_stats(img, cfg)
        gray_rows, gray_spike = _gray_metrics(out, meta)
        pure = _pure_l1(img, out, meta)
        clip = _clip_ratio(out)
        delta_gray128 = _gray_delta_at(img, out, meta, 128)
        hi_clip_d = _high_gray_clip_delta(out_identity, out, meta, {224, 255})

        print(
            f"WA_SEL={wa_sel:3d} side={side:8s} a_factor={a:.4f} "
            f"w(mean/p10/p50/p90)={wstats['mean']:.3f}/{wstats['p10']:.3f}/{wstats['p50']:.3f}/{wstats['p90']:.3f} "
            f"clip_ratio(R/G/B)={clip[0]:.3f}/{clip[1]:.3f}/{clip[2]:.3f} "
            f"pure_l1(mean/max)={pure['mean']:.3f}/{pure['max']:.3f} gray_spike={gray_spike:.3f} gray128_l1={delta_gray128:.3f}"
        )
        gray_line = " ".join(
            [f"Y{int(r['y'])}:RG={r['rg']:.1f},GB={r['gb']:.1f},RB={r['rb']:.1f}" for r in gray_rows]
        )
        print(f"  gray_diffs: {gray_line}")
        print(f"  high_gray_clip_delta(R/G/B): {hi_clip_d[0]:+.4f}/{hi_clip_d[1]:+.4f}/{hi_clip_d[2]:+.4f}")

    print("-" * 80)
    cool_cfg = _base_cfg_for_preset(preset, 110)
    warm_cfg = _base_cfg_for_preset(preset, 20)
    if preset == "current":
        cool_cfg.wa_mode = wa_mode
        warm_cfg.wa_mode = wa_mode
    cool_cfg = _with_optional_adaptive_sat(img, cool_cfg, adaptive_sat)
    warm_cfg = _with_optional_adaptive_sat(img, warm_cfg, adaptive_sat)
    out_cool = wpa_process_rgb_uint8(img, cool_cfg)
    out_warm = wpa_process_rgb_uint8(img, warm_cfg)
    cool_delta = _gray_delta_at(img, out_cool, meta, 128)
    warm_delta = _gray_delta_at(img, out_warm, meta, 128)
    cool_a = _wa_sel_to_side_alpha(110)[1]

    wstats = _w_stats(img, baseline_w_cfg)
    if cool_a <= 0.0:
        cool_reason = "cool ineffective due to a_factor == 0."
    elif cool_delta < 1.0 and wstats["mean"] < 0.2:
        cool_reason = "cool looks weak mainly because saturation weight is too low overall."
    elif cool_delta < 1.0:
        cool_reason = "cool looks weak mainly because gains are too conservative around mid-gray."
    else:
        cool_reason = "cool is effective (a_factor>0 and measurable gray delta)."

    if warm_delta < 1.0 and wstats["mean"] < 0.2:
        warm_reason = "warm weak mainly because saturation weight is too low."
    elif warm_delta < 1.0:
        warm_reason = "warm weak mainly because gain endpoints are conservative."
    else:
        warm_reason = "warm is effective with measurable gray delta."

    print(f"Diagnosis(cool): {cool_reason}")
    print(f"Diagnosis(warm): {warm_reason}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify WPA behavior with synthetic patterns.")
    parser.add_argument(
        "--preset",
        choices=["current", "legacy", "both"],
        default="both",
        help="current=patched defaults, legacy=pre-fix defaults, both=compare both",
    )
    parser.add_argument(
        "--adaptive-sat",
        action="store_true",
        help="Use adaptive sat thresholds (s0=p10, s1=p60) for this offline verification only.",
    )
    parser.add_argument(
        "--wa-mode",
        choices=["diag_rgb", "kelvin_ycocg"],
        default="diag_rgb",
        help="WPA model to verify.",
    )
    args = parser.parse_args()

    if args.preset in ("legacy", "both"):
        run_report("legacy", args.adaptive_sat, "diag_rgb")
    if args.preset in ("current", "both"):
        run_report("current", args.adaptive_sat, args.wa_mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
