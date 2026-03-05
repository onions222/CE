"""Hardware resource statistics for fixed-point WPA pipeline."""

from __future__ import annotations

import argparse

from .config import FixedWPAConfig


def _bytes_from_bits(bits: int) -> int:
    return (bits + 7) // 8


def _interp_uses_generic_divider(luma_nodes: list[int]) -> bool:
    """Return True when luma interpolation cannot be reduced to shifts.

    For the default node set, all adjacent spans are {8,16,32} so the
    interpolation denominator is power-of-two and can be implemented as
    right-shifts (no generic divider).
    """
    if len(luma_nodes) < 2:
        return False
    spans = [int(luma_nodes[i + 1]) - int(luma_nodes[i]) for i in range(len(luma_nodes) - 1)]
    if any(s <= 0 for s in spans):
        return True
    return any((s & (s - 1)) != 0 for s in spans)


def collect_hw_stats(cfg: FixedWPAConfig) -> dict:
    """Collect static and dynamic hardware-cost stats for current config."""
    coeff_w = cfg.coeff_frac_bits + 1  # UQ1.F
    pix_w = cfg.frac_bits + 1          # Q0.F over [0,1]
    t_w = cfg.frac_bits + 1            # interpolation fraction t
    mul_w = pix_w + coeff_w

    n_wa = 3
    n_bins = len(cfg.luma_nodes)
    n_ch = 3

    base_lut_entries = n_wa * n_ch
    base_lut_bits = base_lut_entries * coeff_w
    atten_entries = n_bins
    atten_bits = atten_entries * coeff_w
    runtime_bin_entries = n_bins * n_ch
    runtime_bin_bits = runtime_bin_entries * coeff_w
    luma_nodes_bits = n_bins * 8

    interp_has_divider = _interp_uses_generic_divider(cfg.luma_nodes)

    # Per-pixel dynamic ops for current runtime path.
    # Counts are scalar ops; RGB vector ops are expanded per channel.
    ops = {
        "luma_add": 2,
        "luma_shift": 2,  # (2*G) + final >>2
        "interp_comp": max(n_bins - 1, 0),  # interval search comparisons
        "interp_add": 8,   # t: +1, per-channel interp: +6, total +1
        "interp_sub": 5,   # y-node_lo, node_hi-node_lo, plus 3 channel deltas
        "interp_mul": 3,   # t * delta for RGB
        "interp_div": 1 if interp_has_divider else 0,  # numer // span or shift
        "interp_shift": 2 if interp_has_divider else 3,  # +1 when span is power-of-two
        "apply_mul": 3,    # pixel * gain for RGB
        "apply_add": 3,    # + rounding bias
        "apply_shift": 3,  # >> coeff_frac_bits
        "clip": 1,         # adjusted clip to [0, ONE]
    }
    if cfg.sat_en:
        ops.update(
            {
                "sat_abs": 3,
                "sat_add": 7,    # s sums + weight numer + per-ch blending adds
                "sat_sub": 5,    # channel diffs + s1-s + per-ch delta
                "sat_mul": 3,    # w * delta for RGB
                "sat_div": 1,    # weight division
                "sat_shift": 4,  # weight <<F and per-ch >>F
            }
        )

    variables = {
        "wa_sel": 7,
        "luma_u8": 8,
        "idx_lo_hi": 4,
        "t_interp": t_w,
        "pixel_fix_ch": pix_w,
        "gain_fix_ch": coeff_w,
        "mul_acc_ch": mul_w,
        "adjusted_fix_ch": pix_w,
        "sat_weight": pix_w,
        "sat_proxy_s": 10,  # |R-G|+|G-B|+|B-R| in [0,510]
    }

    return {
        "config": {
            "frac_bits": cfg.frac_bits,
            "coeff_frac_bits": cfg.coeff_frac_bits,
            "bin_count": n_bins,
            "sat_en": cfg.sat_en,
        },
        "static": {
            "wa_base_gain_lut_entries": base_lut_entries,
            "wa_base_gain_lut_bits": base_lut_bits,
            "wa_base_gain_lut_bytes": _bytes_from_bits(base_lut_bits),
            "atten_q_lut_entries": atten_entries,
            "atten_q_lut_bits": atten_bits,
            "atten_q_lut_bytes": _bytes_from_bits(atten_bits),
            "luma_nodes_bits": luma_nodes_bits,
            "luma_nodes_bytes": _bytes_from_bits(luma_nodes_bits),
            "total_bits": base_lut_bits + atten_bits + luma_nodes_bits,
            "total_bytes": _bytes_from_bits(base_lut_bits + atten_bits + luma_nodes_bits),
        },
        "ops_per_wa_update": {
            "wa_interp_mul": 3,
            "wa_interp_add": 3,
            "wa_interp_shift": 3,  # /64 via >>6
            "bin_expand_mul": n_bins * n_ch,
            "bin_expand_add": n_bins * n_ch,
            "bin_expand_shift": n_bins * n_ch,
        },
        "runtime_buffers": {
            "wa_runtime_bin_gain_entries": runtime_bin_entries,
            "wa_runtime_bin_gain_bits": runtime_bin_bits,
            "wa_runtime_bin_gain_bytes": _bytes_from_bits(runtime_bin_bits),
        },
        "variables_bits": variables,
        "ops_per_pixel": ops,
    }


def format_hw_stats_markdown(stats: dict) -> str:
    cfg = stats["config"]
    st = stats["static"]
    rt = stats["runtime_buffers"]
    wu = stats["ops_per_wa_update"]
    var = stats["variables_bits"]
    ops = stats["ops_per_pixel"]

    lines = []
    lines.append("## Hardware Stats")
    lines.append("")
    lines.append(
        f"- config: `frac_bits={cfg['frac_bits']}`, "
        f"`coeff_frac_bits={cfg['coeff_frac_bits']}`, "
        f"`bins={cfg['bin_count']}`, `sat_en={cfg['sat_en']}`"
    )
    lines.append("")
    lines.append("| Item | Value |")
    lines.append("|---|---:|")
    lines.append(f"| WA base gain LUT entries | {st['wa_base_gain_lut_entries']} |")
    lines.append(f"| WA base gain LUT bits | {st['wa_base_gain_lut_bits']} |")
    lines.append(f"| WA base gain LUT bytes | {st['wa_base_gain_lut_bytes']} |")
    lines.append(f"| Atten LUT entries | {st['atten_q_lut_entries']} |")
    lines.append(f"| Atten LUT bits | {st['atten_q_lut_bits']} |")
    lines.append(f"| Atten LUT bytes | {st['atten_q_lut_bytes']} |")
    lines.append(f"| Luma nodes bits | {st['luma_nodes_bits']} |")
    lines.append(f"| Luma nodes bytes | {st['luma_nodes_bytes']} |")
    lines.append(f"| Total static bits | {st['total_bits']} |")
    lines.append(f"| Total static bytes | {st['total_bytes']} |")
    lines.append(f"| Runtime bin gain entries (per WA update) | {rt['wa_runtime_bin_gain_entries']} |")
    lines.append(f"| Runtime bin gain bits (scratch) | {rt['wa_runtime_bin_gain_bits']} |")
    lines.append(f"| Runtime bin gain bytes (scratch) | {rt['wa_runtime_bin_gain_bytes']} |")
    lines.append("")
    lines.append("| Runtime Variable | Bits |")
    lines.append("|---|---:|")
    for k, v in var.items():
        lines.append(f"| `{k}` | {v} |")
    lines.append("")
    lines.append("| Ops / Pixel | Count |")
    lines.append("|---|---:|")
    for k, v in ops.items():
        lines.append(f"| `{k}` | {v} |")
    lines.append("")
    lines.append("| Ops / WA Update | Count |")
    lines.append("|---|---:|")
    for k, v in wu.items():
        lines.append(f"| `{k}` | {v} |")
    lines.append("")
    return "\n".join(lines)


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print fixed WPA hardware resource stats.")
    parser.add_argument("--frac-bits", type=int, default=10)
    parser.add_argument("--coeff-frac-bits", type=int, default=8, choices=[8, 10])
    parser.add_argument("--sat-en", type=int, default=0, choices=[0, 1])
    args = parser.parse_args(argv)

    cfg = FixedWPAConfig(
        frac_bits=args.frac_bits,
        coeff_frac_bits=args.coeff_frac_bits,
        sat_en=bool(args.sat_en),
    )
    stats = collect_hw_stats(cfg)
    print(format_hw_stats_markdown(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
