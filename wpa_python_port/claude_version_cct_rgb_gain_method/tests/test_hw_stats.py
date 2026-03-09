from __future__ import annotations

from wpa_fixed import FixedWPAConfig
from wpa_fixed.hw_stats import collect_hw_stats


def test_static_lut_size_for_uq18() -> None:
    cfg = FixedWPAConfig(coeff_frac_bits=8)
    s = collect_hw_stats(cfg)["static"]
    assert s["wa_base_gain_lut_entries"] == 3 * 3
    assert s["wa_base_gain_lut_bits"] == 3 * 3 * 9
    assert s["wa_base_gain_lut_bytes"] == 11
    assert s["atten_q_lut_entries"] == 12
    assert s["atten_q_lut_bits"] == 12 * 9
    assert s["atten_q_lut_bytes"] == 14
    assert s["total_bytes"] == 36


def test_static_lut_size_for_uq110() -> None:
    cfg = FixedWPAConfig(coeff_frac_bits=10)
    s = collect_hw_stats(cfg)["static"]
    assert s["wa_base_gain_lut_entries"] == 3 * 3
    assert s["wa_base_gain_lut_bits"] == 3 * 3 * 11
    assert s["wa_base_gain_lut_bytes"] == 13
    assert s["atten_q_lut_entries"] == 12
    assert s["atten_q_lut_bits"] == 12 * 11
    assert s["atten_q_lut_bytes"] == 17
    assert s["total_bytes"] == 41


def test_runtime_variable_widths_default() -> None:
    cfg = FixedWPAConfig(frac_bits=10, coeff_frac_bits=8)
    v = collect_hw_stats(cfg)["variables_bits"]
    assert v["pixel_fix_ch"] == 11
    assert v["gain_fix_ch"] == 9
    assert v["mul_acc_ch"] == 20
    assert v["t_interp"] == 11


def test_sat_ops_are_reported_only_when_enabled() -> None:
    ops0 = collect_hw_stats(FixedWPAConfig(sat_en=False))["ops_per_pixel"]
    ops1 = collect_hw_stats(FixedWPAConfig(sat_en=True))["ops_per_pixel"]
    assert "sat_mul" not in ops0
    assert "sat_mul" in ops1


def test_interp_reports_shift_not_div_for_default_nodes() -> None:
    ops = collect_hw_stats(FixedWPAConfig())["ops_per_pixel"]
    assert ops["interp_div"] == 0
    assert ops["interp_shift"] == 3


def test_interp_reports_div_for_non_pow2_nodes() -> None:
    cfg = FixedWPAConfig(luma_nodes=[0, 10, 21, 255])
    ops = collect_hw_stats(cfg)["ops_per_pixel"]
    assert ops["interp_div"] == 1
    assert ops["interp_shift"] == 2
