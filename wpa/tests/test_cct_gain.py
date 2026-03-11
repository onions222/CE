"""Tests for CCT-driven white-point RGB gain generation."""

from __future__ import annotations

import numpy as np

from wpa.config import (
    COOL_GAIN_GLOBAL,
    WARM_GAIN_GLOBAL,
    build_cct_gain_lut,
    cct_to_xy_approx,
    generate_default_bin_gains,
    map_sat_threshold_gamma_to_linear,
    wa_sel_to_cct,
)


def test_wa_sel_to_cct_anchors() -> None:
    assert wa_sel_to_cct(0) == 3000.0
    assert wa_sel_to_cct(64) == 6500.0
    assert wa_sel_to_cct(127) == 9300.0


def test_wa_sel_to_cct_monotonic() -> None:
    vals = np.array([wa_sel_to_cct(i) for i in range(128)], dtype=np.float64)
    assert np.all(np.diff(vals) >= 0.0)


def test_cct_gain_lut_shape() -> None:
    lut = build_cct_gain_lut()
    assert lut.shape == (128, 3)


def test_cct_gain_lut_identity_at_64() -> None:
    lut = build_cct_gain_lut()
    np.testing.assert_allclose(lut[64], np.array([1.0, 1.0, 1.0]), atol=1e-6)


def test_cct_gain_lut_direction() -> None:
    lut = build_cct_gain_lut()
    warm = lut[0]
    cool = lut[127]
    assert warm[0] > warm[2], f"warm endpoint expected R>B, got {warm}"
    assert cool[2] > cool[0], f"cool endpoint expected B>R, got {cool}"


def test_cool_endpoint_green_does_not_exceed_identity() -> None:
    lut = build_cct_gain_lut()
    cool = lut[127]
    assert cool[1] <= 1.0


def test_cool_highlight_bin_green_does_not_exceed_identity() -> None:
    cool_bins = generate_default_bin_gains(COOL_GAIN_GLOBAL)
    for row in cool_bins[-4:]:
        assert row[1] <= 0.98


def test_warm_highlight_bin_green_is_softened_for_pale_yellow() -> None:
    warm_bins = generate_default_bin_gains(WARM_GAIN_GLOBAL)
    for row in warm_bins[-3:]:
        assert row[1] <= 0.94


def test_warm_highlight_bins_taper_red_and_blue_toward_cleaner_warm_white() -> None:
    warm_bins = generate_default_bin_gains(WARM_GAIN_GLOBAL)
    tail = warm_bins[-4:]
    red = [float(row[0]) for row in tail]
    blue = [float(row[2]) for row in tail]
    chroma_span = [float(np.max(row) - np.min(row)) for row in tail]

    assert red[0] > red[1] > red[2] > red[3]
    assert blue[0] < blue[1] < blue[2] < blue[3]
    assert chroma_span[0] > chroma_span[1] > chroma_span[2] > chroma_span[3]


def test_cool_last_highlight_bin_blue_is_softened_to_reduce_cyan() -> None:
    cool_bins = generate_default_bin_gains(COOL_GAIN_GLOBAL)
    assert cool_bins[-1][2] <= 1.10


def test_cool_highlight_bins_taper_green_and_blue_toward_white() -> None:
    cool_bins = generate_default_bin_gains(COOL_GAIN_GLOBAL)
    green_caps = [0.98, 0.975, 0.97, 0.965]
    blue_caps = [1.15, 1.12, 1.08, 1.06]
    for row, g_cap, b_cap in zip(cool_bins[-4:], green_caps, blue_caps):
        assert row[1] <= g_cap
        assert row[2] <= b_cap


def test_cct_gain_lut_bounds() -> None:
    lut = build_cct_gain_lut()
    assert np.all(np.isfinite(lut))
    assert np.all(lut > 0.0)


def test_sat_threshold_mapping_linear_domain_not_div255() -> None:
    mapped = map_sat_threshold_gamma_to_linear(100.0)
    old = 100.0 / 255.0
    assert 0.0 < mapped < 2.0
    assert abs(mapped - old) > 0.05


def test_sat_threshold_mapping_endpoints() -> None:
    assert map_sat_threshold_gamma_to_linear(0.0) == 0.0
    assert map_sat_threshold_gamma_to_linear(510.0) == 2.0


def test_cct_xy_continuity_around_4000k() -> None:
    x1, y1 = cct_to_xy_approx(3999.9)
    x2, y2 = cct_to_xy_approx(4000.1)
    assert abs(x2 - x1) < 2e-5
    assert abs(y2 - y1) < 2e-5


def test_cct_xy_continuity_around_custom_split() -> None:
    x1, y1 = cct_to_xy_approx(4999.9, split_k=5000.0, blend_half_width_k=120.0)
    x2, y2 = cct_to_xy_approx(5000.1, split_k=5000.0, blend_half_width_k=120.0)
    assert abs(x2 - x1) < 2e-5
    assert abs(y2 - y1) < 2e-5
