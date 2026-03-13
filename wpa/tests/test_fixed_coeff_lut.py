from __future__ import annotations

import numpy as np

from wpa_fixed import FixedWPAConfig, wpa_fixed_process


def _make_gradient_image(h: int = 8, w: int = 64) -> np.ndarray:
    row = np.linspace(0, 255, w, dtype=np.uint8)
    plane = np.tile(row, (h, 1))
    return np.stack([plane, plane, plane], axis=-1)


def _mean_rgb(img: np.ndarray) -> np.ndarray:
    return img.reshape(-1, 3).mean(axis=0)


def test_wa_gain_lut_shape_and_identity_anchor() -> None:
    cfg = FixedWPAConfig(coeff_frac_bits=8)
    assert cfg.wa_base_gain_lut_fixed.shape == (3, 3)
    np.testing.assert_array_equal(cfg.wa_base_gain_lut_fixed[1], np.full((3,), 1 << 8))
    np.testing.assert_array_equal(cfg.runtime_base_gain_fixed(0), cfg.wa_base_gain_lut_fixed[0])
    np.testing.assert_array_equal(cfg.runtime_base_gain_fixed(64), cfg.wa_base_gain_lut_fixed[1])
    cool_expected = cfg.wa_base_gain_lut_fixed[1] + (
        ((63 * (cfg.wa_base_gain_lut_fixed[2] - cfg.wa_base_gain_lut_fixed[1])) + 32) >> 6
    )
    np.testing.assert_array_equal(cfg.runtime_base_gain_fixed(127), cool_expected)
    gains = cfg.runtime_bin_gains_fixed(64)
    assert gains.shape == (12, 3)
    np.testing.assert_array_equal(gains, np.full((12, 3), 1 << 8))


def test_wa_gain_lut_supports_10bit_coeff() -> None:
    cfg = FixedWPAConfig(coeff_frac_bits=10)
    assert cfg.wa_base_gain_lut_fixed.shape == (3, 3)
    np.testing.assert_array_equal(cfg.wa_base_gain_lut_fixed[1], np.full((3,), 1 << 10))
    gains = cfg.runtime_bin_gains_fixed(64)
    assert gains.shape == (12, 3)
    np.testing.assert_array_equal(gains, np.full((12, 3), 1 << 10))


def test_fixed_lut_respects_warm_cct_override() -> None:
    cfg_3000 = FixedWPAConfig(coeff_frac_bits=10, cct_warm_k=3000.0)
    cfg_4500 = FixedWPAConfig(coeff_frac_bits=10, cct_warm_k=4500.0)
    warm_3000 = cfg_3000.wa_base_gain_lut_fixed[0]
    warm_4500 = cfg_4500.wa_base_gain_lut_fixed[0]
    assert warm_3000[0] >= warm_4500[0]
    assert warm_3000[2] <= warm_4500[2]


def test_fixed_pipeline_warm_cool_direction_for_8bit_coeff() -> None:
    img = _make_gradient_image()
    out_warm = wpa_fixed_process(img, FixedWPAConfig(wa_sel=0, coeff_frac_bits=8))
    out_cool = wpa_fixed_process(img, FixedWPAConfig(wa_sel=127, coeff_frac_bits=8))

    m_warm = _mean_rgb(out_warm)
    m_cool = _mean_rgb(out_cool)

    assert m_warm[0] > m_warm[2], "warm should bias R over B"
    assert m_cool[2] > m_cool[0], "cool should bias B over R"


def test_fixed_cool_highlight_bins_reduce_green_slightly_below_identity() -> None:
    cfg = FixedWPAConfig(wa_sel=127, coeff_frac_bits=8)
    gains = cfg.runtime_bin_gains_fixed(127)
    green_cap = int(round(0.98 * (1 << cfg.coeff_frac_bits)))
    for row in gains[-4:]:
        assert int(row[1]) <= green_cap


def test_fixed_warm_highlight_bins_soften_green_for_pale_yellow() -> None:
    cfg = FixedWPAConfig(wa_sel=0, coeff_frac_bits=8)
    gains = cfg.runtime_bin_gains_fixed(0)
    green_cap = int(round(0.94 * (1 << cfg.coeff_frac_bits)))
    for row in gains[-3:]:
        assert int(row[1]) <= green_cap


def test_fixed_warm_highlight_bins_taper_red_and_blue_toward_cleaner_warm_white() -> None:
    cfg = FixedWPAConfig(wa_sel=0, coeff_frac_bits=8)
    gains = cfg.runtime_bin_gains_fixed(0)
    tail = gains[-4:].astype(np.int32)
    red = [int(row[0]) for row in tail]
    blue = [int(row[2]) for row in tail]
    chroma_span = [int(row.max() - row.min()) for row in tail]

    assert red[0] > red[1] > red[2] > red[3]
    assert blue[0] < blue[1] < blue[2] < blue[3]
    assert chroma_span[0] > chroma_span[1] > chroma_span[2] > chroma_span[3]


def test_fixed_cool_last_highlight_bin_softens_blue_to_reduce_cyan() -> None:
    cfg = FixedWPAConfig(wa_sel=127, coeff_frac_bits=8)
    gains = cfg.runtime_bin_gains_fixed(127)
    blue_cap = int(round(1.10 * (1 << cfg.coeff_frac_bits)))
    assert int(gains[-1][2]) <= blue_cap


def test_fixed_cool_highlight_bins_taper_green_and_blue_toward_white() -> None:
    cfg = FixedWPAConfig(wa_sel=127, coeff_frac_bits=8)
    gains = cfg.runtime_bin_gains_fixed(127)
    green_caps = [0.98, 0.975, 0.97, 0.965]
    blue_caps = [1.15, 1.12, 1.08, 1.06]
    for row, g_cap, b_cap in zip(gains[-4:], green_caps, blue_caps):
        assert int(row[1]) <= int(round(g_cap * (1 << cfg.coeff_frac_bits)))
        assert int(row[2]) <= int(round(b_cap * (1 << cfg.coeff_frac_bits)))


def test_10bit_coeff_not_worse_than_8bit_vs_float_reference() -> None:
    img = _make_gradient_image(h=16, w=128)

    # Fixed pipeline with 8-bit and 10-bit coefficients.
    cfg8 = FixedWPAConfig(wa_sel=96, coeff_frac_bits=8)
    cfg10 = FixedWPAConfig(wa_sel=96, coeff_frac_bits=10)
    out8 = wpa_fixed_process(img, cfg8).astype(np.int16)
    out10 = wpa_fixed_process(img, cfg10).astype(np.int16)

    # Float reference from existing pipeline.
    from wpa.config import WPAConfig
    from wpa.core import wpa_process_rgb_uint8

    ref = wpa_process_rgb_uint8(img, WPAConfig(wa_sel=96)).astype(np.int16)

    mae8 = np.mean(np.abs(out8 - ref))
    mae10 = np.mean(np.abs(out10 - ref))
    assert mae10 <= mae8 + 1e-6


def test_fixed_warm_near_white_tail_returns_to_lower_chroma_warm_white() -> None:
    vals = [239, 247, 255]
    img = np.array([[[v, v, v] for v in vals]], dtype=np.uint8)

    out = wpa_fixed_process(img, FixedWPAConfig(wa_sel=0, coeff_frac_bits=8)).astype(np.int32)[0]
    spans = [int(rgb.max() - rgb.min()) for rgb in out]

    assert spans[0] > spans[1] > spans[2]
    assert spans[-1] <= 18
    assert int(out[-1, 1]) - int(out[-1, 2]) <= 10


def test_low_luma_gate_bypasses_dark_blue_shoulder_pixel_in_linear_domain() -> None:
    rgb = np.array([[[16, 18, 22]]], dtype=np.uint8)

    out = wpa_fixed_process(rgb, FixedWPAConfig(wa_sel=0, coeff_frac_bits=8, frac_bits=8))

    np.testing.assert_array_equal(out[0, 0], np.array([13, 22, 22], dtype=np.uint8))


def test_low_luma_gate_blends_smoothly_between_31_and_63() -> None:
    rgb = np.array([[[48, 48, 48]]], dtype=np.uint8)
    cfg = FixedWPAConfig(wa_sel=0, coeff_frac_bits=8, frac_bits=8)

    out = wpa_fixed_process(rgb, cfg)

    np.testing.assert_array_equal(out[0, 0], np.array([56, 49, 46], dtype=np.uint8))


def test_low_luma_gate_preserves_existing_behavior_above_transition_window() -> None:
    rgb = np.array([[[80, 80, 80]]], dtype=np.uint8)

    out = wpa_fixed_process(rgb, FixedWPAConfig(wa_sel=0, coeff_frac_bits=8, frac_bits=8))

    np.testing.assert_array_equal(out[0, 0], np.array([99, 77, 64], dtype=np.uint8))
