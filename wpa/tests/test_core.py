"""
Tests for WPA core processing pipeline.

Covers:
- Identity pass-through (WA_EN=False, WA_SEL=64)
- Gamma roundtrip accuracy (sRGB degamma→engamma ≤ 1 LSB)
- Saturation protection effectiveness
"""

from __future__ import annotations

import numpy as np
import pytest

from wpa.config import WPAConfig
from wpa.core import wpa_process_rgb_uint8, _wa_sel_to_alpha
from wpa.gamma import srgb_degamma, srgb_engamma


# ── helpers ──────────────────────────────────────────────────────────────

def _make_gradient_image(h: int = 16, w: int = 256) -> np.ndarray:
    """Create a horizontal gradient image (R=G=B = col index)."""
    row = np.arange(w, dtype=np.uint8)
    plane = np.tile(row, (h, 1))
    return np.stack([plane, plane, plane], axis=-1)


def _make_solid_image(r: int, g: int, b: int,
                      h: int = 4, w: int = 4) -> np.ndarray:
    """Create a small solid-colour image."""
    img = np.empty((h, w, 3), dtype=np.uint8)
    img[..., 0] = r
    img[..., 1] = g
    img[..., 2] = b
    return img


# ── WA_SEL → alpha mapping ──────────────────────────────────────────────

class TestWaSelMapping:
    """Verify WA_SEL → (side, alpha) conversion (DDIC 3-segment rule)."""

    def test_identity(self):
        side, a = _wa_sel_to_alpha(64)
        assert side == "identity"
        assert a == 0.0

    def test_warm_max(self):
        side, a = _wa_sel_to_alpha(0)
        assert side == "warm"
        assert a == pytest.approx(1.0)

    def test_cool_max(self):
        side, a = _wa_sel_to_alpha(127)
        assert side == "cool"
        assert a == pytest.approx(1.0)

    def test_warm_mid(self):
        side, a = _wa_sel_to_alpha(32)
        assert side == "warm"
        assert a == pytest.approx(0.5)

    def test_cool_mid(self):
        """WA_SEL = 64 + 63/2 ≈ 95 → half cool."""
        side, a = _wa_sel_to_alpha(95)
        assert side == "cool"
        assert a == pytest.approx((95 - 64) / 63.0)


# ── Identity tests ───────────────────────────────────────────────────────

class TestIdentity:
    """WA_EN=False or WA_SEL=64 → output == input (bit-exact)."""

    def test_disabled(self):
        img = _make_gradient_image()
        cfg = WPAConfig(wa_en=False, wa_sel=20)
        out = wpa_process_rgb_uint8(img, cfg)
        np.testing.assert_array_equal(out, img)

    def test_sel_64(self):
        img = _make_gradient_image()
        cfg = WPAConfig(wa_en=True, wa_sel=64)
        out = wpa_process_rgb_uint8(img, cfg)
        np.testing.assert_array_equal(out, img)

    def test_solid_grey_disabled(self):
        img = _make_solid_image(128, 128, 128)
        cfg = WPAConfig(wa_en=False)
        out = wpa_process_rgb_uint8(img, cfg)
        np.testing.assert_array_equal(out, img)


# ── Gamma roundtrip ─────────────────────────────────────────────────────

class TestGammaRoundtrip:
    """sRGB degamma → engamma should be ≤ 1 LSB error for all 0..255."""

    def test_srgb_roundtrip_formula(self):
        codes = np.arange(256, dtype=np.float64) / 255.0
        linear = srgb_degamma(codes)
        encoded = srgb_engamma(linear)
        recovered = np.round(encoded * 255.0).astype(np.int32)
        original = np.arange(256, dtype=np.int32)
        max_err = np.max(np.abs(recovered - original))
        assert max_err <= 1, f"sRGB roundtrip max error = {max_err} LSB"

    def test_srgb_roundtrip_image(self):
        """Process a gradient with gamma_mode='none' and ensure roundtrip via
        manual degamma/engamma is ≤ 1 LSB."""
        from wpa.gamma import degamma, engamma
        img = _make_gradient_image()
        linear = degamma(img, "srgb", is_uint8=True)
        encoded = engamma(linear, "srgb")
        recovered = np.clip(np.round(encoded * 255.0), 0, 255).astype(np.uint8)
        diff = np.abs(img.astype(np.int16) - recovered.astype(np.int16))
        assert diff.max() <= 1

    def test_srgb_roundtrip_lut(self):
        """LUT-based degamma → engamma roundtrip ≤ 1 LSB."""
        from wpa.gamma import srgb_degamma_lut, srgb_engamma_lut
        img = _make_gradient_image()
        linear = srgb_degamma_lut(img)
        recovered = srgb_engamma_lut(linear)
        diff = np.abs(img.astype(np.int16) - recovered.astype(np.int16))
        assert diff.max() <= 1, f"LUT roundtrip max error = {diff.max()} LSB"


# ── Saturation protection ───────────────────────────────────────────────

class TestSaturationProtection:
    """Low-saturation (grey) pixels should be affected more than saturated ones."""

    def test_grey_vs_saturated_red(self):
        """Grey (128,128,128) should shift more than pure red (255,0,0)
        under warm WA_SEL=0 with saturation protection ON."""
        grey = _make_solid_image(128, 128, 128)
        red = _make_solid_image(255, 0, 0)

        cfg = WPAConfig(wa_sel=0, sat_en=True)

        out_grey = wpa_process_rgb_uint8(grey, cfg)
        out_red = wpa_process_rgb_uint8(red, cfg)

        # Compute per-pixel L2 change
        delta_grey = np.linalg.norm(
            out_grey.astype(np.float32) - grey.astype(np.float32), axis=-1
        ).mean()
        delta_red = np.linalg.norm(
            out_red.astype(np.float32) - red.astype(np.float32), axis=-1
        ).mean()

        assert delta_grey > delta_red, (
            f"Grey shift ({delta_grey:.2f}) should exceed red shift "
            f"({delta_red:.2f}) with saturation protection"
        )

    def test_sat_protection_off_allows_saturated(self):
        """With sat protection off, saturated pixels should still change."""
        red = _make_solid_image(200, 0, 0)
        cfg = WPAConfig(wa_sel=0, sat_en=False)
        out = wpa_process_rgb_uint8(red, cfg)
        assert not np.array_equal(out, red), \
            "Saturated pixel should change when saturation protection is off"


# ── Warm / Cool direction ───────────────────────────────────────────────

class TestWarmCoolDirection:
    """Sanity-check that warm boosts R and cool boosts B."""

    def test_warm_boosts_red(self):
        grey = _make_solid_image(128, 128, 128)
        cfg = WPAConfig(wa_sel=0)  # max warm
        out = wpa_process_rgb_uint8(grey, cfg)
        # For grey input, R channel should increase, B should decrease
        r_diff = out[0, 0, 0].astype(int) - 128
        b_diff = out[0, 0, 2].astype(int) - 128
        assert r_diff > 0, f"Warm should boost R, got diff={r_diff}"
        assert b_diff < 0, f"Warm should cut B, got diff={b_diff}"

    def test_cool_boosts_blue(self):
        grey = _make_solid_image(128, 128, 128)
        cfg = WPAConfig(wa_sel=127)  # max cool
        out = wpa_process_rgb_uint8(grey, cfg)
        r_diff = out[0, 0, 0].astype(int) - 128
        b_diff = out[0, 0, 2].astype(int) - 128
        assert r_diff < 0, f"Cool should cut R, got diff={r_diff}"
        assert b_diff > 0, f"Cool should boost B, got diff={b_diff}"

    def test_warm_near_white_tail_returns_to_lower_chroma_warm_white(self):
        vals = [239, 247, 255]
        img = np.array([[[v, v, v] for v in vals]], dtype=np.uint8)

        out = wpa_process_rgb_uint8(img, WPAConfig(wa_sel=0, sat_en=False))[0].astype(np.int32)
        spans = [int(rgb.max() - rgb.min()) for rgb in out]

        assert spans[0] > spans[1] > spans[2]
        assert spans[-1] <= 18
        assert int(out[-1, 1]) - int(out[-1, 2]) <= 10


# ── Modes ────────────────────────────────────────────────────────────────

class TestGammaModes:
    """Ensure all gamma modes run without error and produce valid output."""

    @pytest.mark.parametrize("mode", ["srgb", "power", "none"])
    def test_modes_run(self, mode):
        img = _make_gradient_image()
        cfg = WPAConfig(wa_sel=20, gamma_mode=mode)
        out = wpa_process_rgb_uint8(img, cfg)
        assert out.dtype == np.uint8
        assert out.shape == img.shape

    def test_lut_mode(self):
        img = _make_gradient_image()
        cfg = WPAConfig(wa_sel=20, gamma_mode="srgb", use_gamma_lut=True)
        out = wpa_process_rgb_uint8(img, cfg)
        assert out.dtype == np.uint8
        assert out.shape == img.shape
