import numpy as np
import pytest

from wpa.config import WPAConfig
from wpa.core import wpa_process_rgb_uint8
from wpa.gamma import linear_to_srgb_formula, srgb_to_linear_formula


def test_identity_when_disabled_or_center_sel() -> None:
    img = np.array(
        [
            [[0, 0, 0], [32, 64, 128]],
            [[255, 200, 10], [17, 19, 23]],
        ],
        dtype=np.uint8,
    )

    cfg_off = WPAConfig(WA_EN=False, WA_SEL=10)
    out_off = wpa_process_rgb_uint8(img, cfg_off)
    assert np.array_equal(out_off, img)

    cfg_center = WPAConfig(WA_EN=True, WA_SEL=64)
    out_center = wpa_process_rgb_uint8(img, cfg_center)
    assert np.array_equal(out_center, img)


def test_saturation_protection_grayscale_changes_more_than_red() -> None:
    # First pixel gray (low sat), second pixel saturated red (high sat).
    img = np.array([[[128, 128, 128], [255, 0, 0]]], dtype=np.uint8)

    cfg = WPAConfig(
        WA_EN=True,
        WA_SEL=0,  # strongest warm
        sat_s0=0.01,
        sat_s1=0.2,
        sat_weight_domain="gamma",
    )
    out = wpa_process_rgb_uint8(img, cfg)

    delta_gray = np.abs(out[0, 0].astype(np.int32) - img[0, 0].astype(np.int32)).sum()
    delta_red = np.abs(out[0, 1].astype(np.int32) - img[0, 1].astype(np.int32)).sum()
    assert delta_gray > delta_red


def test_srgb_roundtrip_within_1_lsb() -> None:
    codes = np.array([0, 1, 2, 8, 16, 32, 64, 96, 128, 160, 192, 224, 254, 255], dtype=np.uint8)
    x = codes.astype(np.float32) / 255.0
    y = linear_to_srgb_formula(srgb_to_linear_formula(x))
    out_codes = np.rint(y * 255.0).astype(np.int32)
    err = np.abs(out_codes - codes.astype(np.int32))
    assert np.max(err) <= 1


def test_kelvin_side_scale_validation_and_disable_path() -> None:
    img = np.array([[[120, 130, 90], [180, 170, 140]]], dtype=np.uint8)

    with pytest.raises(ValueError):
        WPAConfig(kelvin_warm_side_scale=-0.1)
    with pytest.raises(ValueError):
        WPAConfig(kelvin_cool_side_scale=-0.1)

    cfg_off_cool = WPAConfig(WA_EN=True, WA_SEL=110, wa_mode="kelvin_ycocg", kelvin_cool_side_scale=0.0)
    out_off_cool = wpa_process_rgb_uint8(img, cfg_off_cool)
    assert np.array_equal(out_off_cool, img)

    cfg_off_warm = WPAConfig(WA_EN=True, WA_SEL=20, wa_mode="kelvin_ycocg", kelvin_warm_side_scale=0.0)
    out_off_warm = wpa_process_rgb_uint8(img, cfg_off_warm)
    assert np.array_equal(out_off_warm, img)
