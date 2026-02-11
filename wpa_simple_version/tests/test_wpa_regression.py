import numpy as np

from wpa.config import WPAConfig
from wpa.core import _wa_sel_to_side_alpha, wpa_process_rgb_uint8
from wpa.weights import saturation_proxy, saturation_weight


def _gray_patch(y: int, size: int = 64) -> np.ndarray:
    return np.full((size, size, 3), y, dtype=np.uint8)


def _gray_ramp_image(size: int = 64) -> tuple[np.ndarray, list[int]]:
    ys = [0, 32, 64, 96, 128, 160, 192, 224, 255]
    patches = [_gray_patch(y, size=size) for y in ys]
    return np.concatenate(patches, axis=1), ys


def _gray_channel_diffs(arr: np.ndarray, ys: list[int], size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    rg = []
    gb = []
    for i, _ in enumerate(ys):
        x0 = i * size
        x1 = x0 + size
        p = arr[:, x0:x1, :].astype(np.int32)
        rg.append(float(np.mean(p[..., 0] - p[..., 1])))
        gb.append(float(np.mean(p[..., 1] - p[..., 2])))
    return np.asarray(rg, dtype=np.float32), np.asarray(gb, dtype=np.float32)


def _clip_ratio_rgb(arr: np.ndarray) -> np.ndarray:
    clip = (arr == 0) | (arr == 255)
    return np.array([clip[..., 0].mean(), clip[..., 1].mean(), clip[..., 2].mean()], dtype=np.float32)


def test_identity_regression_paths() -> None:
    img = np.array(
        [[[0, 0, 0], [128, 128, 128]], [[255, 100, 20], [17, 19, 23]]],
        dtype=np.uint8,
    )
    out_off = wpa_process_rgb_uint8(img, WPAConfig(WA_EN=False, WA_SEL=10))
    out_center = wpa_process_rgb_uint8(img, WPAConfig(WA_EN=True, WA_SEL=64))
    assert np.array_equal(out_off, img)
    assert np.array_equal(out_center, img)


def test_wa_sel_mapping_and_bins_independence() -> None:
    side, a = _wa_sel_to_side_alpha(65)
    assert side == "cool"
    assert a > 0.0

    cfg = WPAConfig()
    assert cfg.warm_gains_bins is not cfg.cool_gains_bins
    nodes = np.asarray(cfg.luma_nodes_12, dtype=np.float32)
    idx_127 = int(np.where(nodes == 127)[0][0])
    assert not np.allclose(cfg.warm_gains_bins[idx_127], cfg.cool_gains_bins[idx_127])


def test_cool_visible_on_mid_gray() -> None:
    img = _gray_patch(128, size=64)
    out = wpa_process_rgb_uint8(img, WPAConfig(WA_EN=True, WA_SEL=110))
    delta = np.abs(out.astype(np.int32) - img.astype(np.int32)).sum(axis=-1).mean()
    assert delta >= 3.0


def test_warm_visible_on_mid_gray() -> None:
    img = _gray_patch(128, size=64)
    out = wpa_process_rgb_uint8(img, WPAConfig(WA_EN=True, WA_SEL=20))
    delta = np.abs(out.astype(np.int32) - img.astype(np.int32)).sum(axis=-1).mean()
    assert delta >= 3.0


def test_kelvin_ycocg_mode_visible_and_safe_on_gray_ramp() -> None:
    img, ys = _gray_ramp_image(size=64)
    for wa in (20, 110):
        out = wpa_process_rgb_uint8(img, WPAConfig(WA_EN=True, WA_SEL=wa, wa_mode="kelvin_ycocg"))
        rg, gb = _gray_channel_diffs(out, ys, size=64)
        # Visible change around mid-gray.
        mid = ys.index(128)
        assert abs(float(rg[mid])) + abs(float(gb[mid])) >= 2.0
        # No highlight spike.
        hi_idx = [i for i, y in enumerate(ys) if y >= 223]
        if len(hi_idx) > 1:
            drg = np.diff(rg[hi_idx])
            dgb = np.diff(gb[hi_idx])
            spike = max(float(np.max(np.abs(drg))), float(np.max(np.abs(dgb))))
            assert spike <= 6.0


def test_no_red_yellow_spike_in_highlights_for_gray_ramp() -> None:
    img, ys = _gray_ramp_image(size=64)
    hi_idx = [i for i, y in enumerate(ys) if y >= 223]
    cfgs = [WPAConfig(WA_EN=True, WA_SEL=20), WPAConfig(WA_EN=True, WA_SEL=110)]
    for cfg in cfgs:
        out = wpa_process_rgb_uint8(img, cfg)
        rg, gb = _gray_channel_diffs(out, ys, size=64)
        if len(hi_idx) > 1:
            drg = np.diff(rg[hi_idx])
            dgb = np.diff(gb[hi_idx])
            spike = max(float(np.max(np.abs(drg))), float(np.max(np.abs(dgb))))
            assert spike <= 2.0


def test_clipping_safety_on_highlight_gray_patch() -> None:
    img = _gray_patch(230, size=128)
    base = wpa_process_rgb_uint8(img, WPAConfig(WA_EN=True, WA_SEL=64))
    base_clip = _clip_ratio_rgb(base)

    warm = wpa_process_rgb_uint8(img, WPAConfig(WA_EN=True, WA_SEL=20))
    cool = wpa_process_rgb_uint8(img, WPAConfig(WA_EN=True, WA_SEL=110))
    warm_delta = _clip_ratio_rgb(warm) - base_clip
    cool_delta = _clip_ratio_rgb(cool) - base_clip

    assert float(np.max(warm_delta)) < 0.01
    assert float(np.max(cool_delta)) < 0.01


def test_saturation_weight_distribution_gray_vs_pure() -> None:
    gray = _gray_patch(160, size=64)
    pure_red = np.zeros((64, 64, 3), dtype=np.uint8)
    pure_red[:] = (255, 0, 0)

    cfg = WPAConfig()
    s_gray = saturation_proxy(gray.astype(np.float32) / 255.0)
    s_red = saturation_proxy(pure_red.astype(np.float32) / 255.0)
    w_gray = saturation_weight(s_gray, cfg.sat_s0, cfg.sat_s1)
    w_red = saturation_weight(s_red, cfg.sat_s0, cfg.sat_s1)

    assert float(np.mean(w_gray)) > float(np.mean(w_red))
