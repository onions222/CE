import numpy as np

from wpa.config import WPAConfig
from wpa.weights import interpolate_bins


def test_bin_interpolation_boundaries_and_linear_segments() -> None:
    nodes = np.array([15, 31, 47, 63, 95, 127, 159, 191, 223, 239, 247, 255], dtype=np.float32)
    vals = np.stack([nodes, nodes * 2.0, nodes * 3.0], axis=1).astype(np.float32)

    y = np.array([0.0, 15.0, 23.0, 31.0, 40.0, 255.0, 300.0], dtype=np.float32)
    out = interpolate_bins(y, nodes, vals, interp=True)

    # Below min and above max clamp correctly.
    assert np.allclose(out[0], vals[0])
    assert np.allclose(out[-1], vals[-1])

    # Exact node continuity.
    assert np.allclose(out[1], vals[0])
    assert np.allclose(out[3], vals[1])
    assert np.allclose(out[5], vals[-1])

    # Linear interpolation between 15 and 31 for y=23 (halfway).
    expected_23 = 0.5 * vals[0] + 0.5 * vals[1]
    assert np.allclose(out[2], expected_23)

    # Linear interpolation between 31 and 47 for y=40 (9/16 from 31).
    t = (40.0 - 31.0) / (47.0 - 31.0)
    expected_40 = vals[1] + t * (vals[2] - vals[1])
    assert np.allclose(out[4], expected_40)


def test_default_generated_bins_have_peak_at_mid_luma() -> None:
    cfg = WPAConfig()
    nodes = np.asarray(cfg.luma_nodes_12, dtype=np.float32)
    warm = np.asarray(cfg.warm_gains_bins, dtype=np.float32)
    cool = np.asarray(cfg.cool_gains_bins, dtype=np.float32)

    idx_mid = int(np.where(nodes == 127)[0][0])
    mid_warm_mag = np.max(np.abs(warm[idx_mid] - 1.0))
    mid_cool_mag = np.max(np.abs(cool[idx_mid] - 1.0))

    idx_low = np.where(nodes <= 31)[0]
    idx_high = np.where(nodes >= 239)[0]

    low_warm_mag = np.max(np.abs(warm[idx_low] - 1.0))
    high_warm_mag = np.max(np.abs(warm[idx_high] - 1.0))
    low_cool_mag = np.max(np.abs(cool[idx_low] - 1.0))
    high_cool_mag = np.max(np.abs(cool[idx_high] - 1.0))

    assert mid_warm_mag > low_warm_mag
    assert mid_warm_mag > high_warm_mag
    assert mid_cool_mag > low_cool_mag
    assert mid_cool_mag > high_cool_mag
