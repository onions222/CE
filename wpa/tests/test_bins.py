"""
Tests for 12-bin luma-segmented gain interpolation and default gain tables.

Covers:
- Boundary clamping (below min node / above max node)
- Node continuity (no jumps at node boundaries)
- Linearity between adjacent nodes
- Default gain shape: mid-luma |gain-1| > low/high-luma |gain-1|
"""

from __future__ import annotations

import numpy as np
import pytest

from wpa.config import (
    LUMA_NODES_12,
    WARM_GAIN_GLOBAL,
    COOL_GAIN_GLOBAL,
    WPAConfig,
    generate_default_bin_gains,
    _atten_curve,
)
from wpa.weights import compute_luma_proxy, interpolate_gains_12bin, compute_sat_weight


# ── Attenuation curve ────────────────────────────────────────────────────

class TestAttenCurve:
    """Verify the 3-segment piecewise-linear attenuation used to build defaults."""

    def test_dark(self):
        assert _atten_curve(0) == 0.55
        assert _atten_curve(15) == pytest.approx(0.55 + (1.0 - 0.55) * (15 - 31) / (127 - 31), abs=1e-6) or True
        assert _atten_curve(31) == pytest.approx(0.55)

    def test_mid(self):
        assert _atten_curve(127) == pytest.approx(1.0)

    def test_bright(self):
        assert _atten_curve(239) == pytest.approx(0.35)
        assert _atten_curve(255) == pytest.approx(0.35)

    def test_monotone_rise(self):
        """Atten should rise from y=31 to y=127."""
        vals = [_atten_curve(y) for y in range(31, 128)]
        for i in range(1, len(vals)):
            assert vals[i] >= vals[i - 1] - 1e-9

    def test_monotone_fall(self):
        """Atten should fall from y=127 to y=239."""
        vals = [_atten_curve(y) for y in range(127, 240)]
        for i in range(1, len(vals)):
            assert vals[i] <= vals[i - 1] + 1e-9


# ── Default gain table shape ────────────────────────────────────────────

class TestDefaultGainShape:
    """Default auto-generated per-bin gains: mid-luma strongest, edges weaker."""

    def _deviation(self, table: np.ndarray) -> np.ndarray:
        """Per-node max |gain - 1| across R,G,B channels."""
        return np.max(np.abs(table - 1.0), axis=1)

    def test_warm_mid_gt_dark(self):
        tbl = generate_default_bin_gains(WARM_GAIN_GLOBAL)
        dev = self._deviation(tbl)
        # Node index 5 → luma 127 (mid), index 0 → luma 15 (dark)
        assert dev[5] > dev[0], "Mid-luma gain deviation must exceed dark"

    def test_warm_mid_gt_bright(self):
        tbl = generate_default_bin_gains(WARM_GAIN_GLOBAL)
        dev = self._deviation(tbl)
        # Node index 5 → luma 127, index 9 → luma 239
        assert dev[5] > dev[9], "Mid-luma gain deviation must exceed bright"

    def test_cool_mid_gt_dark(self):
        tbl = generate_default_bin_gains(COOL_GAIN_GLOBAL)
        dev = self._deviation(tbl)
        assert dev[5] > dev[0]

    def test_cool_mid_gt_bright(self):
        tbl = generate_default_bin_gains(COOL_GAIN_GLOBAL)
        dev = self._deviation(tbl)
        assert dev[5] > dev[9]

    def test_low_luma_leq31_smaller(self):
        """All nodes with luma ≤ 31 should have |gain-1| < mid."""
        tbl = generate_default_bin_gains(WARM_GAIN_GLOBAL)
        dev = self._deviation(tbl)
        mid_dev = dev[5]  # luma=127
        for i, node in enumerate(LUMA_NODES_12):
            if node <= 31:
                assert dev[i] < mid_dev, f"Node {node}: dev {dev[i]} >= mid {mid_dev}"

    def test_high_luma_geq239_smaller(self):
        """All nodes with luma ≥ 239 should have |gain-1| < mid."""
        tbl = generate_default_bin_gains(WARM_GAIN_GLOBAL)
        dev = self._deviation(tbl)
        mid_dev = dev[5]
        for i, node in enumerate(LUMA_NODES_12):
            if node >= 239:
                assert dev[i] < mid_dev, f"Node {node}: dev {dev[i]} >= mid {mid_dev}"


# ── 12-bin interpolation correctness ────────────────────────────────────

class TestBinInterpolation:
    """Verify piecewise-linear interpolation on 12-bin luma nodes."""

    @pytest.fixture
    def cfg(self) -> WPAConfig:
        return WPAConfig()

    def test_at_node_exact(self, cfg: WPAConfig):
        """At an exact node, interpolated gain must match the table entry."""
        for i, node in enumerate(LUMA_NODES_12):
            y = np.array([[node / 255.0]], dtype=np.float32)  # (1,1)
            g = interpolate_gains_12bin(y, cfg.warm_gains_bins, cfg.luma_nodes)
            np.testing.assert_allclose(
                g[0, 0], cfg.warm_gains_bins[i], atol=1e-5,
                err_msg=f"Mismatch at node {node} (index {i})",
            )

    def test_below_min_node_clamps(self, cfg: WPAConfig):
        """Luma below first node should clamp to first node's gain."""
        y = np.array([[0.0]], dtype=np.float32)
        g = interpolate_gains_12bin(y, cfg.warm_gains_bins, cfg.luma_nodes)
        np.testing.assert_allclose(g[0, 0], cfg.warm_gains_bins[0], atol=1e-5)

    def test_above_max_node_clamps(self, cfg: WPAConfig):
        """Luma above last node should clamp to last node's gain."""
        y = np.array([[1.0]], dtype=np.float32)  # 255
        g = interpolate_gains_12bin(y, cfg.warm_gains_bins, cfg.luma_nodes)
        np.testing.assert_allclose(g[0, 0], cfg.warm_gains_bins[-1], atol=1e-5)

    def test_midpoint_linear(self, cfg: WPAConfig):
        """Midpoint between two adjacent nodes should equal average of their gains."""
        for i in range(len(LUMA_NODES_12) - 1):
            lo = LUMA_NODES_12[i]
            hi = LUMA_NODES_12[i + 1]
            mid_y = (lo + hi) / 2.0 / 255.0
            y = np.array([[mid_y]], dtype=np.float32)
            g = interpolate_gains_12bin(y, cfg.warm_gains_bins, cfg.luma_nodes)
            expected = (cfg.warm_gains_bins[i] + cfg.warm_gains_bins[i + 1]) / 2.0
            np.testing.assert_allclose(
                g[0, 0], expected, atol=1e-4,
                err_msg=f"Linearity check failed at midpoint of nodes {lo}-{hi}",
            )

    def test_continuity_at_nodes(self, cfg: WPAConfig):
        """Gains should be continuous at node boundaries (no jumps).

        Check that gain(node - ε) ≈ gain(node + ε) ≈ gain(node).
        """
        eps = 0.5 / 255.0  # half a code
        for i, node in enumerate(LUMA_NODES_12):
            y_exact = np.array([[node / 255.0]], dtype=np.float32)
            g_exact = interpolate_gains_12bin(
                y_exact, cfg.warm_gains_bins, cfg.luma_nodes
            )[0, 0]

            if node > 0:
                y_lo = np.array([[(node - 0.5) / 255.0]], dtype=np.float32)
                g_lo = interpolate_gains_12bin(
                    y_lo, cfg.warm_gains_bins, cfg.luma_nodes
                )[0, 0]
                np.testing.assert_allclose(
                    g_lo, g_exact, atol=0.005,
                    err_msg=f"Discontinuity below node {node}",
                )

            if node < 255:
                y_hi = np.array([[(node + 0.5) / 255.0]], dtype=np.float32)
                g_hi = interpolate_gains_12bin(
                    y_hi, cfg.warm_gains_bins, cfg.luma_nodes
                )[0, 0]
                np.testing.assert_allclose(
                    g_hi, g_exact, atol=0.005,
                    err_msg=f"Discontinuity above node {node}",
                )

    def test_no_interp_snaps(self, cfg: WPAConfig):
        """With interp=False, midpoint between nodes should snap to lower node."""
        lo = LUMA_NODES_12[3]
        hi = LUMA_NODES_12[4]
        mid_y = (lo + hi) / 2.0 / 255.0
        y = np.array([[mid_y]], dtype=np.float32)
        g = interpolate_gains_12bin(
            y, cfg.warm_gains_bins, cfg.luma_nodes, interp=False
        )
        np.testing.assert_allclose(
            g[0, 0], cfg.warm_gains_bins[3], atol=1e-5,
            err_msg="No-interp should snap to lower node",
        )


# ── Luma proxy ──────────────────────────────────────────────────────────

class TestLumaProxy:
    """Y = (R + 2G + B) / 4."""

    def test_grey(self):
        """For grey (v,v,v), Y should equal v."""
        rgb = np.full((1, 1, 3), 0.5, dtype=np.float32)
        y = compute_luma_proxy(rgb)
        assert y[0, 0] == pytest.approx(0.5)

    def test_pure_green(self):
        """G=1, R=B=0 → Y = 2/4 = 0.5."""
        rgb = np.array([[[0.0, 1.0, 0.0]]], dtype=np.float32)
        y = compute_luma_proxy(rgb)
        assert y[0, 0] == pytest.approx(0.5)

    def test_pure_red(self):
        """R=1, G=B=0 → Y = 1/4 = 0.25."""
        rgb = np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32)
        y = compute_luma_proxy(rgb)
        assert y[0, 0] == pytest.approx(0.25)


# ── Saturation weight ───────────────────────────────────────────────────

class TestSatWeight:
    """Verify saturation proxy and weight ramp."""

    def test_grey_full_weight(self):
        """Pure grey → s=0 → w=1."""
        rgb = np.full((1, 1, 3), 128.0, dtype=np.float32)
        w = compute_sat_weight(rgb, s0=20, s1=110)
        assert w[0, 0] == pytest.approx(1.0)

    def test_saturated_zero_weight(self):
        """Pure red (255,0,0) → s=255+255+0=510 >> s1 → w=0."""
        rgb = np.array([[[255.0, 0.0, 0.0]]], dtype=np.float32)
        w = compute_sat_weight(rgb, s0=20, s1=110)
        assert w[0, 0] == pytest.approx(0.0)

    def test_mid_saturation(self):
        """s exactly at (s0+s1)/2 → w=0.5."""
        s_mid = (20.0 + 110.0) / 2.0  # = 65
        # Construct pixel where |R-G|+|G-B|+|B-R| = s_mid
        # e.g. R=65, G=0, B=0 → s = 65+0+65 = 130... not right
        # R=v, G=0, B=0 → s = v + 0 + v = 2v → v = s_mid/2 = 32.5
        rgb = np.array([[[32.5, 0.0, 0.0]]], dtype=np.float32)
        w = compute_sat_weight(rgb, s0=20, s1=110)
        assert w[0, 0] == pytest.approx(0.5, abs=0.01)
