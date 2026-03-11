from __future__ import annotations

import numpy as np

from scripts.generate_test_images import GENERATORS


def test_synthetic_generator_catalog_matches_visual_stability_design() -> None:
    assert "near_black_steps" in GENERATORS
    assert "near_white_steps" in GENERATORS
    assert "ui_dark_theme_chart" in GENERATORS
    assert "rgb_cmy_color_bars" in GENERATORS


def test_extended_synthetic_generator_catalog_contains_new_diagnostic_charts() -> None:
    assert "two_axis_neutral_gradient" in GENERATORS
    assert "bin_boundary_triplet_chart" in GENERATORS
    assert "near_node_patch_grid" in GENERATORS
    assert "near_node_ramp_chart" in GENERATORS
    assert "iso_gray_18_70_pair" in GENERATORS
    assert "midtone_neutral_texture" in GENERATORS
    assert "saturation_threshold_ladder" in GENERATORS
    assert "skin_tone_luma_strip" in GENERATORS
    assert "warm_cool_split_field" in GENERATORS
    assert "shadow_with_colored_highlight" in GENERATORS


def test_new_diagnostic_generators_return_expected_shapes() -> None:
    two_axis = GENERATORS["two_axis_neutral_gradient"]()
    triplets = GENERATORS["bin_boundary_triplet_chart"]()
    patch_grid = GENERATORS["near_node_patch_grid"]()
    near_node = GENERATORS["near_node_ramp_chart"]()
    threshold = GENERATORS["saturation_threshold_ladder"]()

    assert two_axis.ndim == 3
    assert two_axis.shape[2] == 3
    assert triplets.ndim == 3
    assert triplets.shape[2] == 3
    assert patch_grid.ndim == 3
    assert patch_grid.shape[2] == 3
    assert near_node.ndim == 3
    assert near_node.shape[2] == 3
    assert threshold.ndim == 3
    assert threshold.shape[2] == 3


def test_near_node_patch_grid_matches_reference_structure() -> None:
    img = GENERATORS["near_node_patch_grid"]()

    assert img.shape == (432, 252, 3)
    assert np.array_equal(img[..., 0], img[..., 1])
    assert np.array_equal(img[..., 1], img[..., 2])

    luma_nodes = [15, 31, 47, 63, 95, 127, 159, 191, 223, 239, 247, 255]
    offsets = list(range(-4, 5))

    for row_idx, node in enumerate(luma_nodes):
        y = row_idx * 36
        row = img[y, :, 0]
        values = [int(row[col * 28]) for col in range(9)]
        expected = [int(np.clip(node + offset, 0, 255)) for offset in offsets]
        assert values == expected
