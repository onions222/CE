from __future__ import annotations

from scripts.generate_test_images import GENERATORS


def test_synthetic_generator_catalog_matches_visual_stability_design() -> None:
    assert "near_black_steps" in GENERATORS
    assert "near_white_steps" in GENERATORS
    assert "ui_dark_theme_chart" in GENERATORS
    assert "rgb_cmy_color_bars" in GENERATORS


def test_extended_synthetic_generator_catalog_contains_new_diagnostic_charts() -> None:
    assert "two_axis_neutral_gradient" in GENERATORS
    assert "bin_boundary_triplet_chart" in GENERATORS
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
    near_node = GENERATORS["near_node_ramp_chart"]()
    threshold = GENERATORS["saturation_threshold_ladder"]()

    assert two_axis.ndim == 3
    assert two_axis.shape[2] == 3
    assert triplets.ndim == 3
    assert triplets.shape[2] == 3
    assert near_node.ndim == 3
    assert near_node.shape[2] == 3
    assert threshold.ndim == 3
    assert threshold.shape[2] == 3
