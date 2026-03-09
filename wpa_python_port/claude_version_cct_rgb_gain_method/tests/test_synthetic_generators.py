from __future__ import annotations

from scripts.generate_test_images import GENERATORS


def test_synthetic_generator_catalog_matches_visual_stability_design() -> None:
    assert "near_black_steps" in GENERATORS
    assert "near_white_steps" in GENERATORS
    assert "ui_dark_theme_chart" in GENERATORS
    assert "rgb_cmy_color_bars" in GENERATORS

