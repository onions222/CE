from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from scripts.generate_comparison_visuals import choose_layout, render_comparison_panel


def test_choose_layout_prefers_vertical_for_wide_strip_images() -> None:
    assert choose_layout(1200, 120) == "vertical"
    assert choose_layout(256, 256) == "horizontal"


def test_render_comparison_panel_uses_horizontal_layout_for_regular_images() -> None:
    img = Image.fromarray(np.full((80, 120, 3), 128, dtype=np.uint8))

    panel, layout = render_comparison_panel(img, image_name="regular.png")

    assert layout == "horizontal"
    assert panel.width > panel.height


def test_render_comparison_panel_uses_vertical_layout_for_wide_strip_images() -> None:
    img = Image.fromarray(np.full((40, 400, 3), 128, dtype=np.uint8))

    panel, layout = render_comparison_panel(img, image_name="wide_strip.png")

    assert layout == "vertical"
    assert panel.width < img.width * 2


def test_render_comparison_panel_can_be_saved(tmp_path: Path) -> None:
    img = Image.fromarray(np.full((48, 96, 3), 128, dtype=np.uint8))

    panel, _ = render_comparison_panel(img, image_name="save_me.png")
    output_path = tmp_path / "save_me_panel.png"
    panel.save(output_path)

    assert output_path.exists()
