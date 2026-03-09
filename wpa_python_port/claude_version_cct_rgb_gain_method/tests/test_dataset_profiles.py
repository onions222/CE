from __future__ import annotations

from validation.dataset_profiles import get_profile_spec


def test_smoke_visual_profile_focuses_on_p0_neutral_stability() -> None:
    spec = get_profile_spec("smoke_visual")

    assert spec["name"] == "smoke_visual"
    assert "near_black_steps" in spec["synthetic"]
    assert "rgb_cmy_color_bars" in spec["synthetic"]


def test_core_visual_profile_exposes_metadata_driven_items() -> None:
    spec = get_profile_spec("core_visual")

    item = next(entry for entry in spec["items"] if entry["name"] == "near_black_steps")
    assert item["dataset_role"] == "neutral_stability_core"
    assert item["priority"] == "P0"
    assert "dark_instability" in item["visual_risk"]
