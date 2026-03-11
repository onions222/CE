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


def test_core_visual_profile_includes_new_failure_mode_driven_items() -> None:
    spec = get_profile_spec("core_visual")

    assert "bin_boundary_triplet_chart" in spec["synthetic"]
    assert "near_node_ramp_chart" in spec["synthetic"]
    assert "saturation_threshold_ladder" in spec["synthetic"]


def test_core_visual_profile_items_include_richer_inspection_metadata() -> None:
    spec = get_profile_spec("core_visual")

    item = next(entry for entry in spec["items"] if entry["name"] == "near_black_steps")
    assert item["inspection_priority"] == "P0"
    assert "failure_modes" in item
    assert "recommended_wa_sel" in item
    assert "pass_hint" in item
