"""Dataset profile definitions for WPA validation image sets."""

from __future__ import annotations

from typing import Literal

from scripts.download_test_images import KODAK_IMAGES


SYNTHETIC_ALL = [
    "01_grey_ramp",
    "02_color_checker",
    "03_channel_sweeps",
    "04_saturation_gradient",
    "05_highlight_lowlight",
    "06_skin_tones",
    "07_smooth_gradient",
    "08_random_noise",
    "09_grey_steps",
    "10_luma_node_chart",
    "11_ui_text_contrast",
    "12_specular_clip_chart",
    "13_mixed_illumination_chart",
]

KODAK_ALL = [f"kodim{i:02d}.png" for i in range(1, 25)]
KODAK_CURATED = list(KODAK_IMAGES.keys())
LEGACY_KODAK_BASELINE = [
    "kodim04.png",
    "kodim05.png",
    "kodim13.png",
    "kodim01.png",
    "kodim10.png",
    "kodim21.png",
]

VISUAL_NEUTRAL_CORE = [
    "01_grey_ramp",
    "09_grey_steps",
    "10_luma_node_chart",
    "bin_boundary_triplet_chart",
    "11_ui_text_contrast",
    "12_specular_clip_chart",
    "near_black_steps",
    "near_white_steps",
    "ui_dark_theme_chart",
    "two_axis_neutral_gradient",
    "iso_gray_18_70_pair",
    "midtone_neutral_texture",
    "warm_cool_split_field",
    "shadow_with_colored_highlight",
]

VISUAL_COLOR_SIDE_EFFECT = [
    "02_color_checker",
    "04_saturation_gradient",
    "06_skin_tones",
    "13_mixed_illumination_chart",
    "rgb_cmy_color_bars",
    "saturation_threshold_ladder",
    "skin_tone_luma_strip",
]

def _visual_meta(
    *,
    dataset_role: str,
    priority: str,
    visual_risk: list[str],
    expected_observation: str,
    expected_artifact_if_fail: str,
    recommended_wa_sel: list[int],
    suggested_metrics: list[str],
    pass_hint: str,
    notes: str = "",
) -> dict:
    return {
        "dataset_role": dataset_role,
        "priority": priority,
        "inspection_priority": priority,
        "visual_risk": visual_risk,
        "failure_modes": visual_risk,
        "expected_observation": expected_observation,
        "expected_artifact_if_fail": expected_artifact_if_fail,
        "recommended_wa_sel": recommended_wa_sel,
        "suggested_metrics": suggested_metrics,
        "pass_hint": pass_hint,
        "notes": notes,
    }


VISUAL_ITEM_METADATA = {
    "01_grey_ramp": _visual_meta(
        dataset_role="neutral_stability_core",
        priority="P0",
        visual_risk=["neutral_cast", "banding"],
        expected_observation="Grey ramp remains neutral and smooth.",
        expected_artifact_if_fail="Large-area ramp shows tinting or visible contouring.",
        recommended_wa_sel=[0, 32, 64, 96, 127],
        suggested_metrics=["channel_delta_mean", "local_luma_jump_max"],
        pass_hint="Look for a smooth grey-only ramp at warm and cool extremes.",
    ),
    "09_grey_steps": _visual_meta(
        dataset_role="neutral_stability_core",
        priority="P0",
        visual_risk=["neutral_cast", "node_discontinuity"],
        expected_observation="Discrete grey steps remain neutral without abrupt jumps.",
        expected_artifact_if_fail="Adjacent steps pick up tint or show uneven spacing.",
        recommended_wa_sel=[0, 64, 127],
        suggested_metrics=["step_delta_uniformity"],
        pass_hint="Each grey step should stay neutral and visually ordered.",
    ),
    "10_luma_node_chart": _visual_meta(
        dataset_role="neutral_stability_core",
        priority="P0",
        visual_risk=["node_discontinuity"],
        expected_observation="Node-aligned greys should not show visible bin transitions.",
        expected_artifact_if_fail="Exact node positions show visible output jumps.",
        recommended_wa_sel=[0, 64, 127],
        suggested_metrics=["node_boundary_jump"],
        pass_hint="Check for abrupt transitions at node-aligned patches.",
    ),
    "bin_boundary_triplet_chart": _visual_meta(
        dataset_role="neutral_stability_core",
        priority="P0",
        visual_risk=["luma_node_discontinuity", "quantization_jump"],
        expected_observation="Triplets around each node vary smoothly across n-1, n, and n+1.",
        expected_artifact_if_fail="One patch in a triplet shifts more than its neighbors.",
        recommended_wa_sel=[0, 64, 127],
        suggested_metrics=["boundary_triplet_delta"],
        pass_hint="Inspect each triplet for monotonic, near-equal transitions.",
    ),
    "11_ui_text_contrast": _visual_meta(
        dataset_role="neutral_stability_core",
        priority="P0",
        visual_risk=["neutral_cast", "ui_artifact"],
        expected_observation="Light-theme UI blocks stay neutral and text edges remain clean.",
        expected_artifact_if_fail="Cards or fine bars develop tint or color fringing.",
        recommended_wa_sel=[0, 64, 127],
        suggested_metrics=["edge_chroma_spread"],
        pass_hint="White and light-grey UI surfaces should stay neutral.",
    ),
    "12_specular_clip_chart": _visual_meta(
        dataset_role="neutral_stability_core",
        priority="P0",
        visual_risk=["highlight_shift"],
        expected_observation="Bright highlights stay stable without obvious tinting near clip.",
        expected_artifact_if_fail="Near-clip blobs shift hue or clip unevenly by channel.",
        recommended_wa_sel=[0, 64, 127],
        suggested_metrics=["highlight_channel_clip_rate"],
        pass_hint="Bright blobs should brighten without picking up obvious color.",
    ),
    "near_black_steps": _visual_meta(
        dataset_role="neutral_stability_core",
        priority="P0",
        visual_risk=["dark_instability", "neutral_cast"],
        expected_observation="Near-black steps stay separable and neutral.",
        expected_artifact_if_fail="Shadow steps merge together or show colored patches.",
        recommended_wa_sel=[0, 32, 64, 96, 127],
        suggested_metrics=["shadow_step_separation", "channel_delta_mean"],
        pass_hint="Dark steps should remain distinct and neutral at both extremes.",
    ),
    "near_white_steps": _visual_meta(
        dataset_role="neutral_stability_core",
        priority="P0",
        visual_risk=["highlight_shift", "neutral_cast"],
        expected_observation="Near-white steps stay neutral before clipping.",
        expected_artifact_if_fail="Upper steps compress suddenly or drift pink/blue/yellow.",
        recommended_wa_sel=[0, 64, 127],
        suggested_metrics=["highlight_step_separation"],
        pass_hint="Near-white patches should stay separated until clipping.",
    ),
    "ui_dark_theme_chart": _visual_meta(
        dataset_role="neutral_stability_core",
        priority="P0",
        visual_risk=["dark_instability", "ui_artifact"],
        expected_observation="Dark-theme UI layers remain stable without color cast.",
        expected_artifact_if_fail="Dark panels separate poorly or pick up colored haze.",
        recommended_wa_sel=[0, 64, 127],
        suggested_metrics=["shadow_panel_contrast"],
        pass_hint="Dark layered panels should remain neutral and legible.",
    ),
    "two_axis_neutral_gradient": _visual_meta(
        dataset_role="neutral_stability_core",
        priority="P0",
        visual_risk=["banding", "area_nonuniformity", "neutral_cast"],
        expected_observation="The 2D neutral gradient remains smooth and achromatic.",
        expected_artifact_if_fail="Contours or tinted regions appear across the field.",
        recommended_wa_sel=[0, 64, 127],
        suggested_metrics=["local_luma_jump_max", "channel_delta_mean"],
        pass_hint="Scan the whole field for stripes, blotches, or color drift.",
    ),
    "iso_gray_18_70_pair": _visual_meta(
        dataset_role="neutral_stability_core",
        priority="P0",
        visual_risk=["luma_dependent_neutral_drift", "neutral_cast"],
        expected_observation="Low and high neutral blocks drift consistently without hue skew.",
        expected_artifact_if_fail="Darker and brighter greys shift in different directions.",
        recommended_wa_sel=[0, 64, 127],
        suggested_metrics=["paired_grey_channel_delta"],
        pass_hint="Compare low and high greys side by side for directionally similar shift.",
    ),
    "midtone_neutral_texture": _visual_meta(
        dataset_role="neutral_stability_core",
        priority="P0",
        visual_risk=["ui_artifact", "detail_collapse", "neutral_cast"],
        expected_observation="Fine neutral details stay crisp without colored edges.",
        expected_artifact_if_fail="Thin lines soften or show colored fringing.",
        recommended_wa_sel=[0, 64, 127],
        suggested_metrics=["edge_chroma_spread", "texture_contrast_retention"],
        pass_hint="Inspect fine lines and boxes for colored halos or contrast loss.",
    ),
    "warm_cool_split_field": _visual_meta(
        dataset_role="neutral_stability_core",
        priority="P0",
        visual_risk=["mixed_illumination_transition", "neutral_cast"],
        expected_observation="The warm-to-cool field transitions smoothly through neutral references.",
        expected_artifact_if_fail="Transition boundary looks abrupt or neutral references skew differently.",
        recommended_wa_sel=[0, 64, 127],
        suggested_metrics=["transition_smoothness"],
        pass_hint="Pay attention to the center transition and neutral blocks.",
    ),
    "shadow_with_colored_highlight": _visual_meta(
        dataset_role="neutral_stability_core",
        priority="P0",
        visual_risk=["near_white_drift", "near_black_instability", "highlight_halo_or_tint"],
        expected_observation="Dark background remains stable while bright highlights keep their families.",
        expected_artifact_if_fail="Highlights bloom into tinted halos or shadows pick up cast.",
        recommended_wa_sel=[0, 64, 127],
        suggested_metrics=["highlight_channel_clip_rate", "shadow_channel_delta"],
        pass_hint="Check both the highlight cores and the surrounding dark field.",
    ),
    "02_color_checker": _visual_meta(
        dataset_role="color_side_effect_set",
        priority="P1",
        visual_risk=["hue_shift"],
        expected_observation="Reference color patches should not drift excessively.",
        expected_artifact_if_fail="Multiple hue families rotate or compress together.",
        recommended_wa_sel=[0, 64, 127],
        suggested_metrics=["patch_delta_rgb"],
        pass_hint="Use as a broad color sanity check, not the main release gate.",
    ),
    "04_saturation_gradient": _visual_meta(
        dataset_role="color_side_effect_set",
        priority="P1",
        visual_risk=["hue_shift"],
        expected_observation="Saturation sweeps should stay smooth without abrupt hue drift.",
        expected_artifact_if_fail="A hue strip bends or changes direction across saturation.",
        recommended_wa_sel=[0, 64, 127],
        suggested_metrics=["hue_sweep_smoothness"],
        pass_hint="Look for abrupt hue turns across each saturation ramp.",
    ),
    "06_skin_tones": _visual_meta(
        dataset_role="color_side_effect_set",
        priority="P1",
        visual_risk=["hue_shift"],
        expected_observation="Skin-tone patches should stay plausible and ordered.",
        expected_artifact_if_fail="One complexion family becomes grey, pink, or green relative to neighbors.",
        recommended_wa_sel=[0, 64, 127],
        suggested_metrics=["skin_patch_delta_rgb"],
        pass_hint="Check whether complexion ordering still looks plausible.",
    ),
    "13_mixed_illumination_chart": _visual_meta(
        dataset_role="color_side_effect_set",
        priority="P1",
        visual_risk=["hue_shift", "neutral_cast"],
        expected_observation="Warm-to-cool transitions should remain smooth.",
        expected_artifact_if_fail="Objects crossing the field shift in inconsistent ways.",
        recommended_wa_sel=[0, 64, 127],
        suggested_metrics=["transition_smoothness"],
        pass_hint="Inspect transition smoothness, especially around the neutral object.",
    ),
    "rgb_cmy_color_bars": _visual_meta(
        dataset_role="color_side_effect_set",
        priority="P1",
        visual_risk=["hue_shift"],
        expected_observation="Primary and secondary bars should preserve their hue families.",
        expected_artifact_if_fail="Bars skew toward adjacent families or clip unevenly.",
        recommended_wa_sel=[0, 64, 127],
        suggested_metrics=["bar_channel_ratio"],
        pass_hint="Each bar should stay within its hue family after adjustment.",
    ),
    "saturation_threshold_ladder": _visual_meta(
        dataset_role="color_side_effect_set",
        priority="P1",
        visual_risk=["saturation_transition_artifact", "hue_shift", "over_protection", "under_protection"],
        expected_observation="Saturation increases smoothly through the protection threshold region.",
        expected_artifact_if_fail="The ladder shows sudden hue or intensity changes around mid saturation.",
        recommended_wa_sel=[0, 64, 127],
        suggested_metrics=["sat_transition_smoothness"],
        pass_hint="Look for the point where protection starts to kick in too abruptly.",
    ),
    "skin_tone_luma_strip": _visual_meta(
        dataset_role="color_side_effect_set",
        priority="P1",
        visual_risk=["skin_hue_shift", "skin_luma_dependency"],
        expected_observation="Skin-tone families remain plausible across luminance levels.",
        expected_artifact_if_fail="Darker or brighter variants of the same family diverge in hue.",
        recommended_wa_sel=[0, 64, 127],
        suggested_metrics=["skin_luma_family_delta"],
        pass_hint="Compare columns within the same skin family before comparing different families.",
    ),
}


def _build_visual_items(names: list[str]) -> list[dict]:
    items: list[dict] = []
    for name in names:
        meta = VISUAL_ITEM_METADATA[name]
        items.append(
            {
                "name": name,
                "source": "synthetic",
                "pass_mode": "visual_only",
                **meta,
            }
        )
    return items


def get_profile_spec(
    profile: Literal["smoke", "core", "full", "smoke_visual", "core_visual", "release_visual"]
) -> dict:
    if profile == "smoke_visual":
        synthetic = [
            "01_grey_ramp",
            "09_grey_steps",
            "10_luma_node_chart",
            "11_ui_text_contrast",
            "near_black_steps",
            "near_white_steps",
            "ui_dark_theme_chart",
            "rgb_cmy_color_bars",
        ]
        return {
            "name": "smoke_visual",
            "synthetic": synthetic,
            "items": _build_visual_items(synthetic),
            "legacy_kodak": [],
            "real_sanity": [],
            "jpeg_ladder": {
                "enabled": False,
                "source_images": [],
                "qualities": [],
            },
        }

    if profile == "core_visual":
        synthetic = list(VISUAL_NEUTRAL_CORE + VISUAL_COLOR_SIDE_EFFECT)
        return {
            "name": "core_visual",
            "synthetic": synthetic,
            "items": _build_visual_items(synthetic),
            "legacy_kodak": [],
            "real_sanity": [],
            "jpeg_ladder": {
                "enabled": False,
                "source_images": [],
                "qualities": [],
            },
        }

    if profile == "release_visual":
        synthetic = list(VISUAL_NEUTRAL_CORE + VISUAL_COLOR_SIDE_EFFECT)
        return {
            "name": "release_visual",
            "synthetic": synthetic,
            "items": _build_visual_items(synthetic),
            "legacy_kodak": list(LEGACY_KODAK_BASELINE),
            "real_sanity": [
                "public_portrait",
                "public_hdr_window",
                "public_night_neon",
                "public_ui_workspace",
                "research_mixed_light",
            ],
            "jpeg_ladder": {
                "enabled": True,
                "source_images": ["kodim04.png", "kodim13.png", "kodim21.png"],
                "qualities": [95, 80, 60, 40],
            },
        }

    if profile == "smoke":
        return {
            "name": "smoke",
            "synthetic": [
                "01_grey_ramp",
                "02_color_checker",
                "05_highlight_lowlight",
                "07_smooth_gradient",
                "10_luma_node_chart",
                "11_ui_text_contrast",
            ],
            "real_kodak": [
                "kodim04.png",
                "kodim05.png",
                "kodim01.png",
                "kodim13.png",
                "kodim10.png",
                "kodim21.png",
            ],
            "jpeg_ladder": {
                "enabled": False,
                "source_images": [],
                "qualities": [],
            },
        }

    if profile == "core":
        return {
            "name": "core",
            "synthetic": list(SYNTHETIC_ALL),
            "real_kodak": list(KODAK_CURATED),
            "jpeg_ladder": {
                "enabled": True,
                "source_images": ["kodim04.png", "kodim13.png", "kodim22.png"],
                "qualities": [95, 80, 60, 40],
            },
        }

    if profile == "full":
        return {
            "name": "full",
            "synthetic": list(SYNTHETIC_ALL),
            "real_kodak": list(KODAK_ALL),
            "jpeg_ladder": {
                "enabled": True,
                "source_images": ["kodim04.png", "kodim13.png", "kodim22.png"],
                "qualities": [95, 80, 60, 40],
            },
        }

    raise ValueError(f"Unknown profile: {profile}")
