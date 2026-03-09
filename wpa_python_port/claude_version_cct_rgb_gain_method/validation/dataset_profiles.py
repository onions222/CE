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
    "11_ui_text_contrast",
    "12_specular_clip_chart",
    "near_black_steps",
    "near_white_steps",
    "ui_dark_theme_chart",
]

VISUAL_COLOR_SIDE_EFFECT = [
    "02_color_checker",
    "04_saturation_gradient",
    "06_skin_tones",
    "13_mixed_illumination_chart",
    "rgb_cmy_color_bars",
]

VISUAL_ITEM_METADATA = {
    "01_grey_ramp": {
        "dataset_role": "neutral_stability_core",
        "priority": "P0",
        "visual_risk": ["neutral_cast", "banding"],
        "expected_observation": "Grey ramp remains neutral and smooth.",
    },
    "09_grey_steps": {
        "dataset_role": "neutral_stability_core",
        "priority": "P0",
        "visual_risk": ["neutral_cast", "node_discontinuity"],
        "expected_observation": "Discrete grey steps remain neutral without abrupt jumps.",
    },
    "10_luma_node_chart": {
        "dataset_role": "neutral_stability_core",
        "priority": "P0",
        "visual_risk": ["node_discontinuity"],
        "expected_observation": "Node-aligned greys should not show visible bin transitions.",
    },
    "11_ui_text_contrast": {
        "dataset_role": "neutral_stability_core",
        "priority": "P0",
        "visual_risk": ["neutral_cast", "ui_artifact"],
        "expected_observation": "Light-theme UI blocks stay neutral and text edges remain clean.",
    },
    "12_specular_clip_chart": {
        "dataset_role": "neutral_stability_core",
        "priority": "P0",
        "visual_risk": ["highlight_shift"],
        "expected_observation": "Bright highlights stay stable without obvious tinting near clip.",
    },
    "near_black_steps": {
        "dataset_role": "neutral_stability_core",
        "priority": "P0",
        "visual_risk": ["dark_instability", "neutral_cast"],
        "expected_observation": "Near-black steps stay separable and neutral.",
    },
    "near_white_steps": {
        "dataset_role": "neutral_stability_core",
        "priority": "P0",
        "visual_risk": ["highlight_shift", "neutral_cast"],
        "expected_observation": "Near-white steps stay neutral before clipping.",
    },
    "ui_dark_theme_chart": {
        "dataset_role": "neutral_stability_core",
        "priority": "P0",
        "visual_risk": ["dark_instability", "ui_artifact"],
        "expected_observation": "Dark-theme UI layers remain stable without color cast.",
    },
    "02_color_checker": {
        "dataset_role": "color_side_effect_set",
        "priority": "P1",
        "visual_risk": ["hue_shift"],
        "expected_observation": "Reference color patches should not drift excessively.",
    },
    "04_saturation_gradient": {
        "dataset_role": "color_side_effect_set",
        "priority": "P1",
        "visual_risk": ["hue_shift"],
        "expected_observation": "Saturation sweeps should stay smooth without abrupt hue drift.",
    },
    "06_skin_tones": {
        "dataset_role": "color_side_effect_set",
        "priority": "P1",
        "visual_risk": ["hue_shift"],
        "expected_observation": "Skin-tone patches should stay plausible and ordered.",
    },
    "13_mixed_illumination_chart": {
        "dataset_role": "color_side_effect_set",
        "priority": "P1",
        "visual_risk": ["hue_shift", "neutral_cast"],
        "expected_observation": "Warm-to-cool transitions should remain smooth.",
    },
    "rgb_cmy_color_bars": {
        "dataset_role": "color_side_effect_set",
        "priority": "P1",
        "visual_risk": ["hue_shift"],
        "expected_observation": "Primary and secondary bars should preserve their hue families.",
    },
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
            "real_sanity": [],
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
