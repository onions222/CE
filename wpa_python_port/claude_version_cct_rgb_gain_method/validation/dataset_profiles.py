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


def get_profile_spec(profile: Literal["smoke", "core", "full"]) -> dict:
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
