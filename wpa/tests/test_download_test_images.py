from __future__ import annotations

from pathlib import Path

from scripts import build_test_set
from scripts.download_test_images import REAL_WORLD_IMAGE_GROUPS
from validation.dataset_profiles import get_profile_spec


def test_download_catalog_exposes_real_world_sanity_groups() -> None:
    assert "public_portrait" in REAL_WORLD_IMAGE_GROUPS
    assert "research_mixed_light" in REAL_WORLD_IMAGE_GROUPS


def test_release_visual_profile_requests_real_world_sanity_images() -> None:
    spec = get_profile_spec("release_visual")

    assert spec["real_sanity"]


def test_build_release_visual_requests_real_world_sanity_images(
    monkeypatch, tmp_path: Path
) -> None:
    requested: list[str] = []

    def fake_download(output_dir: Path, groups: list[str], root: Path) -> dict[str, str]:
        requested.extend(groups)
        out = output_dir / "real_sanity"
        out.mkdir(parents=True, exist_ok=True)
        fp = out / "sample.png"
        fp.write_bytes(b"fake")
        return {"sample.png": str(fp.relative_to(root))}

    monkeypatch.setattr(build_test_set, "download_real_world_images", fake_download)

    root = tmp_path / "dataset"
    result = build_test_set._download_real_world_sanity(  # type: ignore[attr-defined]
        ["public_portrait"], root / "real", root
    )

    assert requested == ["public_portrait"]
    assert result["sample.png"] == "real/real_sanity/sample.png"
