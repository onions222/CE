#!/usr/bin/env python3
"""Build WPA test image set by profile."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

from PIL import Image

# Ensure project root import path when running script directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from validation.dataset_profiles import get_profile_spec
from scripts.generate_test_images import GENERATORS, save as save_synth
from scripts.download_test_images import KODAK_IMAGES, download_kodak


def _build_kodak_dict(names: Iterable[str]) -> dict[str, str]:
    d: dict[str, str] = {}
    for n in names:
        d[n] = KODAK_IMAGES.get(n, f"Kodak {n}")
    return d


def _generate_synthetic(selected: list[str], out_dir: Path, root: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    generated: dict[str, str] = {}
    print("\n[Synthetic]")
    for name in selected:
        if name not in GENERATORS:
            raise ValueError(f"Unknown synthetic generator: {name}")
        img = GENERATORS[name]()
        fp = save_synth(img, out_dir, name)
        generated[name] = str(fp.relative_to(root))
    return generated


def _download_real_kodak(selected: list[str], out_dir: Path, root: Path) -> dict[str, str]:
    print("\n[Real/Kodak]")
    names = _build_kodak_dict(selected)
    files = download_kodak(out_dir, names)
    return {p.name: str(p.relative_to(root)) for p in files}


def _build_jpeg_ladder(
    kodak_dir: Path,
    output_dir: Path,
    source_images: list[str],
    qualities: list[int],
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[str] = []
    print("\n[Derived/JPEG Ladder]")
    for src_name in source_images:
        src = kodak_dir / src_name
        if not src.exists():
            print(f"  ! skip {src_name} (not found)")
            continue
        im = Image.open(src).convert("RGB")
        stem = src.stem
        for q in qualities:
            out = output_dir / f"{stem}_q{q}.jpg"
            im.save(out, format="JPEG", quality=q, optimize=True)
            print(f"  ✓ {out.name}")
            created.append(str(out.relative_to(output_dir.parent.parent)))
    return created


def _write_manifest(output_dir: Path, profile: str, manifest: dict) -> tuple[Path, Path]:
    mdir = output_dir / "manifests"
    mdir.mkdir(parents=True, exist_ok=True)
    json_path = mdir / f"{profile}.json"
    md_path = mdir / f"{profile}.md"

    json_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        f"# WPA Test Set Manifest: {profile}",
        "",
        f"- synthetic: {len(manifest['synthetic_files'])}",
        f"- real_kodak: {len(manifest['real_files'])}",
        f"- derived_jpeg: {len(manifest['derived_files'])}",
        f"- items: {len(manifest['items'])}",
        "",
        "## Items",
    ]
    lines.extend(
        [
            (
                f"- `{item['name']}` ({item['dataset_role']}, {item['priority']})"
                f" -> `{item['path']}`"
            )
            for item in manifest["items"]
        ]
    )
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build WPA test set by profile.")
    parser.add_argument(
        "--profile",
        choices=["smoke", "core", "full", "smoke_visual", "core_visual", "release_visual"],
        default="core_visual",
    )
    parser.add_argument("--output-root", default="test_images", help="Root dir for generated dataset")
    args = parser.parse_args()

    spec = get_profile_spec(args.profile)
    root = Path(args.output_root)
    synth_dir = root / "synthetic"
    real_dir = root / "real"
    derived_dir = root / "derived" / "jpeg_ladder"
    kodak_dir = real_dir / "kodak"

    print(f"Building profile={args.profile} -> {root}")
    synthetic_files = _generate_synthetic(spec["synthetic"], synth_dir, root)
    legacy_kodak = spec.get("legacy_kodak", spec.get("real_kodak", []))
    real_files = _download_real_kodak(legacy_kodak, real_dir, root) if legacy_kodak else {}

    derived_files: list[str] = []
    jl = spec["jpeg_ladder"]
    if jl["enabled"]:
        derived_files = _build_jpeg_ladder(
            kodak_dir=kodak_dir,
            output_dir=derived_dir,
            source_images=jl["source_images"],
            qualities=jl["qualities"],
        )

    items = []
    for entry in spec.get("items", []):
        path = synthetic_files.get(entry["name"]) or real_files.get(entry["name"])
        items.append({**entry, "path": path})

    manifest = {
        "profile": args.profile,
        "synthetic_files": list(synthetic_files.values()),
        "real_files": list(real_files.values()),
        "derived_files": derived_files,
        "items": items,
    }
    json_path, md_path = _write_manifest(root, args.profile, manifest)

    print("\n[Done]")
    print(f"- manifest json: {json_path}")
    print(f"- manifest md  : {md_path}")
    print(
        f"- counts        : synthetic={len(synthetic_files)}, "
        f"real={len(real_files)}, derived={len(derived_files)}"
    )


if __name__ == "__main__":
    main()
