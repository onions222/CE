from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_build_test_set_writes_rich_manifest(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    output_root = tmp_path / "dataset"
    cmd = [
        sys.executable,
        str(root / "scripts" / "build_test_set.py"),
        "--profile",
        "smoke_visual",
        "--output-root",
        str(output_root),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=str(root))
    assert proc.returncode == 0, proc.stderr

    manifest = json.loads((output_root / "manifests" / "smoke_visual.json").read_text(encoding="utf-8"))
    assert manifest["profile"] == "smoke_visual"
    assert manifest["items"]

    first = manifest["items"][0]
    assert first["dataset_role"] == "neutral_stability_core"
    assert "expected_observation" in first


def test_build_test_set_manifest_contains_failure_mode_fields(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    output_root = tmp_path / "dataset"
    cmd = [
        sys.executable,
        str(root / "scripts" / "build_test_set.py"),
        "--profile",
        "core_visual",
        "--output-root",
        str(output_root),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=str(root))
    assert proc.returncode == 0, proc.stderr

    manifest = json.loads((output_root / "manifests" / "core_visual.json").read_text(encoding="utf-8"))
    item = next(entry for entry in manifest["items"] if entry["name"] == "near_black_steps")
    assert "failure_modes" in item
    assert "recommended_wa_sel" in item
    assert "pass_hint" in item
