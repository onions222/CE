from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def test_export_matlab_wpa_fixed_golden_writes_case_manifest(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    input_path = tmp_path / "input.png"
    output_dir = tmp_path / "golden"
    img = np.full((8, 8, 3), 180, dtype=np.uint8)
    Image.fromarray(img).save(input_path)

    cmd = [
        sys.executable,
        str(root / "scripts" / "export_matlab_wpa_fixed_golden.py"),
        "--output-dir",
        str(output_dir),
        "--image",
        str(input_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=str(root))

    assert proc.returncode == 0, proc.stderr

    manifest_path = output_dir / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["cases"]
    assert {case["wa_sel"] for case in manifest["cases"]} == {0, 64, 127}

    first = manifest["cases"][0]
    assert first["source_image"].endswith("input.png")
    assert Path(output_dir / first["output_image"]).exists()
