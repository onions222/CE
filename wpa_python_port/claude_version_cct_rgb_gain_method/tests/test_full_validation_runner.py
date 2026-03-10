from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_full_validation_runner_report_renders_visual_role_sections(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    report_path = tmp_path / "validation_report.md"
    cmd = [
        sys.executable,
        str(root / "validation" / "run_full_validation.py"),
        "--skip-pytest",
        "--height",
        "16",
        "--width",
        "16",
        "--output",
        str(report_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=str(root))

    assert report_path.exists(), proc.stderr
    text = report_path.read_text(encoding="utf-8")
    assert "P0 Neutral Stability" in text
    assert "P1 Color Side Effects" in text
    assert "P2 Real-World Sanity" in text
    assert "Recommended WA_SEL" in text
    assert "Failure-Mode Coverage" in text
