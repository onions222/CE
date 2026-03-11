from __future__ import annotations

from pathlib import Path


def _read(path: str) -> str:
    root = Path(__file__).resolve().parents[1]
    return (root / path).read_text(encoding="utf-8")


def test_matlab_runtime_bin_gains_uses_prequantized_cap_tables() -> None:
    text = _read("matlab/wpa_fixed_runtime_bin_gains.m")
    assert "cfg.warm_highlight_green_caps_fixed" in text
    assert "cfg.cool_highlight_green_caps_fixed" in text
    assert "cfg.cool_highlight_blue_caps_fixed" in text
    assert "0.06" not in text
    assert "0.02" not in text
    assert "0.025" not in text
    assert "0.03" not in text
    assert "0.035" not in text
    assert "0.15" not in text
    assert "0.12" not in text
    assert "0.08" not in text
    assert "double(" not in text


def test_matlab_runtime_config_exposes_fixed_cap_tables() -> None:
    text = _read("matlab/wpa_fixed_config.m")
    assert "'warm_highlight_green_caps_fixed'" in text
    assert "'cool_highlight_green_caps_fixed'" in text
    assert "'cool_highlight_blue_caps_fixed'" in text


def test_matlab_runtime_process_prefers_unsigned_storage_types() -> None:
    text = _read("matlab/wpa_fixed_process_matlab.m")
    assert "uint16" in text
    assert "uint32" in text

