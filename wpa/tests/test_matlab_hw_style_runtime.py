from __future__ import annotations

from pathlib import Path


def _read(path: str) -> str:
    root = Path(__file__).resolve().parents[1]
    return (root / path).read_text(encoding="utf-8")


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_matlab_runtime_bin_gains_uses_prequantized_cap_tables() -> None:
    text = _read("matlab/wpa_fixed_runtime_bin_gains.m")
    assert "cfg.warm_highlight_red_caps_fixed" in text
    assert "cfg.warm_highlight_green_caps_fixed" in text
    assert "cfg.warm_highlight_blue_floors_fixed" in text
    assert "cfg.cool_highlight_green_caps_fixed" in text
    assert "cfg.cool_highlight_blue_caps_fixed" in text
    assert "0.06" not in text
    assert "0.30" not in text
    assert "0.22" not in text
    assert "0.16" not in text
    assert "0.10" not in text
    assert "0.20" not in text
    assert "0.13" not in text
    assert "0.02" not in text
    assert "0.025" not in text
    assert "0.03" not in text
    assert "0.035" not in text
    assert "0.15" not in text
    assert "0.12" not in text
    assert "0.08" not in text
    assert "double(" not in text
    assert "uint16(" not in text
    assert "uint32(" not in text
    assert "int32(" not in text
    assert "'uint16'" not in text


def test_matlab_runtime_config_exposes_fixed_cap_tables() -> None:
    text = _read("matlab/wpa_fixed_config.m")
    assert "'warm_highlight_red_caps_fixed'" in text
    assert "'warm_highlight_green_caps_fixed'" in text
    assert "'warm_highlight_blue_floors_fixed'" in text
    assert "'cool_highlight_green_caps_fixed'" in text
    assert "'cool_highlight_blue_caps_fixed'" in text
    assert "只是 MATLAB 存储容器" not in text
    assert "uint16(" not in text
    assert "uint8(" not in text


def test_matlab_runtime_process_uses_raw_code_language_without_host_integer_casts() -> None:
    text = _read("matlab/wpa_fixed_process_matlab.m")
    assert "storage container" not in text
    assert "effective hw width" not in text
    assert "cfg.pixel_bits" in text
    assert "cfg.coeff_bits" in text
    assert "cfg.mul_bits" in text
    assert "uint16(" not in text
    assert "uint32(" not in text
    assert "int32(" not in text
    assert "int16(" not in text


def test_matlab_fixed_runtime_is_consolidated_into_fewer_files() -> None:
    root = _root()
    removed_helpers = [
        "matlab/wpa_fixed_atten_curve.m",
        "matlab/wpa_fixed_degamma.m",
        "matlab/wpa_fixed_engamma.m",
        "matlab/wpa_fixed_interpolate_gains.m",
        "matlab/wpa_fixed_luma_proxy_u8.m",
        "matlab/wpa_fixed_runtime_base_gain.m",
        "matlab/wpa_fixed_sat_weight.m",
    ]
    for rel in removed_helpers:
        assert not (root / rel).exists(), rel


def test_matlab_fixed_process_uses_raw_code_scale_factor_language() -> None:
    text = _read("matlab/wpa_fixed_process_matlab.m")
    assert "raw code" in text
    assert "scale factor" in text
    assert "pixel_code" in text
    assert "gain_code" in text


def test_matlab_fixed_core_files_include_chinese_bit_width_comments() -> None:
    for rel in [
        "matlab/wpa_fixed_config.m",
        "matlab/wpa_fixed_process_matlab.m",
        "matlab/wpa_fixed_runtime_bin_gains.m",
    ]:
        text = _read(rel)
        assert "位宽" in text, rel
        assert "中文" not in text


def test_matlab_readme_documents_batch_folder_runner() -> None:
    text = _read("matlab/README.md")
    assert "run_wpa_fixed_folder" in text
    assert "文件夹" in text
    assert "raw code" in text


def test_matlab_batch_runner_exists_and_filters_image_extensions() -> None:
    text = _read("matlab/run_wpa_fixed_folder.m")
    assert "dir(" in text
    assert ".png" in text
    assert ".jpg" in text
    assert ".jpeg" in text
    assert "wpa_fixed_process_matlab" in text
    assert "function run_wpa_fixed_folder" not in text
    assert "input_dir =" in text
    assert "output_dir =" in text
    assert "wa_sel =" in text


def test_matlab_batch_runner_documents_user_config_and_wa_presets() -> None:
    text = _read("matlab/run_wpa_fixed_folder.m")
    assert "参数说明" in text
    assert "WA_SEL 常用预设" in text
    assert "0 = 最暖" in text
    assert "64 = 关闭调节" in text
    assert "127 = 最冷" in text


def test_matlab_readme_mentions_wa_presets_for_batch_runner() -> None:
    text = _read("matlab/README.md")
    assert "WA_SEL 常用预设" in text
    assert "64 = 关闭调节" in text


def test_matlab_hw_runtime_directory_contains_independent_core_files() -> None:
    root = _root()
    expected = [
        "matlab/hw_runtime/README.md",
        "matlab/hw_runtime/hw_fixed_config.m",
        "matlab/hw_runtime/hw_fixed_runtime_bin_gains.m",
        "matlab/hw_runtime/hw_fixed_process_image.m",
        "matlab/hw_runtime/run_hw_fixed_image.m",
        "matlab/hw_runtime/run_hw_fixed_folder.m",
        "matlab/hw_runtime/validate_hw_fixed_against_python.m",
    ]
    for rel in expected:
        assert (root / rel).exists(), rel


def test_matlab_hw_runtime_readme_documents_anchor_and_runtime_structure() -> None:
    text = _read("matlab/hw_runtime/README.md")
    assert "3 anchor" in text
    assert "12 luma nodes" in text
    assert "runtime 12x3" in text
    assert "validate_hw_fixed_against_python" in text
    assert "coeff_bits = 9 bit" in text
    assert "pixel_bits = 9 bit" in text
    assert "mul_bits = 18 bit" in text
    assert "Q0.8" in text
    assert "0.1 -> round(0.1 * 256) = 26" in text
    assert "26 / 256" in text


def test_matlab_hw_runtime_is_independent_from_legacy_wpa_fixed_names() -> None:
    for rel in [
        "matlab/hw_runtime/hw_fixed_config.m",
        "matlab/hw_runtime/hw_fixed_runtime_bin_gains.m",
        "matlab/hw_runtime/hw_fixed_process_image.m",
        "matlab/hw_runtime/run_hw_fixed_image.m",
        "matlab/hw_runtime/run_hw_fixed_folder.m",
        "matlab/hw_runtime/validate_hw_fixed_against_python.m",
    ]:
        text = _read(rel)
        assert "wpa_fixed_" not in text, rel
        assert "hw_fixed_" in text or "run_hw_fixed" in text, rel


def test_matlab_hw_runtime_core_files_include_chinese_bit_width_comments() -> None:
    for rel in [
        "matlab/hw_runtime/hw_fixed_config.m",
        "matlab/hw_runtime/hw_fixed_runtime_bin_gains.m",
        "matlab/hw_runtime/hw_fixed_process_image.m",
        "matlab/hw_runtime/run_hw_fixed_folder.m",
        "matlab/hw_runtime/run_hw_fixed_image.m",
        "matlab/hw_runtime/validate_hw_fixed_against_python.m",
    ]:
        text = _read(rel)
        assert "位宽" in text, rel
        assert "raw code" in text, rel
