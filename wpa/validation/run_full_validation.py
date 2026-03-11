from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from validation.linear_interp_basis.evaluate_anchor_interp import evaluate_anchor_interp_error
from validation.dataset_profiles import get_profile_spec
from scripts.download_test_images import REAL_WORLD_IMAGE_GROUPS
from wpa import WPAConfig, wpa_process_rgb_uint8
from wpa_fixed import FixedWPAConfig, wpa_fixed_process
from wpa_fixed.hw_stats import collect_hw_stats


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


REAL_WORLD_GROUP_FOCUS = {
    "public_portrait": ["skin_plausibility", "neutral_cast"],
    "public_hdr_window": ["hdr_transition", "near_white_drift"],
    "public_night_neon": ["mixed_colored_light", "highlight_shift"],
    "public_ui_workspace": ["ui_neutrality", "edge_artifact"],
    "research_mixed_light": ["mixed_illumination_transition", "neutral_cast"],
    "research_outdoor_sanity": ["outdoor_sanity", "shadow_highlight_balance"],
}


def run_pytest() -> tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q"],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(PROJECT_ROOT),
    )
    output = (proc.stdout + "\n" + proc.stderr).strip()
    return proc.returncode == 0, output


def evaluate_float_vs_fixed(height: int, width: int, seed: int, coeff_frac_bits: int) -> dict:
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    wa_values = [0, 32, 64, 96, 127]
    rows = []
    for wa in wa_values:
        float_cfg = WPAConfig(
            wa_sel=wa,
            sat_en=False,
            gamma_mode="srgb",
            use_gamma_lut=False,
        )
        fixed_cfg = FixedWPAConfig(
            wa_sel=wa,
            sat_en=False,
            gamma_mode="srgb",
            coeff_frac_bits=coeff_frac_bits,
            frac_bits=10,
        )
        out_f = wpa_process_rgb_uint8(img, float_cfg).astype(np.int16)
        out_q = wpa_fixed_process(img, fixed_cfg).astype(np.int16)
        diff = np.abs(out_f - out_q).reshape(-1)
        rows.append(
            {
                "wa_sel": wa,
                "mean_abs": float(diff.mean()),
                "p99_abs": float(np.percentile(diff, 99)),
                "max_abs": int(diff.max()),
            }
        )
    return {"coeff_frac_bits": coeff_frac_bits, "rows": rows}


def _format_anchor_table(metrics_8: dict, metrics_10: dict) -> str:
    lines = [
        "| coeff_frac_bits | MAE | P99 Abs | Max Abs |",
        "|---:|---:|---:|---:|",
    ]
    for m in (metrics_8, metrics_10):
        lines.append(
            f"| {m['coeff_frac_bits']} | {m['mae']:.6f} | {m['p99_abs']:.6f} | {m['max_abs']:.6f} |"
        )
    return "\n".join(lines)


def _format_consistency_table(consistency: dict) -> str:
    lines = [
        f"### coeff_frac_bits={consistency['coeff_frac_bits']}",
        "",
        "| WA_SEL | Mean Abs (LSB) | P99 Abs (LSB) | Max Abs (LSB) |",
        "|---:|---:|---:|---:|",
    ]
    for row in consistency["rows"]:
        lines.append(
            f"| {row['wa_sel']} | {row['mean_abs']:.4f} | {row['p99_abs']:.2f} | {row['max_abs']} |"
        )
    return "\n".join(lines)


def _build_checks(
    pytest_enabled: bool,
    pytest_ok: bool,
    anchor_8: dict,
    anchor_10: dict,
    hw8: dict,
    hw10: dict,
    cons8: dict,
    cons10: dict,
) -> list[CheckResult]:
    checks: list[CheckResult] = []
    if pytest_enabled:
        checks.append(
            CheckResult(
                name="Pytest",
                passed=pytest_ok,
                detail="`python -m pytest tests -q` returns exit code 0",
            )
        )
    else:
        checks.append(
            CheckResult(
                name="Pytest",
                passed=True,
                detail="Skipped by `--skip-pytest`",
            )
        )

    for metric in (anchor_8, anchor_10):
        passed = metric["mae"] <= 0.012 and metric["max_abs"] <= 0.03
        checks.append(
            CheckResult(
                name=f"Anchor interpolation (coeff={metric['coeff_frac_bits']})",
                passed=passed,
                detail=f"MAE={metric['mae']:.6f} (<=0.012), MaxAbs={metric['max_abs']:.6f} (<=0.03)",
            )
        )

    checks.append(
        CheckResult(
            name="HW static bytes",
            passed=(hw8["static"]["total_bytes"] == 36 and hw10["static"]["total_bytes"] == 41),
            detail=(
                f"coeff8={hw8['static']['total_bytes']}B (expect 36B), "
                f"coeff10={hw10['static']['total_bytes']}B (expect 41B)"
            ),
        )
    )

    for metrics in (cons8, cons10):
        rows = metrics["rows"]
        identity_max = next(r["max_abs"] for r in rows if r["wa_sel"] == 64)
        mean_max = max(r["mean_abs"] for r in rows)
        p99_max = max(r["p99_abs"] for r in rows)
        max_abs = max(r["max_abs"] for r in rows)
        passed = mean_max < 0.5 and p99_max <= 2.0 and max_abs <= 4 and identity_max == 0
        checks.append(
            CheckResult(
                name=f"Float-vs-fixed consistency (coeff={metrics['coeff_frac_bits']})",
                passed=passed,
                detail=(
                    f"max(mean)={mean_max:.4f} < 0.5, max(p99)={p99_max:.2f} <= 2, "
                    f"global max={max_abs} <= 4, wa64 max={identity_max} == 0"
                ),
            )
        )
    return checks


def _format_checks_table(checks: list[CheckResult]) -> str:
    lines = ["| Check | Status | Detail |", "|---|---|---|"]
    for c in checks:
        status = "PASS" if c.passed else "FAIL"
        lines.append(f"| {c.name} | {status} | {c.detail} |")
    return "\n".join(lines)


def _build_visual_role_table(items: list[dict]) -> str:
    lines = [
        "| Item | Failure Modes | Recommended WA_SEL | Pass Hint |",
        "|---|---|---|---|",
    ]
    for item in items:
        modes = ", ".join(item.get("failure_modes", item.get("visual_risk", [])))
        wa_values = ", ".join(str(v) for v in item.get("recommended_wa_sel", []))
        lines.append(
            f"| {item['name']} | {modes} | {wa_values or '-'} | {item.get('pass_hint', item.get('expected_observation', '-'))} |"
        )
    return "\n".join(lines)


def _build_real_sanity_table(groups: list[str]) -> str:
    lines = [
        "| Group | Assets | Failure Modes | Recommended WA_SEL |",
        "|---|---|---|---|",
    ]
    for group in groups:
        entries = REAL_WORLD_IMAGE_GROUPS.get(group, [])
        asset_names = ", ".join(entry["filename"] for entry in entries) or "-"
        modes = ", ".join(REAL_WORLD_GROUP_FOCUS.get(group, ["real_world_sanity"]))
        lines.append(f"| {group} | {asset_names} | {modes} | 0, 64, 127 |")
    return "\n".join(lines)


def _format_failure_mode_coverage(items: list[dict]) -> str:
    counter: Counter[str] = Counter()
    for item in items:
        counter.update(item.get("failure_modes", item.get("visual_risk", [])))
    if not counter:
        return "- none"
    return ", ".join(f"`{name}`×{count}" for name, count in sorted(counter.items()))


def _render_visual_sections(profile_name: str) -> str:
    spec = get_profile_spec(profile_name)
    items = spec.get("items", [])
    p0_items = [item for item in items if item.get("dataset_role") == "neutral_stability_core"]
    p1_items = [item for item in items if item.get("dataset_role") == "color_side_effect_set"]
    p2_groups = spec.get("real_sanity", [])

    sections = [
        "## P0 Neutral Stability",
        "",
        f"Failure-Mode Coverage: {_format_failure_mode_coverage(p0_items)}",
        "",
        "Recommended WA_SEL: `0, 64, 127` plus any item-specific extremes.",
        "",
        _build_visual_role_table(p0_items),
        "",
        "## P1 Color Side Effects",
        "",
        f"Failure-Mode Coverage: {_format_failure_mode_coverage(p1_items)}",
        "",
        "Recommended WA_SEL: `0, 64, 127` for hue-family comparison.",
        "",
        _build_visual_role_table(p1_items),
        "",
        "## P2 Real-World Sanity",
        "",
        "Failure-Mode Coverage: mixed-light, portrait plausibility, HDR transition, UI neutrality.",
        "",
        "Recommended WA_SEL: `0, 64, 127`.",
        "",
        _build_real_sanity_table(p2_groups),
    ]
    return "\n".join(sections)


def _render_report(
    template_path: Path,
    *,
    timestamp: str,
    pytest_summary: str,
    anchor_table: str,
    hw_table: str,
    consistency_table: str,
    check_table: str,
    visual_sections: str,
    verdict: str,
) -> str:
    text = template_path.read_text(encoding="utf-8")
    replacements = {
        "{{timestamp}}": timestamp,
        "{{pytest_summary}}": pytest_summary,
        "{{anchor_table}}": anchor_table,
        "{{hw_table}}": hw_table,
        "{{consistency_table}}": consistency_table,
        "{{check_table}}": check_table,
        "{{visual_sections}}": visual_sections,
        "{{verdict}}": verdict,
    }
    for key, value in replacements.items():
        text = text.replace(key, value)
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run full WPA validation and write Markdown report.")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "validation_report.md")
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-pytest", action="store_true")
    parser.add_argument("--visual-profile", default="release_visual")
    args = parser.parse_args(argv)

    pytest_ok = True
    pytest_output = "Skipped by flag."
    if not args.skip_pytest:
        pytest_ok, pytest_output = run_pytest()

    anchor_8 = evaluate_anchor_interp_error(coeff_frac_bits=8)
    anchor_10 = evaluate_anchor_interp_error(coeff_frac_bits=10)

    hw8 = collect_hw_stats(FixedWPAConfig(coeff_frac_bits=8, frac_bits=10))
    hw10 = collect_hw_stats(FixedWPAConfig(coeff_frac_bits=10, frac_bits=10))
    hw_table = (
        "| coeff_frac_bits | static bytes | runtime scratch bytes |\n"
        "|---:|---:|---:|\n"
        f"| 8 | {hw8['static']['total_bytes']} | {hw8['runtime_buffers']['wa_runtime_bin_gain_bytes']} |\n"
        f"| 10 | {hw10['static']['total_bytes']} | {hw10['runtime_buffers']['wa_runtime_bin_gain_bytes']} |"
    )

    cons8 = evaluate_float_vs_fixed(args.height, args.width, args.seed, coeff_frac_bits=8)
    cons10 = evaluate_float_vs_fixed(args.height, args.width, args.seed + 1, coeff_frac_bits=10)

    checks = _build_checks(
        pytest_enabled=not args.skip_pytest,
        pytest_ok=pytest_ok,
        anchor_8=anchor_8,
        anchor_10=anchor_10,
        hw8=hw8,
        hw10=hw10,
        cons8=cons8,
        cons10=cons10,
    )
    verdict = "PASS" if all(c.passed for c in checks) else "FAIL"

    template_path = PROJECT_ROOT / "validation" / "validation_report_template.md"
    report = _render_report(
        template_path,
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        pytest_summary=pytest_output,
        anchor_table=_format_anchor_table(anchor_8, anchor_10),
        hw_table=hw_table,
        consistency_table=f"{_format_consistency_table(cons8)}\n\n{_format_consistency_table(cons10)}",
        check_table=_format_checks_table(checks),
        visual_sections=_render_visual_sections(args.visual_profile),
        verdict=verdict,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote report: {args.output}")
    print(f"Final verdict: {verdict}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
