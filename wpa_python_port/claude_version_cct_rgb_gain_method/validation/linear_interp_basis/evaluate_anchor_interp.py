"""Evaluate 3-anchor WA interpolation error against 128-point reference.

This script is intentionally placed under validation/ to avoid affecting
production package/test delivery paths.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import numpy as np

# Ensure project root is importable when running this script directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wpa.config import build_cct_gain_lut
from wpa_fixed.config import FixedWPAConfig


def evaluate_anchor_interp_error(coeff_frac_bits: int = 8) -> dict:
    cfg = FixedWPAConfig(coeff_frac_bits=coeff_frac_bits)
    ref = build_cct_gain_lut().astype(np.float64)  # (128,3)
    est = np.stack(
        [cfg.runtime_base_gain_fixed(wa) for wa in range(128)], axis=0
    ).astype(np.float64) / float(1 << coeff_frac_bits)

    absd = np.abs(est - ref)
    mean_by_ch = absd.mean(axis=0)
    max_by_ch = absd.max(axis=0)
    return {
        "coeff_frac_bits": coeff_frac_bits,
        "mae": float(absd.mean()),
        "max_abs": float(absd.max()),
        "p99_abs": float(np.quantile(absd, 0.99)),
        "mae_by_channel": {
            "R": float(mean_by_ch[0]),
            "G": float(mean_by_ch[1]),
            "B": float(mean_by_ch[2]),
        },
        "max_abs_by_channel": {
            "R": float(max_by_ch[0]),
            "G": float(max_by_ch[1]),
            "B": float(max_by_ch[2]),
        },
    }


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Evaluate 3-anchor WA interpolation error.")
    p.add_argument("--coeff-frac-bits", type=int, default=8, choices=[8, 10])
    p.add_argument("--json", action="store_true", help="Print JSON only")
    args = p.parse_args(argv)

    m = evaluate_anchor_interp_error(coeff_frac_bits=args.coeff_frac_bits)
    if args.json:
        print(json.dumps(m, indent=2, sort_keys=True))
        return 0

    print(f"Anchor Interp Error (coeff_frac_bits={m['coeff_frac_bits']})")
    print(f"MAE      : {m['mae']:.6f}")
    print(f"P99 Abs  : {m['p99_abs']:.6f}")
    print(f"Max Abs  : {m['max_abs']:.6f}")
    print(
        "MAE RGB  : "
        f"R={m['mae_by_channel']['R']:.6f}, "
        f"G={m['mae_by_channel']['G']:.6f}, "
        f"B={m['mae_by_channel']['B']:.6f}"
    )
    print(
        "MAX RGB  : "
        f"R={m['max_abs_by_channel']['R']:.6f}, "
        f"G={m['max_abs_by_channel']['G']:.6f}, "
        f"B={m['max_abs_by_channel']['B']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
