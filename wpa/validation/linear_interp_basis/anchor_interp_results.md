# 3-Anchor Linear Interpolation Validation

## Purpose
Validate whether `warm/neutral/cool` 3-anchor piecewise linear interpolation is a reasonable approximation for the `WA_SEL -> base gain` mapping.

## Reference and Candidate
- Reference: float pipeline `build_cct_gain_lut()` 128-point base gain.
- Candidate: fixed pipeline `runtime_base_gain_fixed()` with 3 anchors:
  - `WA_SEL=0` warm
  - `WA_SEL=64` neutral
  - `WA_SEL=127` cool

## Command
```bash
python validation/linear_interp_basis/evaluate_anchor_interp.py --coeff-frac-bits 8
python validation/linear_interp_basis/evaluate_anchor_interp.py --coeff-frac-bits 10
```

## Results
- `coeff_frac_bits=8`
  - `MAE = 0.010408`
  - `P99 Abs = 0.028085`
  - `Max Abs = 0.028228`
  - `MAE by channel: R=0.014737, G=0.002485, B=0.014000`
  - `Max by channel: R=0.028228, G=0.006948, B=0.025095`

- `coeff_frac_bits=10`
  - `MAE = 0.010742`
  - `P99 Abs = 0.027249`
  - `Max Abs = 0.027910`
  - `MAE by channel: R=0.014964, G=0.003024, B=0.014239`
  - `Max by channel: R=0.027910, G=0.006960, B=0.024634`

## Conclusion
Under current CCT-to-gain model, 3-anchor linear interpolation introduces about 1% average absolute error and about 2.8% worst-case absolute error versus 128-point reference, which is acceptable for current low-resource hardware target.
