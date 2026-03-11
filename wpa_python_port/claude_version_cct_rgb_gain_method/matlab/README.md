# MATLAB Fixed WPA

This directory contains a MATLAB implementation intended to mirror the Python `wpa_fixed` pipeline.

## Main Entry Points

- `wpa_fixed_config.m`
- `wpa_fixed_process_matlab.m`
- `run_wpa_fixed_image.m`
- `validate_wpa_fixed_against_python.m`

## Typical Flow

1. Export Python golden cases:

```bash
python scripts/export_matlab_wpa_fixed_golden.py
```

2. In MATLAB:

```matlab
addpath('matlab');
summary = validate_wpa_fixed_against_python('matlab/golden_cases');
```

3. Run a single image:

```matlab
addpath('matlab');
run_wpa_fixed_image('input.png', 'output.png', 'wa_sel', 0, 'coeff_frac_bits', 8);
```

## Notes

- The MATLAB version mirrors Python `wpa_fixed` semantics: float gamma, integer fixed-point gain path.
- `wa_sel = 64` should remain identity.
- The current local environment used to author these files did not have a MATLAB or Octave runtime, so validation here is limited to Python-side golden export and static implementation.
