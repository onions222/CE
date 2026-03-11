# MATLAB Fixed WPA Design

## 1. Background

The repository already contains two active implementations of White Point Adjustment:

- `wpa`: floating-point reference implementation
- `wpa_fixed`: fixed-point implementation intended to match delivery constraints while still using float degamma/engamma

The user requested a MATLAB version of the fixed-point implementation, with the same functional scope as the current Python fixed-point version rather than a reduced demo.

That means the MATLAB deliverable must preserve:

- fixed-point runtime gain generation
- fixed-point gain interpolation and application
- configurable `frac_bits` and `coeff_frac_bits`
- the same user-facing control surface as `FixedWPAConfig`
- image I/O entry points equivalent in spirit to the Python CLI

## 2. Goal

Build a MATLAB fixed-point WPA implementation that is behaviorally aligned with `wpa_fixed` in Python and can be verified directly against it using shared images and golden-output comparisons.

## 3. Non-Goals

This phase does not aim to:

- replace the Python fixed-point implementation
- produce a pure-hardware MATLAB model with LUT-only gamma
- create Simulink or HDL-targeted artifacts
- redesign the algorithm or change the fixed-point math model

## 4. Design Principles

1. MATLAB structure should mirror Python structure closely enough that discrepancies are easy to localize.
2. Naming should match Python config and helper names wherever practical.
3. Validation must compare MATLAB against Python `wpa_fixed`, not against subjective visual expectations alone.
4. `wa_sel=64` identity behavior must remain exact.
5. Differences caused by MATLAB vs NumPy rounding must be measured explicitly, not hidden by undocumented logic changes.

## 5. Architecture

The MATLAB implementation should mirror the Python fixed-point package as a small module tree under `matlab/`.

Recommended file structure:

- `matlab/wpa_fixed_process_matlab.m`
- `matlab/wpa_fixed_config.m`
- `matlab/wpa_fixed_runtime_base_gain.m`
- `matlab/wpa_fixed_runtime_bin_gains.m`
- `matlab/wpa_fixed_interpolate_gains.m`
- `matlab/wpa_fixed_luma_proxy_u8.m`
- `matlab/wpa_fixed_sat_weight.m`
- `matlab/wpa_fixed_degamma.m`
- `matlab/wpa_fixed_engamma.m`
- `matlab/run_wpa_fixed_image.m`
- `matlab/validate_wpa_fixed_against_python.m`

Optional helper data:

- `matlab/golden_cases/` for shared `.mat` or `.json` fixtures if needed

## 6. Interface Design

### 6.1 Core Function

Primary MATLAB entry point:

```matlab
out = wpa_fixed_process_matlab(img, cfg)
```

Contract:

- `img` is `HxWx3 uint8`
- `cfg` is a struct returned or normalized by `wpa_fixed_config`
- `out` is `HxWx3 uint8`

### 6.2 Config Factory

Recommended constructor-style helper:

```matlab
cfg = wpa_fixed_config(...)
```

The config should accept Python-aligned names:

- `frac_bits`
- `coeff_frac_bits`
- `wa_en`
- `wa_sel`
- `gamma_mode`
- `gamma_power`
- `luma_nodes`
- `bin_interp`
- `luma_domain`
- `sat_en`
- `sat_s0`
- `sat_s1`
- `cct_warm_k`
- `cct_neutral_k`
- `cct_cool_k`
- `cct_xy_split_k`
- `cct_xy_blend_half_width_k`

It should also derive:

- `ONE`
- `HALF`
- `COEFF_ONE`
- `COEFF_HALF`
- `wa_base_gain_lut_fixed`
- `atten_q_lut_fixed`

### 6.3 Run-Image Entry Point

MATLAB image runner:

```matlab
run_wpa_fixed_image(input_path, output_path, varargin)
```

Purpose:

- serve as the MATLAB equivalent of the Python CLI
- support direct visual spot checks on images
- keep the core processing function separate from I/O code

## 7. Behavioral Mapping to Python

The MATLAB implementation should follow the same pipeline as `wpa_fixed/core.py`:

1. degamma input image to float linear domain
2. convert float linear values to fixed-point integer representation
3. compute `luma_proxy_u8`
4. build runtime 12x3 gain table from 3 anchor gains plus attenuation LUT
5. interpolate per-pixel gains across luma nodes
6. apply gain in integer arithmetic
7. apply saturation protection in integer arithmetic if enabled
8. clip to valid fixed-point range
9. convert back to float linear
10. engamma
11. quantize to `uint8`

This must preserve the current Python design choice:

- degamma and engamma remain floating-point
- gain path and saturation-weight path remain fixed-point integer logic

## 8. Fixed-Point Requirements

### 8.1 Supported Precision Modes

The MATLAB version must support:

- `coeff_frac_bits = 8`
- `coeff_frac_bits = 10`

and default to:

- `frac_bits = 10`
- `coeff_frac_bits = 8`

### 8.2 Runtime Gain Construction

The MATLAB implementation must preserve the same RAM-saving runtime strategy as Python:

- store 3 anchor gains: warm, neutral, cool
- build runtime base gain for the selected `wa_sel`
- expand runtime 12x3 bin gains from base gain plus attenuation LUT

### 8.3 Identity Rule

`wa_sel = 64` must produce identity output exactly.

This is both:

- a functional requirement
- a critical alignment point for Python-vs-MATLAB validation

## 9. Gamma Handling

The MATLAB implementation should support the same modes as Python:

- `srgb`
- `power`
- `none`

Design choice:

- implement `srgb` directly in MATLAB formulas first
- keep LUT-specific gamma shortcuts out of scope unless needed for parity later

This is acceptable because the current Python fixed-point path also uses float gamma helpers rather than an all-integer gamma pipeline.

## 10. Saturation Protection

The MATLAB implementation must match the Python fixed-point semantics:

- default `sat_en = false`
- thresholds expressed in the same user-facing scale
- fixed-point weight computation in the gamma-domain integer path

This is important because color-drift discrepancies are likely to show up first in:

- high-saturation colors
- mixed-illumination scenes
- highlight-adjacent colored regions

## 11. Validation Strategy

Validation should be explicit and layered.

### 11.1 Unit-Level MATLAB Checks

MATLAB-only checks should cover:

- `wa_sel -> runtime_base_gain` anchor behavior
- `runtime_bin_gains` shape and identity at `wa_sel=64`
- luma proxy correctness for simple RGB cases
- interpolation continuity at node boundaries

### 11.2 Python-vs-MATLAB Golden Cases

The MATLAB validator should compare against Python `wpa_fixed` using a small curated image set.

Recommended images:

- `test_images/synthetic/01_grey_ramp.png`
- `test_images/synthetic/near_white_steps.png`
- `test_images/synthetic/11_ui_text_contrast.png`
- `test_images/real/real_sanity/public_hdr_window/gfp_sunroom.jpg`

Recommended WA settings:

- `0`
- `64`
- `127`

Recommended metrics:

- `max_abs`
- `mean_abs`
- `p99_abs`

### 11.3 Acceptance Criteria

Suggested targets:

- `wa_sel=64`: bit-exact
- `wa_sel in {0,127}`: `max_abs <= 1` preferred, `<= 2` acceptable if caused by documented rounding differences

If larger discrepancies occur, they must be localized by stage rather than hidden by threshold inflation.

## 12. Delivery Scope

The MATLAB version is complete for this phase when:

- it can process images with the same functional controls as `wpa_fixed`
- it can be run from a MATLAB-side image entry point
- it has a validator that compares outputs against Python `wpa_fixed`
- the repository documents how to run the MATLAB implementation and how to compare results

## 13. Risks

Main risks:

- MATLAB and NumPy differ subtly in default rounding and integer promotion behavior
- MATLAB image I/O and array class handling can introduce accidental type conversions
- shape conventions may diverge if helper functions are not kept close to Python semantics

Mitigations:

- normalize all key intermediate types explicitly
- validate step-by-step, not only final images
- keep MATLAB helper names and argument roles close to Python counterparts
