# MATLAB Fixed WPA Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a MATLAB implementation of the fixed-point WPA pipeline that matches the current Python `wpa_fixed` feature set closely enough for direct output comparison.

**Architecture:** Mirror the Python `wpa_fixed` package structure inside a new `matlab/` directory and keep helper boundaries aligned with Python concepts: config normalization, runtime gain generation, interpolation, saturation weighting, gamma handling, and image runner entry point. Validate incrementally by checking MATLAB helper semantics first and then comparing end-to-end MATLAB outputs against Python-generated golden outputs.

**Tech Stack:** MATLAB `.m` files, Python `wpa_fixed`, pytest, repository test images, optional `.mat`/`.json` golden fixtures

---

### Task 1: Add Python-Side Golden Export for MATLAB Validation

**Files:**
- Create: `scripts/export_matlab_wpa_fixed_golden.py`
- Create: `tests/test_export_matlab_wpa_fixed_golden.py`

**Step 1: Write the failing test**

Create a pytest file that verifies the export script writes deterministic metadata for a small set of inputs:

```python
def test_export_matlab_wpa_fixed_golden_writes_case_manifest(tmp_path: Path) -> None:
    ...
    assert manifest["cases"]
    assert manifest["cases"][0]["wa_sel"] in [0, 64, 127]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_export_matlab_wpa_fixed_golden.py -v`
Expected: FAIL because the exporter script does not exist yet.

**Step 3: Write minimal implementation**

Implement `scripts/export_matlab_wpa_fixed_golden.py` to export:

- selected source image paths
- configuration values
- Python `wpa_fixed` output images for `wa_sel = 0, 64, 127`
- optional scalar summaries (`max`, `mean`, shape, dtype)

Keep the format simple and MATLAB-readable.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_export_matlab_wpa_fixed_golden.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_export_matlab_wpa_fixed_golden.py scripts/export_matlab_wpa_fixed_golden.py
git commit -m "feat: add matlab golden export for fixed wpa"
```

### Task 2: Create MATLAB Config and Runtime Gain Helpers

**Files:**
- Create: `matlab/wpa_fixed_config.m`
- Create: `matlab/wpa_fixed_runtime_base_gain.m`
- Create: `matlab/wpa_fixed_runtime_bin_gains.m`
- Create: `matlab/wpa_fixed_atten_curve.m`
- Modify: `docs/plans/2026-03-10-matlab-fixed-wpa-design.md` only if names must change

**Step 1: Write the failing check**

Use a MATLAB smoke script or a minimal `matlab -batch` invocation that expects:

```matlab
cfg = wpa_fixed_config('wa_sel', 0, 'coeff_frac_bits', 8);
assert(all(size(cfg.wa_base_gain_lut_fixed) == [3 3]));
assert(all(size(wpa_fixed_runtime_bin_gains(cfg, 64)) == [12 3]));
```

**Step 2: Run the check to verify it fails**

Run: `matlab -batch "addpath('matlab'); run('matlab/smoke_runtime_gain_test.m')"`
Expected: FAIL because the MATLAB helpers do not exist yet.

**Step 3: Write minimal implementation**

Implement the config and runtime gain helpers so they mirror Python semantics:

- same defaults
- same anchor interpolation
- same attenuation curve
- exact identity at `wa_sel=64`

Avoid over-engineering the config layer beyond what the fixed pipeline needs.

**Step 4: Run the check to verify it passes**

Run the same MATLAB batch command.
Expected: PASS

**Step 5: Commit**

```bash
git add matlab/wpa_fixed_config.m matlab/wpa_fixed_runtime_base_gain.m matlab/wpa_fixed_runtime_bin_gains.m matlab/wpa_fixed_atten_curve.m
git commit -m "feat: add matlab fixed wpa config helpers"
```

### Task 3: Add MATLAB Luma, Interpolation, and Saturation Helpers

**Files:**
- Create: `matlab/wpa_fixed_luma_proxy_u8.m`
- Create: `matlab/wpa_fixed_interpolate_gains.m`
- Create: `matlab/wpa_fixed_sat_weight.m`
- Create: `matlab/smoke_helper_test.m`

**Step 1: Write the failing check**

Add a MATLAB smoke script that checks:

- grey luma returns expected value
- exact-node interpolation matches row values
- saturation weight is `1` for neutral and `0` for strong saturated red

**Step 2: Run the check to verify it fails**

Run: `matlab -batch "addpath('matlab'); run('matlab/smoke_helper_test.m')"`
Expected: FAIL because the helper functions do not exist yet.

**Step 3: Write minimal implementation**

Implement MATLAB helpers with Python-aligned semantics:

- `luma_proxy_u8`
- 12-bin interpolation with optional linear interpolation
- fixed-point saturation protection weight

Keep arithmetic and rounding explicit.

**Step 4: Run the check to verify it passes**

Run the same MATLAB batch command.
Expected: PASS

**Step 5: Commit**

```bash
git add matlab/wpa_fixed_luma_proxy_u8.m matlab/wpa_fixed_interpolate_gains.m matlab/wpa_fixed_sat_weight.m matlab/smoke_helper_test.m
git commit -m "feat: add matlab fixed wpa interpolation helpers"
```

### Task 4: Add MATLAB Gamma Helpers and Core Processing Function

**Files:**
- Create: `matlab/wpa_fixed_degamma.m`
- Create: `matlab/wpa_fixed_engamma.m`
- Create: `matlab/wpa_fixed_process_matlab.m`
- Create: `matlab/smoke_process_test.m`

**Step 1: Write the failing check**

Use a MATLAB smoke script that runs a tiny RGB image through:

```matlab
img = uint8(repmat(reshape(uint8([128 128 128]), [1 1 3]), [4 4 1]));
cfg = wpa_fixed_config('wa_sel', 0);
out = wpa_fixed_process_matlab(img, cfg);
assert(isa(out, 'uint8'));
assert(all(size(out) == size(img)));
```

**Step 2: Run the check to verify it fails**

Run: `matlab -batch "addpath('matlab'); run('matlab/smoke_process_test.m')"`
Expected: FAIL because the core process function does not exist yet.

**Step 3: Write minimal implementation**

Implement:

- `srgb`, `power`, and `none` gamma modes
- float degamma/engamma
- integer fixed-point middle path
- exact identity shortcut for `wa_sel=64`

Follow Python stage order as closely as possible.

**Step 4: Run the check to verify it passes**

Run the same MATLAB batch command.
Expected: PASS

**Step 5: Commit**

```bash
git add matlab/wpa_fixed_degamma.m matlab/wpa_fixed_engamma.m matlab/wpa_fixed_process_matlab.m matlab/smoke_process_test.m
git commit -m "feat: add matlab fixed wpa core pipeline"
```

### Task 5: Add MATLAB Image Runner and End-to-End Validator

**Files:**
- Create: `matlab/run_wpa_fixed_image.m`
- Create: `matlab/validate_wpa_fixed_against_python.m`
- Create: `matlab/smoke_validate_against_python.m`
- Modify: `scripts/export_matlab_wpa_fixed_golden.py` if validator needs extra metadata

**Step 1: Write the failing check**

Prepare a golden export and then run a MATLAB validator that expects:

- `wa_sel=64` exact output match
- bounded differences on `wa_sel=0` and `wa_sel=127`

Example MATLAB assertions:

```matlab
assert(case64.max_abs == 0);
assert(case0.max_abs <= 2);
assert(case127.max_abs <= 2);
```

**Step 2: Run the check to verify it fails**

Run:

```bash
python scripts/export_matlab_wpa_fixed_golden.py
matlab -batch "addpath('matlab'); run('matlab/smoke_validate_against_python.m')"
```

Expected: FAIL until the validator and/or runner exist.

**Step 3: Write minimal implementation**

Implement:

- MATLAB image-runner function
- MATLAB validator that reads Python-generated outputs
- per-case metrics: `max_abs`, `mean_abs`, `p99_abs`

Keep the validator report machine-readable and concise.

**Step 4: Run the check to verify it passes**

Run the same commands again.
Expected: PASS or PASS-with-documented tolerated non-identity differences.

**Step 5: Commit**

```bash
git add matlab/run_wpa_fixed_image.m matlab/validate_wpa_fixed_against_python.m matlab/smoke_validate_against_python.m scripts/export_matlab_wpa_fixed_golden.py
git commit -m "feat: add matlab validator for fixed wpa"
```

### Task 6: Document MATLAB Usage and Verification Flow

**Files:**
- Modify: `README.md`
- Optionally create: `matlab/README.md`
- Optionally update: `commands.md`

**Step 1: Write the failing verification target**

The documentation should be considered incomplete until these commands are documented and executable:

```bash
python scripts/export_matlab_wpa_fixed_golden.py
matlab -batch "addpath('matlab'); run('matlab/smoke_validate_against_python.m')"
```

**Step 2: Write minimal implementation**

Document:

- where MATLAB files live
- how to run a single image
- how to export Python golden cases
- how to compare MATLAB against Python fixed outputs

**Step 3: Run verification**

Run:

```bash
pytest tests/test_export_matlab_wpa_fixed_golden.py -v
python scripts/export_matlab_wpa_fixed_golden.py
matlab -batch "addpath('matlab'); run('matlab/smoke_validate_against_python.m')"
```

Expected:

- pytest PASS
- export completes
- MATLAB validation runs successfully

**Step 4: Commit**

```bash
git add README.md matlab/README.md commands.md
git commit -m "docs: add matlab fixed wpa usage"
```
