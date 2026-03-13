# WPA Low-Luma Bypass Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a low-luma bypass/blend stage that avoids unstable `Q0.8` color artifacts for `Y <= 31`, with a smooth transition through `Y < 63`.

**Architecture:** Keep the existing gain tables and fixed-point interpolation intact, but add a post-process blend gate driven by gamma-domain luma proxy `Y`. The gate returns the original input pixel for the darkest region and linearly blends toward the normal WPA output through a short transition window.

**Tech Stack:** Python fixed-point WPA, MATLAB `hw_runtime`, pytest, Pillow, NumPy, MATLAB batch verification

---

## Chunk 1: Test Coverage

### Task 1: Add fixed-point regression tests for low-luma bypass/blend

**Files:**
- Modify: `tests/test_fixed_coeff_lut.py`

- [ ] **Step 1: Write failing tests for representative low-luma pixels**

Add tests that cover:

- a dark blue-biased shoulder pixel such as `(16, 18, 22)` under `wa_sel=0`
- a neutral dark pixel such as `(14, 16, 20)` or a low neutral ramp sample
- a pixel above the blend window to ensure the existing path is preserved

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `pytest tests/test_fixed_coeff_lut.py -k low_luma -v`

Expected: FAIL because the current fixed-point path still emits the colored ring behavior.

- [ ] **Step 3: Add a test for the transition behavior**

Assert that:

- `Y <= 31` returns input unchanged
- `31 < Y < 63` is neither equal to input nor equal to base WPA output for a low-luma sample
- `Y >= 63` matches the legacy WPA path

- [ ] **Step 4: Run the targeted tests again**

Run: `pytest tests/test_fixed_coeff_lut.py -k low_luma -v`

Expected: FAIL until the implementation is added.

## Chunk 2: Python Fixed-Point Implementation

### Task 2: Add low-luma bypass/blend control to the fixed-point Python path

**Files:**
- Modify: `wpa_fixed/config.py`
- Modify: `wpa_fixed/__init__.py` if new config options need export
- Modify: `wpa_fixed/process.py` or the current fixed-point processing file, if separate

- [ ] **Step 1: Add config fields for the low-luma transition**

Introduce defaults for:

- enable flag
- bypass threshold `31`
- blend end threshold `63`

- [ ] **Step 2: Implement the gate in the fixed-point processing path**

Requirements:

- Use the same gamma-domain luma proxy already used by the runtime path
- For `Y <= 31`, return the original input pixel
- For `31 < Y < 63`, linearly blend between input and computed WPA output
- For `Y >= 63`, keep the existing output

- [ ] **Step 3: Keep the implementation local and minimal**

Do not refactor unrelated gain generation or interpolation logic.

- [ ] **Step 4: Run the new fixed-point regression tests**

Run: `pytest tests/test_fixed_coeff_lut.py -k low_luma -v`

Expected: PASS

## Chunk 3: MATLAB hw_runtime Parity

### Task 3: Mirror the same gate in MATLAB `hw_runtime`

**Files:**
- Modify: `matlab/hw_runtime/hw_fixed_config.m`
- Modify: `matlab/hw_runtime/hw_fixed_process_image.m`

- [ ] **Step 1: Add matching config fields to MATLAB**

Expose the same thresholds and enable flag used in Python.

- [ ] **Step 2: Implement the low-luma bypass/blend gate**

Requirements:

- Use gamma-domain `luma_u8`
- For `Y <= 31`, output the original input pixel
- For `31 < Y < 63`, blend input and WPA output in gamma domain
- Preserve the existing path above the transition window

- [ ] **Step 3: Keep behavior aligned with Python fixed-point**

Use the same thresholds and rounding direction where practical.

- [ ] **Step 4: Run a MATLAB batch sanity check**

Run:

```bash
/Applications/MATLAB_R2025a.app/bin/matlab -batch "cd('/Users/onion/Desktop/code/CE/wpa'); addpath('matlab/hw_runtime'); run_hw_fixed_image('test_images/synthetic/12_specular_clip_chart.png','/tmp/12_specular_clip_chart_matlab_low_luma_gate.png','wa_sel',0);"
```

Expected: command exits `0` and writes the gated output image.

## Chunk 4: Verification

### Task 4: Re-run targeted visual verification on sensitive images

**Files:**
- No code changes required

- [ ] **Step 1: Generate updated outputs for sensitive synthetic cases**

At minimum:

- `12_specular_clip_chart`
- `near_black_steps`
- `01_grey_ramp`
- `ui_dark_theme_chart`
- `shadow_with_colored_highlight`

- [ ] **Step 2: Compare against the old `Q0.8` outputs**

Check:

- green-dominant pixel counts
- representative sample points
- visual panels/crops around highlight shoulders

- [ ] **Step 3: Run focused pytest coverage**

Run:

```bash
pytest tests/test_fixed_coeff_lut.py -v
```

Expected: PASS

- [ ] **Step 4: Summarize outcomes and risks**

Document:

- whether the ring is removed
- whether the transition introduces any new discontinuity
- any remaining artifacts in dark UI or grey-step scenes
