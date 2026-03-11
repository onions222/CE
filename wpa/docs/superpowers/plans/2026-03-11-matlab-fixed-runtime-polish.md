# MATLAB Fixed Runtime Polish Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add detailed Chinese bit-width comments, rewrite the MATLAB README, and add a folder-level batch runner for the MATLAB fixed WPA flow.

**Architecture:** Keep the current condensed MATLAB structure (`config` / `process` / `runtime_bin_gains` plus entrypoints), but improve readability and usability. Treat comments and README as part of the public interface, and add a dedicated batch entrypoint rather than overloading the single-image runner.

**Tech Stack:** MATLAB `.m` files, Python `pytest` text-based tests, existing MATLAB batch validation flow.

---

## Chunk 1: Tests And Docs Surface

### Task 1: Add failing tests for MATLAB docs and batch runner

**Files:**
- Modify: `tests/test_matlab_hw_style_runtime.py`
- Create: `matlab/run_wpa_fixed_folder.m`
- Modify: `matlab/README.md`

- [ ] **Step 1: Write failing tests**
- [ ] **Step 2: Run `python -m pytest tests/test_matlab_hw_style_runtime.py -q` and verify failure**
- [ ] **Step 3: Implement the minimal code/docs to satisfy the tests**
- [ ] **Step 4: Re-run `python -m pytest tests/test_matlab_hw_style_runtime.py -q` and verify pass**

## Chunk 2: Comment Upgrade

### Task 2: Add detailed Chinese comments with bit-width callouts

**Files:**
- Modify: `matlab/wpa_fixed_config.m`
- Modify: `matlab/wpa_fixed_process_matlab.m`
- Modify: `matlab/wpa_fixed_runtime_bin_gains.m`

- [ ] **Step 1: Expand comments so raw-code meaning, scale factor, and effective bit width are explicit**
- [ ] **Step 2: Ensure key intermediate variables document container type vs. effective width**
- [ ] **Step 3: Re-run the text-based pytest target**

## Chunk 3: Batch Entry

### Task 3: Add folder batch processing entrypoint

**Files:**
- Create: `matlab/run_wpa_fixed_folder.m`
- Modify: `matlab/README.md`

- [ ] **Step 1: Implement a non-recursive folder scanner for common image extensions**
- [ ] **Step 2: Reuse `wpa_fixed_process_matlab` to process each image**
- [ ] **Step 3: Save outputs into a target folder with stable names and console summary**
- [ ] **Step 4: Document single-image and folder usage in README**

## Chunk 4: Verification

### Task 4: Validate the polished MATLAB surface

**Files:**
- Modify: `tests/test_matlab_hw_style_runtime.py`
- Modify: `matlab/README.md`

- [ ] **Step 1: Run `python -m pytest tests/test_matlab_hw_style_runtime.py -q`**
- [ ] **Step 2: Run `python -m pytest tests/test_fixed_coeff_lut.py tests/test_matlab_hw_style_runtime.py -q`**
- [ ] **Step 3: Summarize any remaining gaps, especially around full MATLAB runtime execution**
