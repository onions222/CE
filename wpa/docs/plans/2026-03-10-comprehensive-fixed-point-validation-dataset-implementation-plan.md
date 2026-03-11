# Comprehensive Fixed-Point Validation Dataset Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expand the repository's synthetic-first validation dataset, add a small real-world sanity layer, and upgrade manifest/report outputs so each test image carries explicit diagnostic meaning.

**Architecture:** Keep the current script layout intact and extend it in four layers: synthetic generator catalog, profile/manifest metadata, real-image acquisition, and manifest-driven report rendering. Implement through small TDD increments so generator behavior, profile semantics, and report output stay deterministic before any network-backed asset fetching happens.

**Tech Stack:** Python 3, Pillow, pytest, repository scripts under `scripts/` and `validation/`

---

### Task 1: Extend the Synthetic Generator Catalog

**Files:**
- Modify: `scripts/generate_test_images.py`
- Modify: `tests/test_synthetic_generators.py`

**Step 1: Write the failing test**

Add coverage in `tests/test_synthetic_generators.py` for the new generator names and one or two shape-level expectations:

```python
from scripts.generate_test_images import GENERATORS


def test_extended_synthetic_generator_catalog_contains_new_diagnostic_charts() -> None:
    assert "two_axis_neutral_gradient" in GENERATORS
    assert "bin_boundary_triplet_chart" in GENERATORS
    assert "saturation_threshold_ladder" in GENERATORS
    assert "shadow_with_colored_highlight" in GENERATORS
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_synthetic_generators.py -v`
Expected: FAIL because the new generator keys do not exist yet.

**Step 3: Write minimal implementation**

Extend `scripts/generate_test_images.py` with generators for:

- `two_axis_neutral_gradient`
- `bin_boundary_triplet_chart`
- `iso_gray_18_70_pair`
- `midtone_neutral_texture`
- `saturation_threshold_ladder`
- `skin_tone_luma_strip`
- `warm_cool_split_field`
- `shadow_with_colored_highlight`

Also register them in the `GENERATORS` mapping and keep output names ASCII-safe.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_synthetic_generators.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_synthetic_generators.py scripts/generate_test_images.py
git commit -m "feat: add fixed-point diagnostic synthetic charts"
```

### Task 2: Extend Profile Definitions and Manifest Metadata

**Files:**
- Modify: `validation/dataset_profiles.py`
- Modify: `scripts/build_test_set.py`
- Modify: `tests/test_dataset_profiles.py`
- Modify: `tests/test_build_test_set.py`

**Step 1: Write the failing tests**

Add tests for the richer metadata contract and for the new synthetic items being present in the intended profiles:

```python
def test_core_visual_profile_includes_new_failure_mode_driven_items() -> None:
    spec = get_profile_spec("core_visual")
    assert "bin_boundary_triplet_chart" in spec["synthetic"]
    assert "saturation_threshold_ladder" in spec["synthetic"]


def test_build_test_set_manifest_contains_failure_mode_fields(tmp_path: Path) -> None:
    ...
    item = next(entry for entry in manifest["items"] if entry["name"] == "bin_boundary_triplet_chart")
    assert "failure_modes" in item
    assert "recommended_wa_sel" in item
    assert "pass_hint" in item
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dataset_profiles.py tests/test_build_test_set.py -v`
Expected: FAIL because the profile lists and manifest schema are still incomplete.

**Step 3: Write minimal implementation**

Update `validation/dataset_profiles.py` to:

- add the new synthetic chart names to `VISUAL_NEUTRAL_CORE` / `VISUAL_COLOR_SIDE_EFFECT`
- expand metadata from the current `priority` + `visual_risk` model to the richer schema:
  - `inspection_priority`
  - `failure_modes`
  - `expected_artifact_if_fail`
  - `recommended_wa_sel`
  - `suggested_metrics`
  - `pass_hint`
  - `notes`
- keep backward-compatible fields only if still used elsewhere

Update `scripts/build_test_set.py` so manifest writing preserves the richer item dictionaries and adds profile-level summary keys if needed.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_dataset_profiles.py tests/test_build_test_set.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add validation/dataset_profiles.py scripts/build_test_set.py tests/test_dataset_profiles.py tests/test_build_test_set.py
git commit -m "feat: enrich validation dataset metadata"
```

### Task 3: Add Real-World Sanity Image Acquisition

**Files:**
- Modify: `scripts/download_test_images.py`
- Modify: `scripts/build_test_set.py`
- Create: `tests/test_download_test_images.py`
- Modify: `validation/dataset_profiles.py`

**Step 1: Write the failing test**

Create `tests/test_download_test_images.py` with unit coverage around source dictionaries and destination selection, avoiding live network calls by patching the download function:

```python
def test_download_catalog_exposes_real_world_sanity_groups() -> None:
    assert "public_portrait" in REAL_WORLD_IMAGE_GROUPS
    assert "research_mixed_light" in REAL_WORLD_IMAGE_GROUPS


def test_build_release_visual_requests_real_world_sanity_images(monkeypatch, tmp_path: Path) -> None:
    ...
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_download_test_images.py -v`
Expected: FAIL because the grouped real-image catalog and build path do not exist yet.

**Step 3: Write minimal implementation**

Extend `scripts/download_test_images.py` to support:

- grouped real-image metadata for direct-download public images
- curated research-subset image definitions
- role- and source-aware output placement

Keep Kodak as `legacy_baseline` and add new helper functions instead of overloading `download_kodak`.

Update `validation/dataset_profiles.py` and `scripts/build_test_set.py` so `release_visual` can request:

- `real_world_sanity`
- `legacy_kodak`
- optional derived JPEG assets

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_download_test_images.py tests/test_build_test_set.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/download_test_images.py scripts/build_test_set.py validation/dataset_profiles.py tests/test_download_test_images.py tests/test_build_test_set.py
git commit -m "feat: add real-world sanity image sources"
```

### Task 4: Render Manifest-Driven Visual Sections in the Validation Report

**Files:**
- Modify: `validation/run_full_validation.py`
- Modify: `validation/validation_report_template.md`
- Modify: `tests/test_full_validation_runner.py`

**Step 1: Write the failing test**

Replace the current placeholder-only expectations with assertions that the report renders grouped visual sections and item-level guidance:

```python
def test_full_validation_runner_report_renders_visual_role_sections(tmp_path: Path) -> None:
    ...
    assert "P0 Neutral Stability" in text
    assert "failure-mode coverage" in text.lower()
    assert "recommended wa_sel" in text.lower()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_full_validation_runner.py -v`
Expected: FAIL because the report still renders static `TBD` placeholders.

**Step 3: Write minimal implementation**

Update `validation/run_full_validation.py` to:

- load an existing manifest or derive visual item summaries from `validation/dataset_profiles.py`
- group visual items by role/priority
- render compact Markdown tables for:
  - `P0 Neutral Stability`
  - `P1 Color Side Effects`
  - `P2 Real-World Sanity`
- keep the existing numeric validation sections unchanged

Update `validation/validation_report_template.md` placeholders accordingly.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_full_validation_runner.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add validation/run_full_validation.py validation/validation_report_template.md tests/test_full_validation_runner.py
git commit -m "feat: add manifest-driven visual validation summary"
```

### Task 5: Update Recommendations Docs and Rebuild Representative Datasets

**Files:**
- Modify: `validation/test_set_recommendations.md`
- Optionally modify: `README.md`
- Update generated outputs under: `test_images/` (if tracked)
- Update generated report: `outputs/validation_report.md`

**Step 1: Write the failing check**

Use a doc/runtime verification target rather than a new unit test:

Run:

```bash
python scripts/build_test_set.py --profile smoke_visual
python scripts/build_test_set.py --profile core_visual
python validation/run_full_validation.py --skip-pytest
```

Expected initially: output is missing the new charts, new manifest fields, or new visual summaries.

**Step 2: Write minimal implementation**

Refresh `validation/test_set_recommendations.md` to match the expanded chart taxonomy and real-world sanity strategy. Update `README.md` only if command examples or profile descriptions have become inaccurate.

**Step 3: Run verification**

Run:

```bash
pytest tests/test_synthetic_generators.py tests/test_dataset_profiles.py tests/test_build_test_set.py tests/test_download_test_images.py tests/test_full_validation_runner.py -v
python scripts/build_test_set.py --profile smoke_visual
python scripts/build_test_set.py --profile core_visual
python validation/run_full_validation.py --skip-pytest
```

Expected:

- all targeted tests PASS
- manifests include richer visual metadata
- report includes grouped visual validation sections

**Step 4: Commit**

```bash
git add validation/test_set_recommendations.md README.md outputs/validation_report.md test_images
git commit -m "docs: refresh validation dataset guidance"
```

### Task 6: Fetch the Release-Level Real-World Sanity Assets

**Files:**
- No code changes required unless URL fixes are needed
- Generated outputs under: `test_images/real/`

**Step 1: Run the fetch/build command**

Run:

```bash
python scripts/build_test_set.py --profile release_visual
```

**Step 2: Verify asset layout**

Check that the manifest contains:

- synthetic diagnostic charts
- curated real-world sanity images
- legacy Kodak baseline
- any intended derived JPEG ladder outputs

**Step 3: Commit generated dataset pointers only if this repository tracks them**

```bash
git add test_images/manifests
git commit -m "data: refresh release visual manifest"
```

If the repository does not track fetched assets, skip the commit and document the fetch command in the final handoff.
