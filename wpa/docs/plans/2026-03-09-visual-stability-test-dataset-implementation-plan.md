# Visual Stability Test Dataset Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the current Kodak-centric validation dataset with a visual-stability-first dataset centered on synthetic diagnostic scenes, richer manifests, and purpose-based profiles.

**Architecture:** Keep the existing dataset build flow, but replace the generator catalog, profile definitions, and manifest schema so that `scripts/build_test_set.py` assembles `smoke_visual`, `core_visual`, and `release_visual` from metadata-driven chart definitions. Preserve a small Kodak path only as `legacy_baseline`, while leaving numeric validation in `validation/run_full_validation.py` intact for implementation correctness.

**Tech Stack:** Python 3, Pillow, NumPy, pytest, Markdown docs

---

### Task 1: Lock the New Synthetic Catalog in Tests

**Files:**
- Modify: `tests/test_synthetic_generators.py`
- Modify: `tests/test_dataset_profiles.py`

**Step 1: Write the failing test**

Update the generator-catalog test to assert the presence of the new visual-stability charts and the removal or demotion of low-signal charts from the main profile.

```python
def test_synthetic_generator_catalog_matches_visual_stability_design() -> None:
    assert "near_black_steps" in GENERATORS
    assert "near_white_steps" in GENERATORS
    assert "ui_dark_theme_chart" in GENERATORS
    assert "rgb_cmy_color_bars" in GENERATORS
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_synthetic_generators.py tests/test_dataset_profiles.py -v`
Expected: FAIL because the current generator names and profile names still reflect the old dataset design.

**Step 3: Write minimal implementation**

Do not change production code yet. Only land the failing expectations first so the profile and generator redesign has a target.

**Step 4: Run test to verify it fails**

Run: `pytest tests/test_synthetic_generators.py tests/test_dataset_profiles.py -v`
Expected: FAIL with missing generator/profile assertions.

**Step 5: Commit**

```bash
git add tests/test_synthetic_generators.py tests/test_dataset_profiles.py
git commit -m "test: define visual stability dataset expectations"
```

### Task 2: Rebuild `scripts/generate_test_images.py` Around Visual Stability Charts

**Files:**
- Modify: `scripts/generate_test_images.py`
- Test: `tests/test_synthetic_generators.py`

**Step 1: Write the failing test**

Add shape- and name-based assertions for the new chart generators, including dark-mode UI, near-black steps, near-white steps, and RGB/CMY bars.

```python
def test_new_visual_stability_generators_produce_uint8_rgb() -> None:
    for name in ["near_black_steps", "near_white_steps", "ui_dark_theme_chart", "rgb_cmy_color_bars"]:
        img = GENERATORS[name]()
        assert img.dtype == np.uint8
        assert img.ndim == 3 and img.shape[2] == 3
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_synthetic_generators.py -v`
Expected: FAIL because the current catalog does not expose the new generators.

**Step 3: Write minimal implementation**

Implement the new chart generators and update `GENERATORS` to reflect the visual-stability-first taxonomy. Remove `08_random_noise` from formal profiles, and keep demoted charts available only if they still serve developer-only workflows.

```python
GENERATORS = {
    "grey_ramp_8bit": gen_grey_ramp_8bit,
    "grey_steps_16": gen_grey_steps_16,
    "near_black_steps": gen_near_black_steps,
    "near_white_steps": gen_near_white_steps,
    "ui_dark_theme_chart": gen_ui_dark_theme_chart,
    "rgb_cmy_color_bars": gen_rgb_cmy_color_bars,
}
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_synthetic_generators.py -v`
Expected: PASS with the updated generator catalog.

**Step 5: Commit**

```bash
git add scripts/generate_test_images.py tests/test_synthetic_generators.py
git commit -m "feat: rebuild synthetic charts for visual stability validation"
```

### Task 3: Replace Quantity-Based Profiles With Purpose-Based Profiles

**Files:**
- Modify: `validation/dataset_profiles.py`
- Test: `tests/test_dataset_profiles.py`

**Step 1: Write the failing test**

Replace the old `smoke/core/full` expectations with new `smoke_visual/core_visual/release_visual` expectations, including role and priority assertions.

```python
def test_smoke_visual_profile_focuses_on_p0_neutral_stability() -> None:
    spec = get_profile_spec("smoke_visual")
    assert spec["name"] == "smoke_visual"
    assert "near_black_steps" in spec["synthetic"]
    assert "rgb_cmy_color_bars" in spec["synthetic"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_dataset_profiles.py -v`
Expected: FAIL because the current profile names and schema are still `smoke/core/full`.

**Step 3: Write minimal implementation**

Redefine profile specs around validation purpose. Keep a small Kodak set only in `release_visual`, and add metadata blocks for `dataset_role`, `priority`, and `visual_risk`.

```python
def get_profile_spec(profile: Literal["smoke_visual", "core_visual", "release_visual"]) -> dict:
    ...
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_dataset_profiles.py -v`
Expected: PASS with the new profile schema.

**Step 5: Commit**

```bash
git add validation/dataset_profiles.py tests/test_dataset_profiles.py
git commit -m "feat: add visual stability dataset profiles"
```

### Task 4: Upgrade `scripts/build_test_set.py` to Emit Rich Manifests

**Files:**
- Modify: `scripts/build_test_set.py`
- Modify: `validation/test_set_recommendations.md`
- Test: `tests/test_dataset_profiles.py`
- Create: `tests/test_build_test_set.py`

**Step 1: Write the failing test**

Create a new test that builds a small profile and asserts that manifest entries include role, risk, priority, and expected-observation metadata.

```python
def test_build_test_set_writes_rich_manifest(tmp_path: Path) -> None:
    ...
    assert manifest["items"][0]["dataset_role"] == "neutral_stability_core"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_build_test_set.py -v`
Expected: FAIL because the current manifest only contains file lists.

**Step 3: Write minimal implementation**

Refactor manifest generation from grouped file arrays into an `items` list with structured metadata. Preserve compatibility fields only if downstream consumers still require them.

```python
manifest = {
    "profile": args.profile,
    "items": [
        {
            "path": "...",
            "dataset_role": "neutral_stability_core",
            "visual_risk": ["neutral_cast", "banding"],
            "priority": "P0",
            "expected_observation": "Grey ramp remains neutral and smooth.",
        }
    ],
}
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_build_test_set.py tests/test_dataset_profiles.py -v`
Expected: PASS with the richer manifest schema.

**Step 5: Commit**

```bash
git add scripts/build_test_set.py validation/test_set_recommendations.md tests/test_build_test_set.py tests/test_dataset_profiles.py
git commit -m "feat: emit metadata-rich validation dataset manifests"
```

### Task 5: Demote Kodak to Legacy Baseline Handling

**Files:**
- Modify: `scripts/download_test_images.py`
- Modify: `validation/dataset_profiles.py`
- Test: `tests/test_dataset_profiles.py`

**Step 1: Write the failing test**

Add assertions that Kodak is no longer the default real-image source and appears only in the release-level baseline path.

```python
def test_release_visual_profile_uses_small_legacy_kodak_baseline() -> None:
    spec = get_profile_spec("release_visual")
    assert len(spec["legacy_kodak"]) <= 6
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_dataset_profiles.py -v`
Expected: FAIL because Kodak is still the main real-image collection.

**Step 3: Write minimal implementation**

Rename or reframe Kodak helper logic so it is explicitly a legacy baseline path. Keep the download code, but stop presenting it as the recommended real-image dataset.

```python
LEGACY_KODAK_BASELINE = {
    "kodim04.png": "...",
    "kodim05.png": "...",
}
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_dataset_profiles.py -v`
Expected: PASS with Kodak limited to legacy use.

**Step 5: Commit**

```bash
git add scripts/download_test_images.py validation/dataset_profiles.py tests/test_dataset_profiles.py
git commit -m "refactor: demote kodak to legacy validation baseline"
```

### Task 6: Document the New Visual Verdict Model

**Files:**
- Modify: `validation/validation_report_template.md`
- Modify: `outputs/validation_report.md`
- Modify: `README.md`
- Test: `tests/test_full_validation_runner.py`

**Step 1: Write the failing test**

Extend the validation-runner test to expect separate verdict sections for implementation correctness and visual validation placeholders or status blocks.

```python
def test_full_validation_runner_report_mentions_visual_verdict_sections(tmp_path: Path) -> None:
    ...
    assert "P0 stability verdict" in text
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_full_validation_runner.py -v`
Expected: FAIL because the current template only reports a single final verdict.

**Step 3: Write minimal implementation**

Add report sections that explicitly separate numeric correctness from future visual-stability review output. If visual verdicts are placeholders in this phase, label them clearly.

```markdown
## Visual Validation

- P0 stability verdict: `TBD`
- P1 side-effect verdict: `TBD`
- Real-world sanity verdict: `TBD`
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_full_validation_runner.py -v`
Expected: PASS with the updated report structure.

**Step 5: Commit**

```bash
git add validation/validation_report_template.md outputs/validation_report.md README.md tests/test_full_validation_runner.py
git commit -m "docs: add visual validation verdict structure"
```

### Task 7: Run the Focused Verification Suite

**Files:**
- Test: `tests/test_synthetic_generators.py`
- Test: `tests/test_dataset_profiles.py`
- Test: `tests/test_build_test_set.py`
- Test: `tests/test_full_validation_runner.py`

**Step 1: Run focused tests**

Run: `pytest tests/test_synthetic_generators.py tests/test_dataset_profiles.py tests/test_build_test_set.py tests/test_full_validation_runner.py -v`
Expected: PASS with all dataset and report tests green.

**Step 2: Run dataset build smoke test**

Run: `python scripts/build_test_set.py --profile smoke_visual --output-root /tmp/wpa_visual_dataset_smoke`
Expected: manifest files written successfully with structured metadata.

**Step 3: Review generated manifest**

Run: `python - <<'PY'\nimport json\nfrom pathlib import Path\np = Path('/tmp/wpa_visual_dataset_smoke/manifests/smoke_visual.json')\nprint(json.loads(p.read_text())['profile'])\nPY`
Expected: prints `smoke_visual`.

**Step 4: Commit**

```bash
git add scripts tests validation README.md
git commit -m "test: verify visual stability dataset workflow"
```

### Task 8: Final Documentation Check

**Files:**
- Modify: `validation/test_set_recommendations.md`
- Modify: `README.md`
- Modify: `docs/plans/2026-03-09-visual-stability-test-dataset-design.md`

**Step 1: Review documentation for consistency**

Confirm that the public docs no longer describe Kodak as the primary recommended real-image set and that profile names match the implemented API.

**Step 2: Run markdown sanity checks**

Run: `rg "smoke|core|full|Kodak image set" README.md validation docs/plans`
Expected: only intentional legacy references remain.

**Step 3: Commit**

```bash
git add README.md validation/test_set_recommendations.md docs/plans/2026-03-09-visual-stability-test-dataset-design.md
git commit -m "docs: align dataset documentation with visual stability workflow"
```
