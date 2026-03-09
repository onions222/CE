# Visual Stability Test Dataset Design

## 1. Background

The current validation dataset mixes two very different goals:

- synthetic charts for algorithm edge cases
- Kodak images as the only real-image source

This is no longer aligned with the intended use of the repository.

The original Kodak subset is useful as a historical reference, but it is not the right primary dataset for evaluating visual stability. The user clarified that the highest priority is not broad real-world coverage, but stable behavior in common diagnostic scenes such as grayscale, color bars, UI-like content, dark regions, and highlight regions.

## 2. Problem Statement

The current dataset design has four issues:

1. `scripts/download_test_images.py` treats Kodak as the default real-image dataset.
2. `validation/dataset_profiles.py` organizes profiles by quantity (`smoke/core/full`) instead of verification purpose.
3. `scripts/generate_test_images.py` contains useful probes and low-signal probes in the same tier.
4. Existing reporting validates implementation correctness, but does not express visual-stability priorities such as neutral preservation, banding, or node discontinuity risk.

## 3. Goals

Primary goal:

- redesign the test dataset so that visual stability is the main evaluation target

Secondary goals:

- keep a small amount of color-side-effect coverage
- keep a small real-image sanity set
- retain a minimal Kodak baseline only for historical comparison
- make profile definitions and manifests reflect verification intent

Non-goals:

- building a large modern-photo benchmark as the main gate
- replacing numeric validation of float/fixed consistency
- introducing subjective image-quality scoring in this phase

## 4. Design Principles

The new dataset should follow these rules:

1. Special diagnostic scenes are the primary gate.
2. Neutral stability is weighted higher than content diversity.
3. Each synthetic chart must correspond to a clear failure mode.
4. Real images are a sanity layer, not the main decision layer.
5. Kodak remains only as `legacy_baseline`.

## 5. Dataset Architecture

The dataset is split into three layers:

### 5.1 `neutral_stability_core`

This is the primary dataset and carries the highest weight in evaluation.

Purpose:

- reveal grayscale color cast
- reveal banding and smoothness issues
- reveal 12-bin node discontinuities
- reveal dark-region instability
- reveal highlight-region instability
- reveal UI neutrality regressions

Suggested weight:

- 70% of the final visual judgment

### 5.2 `color_side_effect_set`

This is a secondary dataset used to ensure the algorithm does not become visually stable by damaging color behavior.

Purpose:

- detect hue drift on primary and secondary colors
- detect over-protection or under-protection on saturated regions
- detect skin-tone side effects
- detect mixed-light artifacts

Suggested weight:

- 20% of the final visual judgment

### 5.3 `real_world_sanity_set`

This is a small sanity layer used to confirm that chart-level conclusions do not break obvious real content.

Purpose:

- human sanity check after chart validation
- spot gross regressions on portrait, indoor light, UI capture, HDR, and night scenes

Suggested weight:

- 10% of the final visual judgment

## 6. Synthetic Dataset Redesign

### 6.1 `neutral_stability_core`

Recommended core charts:

1. `grey_ramp_8bit`
   - horizontal 0-255 neutral ramp
   - checks global color cast and continuity
2. `grey_steps_16`
   - 16-step neutral blocks
   - checks discrete stepping and neutrality
3. `near_black_steps`
   - dense grayscale steps in the low end, e.g. 0-32
   - checks shadow instability and dark-region color cast
4. `near_white_steps`
   - dense grayscale steps in the high end, e.g. 223-255
   - checks highlight compression and near-white tinting
5. `two_axis_neutral_gradient`
   - large-area 2D neutral gradient
   - checks banding and area non-uniformity
6. `luma_node_chart`
   - aligned to the 12 luma nodes used by the algorithm
   - checks visible bin-boundary discontinuity
7. `ui_light_theme_chart`
   - light background, dark text bars, grey separators
   - checks white-background UI stability
8. `ui_dark_theme_chart`
   - dark background, light text bars, dark-grey layers
   - checks dark-mode neutrality and separation
9. `specular_on_dark_chart`
   - neutral dark field with neutral/warm/cool highlights
   - checks clipping-adjacent highlight behavior
10. `hdr_window_scene_chart`
   - interior shadows, bright window, neutral wall/object regions
   - checks high dynamic range stability

### 6.2 `color_side_effect_set`

Recommended secondary charts:

1. `rgb_cmy_color_bars`
   - checks hue stability on primaries and secondaries
2. `saturation_sweep_bars`
   - checks low-to-high saturation response for each hue group
3. `skin_tone_strip`
   - multiple skin-tone patches from light to dark and warm to cool
4. `foliage_sky_dual_patch`
   - paired green/blue patches for common visual side effects
5. `mixed_warm_cool_blocks`
   - neutral grey and white blocks under warm/cool lighting assumptions
6. `led_extreme_color_patch`
   - checks catastrophic behavior under extreme colored light

### 6.3 Real-Image Sanity Layer

The real-image layer should be small and purpose-specific:

- 2 portrait/selfie images
- 1 indoor LED image
- 1 night/neon image
- 1 white-background UI or screen-capture image
- 1 outdoor HDR image

Optional:

- 1-2 Kodak images as `legacy_baseline`

## 7. Disposition of Existing Synthetic Charts

### 7.1 Keep and upgrade

- `01_grey_ramp`
- `09_grey_steps`
- `10_luma_node_chart`
- `11_ui_text_contrast`
- `12_specular_clip_chart`

These are still relevant, but some should be renamed or redrawn to better match the new taxonomy.

### 7.2 Keep but redraw

- `06_skin_tones`
- `07_smooth_gradient`
- `13_mixed_illumination_chart`

These have value, but should be redrawn with tighter visual-stability intent.

### 7.3 Demote from the main gate

- `02_color_checker`
- `03_channel_sweeps`
- `04_saturation_gradient`

These may still be useful during development, but should not dominate the formal visual-stability gate.

### 7.4 Remove from formal visual-stability profiles

- `08_random_noise`

This is low signal for the stated objective.

### 7.5 New charts to add

- `near_black_steps`
- `near_white_steps`
- `ui_dark_theme_chart`
- `hdr_window_scene_chart`
- `rgb_cmy_color_bars`

Recommended total chart count:

- 16-18 images across core and side-effect layers

## 8. Profile Redesign

Replace the current quantity-based profile strategy with purpose-based profiles:

### 8.1 `smoke_visual`

Fast local regression profile.

Recommended contents:

- `grey_ramp_8bit`
- `grey_steps_16`
- `near_black_steps`
- `near_white_steps`
- `luma_node_chart`
- `ui_light_theme_chart`
- `ui_dark_theme_chart`
- `rgb_cmy_color_bars`

### 8.2 `core_visual`

Default daily validation profile.

Contents:

- all of `neutral_stability_core`
- selected high-value charts from `color_side_effect_set`

Target size:

- 16-18 images

### 8.3 `release_visual`

Release-level profile.

Contents:

- full `core_visual`
- `real_world_sanity_set`
- small `legacy_kodak_baseline`

## 9. Manifest Redesign

The current manifest fields are too coarse:

- `synthetic_files`
- `real_files`
- `derived_files`

The new manifest should attach per-item metadata such as:

- `dataset_role`
  - `neutral_stability_core`
  - `color_side_effect_set`
  - `real_world_sanity_set`
  - `legacy_baseline`
- `visual_risk`
  - `neutral_cast`
  - `banding`
  - `node_discontinuity`
  - `dark_instability`
  - `highlight_shift`
  - `hue_shift`
  - `ui_artifact`
- `expected_observation`
  - short human-readable statement of what should remain stable
- `priority`
  - `P0`, `P1`, `P2`
- `pass_mode`
  - `visual_only`
  - `visual_plus_metric`

This enables result grouping by visual failure mode instead of by filename only.

## 10. Validation Output Redesign

Current full validation primarily proves implementation correctness:

- pytest pass/fail
- anchor interpolation error
- hardware resource snapshot
- float-vs-fixed consistency

That remains necessary, but visual dataset reporting should introduce a separate verdict structure:

- `P0 stability verdict`
- `P1 side-effect verdict`
- `real-world sanity verdict`

Suggested interpretation:

- `P0` must pass
- `P1` allows only minor acceptable change
- real-world sanity must show no obvious abnormal artifact

## 11. Required Codebase Changes

Expected file-level changes:

- `scripts/generate_test_images.py`
  - replace the current mixed generator catalog with the new visual-stability chart set
- `validation/dataset_profiles.py`
  - replace `smoke/core/full` emphasis with `smoke_visual/core_visual/release_visual`
- `scripts/build_test_set.py`
  - emit richer manifests with role/risk/priority metadata
- `scripts/download_test_images.py`
  - demote Kodak to legacy baseline handling
- `validation/test_set_recommendations.md`
  - document the new structure and rationale
- `tests/test_synthetic_generators.py`
  - update the generator catalog assertions
- `tests/test_dataset_profiles.py`
  - update profile expectations and metadata checks

## 12. Final Recommendation

Adopt a visual-stability-first validation strategy:

- make special synthetic scenes the primary gate
- make neutral stability the dominant decision criterion
- keep a smaller color side-effect layer
- reduce real images to sanity validation only
- keep Kodak only as a historical baseline

This structure is better aligned with the actual decision question:

"Is the algorithm visually stable and effective on the diagnostic scenes that most clearly reveal failure?"
