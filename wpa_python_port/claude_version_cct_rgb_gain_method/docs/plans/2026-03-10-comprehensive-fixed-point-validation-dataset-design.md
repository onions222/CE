# Comprehensive Fixed-Point Validation Dataset Design

## 1. Background

The repository already validates:

- float vs fixed consistency
- interpolation approximation error
- static hardware resource usage
- a visual-stability-oriented synthetic image set

That foundation is useful, but still incomplete for fixed-point delivery review.

The current dataset is strongest at broad neutral-stability checks and weakest at:

- explicit bin-boundary stress around the 12-node interpolation structure
- saturation-threshold transition behavior
- skin-tone behavior across luminance levels
- small but targeted real-image sanity coverage for mixed illumination and highlight/shadow stress
- manifest/report metadata that explains what each image is meant to catch

The user clarified two priorities:

1. expand validation depth primarily through diagnostic images, not by building a large natural-image benchmark
2. extend both the dataset and the manifest/report structure so the validation output is easier to inspect and reason about

## 2. Goals

Primary goals:

- make fixed-point failure modes easier to expose visually
- map each image to one or more explicit diagnostic risks
- add a small real-world sanity layer without making it the primary gate
- upgrade manifests and reports from file inventories into validation guidance

Secondary goals:

- keep existing profiles usable
- preserve current legacy Kodak support as a historical baseline
- keep generation and download flows scriptable from the repository

Non-goals:

- replacing numeric float-vs-fixed validation with subjective image review
- introducing a large-scale benchmark ingestion framework
- building a learned or automated perceptual scoring system in this phase

## 3. External Reference Summary

The proposed structure is consistent with common white-balance and color-evaluation practice:

- Imatest `Colorcheck` and `ColorGray` workflows emphasize grayscale, neutral patches, and performance across luminance levels rather than relying on a single white patch
- chart-based workflows commonly keep skin tones and high-saturation patches as side-effect checks rather than the primary white-balance gate
- modern color-constancy datasets such as LSMI and Cube++ highlight mixed illumination, overexposure, and shadow-heavy scenes as important real-image stress cases

Sources:

- Imatest Colorcheck: <https://www.imatest.com/docs/guides/modules/colorcheck>
- Imatest ColorGray-44: <https://www.imatest.com/product/colorgray-44/>
- Imatest ColorGauge: <https://www.imatest.com/product/colorgauge/>
- LSMI (ICCV 2021): <https://openaccess.thecvf.com/content/ICCV2021/html/Kim_Large_Scale_Multi-Illuminant_LSMI_Dataset_for_Developing_White_Balance_Algorithm_ICCV_2021_paper.html>
- Cube++: <https://github.com/Visillect/CubePlusPlus>

Inference from those sources:

- for this repository, diagnostic synthetic charts should remain the main gate
- real photographs should be kept as a sanity layer focused on mixed illuminant, highlight, shadow, and human-subject plausibility

## 4. Validation Model

The dataset should be organized into three layers:

### 4.1 `neutral_stability_core`

Purpose:

- detect neutral cast
- detect node-boundary discontinuity
- detect banding/posterization
- detect near-black instability
- detect near-white drift
- detect UI-like edge artifacts

This remains the primary release gate.

### 4.2 `color_side_effect_set`

Purpose:

- detect hue-family drift
- detect saturation-protection transition artifacts
- detect skin-tone plausibility issues
- detect warm/cool transition artifacts

This remains a secondary gate.

### 4.3 `real_world_sanity_set`

Purpose:

- confirm that chart-level conclusions hold on a small number of realistic scenes
- expose obvious regressions in portrait, mixed-light, HDR, and night content

This is intentionally small and review-oriented.

## 5. Synthetic Dataset Expansion

The current repository already includes useful charts such as:

- `01_grey_ramp`
- `09_grey_steps`
- `10_luma_node_chart`
- `11_ui_text_contrast`
- `12_specular_clip_chart`
- `near_black_steps`
- `near_white_steps`
- `ui_dark_theme_chart`
- `rgb_cmy_color_bars`

The next expansion should add the following charts.

### 5.1 `two_axis_neutral_gradient`

Intent:

- reveal large-area banding
- reveal 2D non-uniform tint
- reveal subtle quantization contouring

Construction:

- horizontal axis varies from dark to bright neutral
- vertical axis varies independently with a smaller luminance range perturbation
- image remains neutral in RGB input space

Primary failure modes:

- `banding`
- `neutral_cast`
- `area_nonuniformity`

### 5.2 `bin_boundary_triplet_chart`

Intent:

- explicitly stress interpolation continuity around every luma node

Construction:

- for each node `n`, place triplets at `n-1`, `n`, and `n+1`
- include separators so adjacent triplets are easy to inspect visually

Primary failure modes:

- `luma_node_discontinuity`
- `quantization_jump`

### 5.3 `iso_gray_18_70_pair`

Intent:

- compare low/mid and high neutral behavior without texture or hue distractions

Construction:

- paired neutral blocks near common grey-card style reflectance regions
- repeated across the frame for side-by-side inspection

Primary failure modes:

- `luma_dependent_neutral_drift`
- `neutral_cast`

### 5.4 `midtone_neutral_texture`

Intent:

- catch colored edge artifacts and tonal collapse in practical neutral-detail regions

Construction:

- mid-grey field
- neutral fine lines, boxes, and subtle contrast texture

Primary failure modes:

- `ui_edge_artifact`
- `detail_collapse`
- `neutral_cast`

### 5.5 `saturation_threshold_ladder`

Intent:

- stress the protection threshold region instead of only endpoints

Construction:

- several hue families
- within each hue, saturation increases gradually across the row
- luminance held roughly stable

Primary failure modes:

- `saturation_transition_artifact`
- `hue_shift`
- `over_protection`
- `under_protection`

### 5.6 `skin_tone_luma_strip`

Intent:

- verify that skin-tone behavior stays plausible across both complexion and brightness

Construction:

- several representative skin-tone families
- each family repeated at multiple luminance levels

Primary failure modes:

- `skin_hue_shift`
- `skin_luma_dependency`

### 5.7 `warm_cool_split_field`

Intent:

- mimic mixed illumination with a clear spatial transition

Construction:

- warm-tinted side
- cool-tinted side
- neutral references crossing the boundary

Primary failure modes:

- `mixed_illumination_transition`
- `neutral_cast`

### 5.8 `shadow_with_colored_highlight`

Intent:

- combine dark-region stability with highlight-adjacent color stress

Construction:

- dark neutral base
- bright neutral, warm, and cool highlight patches near clip

Primary failure modes:

- `near_white_drift`
- `near_black_instability`
- `highlight_halo_or_tint`

## 6. Real-Image Sanity Layer

The real-image layer should stay small, explicit, and easy to refresh.

### 6.1 Scene Categories

Recommended minimum categories:

- portrait / face
- indoor LED or mixed indoor light
- night / neon
- HDR window or backlit interior
- screen or UI-like capture
- outdoor neutral object or architecture

### 6.2 Source Strategy

Use two acquisition paths:

1. stable direct-download images from public sources for easy local setup
2. curated small subsets from research datasets for higher-value edge cases

Recommended target sources:

- Cube++ for shadows, overexposure, and outdoor scenes
- LSMI for mixed-illuminant indoor scenes
- a small number of directly downloadable portrait / UI-like examples from permissive public sources
- existing Kodak subset retained only as `legacy_baseline`

### 6.3 Naming and Roles

Real-image items should be explicitly tagged by purpose rather than origin.

Examples:

- `portrait_daylight_public_01`
- `mixed_led_lsmi_01`
- `night_neon_public_01`
- `hdr_window_cubepp_01`
- `ui_capture_public_01`

Origin should remain metadata, not the dataset role.

## 7. Manifest Redesign

The current manifest is useful but too shallow for guided review.

Each item should include:

- `name`
- `path`
- `source`
- `source_group`
- `dataset_role`
- `inspection_priority`
- `failure_modes`
- `expected_observation`
- `expected_artifact_if_fail`
- `recommended_wa_sel`
- `suggested_metrics`
- `pass_hint`
- `notes`

### 7.1 Field Semantics

`inspection_priority`

- `P0`: release-blocking neutral stability risk
- `P1`: important side-effect risk
- `P2`: sanity-only or informational

`failure_modes`

- normalized short identifiers such as `neutral_cast`, `banding`, `luma_node_discontinuity`

`recommended_wa_sel`

- a list of WA settings to inspect, typically including neutral, warm extreme, cool extreme, and one intermediate setting

`suggested_metrics`

- optional numeric hints such as mean channel delta, max local jump, or saturation-drift probes

`pass_hint`

- short operator-facing statement describing what acceptable output should look like

## 8. Report Redesign

`validation/validation_report_template.md` and the generated report should evolve from static placeholders into guided review sections.

### 8.1 New Visual Sections

- `P0 Neutral Stability`
- `P1 Color Side Effects`
- `P2 Real-World Sanity`

Each section should summarize:

- which items belong to that role
- which failure modes are covered
- which WA settings are recommended for inspection

### 8.2 Manifest-Driven Rendering

The report generator should read manifest metadata and render:

- grouped item tables by role
- counts by priority
- a compact failure-mode coverage summary

This keeps the visual validation summary synchronized with the dataset definition.

## 9. Profile Changes

### 9.1 `smoke_visual`

Should remain fast, but include at least:

- one global neutral continuity chart
- one bin-boundary chart
- one near-black chart
- one near-white chart
- one UI-like chart
- one color-side-effect chart

### 9.2 `core_visual`

Should include:

- the full neutral core
- selected saturation, skin, and mixed-light charts

### 9.3 `release_visual`

Should include:

- all `core_visual` synthetic items
- curated `real_world_sanity_set`
- minimal `legacy_kodak_baseline`
- optional derived perturbations such as JPEG ladder where still useful

## 10. Implementation Strategy

Implementation should proceed in this order:

1. extend synthetic generators and their tests
2. extend manifest/profile metadata and tests
3. add real-image download definitions and build-path support
4. upgrade report rendering to consume manifest metadata
5. rebuild representative profiles and verify outputs

This ordering keeps early changes deterministic and testable before touching network-backed assets.

## 11. Risks

- external image URLs may drift or rate-limit
- research datasets may have packaging friction
- report metadata can become noisy if fields are too verbose

Mitigations:

- keep a minimal, explicit curated list
- separate direct-download sources from research-subset sources
- keep metadata vocabularies normalized and short

## 12. Acceptance Criteria

The redesign is complete when:

- the repository can build an expanded synthetic-first validation dataset from scripts
- profiles express diagnostic intent rather than only quantity
- manifests explicitly describe failure modes and review guidance
- the validation report summarizes visual coverage by role and priority
- a small real-world sanity layer can be fetched reproducibly
