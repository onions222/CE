# WPA Test Set Recommendations

## Current Status
- Primary gate: visual-stability synthetic charts
- Secondary gate: color side-effect charts
- Legacy baseline: small Kodak subset kept only for historical comparison

## Profile Strategy
- `smoke_visual`: fast local regression on the most sensitive neutral-stability charts
- `core_visual`: daily default profile with neutral-stability core plus color side-effect checks
- `release_visual`: release gate with `core_visual` + curated real-world sanity groups + legacy Kodak baseline + derived JPEG ladder

## Recommended Core Charts
- `01_grey_ramp`: global neutral cast and continuity
- `09_grey_steps`: discrete grey-step stability
- `10_luma_node_chart`: 12-bin boundary continuity
- `bin_boundary_triplet_chart`: exact node `n-1 / n / n+1` continuity stress
- `11_ui_text_contrast`: light-theme UI neutrality and edge integrity
- `near_black_steps`: dark-region neutrality and separability
- `near_white_steps`: near-white stability before clipping
- `ui_dark_theme_chart`: dark-theme neutrality and layer separation
- `two_axis_neutral_gradient`: large-area banding and tint detection
- `iso_gray_18_70_pair`: low/high neutral drift comparison
- `midtone_neutral_texture`: neutral fine-detail and edge-coloring check
- `warm_cool_split_field`: warm/cool transition continuity
- `shadow_with_colored_highlight`: shadow stability with near-clip highlights

## Recommended Side-Effect Charts
- `12_specular_clip_chart`: highlight clipping and near-white drift
- `13_mixed_illumination_chart`: warm/cool transition behavior
- `rgb_cmy_color_bars`: primary/secondary hue stability
- `06_skin_tones`: skin-tone plausibility
- `04_saturation_gradient`: saturation protection transition
- `saturation_threshold_ladder`: sat-protection threshold smoothness
- `skin_tone_luma_strip`: skin plausibility across luminance

## Real-World Sanity Groups
- `public_portrait`: daylight skin-tone plausibility sanity
- `public_hdr_window`: backlit / bright-window HDR transition
- `public_night_neon`: night mixed-color illumination stress
- `public_ui_workspace`: screen-like neutral/UI sanity
- `research_mixed_light`: official mixed-illuminant sample reference

## Manifest Guidance
- `inspection_priority`: P0/P1/P2 review priority
- `failure_modes`: normalized failure categories for that image
- `recommended_wa_sel`: WA values to inspect first
- `pass_hint`: short operator-facing acceptance hint
- `expected_artifact_if_fail`: what a regression usually looks like

## Build Commands
```bash
python scripts/build_test_set.py --profile smoke_visual
python scripts/build_test_set.py --profile core_visual
python scripts/build_test_set.py --profile release_visual
```
