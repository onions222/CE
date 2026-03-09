# WPA Test Set Recommendations

## Current Status
- Primary gate: visual-stability synthetic charts
- Secondary gate: color side-effect charts
- Legacy baseline: small Kodak subset kept only for historical comparison

## Profile Strategy
- `smoke_visual`: fast local regression on the most sensitive neutral-stability charts
- `core_visual`: daily default profile with neutral-stability core plus color side-effect checks
- `release_visual`: release gate with `core_visual` + legacy Kodak baseline + derived JPEG ladder

## Recommended Core Charts
- `01_grey_ramp`: global neutral cast and continuity
- `09_grey_steps`: discrete grey-step stability
- `10_luma_node_chart`: 12-bin boundary continuity
- `11_ui_text_contrast`: light-theme UI neutrality and edge integrity
- `near_black_steps`: dark-region neutrality and separability
- `near_white_steps`: near-white stability before clipping
- `ui_dark_theme_chart`: dark-theme neutrality and layer separation

## Recommended Side-Effect Charts
- `12_specular_clip_chart`: highlight clipping and near-white drift
- `13_mixed_illumination_chart`: warm/cool transition behavior
- `rgb_cmy_color_bars`: primary/secondary hue stability
- `06_skin_tones`: skin-tone plausibility
- `04_saturation_gradient`: saturation protection transition

## Build Command
```bash
python scripts/build_test_set.py --profile smoke_visual
python scripts/build_test_set.py --profile core_visual
python scripts/build_test_set.py --profile release_visual
```
