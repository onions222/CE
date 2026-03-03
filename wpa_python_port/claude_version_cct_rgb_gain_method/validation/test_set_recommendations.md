# WPA Test Set Recommendations

## Current Status
- Synthetic: 13 charts (includes UI/text contrast, specular highlight, mixed illumination)
- Real: Kodak curated 15 by default, optional all 24

## Profile Strategy
- `smoke`: fast local sanity run
- `core`: daily validation (recommended default)
- `full`: release gate with all Kodak + JPEG degradation ladder

## Why New Additions
- `11_ui_text_contrast`: validate neutral text/line integrity and edge artifacts
- `12_specular_clip_chart`: validate highlight clipping and near-white color drift
- `13_mixed_illumination_chart`: validate warm/cool coexistence and transition behavior

## Build Command
```bash
python scripts/build_test_set.py --profile smoke
python scripts/build_test_set.py --profile core
python scripts/build_test_set.py --profile full
```
