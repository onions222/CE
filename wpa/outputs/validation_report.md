# WPA Full Validation Report

- Generated at: `2026-03-10 11:47:38 UTC`

## Scope

- Validate current repository algorithm implementation:
  - Float reference path: `wpa`
  - Fixed-point delivery path: `wpa_fixed`

## Pytest

```text
Skipped by flag.
```

## Anchor Interpolation Error

| coeff_frac_bits | MAE | P99 Abs | Max Abs |
|---:|---:|---:|---:|
| 8 | 0.034698 | 0.151998 | 0.153985 |
| 10 | 0.035042 | 0.153037 | 0.154223 |

## Hardware Resource Snapshot

| coeff_frac_bits | static bytes | runtime scratch bytes |
|---:|---:|---:|
| 8 | 36 | 41 |
| 10 | 41 | 50 |

## Float vs Fixed Consistency

### coeff_frac_bits=8

| WA_SEL | Mean Abs (LSB) | P99 Abs (LSB) | Max Abs (LSB) |
|---:|---:|---:|---:|
| 0 | 0.2373 | 2.00 | 3 |
| 32 | 0.2252 | 2.00 | 4 |
| 64 | 0.0000 | 0.00 | 0 |
| 96 | 0.3293 | 2.00 | 3 |
| 127 | 0.2402 | 2.00 | 4 |

### coeff_frac_bits=10

| WA_SEL | Mean Abs (LSB) | P99 Abs (LSB) | Max Abs (LSB) |
|---:|---:|---:|---:|
| 0 | 0.2131 | 2.00 | 3 |
| 32 | 0.1975 | 2.00 | 4 |
| 64 | 0.0000 | 0.00 | 0 |
| 96 | 0.1877 | 2.00 | 3 |
| 127 | 0.2443 | 2.00 | 4 |

## Gate Checks

| Check | Status | Detail |
|---|---|---|
| Pytest | PASS | Skipped by `--skip-pytest` |
| Anchor interpolation (coeff=8) | FAIL | MAE=0.034698 (<=0.012), MaxAbs=0.153985 (<=0.03) |
| Anchor interpolation (coeff=10) | FAIL | MAE=0.035042 (<=0.012), MaxAbs=0.154223 (<=0.03) |
| HW static bytes | PASS | coeff8=36B (expect 36B), coeff10=41B (expect 41B) |
| Float-vs-fixed consistency (coeff=8) | PASS | max(mean)=0.3293 < 0.5, max(p99)=2.00 <= 2, global max=4 <= 4, wa64 max=0 == 0 |
| Float-vs-fixed consistency (coeff=10) | PASS | max(mean)=0.2443 < 0.5, max(p99)=2.00 <= 2, global max=4 <= 4, wa64 max=0 == 0 |

## Visual Validation

## P0 Neutral Stability

Failure-Mode Coverage: `area_nonuniformity`×1, `banding`×2, `dark_instability`×2, `detail_collapse`×1, `highlight_halo_or_tint`×1, `highlight_shift`×2, `luma_dependent_neutral_drift`×1, `luma_node_discontinuity`×1, `mixed_illumination_transition`×1, `near_black_instability`×1, `near_white_drift`×1, `neutral_cast`×9, `node_discontinuity`×2, `quantization_jump`×1, `ui_artifact`×3

Recommended WA_SEL: `0, 64, 127` plus any item-specific extremes.

| Item | Failure Modes | Recommended WA_SEL | Pass Hint |
|---|---|---|---|
| 01_grey_ramp | neutral_cast, banding | 0, 32, 64, 96, 127 | Look for a smooth grey-only ramp at warm and cool extremes. |
| 09_grey_steps | neutral_cast, node_discontinuity | 0, 64, 127 | Each grey step should stay neutral and visually ordered. |
| 10_luma_node_chart | node_discontinuity | 0, 64, 127 | Check for abrupt transitions at node-aligned patches. |
| bin_boundary_triplet_chart | luma_node_discontinuity, quantization_jump | 0, 64, 127 | Inspect each triplet for monotonic, near-equal transitions. |
| 11_ui_text_contrast | neutral_cast, ui_artifact | 0, 64, 127 | White and light-grey UI surfaces should stay neutral. |
| 12_specular_clip_chart | highlight_shift | 0, 64, 127 | Bright blobs should brighten without picking up obvious color. |
| near_black_steps | dark_instability, neutral_cast | 0, 32, 64, 96, 127 | Dark steps should remain distinct and neutral at both extremes. |
| near_white_steps | highlight_shift, neutral_cast | 0, 64, 127 | Near-white patches should stay separated until clipping. |
| ui_dark_theme_chart | dark_instability, ui_artifact | 0, 64, 127 | Dark layered panels should remain neutral and legible. |
| two_axis_neutral_gradient | banding, area_nonuniformity, neutral_cast | 0, 64, 127 | Scan the whole field for stripes, blotches, or color drift. |
| iso_gray_18_70_pair | luma_dependent_neutral_drift, neutral_cast | 0, 64, 127 | Compare low and high greys side by side for directionally similar shift. |
| midtone_neutral_texture | ui_artifact, detail_collapse, neutral_cast | 0, 64, 127 | Inspect fine lines and boxes for colored halos or contrast loss. |
| warm_cool_split_field | mixed_illumination_transition, neutral_cast | 0, 64, 127 | Pay attention to the center transition and neutral blocks. |
| shadow_with_colored_highlight | near_white_drift, near_black_instability, highlight_halo_or_tint | 0, 64, 127 | Check both the highlight cores and the surrounding dark field. |

## P1 Color Side Effects

Failure-Mode Coverage: `hue_shift`×6, `neutral_cast`×1, `over_protection`×1, `saturation_transition_artifact`×1, `skin_hue_shift`×1, `skin_luma_dependency`×1, `under_protection`×1

Recommended WA_SEL: `0, 64, 127` for hue-family comparison.

| Item | Failure Modes | Recommended WA_SEL | Pass Hint |
|---|---|---|---|
| 02_color_checker | hue_shift | 0, 64, 127 | Use as a broad color sanity check, not the main release gate. |
| 04_saturation_gradient | hue_shift | 0, 64, 127 | Look for abrupt hue turns across each saturation ramp. |
| 06_skin_tones | hue_shift | 0, 64, 127 | Check whether complexion ordering still looks plausible. |
| 13_mixed_illumination_chart | hue_shift, neutral_cast | 0, 64, 127 | Inspect transition smoothness, especially around the neutral object. |
| rgb_cmy_color_bars | hue_shift | 0, 64, 127 | Each bar should stay within its hue family after adjustment. |
| saturation_threshold_ladder | saturation_transition_artifact, hue_shift, over_protection, under_protection | 0, 64, 127 | Look for the point where protection starts to kick in too abruptly. |
| skin_tone_luma_strip | skin_hue_shift, skin_luma_dependency | 0, 64, 127 | Compare columns within the same skin family before comparing different families. |

## P2 Real-World Sanity

Failure-Mode Coverage: mixed-light, portrait plausibility, HDR transition, UI neutrality.

Recommended WA_SEL: `0, 64, 127`.

| Group | Assets | Failure Modes | Recommended WA_SEL |
|---|---|---|---|
| public_portrait | portrait_of_woman.jpg | skin_plausibility, neutral_cast | 0, 64, 127 |
| public_hdr_window | gfp_sunroom.jpg | hdr_transition, near_white_drift | 0, 64, 127 |
| public_night_neon | led_and_neon_signs_on_portland_street_at_night.jpg | mixed_colored_light, highlight_shift | 0, 64, 127 |
| public_ui_workspace | desk_setup_unsplash.jpg | ui_neutrality, edge_artifact | 0, 64, 127 |
| research_mixed_light | lsmi_mixed_light_sample.png | mixed_illumination_transition, neutral_cast | 0, 64, 127 |

## Final Verdict

`FAIL`
