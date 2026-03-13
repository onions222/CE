# WPA Low-Luma Bypass Design

## Context

This note records the investigation status for the warm halo issue observed on
`12_specular_clip_chart` under the MATLAB `hw_runtime` path.

Verified findings:

- MATLAB `hw_runtime` and the current release outputs match for
  `12_specular_clip_chart`.
- The visible warm halo is not introduced by the review panel only. The review
  panel resize can perturb exact pixel values, but the underlying
  `wa=0` output already contains a colored ring.
- The ring is not explained by "neutral input turns green" in the abstract.
  The critical shoulder pixels are slightly blue-biased dark inputs such as
  `(16, 18, 22)`.
- In the current fixed-point path with `frac_bits=8`, these shoulder pixels
  quantize to the linear-domain code `[1, 2, 2]`.
- After applying warm low-luma gain and re-encoding, `[1, 2, 2]` can collapse
  to `[1, 2, 1]`, which appears as `(13, 22, 13)`.
- Raising internal linear precision to `frac_bits=10` removes the green ring on
  `12_specular_clip_chart`, which confirms that low-code quantization is a
  direct contributor.

## Rejected Adjustments

### Only lowering the low-luma attenuation plateau

Reducing the low-luma attenuation from `0.55` to `0.50` is not sufficient as a
final solution.

What it does:

- It removes the most obviously green-dominant output on the synthetic case.

What it does not do:

- It does not remove the visible colored ring.
- It does not address the fact that even an identity low-luma gain still passes
  through the `Q0.8` linear-domain round-trip.

Even with low-luma attenuation forced to `0.00`, representative pixels such as
`(16, 18, 22)` still become `(13, 22, 22)` if they continue to traverse the
fixed-point round-trip. This means a table-only solution has reached its limit.

## Chosen Direction

Use a low-luma bypass/blend strategy instead of further tuning the low-luma
gain table:

- `Y <= 31`: bypass WPA and return the original gamma-domain input pixel.
- `31 < Y < 63`: blend smoothly from input to the normal WPA output.
- `Y >= 63`: keep the existing WPA path unchanged.

## Why This Direction

- It directly avoids the unstable `Q0.8` round-trip for the most vulnerable
  low-code pixels.
- It preserves the current highlight and midtone behavior above the transition
  window.
- It avoids locking the product into a hard cutoff at `Y=31`, which would
  likely create visible node discontinuities.

## Validation Summary So Far

Prototype results on `12_specular_clip_chart`:

- Base `Q0.8` path: visible green ring, `8656` green-dominant pixels.
- Hard bypass for `Y <= 31`: ring removed, `0` green-dominant pixels.
- Blend for `31 < Y < 63`: ring removed, `0` green-dominant pixels.

Prototype results on `near_black_steps`:

- Hard bypass removes the low-luma artifact but introduces a hard boundary near
  the threshold.
- The `31..63` blend avoids that abrupt jump and is the preferred landing shape.

## Implementation Scope

The next implementation step is to introduce the bypass/blend behavior in the
Python fixed-point path and the MATLAB `hw_runtime` path, then re-run targeted
visual verification on the sensitive synthetic images.
