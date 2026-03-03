# WPA Full Validation Report

- Generated at: `2026-03-03 06:20:28 UTC`

## Scope

- Validate current repository algorithm implementation:
  - Float reference path: `wpa`
  - Fixed-point delivery path: `wpa_fixed`

## Pytest

```text
............................................................             [100%]
60 passed in 0.19s
```

## Anchor Interpolation Error

| coeff_frac_bits | MAE | P99 Abs | Max Abs |
|---:|---:|---:|---:|
| 8 | 0.009819 | 0.028085 | 0.028228 |
| 10 | 0.010125 | 0.027249 | 0.027910 |

## Hardware Resource Snapshot

| coeff_frac_bits | static bytes | runtime scratch bytes |
|---:|---:|---:|
| 8 | 36 | 41 |
| 10 | 41 | 50 |

## Float vs Fixed Consistency

### coeff_frac_bits=8

| WA_SEL | Mean Abs (LSB) | P99 Abs (LSB) | Max Abs (LSB) |
|---:|---:|---:|---:|
| 0 | 0.2444 | 2.00 | 4 |
| 32 | 0.2335 | 2.00 | 3 |
| 64 | 0.0000 | 0.00 | 0 |
| 96 | 0.3133 | 2.00 | 3 |
| 127 | 0.2323 | 2.00 | 4 |

### coeff_frac_bits=10

| WA_SEL | Mean Abs (LSB) | P99 Abs (LSB) | Max Abs (LSB) |
|---:|---:|---:|---:|
| 0 | 0.2010 | 2.00 | 4 |
| 32 | 0.1901 | 2.00 | 3 |
| 64 | 0.0000 | 0.00 | 0 |
| 96 | 0.1647 | 2.00 | 3 |
| 127 | 0.1875 | 2.00 | 4 |

## Gate Checks

| Check | Status | Detail |
|---|---|---|
| Pytest | PASS | `python -m pytest tests -q` returns exit code 0 |
| Anchor interpolation (coeff=8) | PASS | MAE=0.009819 (<=0.012), MaxAbs=0.028228 (<=0.03) |
| Anchor interpolation (coeff=10) | PASS | MAE=0.010125 (<=0.012), MaxAbs=0.027910 (<=0.03) |
| HW static bytes | PASS | coeff8=36B (expect 36B), coeff10=41B (expect 41B) |
| Float-vs-fixed consistency (coeff=8) | PASS | max(mean)=0.3133 < 0.5, max(p99)=2.00 <= 2, global max=4 <= 4, wa64 max=0 == 0 |
| Float-vs-fixed consistency (coeff=10) | PASS | max(mean)=0.2010 < 0.5, max(p99)=2.00 <= 2, global max=4 <= 4, wa64 max=0 == 0 |

## Final Verdict

`PASS`
