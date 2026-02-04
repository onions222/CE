import numpy as np
import colorsys

# ============================================================
# 12-bin (30° each) rule-table verifier + transform comparison
# ============================================================
# This script verifies:
#   (1) The 12-bin decision rules (RGB comparisons + one mid test)
#       match standard HSV hue binning for non-boundary samples.
#   (2) Pure YCoCg chroma rotation will often CROSS bins.
#   (3) A sector-preserving "extra transform" in local (max,mid,min)
#       coordinates can GUARANTEE staying in the same 30° bin and
#       matches strict HSV hue adjustment with clamping.

# ---------------------------
# Helpers: HSV hue bin (0..12)
# ---------------------------

def hsv_bin_30deg(rgb_u8: np.ndarray) -> np.ndarray:
    """Return 12-bin id in {1..12}, or 0 for gray (delta==0).
    Uses Python colorsys.rgb_to_hsv (h in [0,1))."""
    rgb = rgb_u8.astype(np.float32) / 255.0
    out = np.zeros((rgb.shape[0],), dtype=np.int16)
    for i in range(rgb.shape[0]):
        r, g, b = rgb[i]
        mx = max(r, g, b)
        mn = min(r, g, b)
        if mx == mn:
            out[i] = 0
        else:
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            deg = (h * 360.0) % 360.0
            out[i] = int(np.floor(deg / 30.0)) + 1
    return out

# ---------------------------------------------
# Rule-table 12-bin id in {1..12}, or 0 for gray
# ---------------------------------------------

def rule_bin_12(rgb_u8: np.ndarray) -> np.ndarray:
    """Implements the 12-bin rule table:
    Decide 60° sector via channel ordering (max/mid/min), then split
    each 60° into two 30° bins using one comparison: 2*mid ? max+min

    Bin ordering:
      1:  [0,30)   2:[30,60)   3:[60,90)   4:[90,120)
      5:  [120,150) 6:[150,180) 7:[180,210) 8:[210,240)
      9:  [240,270) 10:[270,300) 11:[300,330) 12:[330,360)
    """
    R = rgb_u8[:, 0].astype(np.int32)
    G = rgb_u8[:, 1].astype(np.int32)
    B = rgb_u8[:, 2].astype(np.int32)

    mx = np.maximum(np.maximum(R, G), B)
    mn = np.minimum(np.minimum(R, G), B)
    out = np.zeros((rgb_u8.shape[0],), dtype=np.int16)

    # Gray
    gray = (mx == mn)
    # Non-gray mask
    m = ~gray

    # For deterministic binning on ties, we still evaluate with >= / <=.
    # Boundary cases (equalities) are inherently ambiguous vs HSV; the
    # main verification filters them out.

    # Sector A: R >= G >= B  (bins 1,2)
    s = m & (R >= G) & (G >= B)
    if np.any(s):
        low = (2 * G[s] <= (R[s] + B[s]))
        out[s] = np.where(low, 1, 2)

    # Sector B: G >= R >= B (bins 3,4)
    s = m & (G >= R) & (R >= B)
    if np.any(s):
        low = (2 * R[s] >= (G[s] + B[s]))
        out[s] = np.where(low, 3, 4)

    # Sector C: G >= B >= R (bins 5,6)
    s = m & (G >= B) & (B >= R)
    if np.any(s):
        low = (2 * B[s] <= (G[s] + R[s]))
        out[s] = np.where(low, 5, 6)

    # Sector D: B >= G >= R (bins 7,8)
    s = m & (B >= G) & (G >= R)
    if np.any(s):
        low = (2 * G[s] >= (B[s] + R[s]))
        out[s] = np.where(low, 7, 8)

    # Sector E: B >= R >= G (bins 9,10)
    s = m & (B >= R) & (R >= G)
    if np.any(s):
        low = (2 * R[s] <= (B[s] + G[s]))
        out[s] = np.where(low, 9, 10)

    # Sector F: R >= B >= G (bins 11,12)
    s = m & (R >= B) & (B >= G)
    if np.any(s):
        low = (2 * B[s] >= (R[s] + G[s]))
        out[s] = np.where(low, 11, 12)

    return out

# ----------------------------------------------------
# YCoCg (float) forward/inverse for rotation experiment
# ----------------------------------------------------

def rgb_to_ycocg(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """rgb in [0,255] float32. Returns (Y, Co, Cg) float32."""
    R, G, B = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    Co = R - B
    Cg = G - 0.5 * (R + B)
    Y = 0.25 * (R + 2.0 * G + B)
    return Y, Co, Cg


def ycocg_to_rgb(Y: np.ndarray, Co: np.ndarray, Cg: np.ndarray) -> np.ndarray:
    """Inverse of the above (float). Returns rgb float32."""
    R = Y + 0.5 * Co - Cg
    G = Y + Cg
    B = Y - 0.5 * Co - Cg
    rgb = np.stack([R, G, B], axis=1)
    return rgb


def rotate_cocg(rgb_u8: np.ndarray, theta_deg: float) -> np.ndarray:
    """Pure chroma rotation in (Co,Cg), keep Y constant."""
    rgb = rgb_u8.astype(np.float32)
    Y, Co, Cg = rgb_to_ycocg(rgb)
    th = np.deg2rad(theta_deg)
    c, s = np.cos(th), np.sin(th)
    Co2 = c * Co - s * Cg
    Cg2 = s * Co + c * Cg
    rgb2 = ycocg_to_rgb(Y, Co2, Cg2)
    rgb2 = np.clip(np.round(rgb2), 0, 255).astype(np.uint8)
    return rgb2

# -----------------------------------------------------------------
# Sector-preserving hue tweak (extra transform) that stays in 30° bin
# -----------------------------------------------------------------

def sector_preserving_hue(rgb_u8: np.ndarray, delta_h_deg: float) -> np.ndarray:
    """Implements strict HSV-like hue adjustment WITHIN the current 30° bin.

    Key idea:
      - Determine (M, m, mid) and which 60° sector (max/min identity).
      - In local coords: Co = M - m, Cg = mid - (M+m)/2.
      - Within each 60° sector, the HSV hue parameter can be written as
            t = 1/2 + s * (Cg/Co),  with s in {+1,-1} depending on sector.
      - A 30° bin corresponds to t in [0,0.5] or [0.5,1].
      - Apply desired hue shift: t' = clip(t + delta_h/60, bin_lower, bin_upper)
      - Map back: Cg' = (t' - 1/2) * Co / s
      - Reconstruct rgb with same max/min, updated mid.

    This guarantees the output stays in the SAME 30° bin by construction.
    """
    rgb = rgb_u8.astype(np.float32)
    R, G, B = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    # Compute max/min and identify their channels
    M = np.maximum(np.maximum(R, G), B)
    m = np.minimum(np.minimum(R, G), B)
    Co = M - m

    out = rgb.copy()

    # gray: no-op
    gray = (Co == 0)
    if np.all(gray):
        return rgb_u8.copy()

    # Determine which channel is max/min and thus which is mid
    # We'll build masks for the 6 sectors (max,min identities) with tie-friendly >=/<=.

    # max=R, min=B
    s1 = (~gray) & (R >= G) & (R >= B) & (B <= G) & (B <= R)
    # max=G, min=B
    s2 = (~gray) & (G >= R) & (G >= B) & (B <= R) & (B <= G)
    # max=G, min=R
    s3 = (~gray) & (G >= R) & (G >= B) & (R <= B) & (R <= G)
    # max=B, min=R
    s4 = (~gray) & (B >= R) & (B >= G) & (R <= G) & (R <= B)
    # max=B, min=G
    s5 = (~gray) & (B >= R) & (B >= G) & (G <= R) & (G <= B)
    # max=R, min=G
    s6 = (~gray) & (R >= G) & (R >= B) & (G <= B) & (G <= R)

    # Sector sign mapping s (see derivation):
    # (max,min) = (R,B): +1
    # (G,B): -1
    # (G,R): +1
    # (B,R): -1
    # (B,G): +1
    # (R,G): -1

    k = delta_h_deg / 60.0

    def process(mask: np.ndarray, maxc: str, midc: str, minc: str, sgn: float):
        if not np.any(mask):
            return
        # fetch channels
        ch = {'R': R, 'G': G, 'B': B}
        Mv = ch[maxc][mask]
        mv = ch[minc][mask]
        midv = ch[midc][mask]
        Cov = (Mv - mv)
        # local Cg
        Cgv = midv - 0.5 * (Mv + mv)
        # t in [0,1]
        t = 0.5 + sgn * (Cgv / Cov)
        # determine current 30° sub-bin
        # low: t < 0.5 ; high: t >= 0.5
        low = (t < 0.5)
        lo = np.where(low, 0.0, 0.5)
        hi = np.where(low, 0.5, 1.0)
        # apply shift and clamp to current 30° bin
        tp = np.clip(t + k, lo, hi)
        # map back
        Cgp = (tp - 0.5) * Cov / sgn
        midp = Cgp + 0.5 * (Mv + mv)

        # write back (keep max/min fixed)
        out_mask = out[mask]
        idx = {'R': 0, 'G': 1, 'B': 2}
        out_mask[:, idx[maxc]] = Mv
        out_mask[:, idx[minc]] = mv
        out_mask[:, idx[midc]] = midp
        out[mask] = out_mask

    process(s1, 'R', 'G', 'B', +1.0)
    process(s2, 'G', 'R', 'B', -1.0)
    process(s3, 'G', 'B', 'R', +1.0)
    process(s4, 'B', 'G', 'R', -1.0)
    process(s5, 'B', 'R', 'G', +1.0)
    process(s6, 'R', 'B', 'G', -1.0)

    out_u8 = np.clip(np.round(out), 0, 255).astype(np.uint8)
    return out_u8

# --------------------------
# Verification entry point
# --------------------------

def main(seed: int = 0, n: int = 200_000, delta_h: float = 10.0):
    rng = np.random.default_rng(seed)
    rgb = rng.integers(0, 256, size=(n, 3), dtype=np.uint8)

    # (1) Verify rule table vs HSV bins (filter non-boundary samples)
    hsv_bins = hsv_bin_30deg(rgb)
    rule_bins = rule_bin_12(rgb)

    # Filter out ambiguous boundary samples where ties occur or on exact sub-bin boundary.
    R = rgb[:, 0].astype(np.int32)
    G = rgb[:, 1].astype(np.int32)
    B = rgb[:, 2].astype(np.int32)
    mx = np.maximum(np.maximum(R, G), B)
    mn = np.minimum(np.minimum(R, G), B)
    delta = mx - mn

    # distinct channels + non-gray
    distinct = (R != G) & (G != B) & (R != B) & (delta != 0)

    # exclude exact 30° boundaries in local coords: mid == (max+min)/2
    # Compute mid and exclude if 2*mid == max+min
    # First determine mid for each sample
    mid = (R + G + B) - mx - mn
    non_boundary = (2 * mid != (mx + mn))

    strict = distinct & non_boundary

    match = (hsv_bins[strict] == rule_bins[strict])
    match_rate = match.mean() if match.size else float('nan')

    # (2) Compare bin stability under pure CoCg rotation vs sector-preserving transform
    rot = rotate_cocg(rgb, theta_deg=delta_h)
    sp  = sector_preserving_hue(rgb, delta_h_deg=delta_h)

    b0 = hsv_bin_30deg(rgb)
    b_rot = hsv_bin_30deg(rot)
    b_sp  = hsv_bin_30deg(sp)

    # For stability test, ignore grays (bin 0)
    mask = (b0 != 0)
    same_rot = (b_rot[mask] == b0[mask]).mean()
    same_sp  = (b_sp[mask]  == b0[mask]).mean()

    # (3) Check that sector-preserving output matches strict HSV hue adjustment WITHIN current 30° bin.
    # We'll build a reference HSV transform that clamps hue inside the original 30° bin.
    ref = rgb.astype(np.float32)
    ref_out = np.zeros_like(ref)

    for i in range(n):
        r, g, b = ref[i] / 255.0
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        if s == 0.0:
            ref_out[i] = ref[i]
            continue
        deg = (h * 360.0) % 360.0
        bin_id = int(np.floor(deg / 30.0))
        lo = bin_id * 30.0
        hi = lo + 30.0
        # apply shift and clamp within [lo,hi)
        deg2 = deg + delta_h
        # clamp (note: within the bin, we treat it as closed interval for simplicity)
        deg2 = min(max(deg2, lo), hi)
        h2 = (deg2 % 360.0) / 360.0
        r2, g2, b2 = colorsys.hsv_to_rgb(h2, s, v)
        ref_out[i] = np.array([r2, g2, b2]) * 255.0

    ref_u8 = np.clip(np.round(ref_out), 0, 255).astype(np.uint8)

    mae = np.mean(np.abs(ref_u8.astype(np.int16) - sp.astype(np.int16)), axis=0)
    maxe = np.max(np.abs(ref_u8.astype(np.int16) - sp.astype(np.int16)), axis=0)

    print("=== (1) Rule-table vs HSV bin verification ===")
    print(f"Total samples: {n}")
    print(f"Strict (non-gray, no ties, not on sub-bin boundary): {strict.sum()}")
    print(f"Match rate on strict set: {match_rate:.8f}")
    print()

    print("=== (2) Bin stability under transforms (ignore gray) ===")
    print(f"Delta hue = {delta_h} deg")
    print(f"Same-bin rate: Pure YCoCg rotation: {same_rot:.6f}")
    print(f"Same-bin rate: Sector-preserving extra transform: {same_sp:.6f}")
    print()

    print("=== (3) Sector-preserving vs HSV(clamped within original 30° bin) ===")
    print(f"RGB MAE per channel: {mae}")
    print(f"RGB MaxErr per channel: {maxe}")


if __name__ == "__main__":
    main(seed=0, n=200_000, delta_h=5.0)
