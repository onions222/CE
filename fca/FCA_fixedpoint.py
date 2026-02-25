"""
FCA_fixedpoint.py – Fixed-point implementation of FCA hue shift.

All core hue-shift computations use integer-only arithmetic.
Configurable precision via FRAC_BITS.

Internal hue representation:
  hue_hu = sector * ONE + t_fp,  range [0, 6*ONE)
  where ONE = 1 << FRAC_BITS, each sector spans ONE hue-units (= 60°).
  This avoids expensive divide-by-60: sector = hue_hu >> FRAC_BITS,
  t_fp = hue_hu & (ONE - 1).
"""

import numpy as np
from PIL import Image


def hue_shift_fixedpoint(
    img_u8: np.ndarray,
    deltaH_deg: float,
    use_hue_range: bool = False,
    hue_min_deg: float = 0.0,
    hue_max_deg: float = 360.0,
    taper_width_deg: float = 0.0,
    delta_eps: int = 1,
    max_abs_deltaH_deg: float = 15.0,
    frac_bits: int = 10,
):
    """
    Fixed-point hue shift preserving V and S.
    All core math is integer-only (no float in the processing pipeline).

    Parameters
    ----------
    img_u8 : (H, W, 3) uint8 RGB image
    deltaH_deg : hue offset in degrees, clamped to ±max_abs_deltaH_deg
    use_hue_range, hue_min_deg, hue_max_deg : optional hue range filter
    taper_width_deg : soft taper at range boundaries (degrees)
    delta_eps : minimum Delta threshold (integer, default 1 = one 8-bit LSB)
    max_abs_deltaH_deg : hard limit on |deltaH| in degrees (default 15)
    frac_bits : fractional bits for fixed-point precision (default 10)

    Returns
    -------
    out_u8 : (H, W, 3) uint8 output image
    dbg : dict with debug info
    """
    ONE = np.int64(1 << frac_bits)
    HUE_FULL = np.int64(6 * ONE)       # full circle in hue units
    HALF = np.int64(1 << (frac_bits - 1))  # 0.5 for rounding

    # --- 0) Hard-clamp deltaH, convert all params to hue units (integer) ---
    deltaH_deg = max(-max_abs_deltaH_deg, min(deltaH_deg, max_abs_deltaH_deg))

    def deg2hu(deg):
        """Convert degrees to integer hue units: hu = round(deg * 6 * ONE / 360)."""
        return int(round(deg * float(HUE_FULL) / 360.0))

    deltaH_hu  = np.int64(deg2hu(deltaH_deg))
    hmin_hu    = np.int64(deg2hu(hue_min_deg))
    hmax_hu    = np.int64(deg2hu(hue_max_deg))
    taper_hu   = np.int64(deg2hu(taper_width_deg))

    # --- 1) Extract channels as int32 ---
    R = img_u8[:, :, 0].astype(np.int64)
    G = img_u8[:, :, 1].astype(np.int64)
    B = img_u8[:, :, 2].astype(np.int64)

    # --- 2) V, m, Delta ---
    V = np.maximum(np.maximum(R, G), B)
    m = np.minimum(np.minimum(R, G), B)
    Delta = V - m

    active = Delta >= delta_eps

    # --- Sector classification (identical to float version) ---
    isRmax = active & (R >= G) & (R >= B)
    isGmax = active & (G > R)  & (G >= B)
    isBmax = active & ~(isRmax | isGmax)

    s0 = isRmax & (G >= B)     # 0..60
    s5 = isRmax & ~(G >= B)    # 300..360
    s1 = isGmax & (B <= R)     # 60..120
    s2 = isGmax & ~(B <= R)    # 120..180
    s3 = isBmax & (R <= G)     # 180..240
    s4 = isBmax & ~(R <= G)    # 240..300

    # --- 3) Compute t in fixed-point: t_fp = round((diff << frac_bits) / Delta) ---
    t_fp = np.zeros_like(V, dtype=np.int64)

    sector_diffs = [
        (s0, G - B),   # s0: V=R, min=B
        (s1, G - R),   # s1: V=G, min=B
        (s2, B - R),   # s2: V=G, min=R
        (s3, B - G),   # s3: V=B, min=R
        (s4, R - G),   # s4: V=B, min=G
        (s5, R - B),   # s5: V=R, min=G
    ]
    for mask, diff in sector_diffs:
        if np.any(mask):
            num = diff[mask] << frac_bits
            den = Delta[mask]
            t_fp[mask] = (num + (den >> 1)) // den   # round-to-nearest

    t_fp = np.clip(t_fp, 0, ONE)

    # --- Hue in hue units: hue_hu = sector * ONE + t_fp ---
    hue_hu = np.zeros_like(V, dtype=np.int64)
    for i, sx in enumerate([s0, s1, s2, s3, s4, s5]):
        hue_hu[sx] = np.int64(i) * ONE + t_fp[sx]

    # --- Hue range filter ---
    if use_hue_range:
        if hmin_hu <= hmax_hu:
            enable = (hue_hu >= hmin_hu) & (hue_hu <= hmax_hu)
        else:
            enable = (hue_hu >= hmin_hu) | (hue_hu <= hmax_hu)
        active = active & enable

    # --- 4) Effective deltaH with soft taper (all integer) ---
    eff_dH = np.full_like(V, deltaH_hu, dtype=np.int64)

    if use_hue_range and taper_hu > 0:
        # Auto-clamp taper
        if hmin_hu <= hmax_hu:
            rw = hmax_hu - hmin_hu
        else:
            rw = (HUE_FULL - hmin_hu) + hmax_hu
        eff_taper = min(taper_hu, rw // 2)

        if eff_taper > 0:
            # Edge distance in hue units
            edge_dist = np.zeros_like(V, dtype=np.int64)
            h = hue_hu[active]

            if hmin_hu <= hmax_hu:
                d_lo = h - hmin_hu
                d_hi = hmax_hu - h
            else:
                d_lo = np.where(h >= hmin_hu, h - hmin_hu, h + HUE_FULL - hmin_hu)
                d_hi = np.where(h <= hmax_hu, hmax_hu - h, hmax_hu + HUE_FULL - h)

            edge_dist[active] = np.minimum(d_lo, d_hi)

            # scale = clip((edge_dist << frac_bits) / eff_taper, 0, ONE)
            scale = np.zeros_like(V, dtype=np.int64)
            scale[active] = np.clip(
                (edge_dist[active] << frac_bits) // eff_taper,
                0, ONE,
            )

            # eff_dH = (deltaH_hu * scale) >> frac_bits
            eff_dH[active] = (deltaH_hu * scale[active]) >> frac_bits

    # --- New hue = (hue + eff_dH) mod HUE_FULL ---
    new_hue = np.zeros_like(V, dtype=np.int64)
    new_hue[active] = (hue_hu[active] + eff_dH[active]) % HUE_FULL

    # --- Derive new sector (shift) and new t (mask) — no division needed ---
    new_sec = np.zeros_like(V, dtype=np.int64)
    new_t   = np.zeros_like(V, dtype=np.int64)
    new_sec[active] = new_hue[active] >> frac_bits       # = new_hue // ONE
    new_t[active]   = new_hue[active] & (ONE - 1)        # = new_hue % ONE
    new_sec = np.clip(new_sec, 0, 5)

    # --- New sector masks ---
    ns = [active & (new_sec == i) for i in range(6)]

    # --- 5) Reconstruct RGB (integer only) ---
    R2 = R.copy(); G2 = G.copy(); B2 = B.copy()

    def dt_mul(mask):
        """(Delta * new_t + HALF) >> frac_bits — rounded fixed-point multiply."""
        if not np.any(mask):
            return np.array([], dtype=np.int64)
        return (Delta[mask] * new_t[mask] + HALF) >> frac_bits

    # ns0: V=R, min=B, mid=G = m + Delta*t
    R2[ns[0]] = V[ns[0]]
    B2[ns[0]] = m[ns[0]]
    if np.any(ns[0]): G2[ns[0]] = m[ns[0]] + dt_mul(ns[0])

    # ns1: V=G, min=B, mid=R = V - Delta*t
    G2[ns[1]] = V[ns[1]]
    B2[ns[1]] = m[ns[1]]
    if np.any(ns[1]): R2[ns[1]] = V[ns[1]] - dt_mul(ns[1])

    # ns2: V=G, min=R, mid=B = m + Delta*t
    G2[ns[2]] = V[ns[2]]
    R2[ns[2]] = m[ns[2]]
    if np.any(ns[2]): B2[ns[2]] = m[ns[2]] + dt_mul(ns[2])

    # ns3: V=B, min=R, mid=G = V - Delta*t
    B2[ns[3]] = V[ns[3]]
    R2[ns[3]] = m[ns[3]]
    if np.any(ns[3]): G2[ns[3]] = V[ns[3]] - dt_mul(ns[3])

    # ns4: V=B, min=G, mid=R = m + Delta*t
    B2[ns[4]] = V[ns[4]]
    G2[ns[4]] = m[ns[4]]
    if np.any(ns[4]): R2[ns[4]] = m[ns[4]] + dt_mul(ns[4])

    # ns5: V=R, min=G, mid=B = V - Delta*t
    R2[ns[5]] = V[ns[5]]
    G2[ns[5]] = m[ns[5]]
    if np.any(ns[5]): B2[ns[5]] = V[ns[5]] - dt_mul(ns[5])

    # Clamp to [0, 255]
    R2 = np.clip(R2, 0, 255)
    G2 = np.clip(G2, 0, 255)
    B2 = np.clip(B2, 0, 255)

    out_u8 = np.stack([R2, G2, B2], axis=-1).astype(np.uint8)

    dbg = {"frac_bits": frac_bits, "deltaH_hu": int(deltaH_hu), "HUE_FULL": int(HUE_FULL)}
    return out_u8, dbg


# ===================== main: run and compare against float version =====================
if __name__ == "__main__":
    from FCA import hue_shift_v_s_strict_ycocg

    img_path = "grass.jpg"
    img_u8 = np.asarray(Image.open(img_path))
    img_f64 = img_u8.astype(np.float64) / 255.0

    deltaH = +10.0
    use_hue_range = True
    hue_min = 50.0
    hue_max = 170.0
    taper_width = 15.0

    # --- Fixed-point ---
    out_fp, dbg_fp = hue_shift_fixedpoint(
        img_u8, deltaH,
        use_hue_range=use_hue_range,
        hue_min_deg=hue_min,
        hue_max_deg=hue_max,
        taper_width_deg=taper_width,
    )

    # --- Floating-point reference ---
    out_ref_f64, dbg_ref = hue_shift_v_s_strict_ycocg(
        img_f64, deltaH,
        use_hue_range=use_hue_range,
        hue_min=hue_min,
        hue_max=hue_max,
        taper_width=taper_width,
    )
    out_ref_u8 = (np.clip(out_ref_f64, 0.0, 1.0) * 255.0).round().astype(np.uint8)

    # --- Compare ---
    diff = np.abs(out_fp.astype(np.int16) - out_ref_u8.astype(np.int16))
    print(f"=== Fixed-point vs Float (FRAC_BITS={dbg_fp['frac_bits']}) ===")
    print(f"  MAE  :  R={diff[:,:,0].mean():.4f}  G={diff[:,:,1].mean():.4f}  B={diff[:,:,2].mean():.4f}")
    print(f"  MaxE :  R={diff[:,:,0].max()}  G={diff[:,:,1].max()}  B={diff[:,:,2].max()}")
    print(f"  Float ref — Max err V: {dbg_ref['maxErrV']:.3g}, S: {dbg_ref['maxErrS']:.3g}")

    Image.fromarray(out_fp).save("out_fixedpoint.png")
    print("Saved: out_fixedpoint.png")
