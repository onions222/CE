"""
FCA.py – Python port of FCA.m (improved: cross-sector hue shift)
Hue-shift in HSV domain (6 sectors, allows cross-sector), preserving V and S strictly.
Round-trips through YCoCg at the output stage.
"""

import numpy as np
from PIL import Image


# ======= YCoCg forward =======
def rgb2ycocg(img: np.ndarray):
    """img: (H, W, 3) float64 in [0,1]. Returns (Y, Co, Cg)."""
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    Y  = 0.25 * R + 0.50 * G + 0.25 * B
    Co = 0.50 * (R - B)
    Cg = -0.25 * R + 0.50 * G - 0.25 * B
    return Y, Co, Cg


# ======= YCoCg inverse (verified) =======
def ycocg2rgb(Y: np.ndarray, Co: np.ndarray, Cg: np.ndarray) -> np.ndarray:
    R = Y + Co - Cg
    G = Y + Cg
    B = Y - Co - Cg
    return np.stack([R, G, B], axis=-1)


def _hue_dist_to_range_edge(
    hue: np.ndarray,
    hue_min: float,
    hue_max: float,
    mask: np.ndarray,
) -> np.ndarray:
    """Compute per-pixel angular distance to the nearest edge of the hue range.

    Returns an array (same shape as hue) with distance in degrees.
    Non-masked pixels get 0.
    Handles wrap-around ranges (hue_min > hue_max, e.g. [300, 30]).
    """
    dist = np.zeros_like(hue)
    if not np.any(mask):
        return dist

    h = hue[mask]

    if hue_min <= hue_max:
        # Normal range: [60, 180]
        d_lo = h - hue_min         # distance to lower edge
        d_hi = hue_max - h         # distance to upper edge
    else:
        # Wrap-around range: [300, 30]  — spans across 0°
        # Pixels satisfy (h >= hue_min) | (h <= hue_max)
        # Distance to lower edge (hue_min), going backward through 360
        d_lo = np.where(h >= hue_min, h - hue_min, h + 360.0 - hue_min)
        # Distance to upper edge (hue_max), going forward through 0
        d_hi = np.where(h <= hue_max, hue_max - h, hue_max + 360.0 - h)

    dist[mask] = np.minimum(d_lo, d_hi)
    return dist


def hue_shift_v_s_strict_ycocg(
    Iin: np.ndarray,
    deltaH: float,
    use_hue_range: bool = False,
    hue_min: float = 0.0,
    hue_max: float = 360.0,
    taper_width: float = 0.0,
    delta_eps: float = 1.0 / 255.0,
    max_abs_deltaH: float = 15.0,
    epsT: float = 1e-6,
):
    """
    Strict HSV hue shift via 6 sectors, preserving V and S exactly.
    Supports cross-sector shifting — no clamping at sector boundaries.

    Parameters
    ----------
    Iin : (H, W, 3) float64 in [0, 1]
    deltaH : hue offset in degrees. Hard-clamped to [-max_abs_deltaH, +max_abs_deltaH].
    use_hue_range : only shift pixels whose original hue is in [hue_min, hue_max]
    hue_min, hue_max : range in [0, 360]
    taper_width : degrees (>= 0). When > 0 and use_hue_range is True, deltaH
        is linearly scaled from 0 (at the hue-range edge) to full deltaH
        (at taper_width degrees inside the range). Prevents "color escape"
        at range boundaries.
    delta_eps : minimum (max - min) threshold in [0, 1]. Pixels with
        Delta < delta_eps are treated as gray and skipped. Default 1/255
        (one 8-bit LSB) to avoid hue noise on near-gray pixels.
    max_abs_deltaH : hard limit on |deltaH| in degrees. Default 30°.
    epsT : small epsilon for t clamp at 360°/0° wrap boundary

    Returns
    -------
    Iout : (H, W, 3) float64 in [0, 1]
    dbg  : dict with maxErrV and maxErrS
    """
    # --- 0) Hard-clamp deltaH ---
    deltaH = float(np.clip(deltaH, -max_abs_deltaH, max_abs_deltaH))

    # --- 1) RGB -> YCoCg (interface only) ---
    _Y, _Co, _Cg = rgb2ycocg(Iin)  # noqa: F841

    # --- 2) V, Delta, S and sector classification ---
    R = Iin[:, :, 0]
    G = Iin[:, :, 1]
    B = Iin[:, :, 2]

    V = np.maximum(np.maximum(R, G), B)
    m = np.minimum(np.minimum(R, G), B)
    Delta = V - m

    S = np.zeros_like(V)
    nzV = V > 0
    S[nzV] = Delta[nzV] / V[nzV]

    active = Delta >= delta_eps  # skip gray / near-gray pixels (hue unstable)

    # Sector masks (consistent with HSV standard; ties use >= for stability)
    isRmax = active & (R >= G) & (R >= B)
    isGmax = active & (G > R) & (G >= B)       # strict > to avoid Rmax/Gmax both true
    isBmax = active & ~(isRmax | isGmax)

    s0 = isRmax & (G >= B)    # 0..60
    s5 = isRmax & ~(G >= B)   # 300..360

    s1 = isGmax & (B <= R)    # 60..120  (min=B)
    s2 = isGmax & ~(B <= R)   # 120..180 (min=R)

    s3 = isBmax & (R <= G)    # 180..240 (min=R)
    s4 = isBmax & ~(R <= G)   # 240..300 (min=G)

    # --- 3) Intra-sector parameter t and compute original Hue ---
    t = np.zeros_like(V)

    t[s0] = (G[s0] - B[s0]) / Delta[s0]
    t[s1] = (G[s1] - R[s1]) / Delta[s1]
    t[s2] = (B[s2] - R[s2]) / Delta[s2]
    t[s3] = (B[s3] - G[s3]) / Delta[s3]
    t[s4] = (R[s4] - G[s4]) / Delta[s4]
    t[s5] = (R[s5] - B[s5]) / Delta[s5]

    t = np.clip(t, 0.0, 1.0)

    # Hue in degrees [0, 360)
    Hue = np.zeros_like(V)
    Hue[s0] = 60.0 * (0 + t[s0])
    Hue[s1] = 60.0 * (1 + t[s1])
    Hue[s2] = 60.0 * (2 + t[s2])
    Hue[s3] = 60.0 * (3 + t[s3])
    Hue[s4] = 60.0 * (4 + t[s4])
    Hue[s5] = 60.0 * (5 + t[s5])
    Hue = np.mod(Hue, 360.0)

    if use_hue_range:
        if hue_min <= hue_max:
            enable = (Hue >= hue_min) & (Hue <= hue_max)
        else:
            # wrap-around, e.g. [300, 30]
            enable = (Hue >= hue_min) | (Hue <= hue_max)
        active = active & enable

    # --- 4) Hue adjustment: cross-sector allowed, with optional soft taper ---
    # Compute per-pixel effective deltaH
    eff_dH = np.full_like(V, deltaH)

    if use_hue_range and taper_width > 0:
        # Auto-clamp taper_width so the range center always gets full deltaH
        if hue_min <= hue_max:
            range_width = hue_max - hue_min
        else:
            range_width = (360.0 - hue_min) + hue_max  # wrap-around
        eff_taper = min(taper_width, range_width / 2.0)

        edge_dist = _hue_dist_to_range_edge(Hue, hue_min, hue_max, active)
        scale = np.clip(edge_dist / eff_taper, 0.0, 1.0) if eff_taper > 0 else np.ones_like(V)
        eff_dH[active] = deltaH * scale[active]

    # Compute new absolute hue, wrap to [0, 360)
    new_hue = np.zeros_like(V)
    new_hue[active] = np.mod(Hue[active] + eff_dH[active], 360.0)

    # Derive new sector index (0..5) and new t within that sector
    new_sector = np.zeros_like(V, dtype=np.int32)
    new_t = np.zeros_like(V)

    new_sector[active] = np.floor(new_hue[active] / 60.0).astype(np.int32)
    new_sector[active] = np.clip(new_sector[active], 0, 5)  # safety for hue==360
    new_t[active] = (new_hue[active] - new_sector[active] * 60.0) / 60.0
    new_t[active] = np.clip(new_t[active], 0.0, 1.0 - epsT)

    # New sector masks (based on where the shifted hue lands)
    ns0 = active & (new_sector == 0)  # 0..60:   V→R, min→B, mid→G
    ns1 = active & (new_sector == 1)  # 60..120:  V→G, min→B, mid→R
    ns2 = active & (new_sector == 2)  # 120..180: V→G, min→R, mid→B
    ns3 = active & (new_sector == 3)  # 180..240: V→B, min→R, mid→G
    ns4 = active & (new_sector == 4)  # 240..300: V→B, min→G, mid→R
    ns5 = active & (new_sector == 5)  # 300..360: V→R, min→G, mid→B

    # --- 5) Reconstruct RGB preserving V and Delta strictly ---
    # Each sector assigns V to its max channel, (V - Delta) to its min channel,
    # and interpolates the mid channel using new_t.
    R2 = R.copy(); G2 = G.copy(); B2 = B.copy()

    # ns0: V=R, min=B, mid=G
    R2[ns0] = V[ns0]
    B2[ns0] = V[ns0] - Delta[ns0]
    G2[ns0] = B2[ns0] + Delta[ns0] * new_t[ns0]

    # ns1: V=G, min=B, mid=R
    G2[ns1] = V[ns1]
    B2[ns1] = V[ns1] - Delta[ns1]
    R2[ns1] = V[ns1] - Delta[ns1] * new_t[ns1]

    # ns2: V=G, min=R, mid=B
    G2[ns2] = V[ns2]
    R2[ns2] = V[ns2] - Delta[ns2]
    B2[ns2] = R2[ns2] + Delta[ns2] * new_t[ns2]

    # ns3: V=B, min=R, mid=G
    B2[ns3] = V[ns3]
    R2[ns3] = V[ns3] - Delta[ns3]
    G2[ns3] = V[ns3] - Delta[ns3] * new_t[ns3]

    # ns4: V=B, min=G, mid=R
    B2[ns4] = V[ns4]
    G2[ns4] = V[ns4] - Delta[ns4]
    R2[ns4] = G2[ns4] + Delta[ns4] * new_t[ns4]

    # ns5: V=R, min=G, mid=B
    R2[ns5] = V[ns5]
    G2[ns5] = V[ns5] - Delta[ns5]
    B2[ns5] = V[ns5] - Delta[ns5] * new_t[ns5]

    Iout = np.stack([R2, G2, B2], axis=-1)

    # --- 6) Round-trip through YCoCg (provides bidirectional capability) ---
    Y2, Co2, Cg2 = rgb2ycocg(Iout)
    Iout = ycocg2rgb(Y2, Co2, Cg2)

    # Clamp to [0, 1]
    Iout = np.clip(Iout, 0.0, 1.0)

    # --- 7) Verify V, S preservation ---
    Rf, Gf, Bf = Iout[:, :, 0], Iout[:, :, 1], Iout[:, :, 2]
    Vf = np.maximum(np.maximum(Rf, Gf), Bf)
    mf = np.minimum(np.minimum(Rf, Gf), Bf)
    Df = Vf - mf

    Sf = np.zeros_like(Vf)
    nzVf = Vf > 0
    Sf[nzVf] = Df[nzVf] / Vf[nzVf]

    dbg = {
        "maxErrV": np.max(np.abs(Vf - V)),
        "maxErrS": np.max(np.abs(Sf - S)),
    }

    return Iout, dbg


# ===================== main script =====================
if __name__ == "__main__":
    img_path = "soap-bubbles-nature.jpg"
    I = np.asarray(Image.open(img_path)).astype(np.float64) / 255.0

    deltaH = +10.0  # Hue offset (degrees), positive or negative

    # Optional: only affect a certain hue range (degrees, 0..360)
    use_hue_range = True
    hue_min = 50.0    # e.g. only touch green/cyan
    hue_max = 170.0
    taper_width = 15.0  # soft-taper: fade deltaH to 0 within 15° of range edges
    epsT = 1e-6         # small epsilon at t boundary

    J, dbg = hue_shift_v_s_strict_ycocg(
        I, deltaH,
        use_hue_range=use_hue_range,
        hue_min=hue_min,
        hue_max=hue_max,
        taper_width=taper_width,
        epsT=epsT,
    )

    # Save output
    out_path = "out_hue_strict_vs.png"
    J_u8 = (np.clip(J, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    Image.fromarray(J_u8).save(out_path)
    print(f"Saved: {out_path}")

    # Error report (should be ~1e-15; slightly larger if clamped to [0,1])
    print(f"Max abs err V: {dbg['maxErrV']:.3g}")
    print(f"Max abs err S: {dbg['maxErrS']:.3g}")
