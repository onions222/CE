from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .tables import KelvinTables12Bin

Array = np.ndarray

@dataclass
class WPAParams:
    RGB_MAX: float = 255.0

    # brightness protection
    Y_dark2: float = 8.0
    Y_dark1: float = 16.0
    Y_bright1: float = 240.0
    Y_bright2: float = 248.0
    bright_minfac: float = 0.25

    keepY: bool = False

    # bin mapping
    bin_mode: str = "doc_step"  # "uniform_linear" | "doc_step" | "doc_linear"
    docY: tuple[int, ...] = (15, 31, 47, 63, 95, 127, 159, 191, 223, 239, 247, 255)

    # strength quantization (hardware simulation)
    s_quant_en: bool = False
    s_quant_Q: int = 64

    # optional: warm yellowing / anti-reddish tint compensation
    tint_k: float = 0.0  # 0.10~0.30 for "more yellow, less red"
    # optional: prevent "washed out"
    chroma_pullback: float = 0.0  # 0.2~0.5

def wpa_apply_ycocg_lms_nobanding(
    Y: Array, Co: Array, Cg: Array,
    wa_sel: int, wa_en: int,
    tbl: KelvinTables12Bin,
    p: WPAParams = WPAParams(),
) -> tuple[Array, Array, Array]:
    """
    Python port of wpa_apply_ycocg_lms_nobanding.m

    Notes:
    - Keeps the original bin-loop structure (12 bins) but vectorizes within each bin.
    - Adds optional `tint_k` and `chroma_pullback` without changing default behavior.
    """
    if wa_en == 0:
        return Y.copy(), Co.copy(), Cg.copy()

    Y = np.asarray(Y, dtype=np.float64)
    Co = np.asarray(Co, dtype=np.float64)
    Cg = np.asarray(Cg, dtype=np.float64)

    sel = int(np.clip(wa_sel, 0, 127))
    if sel < 64:
        side = 1  # warm
        t_global = (64 - sel) / 64.0
    else:
        side = 2  # cool
        t_global = (sel - 64) / 64.0

    Y2 = Y.copy()
    Co2 = Co.copy()
    Cg2 = Cg.copy()

    b_clamp, a_clamp = map_luma_to_bin_alpha(Y, p)

    # bins 0..10 blend; bin 11 no blend
    for bb in range(0, 11):
        mask = (b_clamp == bb)
        if not np.any(mask):
            continue

        Yv = Y[mask].reshape(-1)
        Cov = Co[mask].reshape(-1)
        Cgv = Cg[mask].reshape(-1)

        y0 = np.vstack([Yv, Cov, Cgv])  # 3xN
        aa = a_clamp[mask].reshape(-1)  # N

        w1 = float(tbl.w[bb])
        w2 = float(tbl.w[bb + 1])
        wbin = w1 + aa * (w2 - w1)  # N

        if side == 1:
            D1 = tbl.DmatWarm[bb]
            D2 = tbl.DmatWarm[bb + 1]
        else:
            D1 = tbl.DmatCool[bb]
            D2 = tbl.DmatCool[bb + 1]

        delta1 = D1 @ y0
        deltaD = (D2 - D1) @ y0
        delta = delta1 + deltaD * aa[None, :]

        # ---- optional tint compensation (warm -> more yellow, cool -> symmetric) ----
        if p.tint_k != 0.0:
            k = float(p.tint_k)
            if side == 1:
                delta[2, :] = delta[2, :] + k * delta[1, :]
            else:
                delta[2, :] = delta[2, :] - k * delta[1, :]

        s = (t_global * wbin)  # N
        bfac = brightness_factor_smooth(Yv, p)  # N
        s = s * bfac

        # If keepY, Y channel won't be applied; avoid over-conservative headroom limiting by ignoring dY.
        if p.keepY:
            delta_for_limit = delta.copy()
            delta_for_limit[0, :] = 0.0
        else:
            delta_for_limit = delta

        s = limit_strength_by_rgb_headroom(y0, delta_for_limit, s, p.RGB_MAX)  # N

        if p.s_quant_en:
            Q = int(p.s_quant_Q)
            s = np.round(s * Q) / Q

        s = np.clip(s, 0.0, 1.0)

        y_cur = y0 + delta * s[None, :]

        if p.keepY:
            y_cur[0, :] = y0[0, :]

        # optional chroma pull-back to reduce wash-out
        if p.chroma_pullback != 0.0:
            beta = float(np.clip(p.chroma_pullback, 0.0, 1.0))
            y_cur[1, :] = (1 - beta) * y_cur[1, :] + beta * y0[1, :]
            y_cur[2, :] = (1 - beta) * y_cur[2, :] + beta * y0[2, :]

        # scatter back
        Y2[mask] = y_cur[0, :].reshape(Y2[mask].shape)
        Co2[mask] = y_cur[1, :].reshape(Co2[mask].shape)
        Cg2[mask] = y_cur[2, :].reshape(Cg2[mask].shape)

    # last bin 11 (index 11)
    mask = (b_clamp == 11)
    if np.any(mask):
        Yv = Y[mask].reshape(-1)
        Cov = Co[mask].reshape(-1)
        Cgv = Cg[mask].reshape(-1)
        y0 = np.vstack([Yv, Cov, Cgv])  # 3xN

        if side == 1:
            D = tbl.DmatWarm[11]
        else:
            D = tbl.DmatCool[11]

        delta = D @ y0

        if p.tint_k != 0.0:
            k = float(p.tint_k)
            if side == 1:
                delta[2, :] = delta[2, :] + k * delta[1, :]
            else:
                delta[2, :] = delta[2, :] - k * delta[1, :]

        wbin = float(tbl.w[11])
        s = (t_global * wbin) * np.ones_like(Yv, dtype=np.float64)
        s = s * brightness_factor_smooth(Yv, p)

        if p.keepY:
            delta_for_limit = delta.copy()
            delta_for_limit[0, :] = 0.0
        else:
            delta_for_limit = delta

        s = limit_strength_by_rgb_headroom(y0, delta_for_limit, s, p.RGB_MAX)

        if p.s_quant_en:
            Q = int(p.s_quant_Q)
            s = np.round(s * Q) / Q
        s = np.clip(s, 0.0, 1.0)

        y_cur = y0 + delta * s[None, :]
        if p.keepY:
            y_cur[0, :] = y0[0, :]

        if p.chroma_pullback != 0.0:
            beta = float(np.clip(p.chroma_pullback, 0.0, 1.0))
            y_cur[1, :] = (1 - beta) * y_cur[1, :] + beta * y0[1, :]
            y_cur[2, :] = (1 - beta) * y_cur[2, :] + beta * y0[2, :]

        Y2[mask] = y_cur[0, :].reshape(Y2[mask].shape)
        Co2[mask] = y_cur[1, :].reshape(Co2[mask].shape)
        Cg2[mask] = y_cur[2, :].reshape(Cg2[mask].shape)

    return Y2, Co2, Cg2

# ================= helpers =================

def brightness_factor_smooth(Yv: Array, p: WPAParams) -> Array:
    """Continuous piecewise-linear brightness protection, ported 1:1."""
    Yv = np.asarray(Yv, dtype=np.float64)
    minfac = float(p.bright_minfac)
    bfac = np.ones_like(Yv, dtype=np.float64)

    if p.Y_dark1 > p.Y_dark2:
        bfac[Yv <= p.Y_dark2] = minfac
        m = (Yv > p.Y_dark2) & (Yv < p.Y_dark1)
        bfac[m] = minfac + (Yv[m] - p.Y_dark2) * (1 - minfac) / (p.Y_dark1 - p.Y_dark2)

    if p.Y_bright2 > p.Y_bright1:
        bfac[Yv >= p.Y_bright2] = minfac
        m = (Yv > p.Y_bright1) & (Yv < p.Y_bright2)
        bfac[m] = 1 - (Yv[m] - p.Y_bright1) * (1 - minfac) / (p.Y_bright2 - p.Y_bright1)

    return np.clip(bfac, minfac, 1.0)

def limit_strength_by_rgb_headroom(y0: Array, delta: Array, s: Array, RGB_MAX: float) -> Array:
    """
    Continuous overflow limiting by analytic s_max.
    y0, delta: 3xN, s: (N,)
    """
    s = np.asarray(s, dtype=np.float64).reshape(-1)
    R0, G0, B0 = ycocg2rgb_sep(y0[0, :], y0[1, :], y0[2, :])
    dR, dG, dB = ycocg2rgb_sep(delta[0, :], delta[1, :], delta[2, :])

    smax = np.full_like(s, np.inf, dtype=np.float64)
    smax = np.minimum(smax, bound_channel(R0, dR, RGB_MAX))
    smax = np.minimum(smax, bound_channel(G0, dG, RGB_MAX))
    smax = np.minimum(smax, bound_channel(B0, dB, RGB_MAX))

    s2 = np.minimum(s, smax)
    return np.maximum(s2, 0.0)

def bound_channel(C0: Array, dC: Array, RGB_MAX: float) -> Array:
    C0 = np.asarray(C0, dtype=np.float64).reshape(-1)
    dC = np.asarray(dC, dtype=np.float64).reshape(-1)

    bnd = np.full_like(C0, np.inf, dtype=np.float64)
    epsv = 1e-12
    pos = dC > epsv
    neg = dC < -epsv

    bnd[pos] = (RGB_MAX - C0[pos]) / dC[pos]
    bnd[neg] = (0.0 - C0[neg]) / dC[neg]

    bad0 = (C0 < 0.0) | (C0 > RGB_MAX)
    bnd[bad0] = 0.0
    return np.maximum(bnd, 0.0)

def ycocg2rgb_sep(Y: Array, Co: Array, Cg: Array) -> tuple[Array, Array, Array]:
    R = Y + Co - Cg
    G = Y + Cg
    B = Y - Co - Cg
    return R, G, B

def map_luma_to_bin_alpha(Y: Array, p: WPAParams) -> tuple[Array, Array]:
    """
    Map luma Y (HxW) to bin index b in [0..11] and alpha in [0..1] per MATLAB.
    Returns:
      b_clamp: int array HxW
      a_clamp: float array HxW
    """
    mode = str(p.bin_mode).lower()
    Y = np.asarray(Y, dtype=np.float64)

    Ynode = np.asarray(p.docY, dtype=np.float64).reshape(-1)
    if Ynode.size != 12:
        raise ValueError("p.docY must have 12 nodes.")
    if np.any(np.diff(Ynode) <= 0):
        raise ValueError("p.docY must be strictly increasing.")

    if mode == "uniform_linear":
        u = Y * 12.0 / 256.0
        b = np.floor(u)
        a = u - b

        b_clamp = b.astype(np.int32)
        a_clamp = a.astype(np.float64)

        mlow = b_clamp < 0
        b_clamp[mlow] = 0
        a_clamp[mlow] = 0.0

        mlast = b_clamp >= 11
        b_clamp[mlast] = 11
        a_clamp[mlast] = 0.0

        a_clamp = np.clip(a_clamp, 0.0, 1.0)
        return b_clamp, a_clamp

    if mode == "doc_step":
        b_clamp = np.zeros_like(Y, dtype=np.int32)
        a_clamp = np.zeros_like(Y, dtype=np.float64)

        assigned = np.zeros_like(Y, dtype=bool)
        for j in range(12):
            mj = (~assigned) & (Y <= Ynode[j])
            if np.any(mj):
                b_clamp[mj] = j
                assigned[mj] = True
        mrem = ~assigned
        if np.any(mrem):
            b_clamp[mrem] = 11
        return b_clamp, a_clamp

    if mode == "doc_linear":
        b_clamp = np.zeros_like(Y, dtype=np.int32)
        a_clamp = np.zeros_like(Y, dtype=np.float64)

        mlow = Y <= Ynode[0]
        mhi = Y >= Ynode[11]
        b_clamp[mlow] = 0
        a_clamp[mlow] = 0.0
        b_clamp[mhi] = 11
        a_clamp[mhi] = 0.0

        mmid = ~(mlow | mhi)
        if np.any(mmid):
            Ymid = Y[mmid].reshape(-1)
            btmp = np.zeros_like(Ymid, dtype=np.int32)
            atmp = np.zeros_like(Ymid, dtype=np.float64)

            # loop segments (11 segments)
            for k in range(11):
                Yk = Ynode[k]
                Yk1 = Ynode[k + 1]
                mk = (Ymid >= Yk) & (Ymid < Yk1)
                if np.any(mk):
                    btmp[mk] = k
                    atmp[mk] = (Ymid[mk] - Yk) / (Yk1 - Yk)

            b_clamp[mmid] = btmp.reshape(b_clamp[mmid].shape)
            a_clamp[mmid] = atmp.reshape(a_clamp[mmid].shape)

        a_clamp = np.clip(a_clamp, 0.0, 1.0)
        return b_clamp, a_clamp

    raise ValueError(f"Unknown p.bin_mode: {p.bin_mode} (use uniform_linear/doc_step/doc_linear)")
