from __future__ import annotations
from dataclasses import dataclass
import numpy as np

Array = np.ndarray

@dataclass(frozen=True)
class KelvinTableOptions:
    T_warm_end: float = 3500.0
    T_cool_end: float = 9000.0
    xy_base: tuple[float, float] = (0.3127, 0.3290)  # D65
    w_shadow: float = 0.60
    w_mid: float = 1.00
    w_highlight: float = 0.70
    bin_mid: int = 6

@dataclass
class KelvinTables12Bin:
    opts: KelvinTableOptions
    w: Array                 # (12,)
    T_base: float
    Tw: Array                # (12,)
    Tc: Array                # (12,)
    xy_warm: Array           # (12,2)
    xy_cool: Array           # (12,2)
    Twarm: list[Array]       # 12 of (3,3)
    Tcool: list[Array]       # 12 of (3,3)
    DmatWarm: list[Array]    # 12 of (3,3)
    DmatCool: list[Array]    # 12 of (3,3)

def build_tables_kelvin_12bin(opts: KelvinTableOptions = KelvinTableOptions()) -> KelvinTables12Bin:
    """
    Python port of wpa_build_tables_kelvin_12bin.m
    Build 12-bin endpoint tables for WPA in YCoCg domain, derived from Bradford CAT in XYZ/LMS.
    """
    # ---- YCoCg matrices (match rgb2ycocg/ycocg2rgb) ----
    M_R2Y = np.array([
        [ 0.25,  0.50,  0.25],
        [ 0.50,  0.00, -0.50],
        [-0.25,  0.50, -0.25],
    ], dtype=np.float64)
    M_Y2R = np.array([
        [1,  1, -1],
        [1,  0,  1],
        [1, -1, -1],
    ], dtype=np.float64)

    # ---- linear sRGB / Rec.709 RGB <-> XYZ ----
    M_R2X = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], dtype=np.float64)
    M_X2R = np.linalg.inv(M_R2X)

    # ---- Bradford CAT XYZ <-> LMS ----
    M_X2L = np.array([
        [ 0.8951,  0.2664, -0.1614],
        [-0.7502,  1.7135,  0.0367],
        [ 0.0389, -0.0685,  1.0296],
    ], dtype=np.float64)
    M_L2X = np.linalg.inv(M_X2L)

    x0, y0 = opts.xy_base
    XYZ_base = xy_to_XYZ_Y1(x0, y0)
    LMS_base = M_X2L @ XYZ_base

    w = build_weights_12(opts.w_shadow, opts.w_mid, opts.w_highlight, opts.bin_mid)

    T_base = xy_to_cct_mccamy(x0, y0)
    Tw = interp_cct_mired(T_base, opts.T_warm_end, w)
    Tc = interp_cct_mired(T_base, opts.T_cool_end, w)

    xy_warm = np.zeros((12, 2), dtype=np.float64)
    xy_cool = np.zeros((12, 2), dtype=np.float64)

    Twarm: list[Array] = []
    Tcool: list[Array] = []
    DmatWarm: list[Array] = []
    DmatCool: list[Array] = []

    for i in range(12):
        xy_w = cct_to_xy_kang2002(float(Tw[i]))
        xy_c = cct_to_xy_kang2002(float(Tc[i]))
        xy_warm[i, :] = xy_w
        xy_cool[i, :] = xy_c

        XYZ_w = xy_to_XYZ_Y1(xy_w[0], xy_w[1])
        XYZ_c = xy_to_XYZ_Y1(xy_c[0], xy_c[1])

        LMS_w = M_X2L @ XYZ_w
        LMS_c = M_X2L @ XYZ_c

        r_w = LMS_w / LMS_base  # 3,
        r_c = LMS_c / LMS_base

        A_w = M_L2X @ np.diag(r_w) @ M_X2L
        A_c = M_L2X @ np.diag(r_c) @ M_X2L

        # YCoCg-domain endpoint transforms:
        T_w = M_R2Y @ M_X2R @ A_w @ M_R2X @ M_Y2R
        T_c = M_R2Y @ M_X2R @ A_c @ M_R2X @ M_Y2R

        Twarm.append(T_w)
        Tcool.append(T_c)
        DmatWarm.append(T_w - np.eye(3))
        DmatCool.append(T_c - np.eye(3))

    return KelvinTables12Bin(
        opts=opts, w=w, T_base=float(T_base), Tw=Tw, Tc=Tc,
        xy_warm=xy_warm, xy_cool=xy_cool,
        Twarm=Twarm, Tcool=Tcool, DmatWarm=DmatWarm, DmatCool=DmatCool
    )

# ================= helpers =================

def build_weights_12(w_shadow: float, w_mid: float, w_highlight: float, bin_mid: int) -> Array:
    w = np.zeros(12, dtype=np.float64)
    for i in range(12):
        ii = i + 1  # MATLAB 1-based
        if ii <= bin_mid:
            t = (ii - 1) / max(1, (bin_mid - 1))
            w[i] = w_shadow + t * (w_mid - w_shadow)
        else:
            t = (ii - bin_mid) / max(1, (12 - bin_mid))
            w[i] = w_mid + t * (w_highlight - w_mid)
    return np.clip(w, 0.0, 1.0)

def interp_cct_mired(T_base: float, T_end: float, w: Array) -> Array:
    m_base = 1e6 / float(T_base)
    m_end  = 1e6 / float(T_end)
    m = (1.0 - w) * m_base + w * m_end
    return 1e6 / m

def xy_to_XYZ_Y1(x: float, y: float) -> Array:
    Y = 1.0
    X = (x / y) * Y
    Z = ((1.0 - x - y) / y) * Y
    return np.array([X, Y, Z], dtype=np.float64)

def xy_to_cct_mccamy(x: float, y: float) -> float:
    n = (x - 0.3320) / (0.1858 - y)
    return float(449 * n**3 + 3525 * n**2 + 6823.3 * n + 5520.33)

def cct_to_xy_kang2002(T: float) -> tuple[float, float]:
    if T < 1667 or T > 25000:
        raise ValueError("Kang2002 valid range is 1667..25000K.")
    if T <= 4000:
        x = -0.2661239e9 / T**3 - 0.2343589e6 / T**2 + 0.8776956e3 / T + 0.179910
    else:
        x = -3.0258469e9 / T**3 + 2.1070379e6 / T**2 + 0.2226347e3 / T + 0.240390

    if T <= 2222:
        y = -1.1063814*x**3 - 1.34811020*x**2 + 2.18555832*x - 0.20219683
    elif T <= 4000:
        y = -0.9549476*x**3 - 1.37418593*x**2 + 2.09137015*x - 0.16748867
    else:
        y =  3.0817580*x**3 - 5.8733867*x**2 + 3.75112997*x - 0.37001483
    return float(x), float(y)
