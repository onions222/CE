from __future__ import annotations
import argparse
from dataclasses import asdict
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from wpa.colorspace import srgb_to_linear, linear_to_srgb, rgb_to_ycocg, ycocg_to_rgb, clip_u8
from wpa.tables import KelvinTableOptions, build_tables_kelvin_12bin
from wpa.apply import WPAParams, wpa_apply_ycocg_lms_nobanding

def load_rgb_u8(path: str) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    return np.array(im, dtype=np.uint8)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default='grass.jpg', help="Input image path (sRGB)")
    ap.add_argument("--wa_sel", type=int, default=127, help="0..127, 64 neutral")
    ap.add_argument("--wa_en", type=int, default=1, help="0/1")
    ap.add_argument("--warm_end", type=float, default=4500.0)
    ap.add_argument("--cool_end", type=float, default=9000.0)
    ap.add_argument("--bin_mode", type=str, default="doc_step", choices=["uniform_linear","doc_step","doc_linear"])

    # behavior tuning
    ap.add_argument("--linearize", action="store_true", help="Apply WPA in linear RGB domain (recommended)")
    ap.add_argument("--keepY", action="store_true", help="Keep Y (avoid wash-out)")
    ap.add_argument("--tint_k", type=float, default=0.0, help="0.10~0.30 to push warm toward yellow")
    ap.add_argument("--chroma_pullback", type=float, default=0.0, help="0.2~0.5 to reduce wash-out")
    ap.add_argument("--save", type=str, default=None, help="Optional: save output image to this path")
    args = ap.parse_args()

    rgb_u8 = load_rgb_u8(args.inp)  # HxWx3 uint8

    if args.linearize:
        I = rgb_u8.astype(np.float64) / 255.0
        Ilin = srgb_to_linear(I)
        rgb_for_wpa = Ilin * 255.0
    else:
        rgb_for_wpa = rgb_u8.astype(np.float64)

    Y, Co, Cg = rgb_to_ycocg(rgb_for_wpa)

    tbl = build_tables_kelvin_12bin(KelvinTableOptions(
        T_warm_end=args.warm_end,
        T_cool_end=args.cool_end,
    ))

    p = WPAParams(
        keepY=bool(args.keepY),
        bin_mode=args.bin_mode,
        tint_k=float(args.tint_k),
        chroma_pullback=float(args.chroma_pullback),
    )

    Y2, Co2, Cg2 = wpa_apply_ycocg_lms_nobanding(Y, Co, Cg, args.wa_sel, args.wa_en, tbl, p)

    rgb2 = ycocg_to_rgb(Y2, Co2, Cg2)

    if args.linearize:
        # rgb2 is still linear domain (scaled 0..255). Convert back to sRGB for display.
        lin = np.clip(rgb2 / 255.0, 0.0, 1.0)
        srgb = linear_to_srgb(lin)
        out_u8 = clip_u8(srgb * 255.0)
    else:
        out_u8 = clip_u8(rgb2)

    # 显示原图和处理后的图片对比
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    axes[0].imshow(rgb_u8)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(out_u8)
    axes[1].set_title('WPA Applied')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)  # 让窗口渲染
    
    # 如果指定了保存路径，保存图片
    if args.save:
        from PIL import Image as PILImage
        PILImage.fromarray(out_u8, mode="RGB").save(args.save)
        print(f"已保存: {args.save}")
    
    input("按 Enter 键关闭图片并退出...")
    

if __name__ == "__main__":
    main()
