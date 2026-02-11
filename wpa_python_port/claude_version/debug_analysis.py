"""Diagnostic script: why does WA_SEL=127 (max cool) produce no visible change?"""
import numpy as np
from PIL import Image
from wpa.config import WPAConfig
from wpa.core import wpa_process_rgb_uint8
from wpa.weights import compute_sat_weight, compute_luma_proxy

img = np.array(Image.open("../grass.jpg").convert("RGB"), dtype=np.uint8)
print(f"Image shape: {img.shape}, dtype: {img.dtype}")

# --- 1. Analyze pixel statistics ---
print("\n=== Pixel Statistics ===")
print(f"R  mean={img[...,0].mean():.1f}  G  mean={img[...,1].mean():.1f}  B  mean={img[...,2].mean():.1f}")

# --- 2. Saturation proxy distribution ---
r, g, b = img[...,0].astype(np.float32), img[...,1].astype(np.float32), img[...,2].astype(np.float32)
s = np.abs(r - g) + np.abs(g - b) + np.abs(b - r)
print(f"\n=== Saturation Proxy s = |R-G|+|G-B|+|B-R| ===")
print(f"  min={s.min():.1f}  mean={s.mean():.1f}  median={np.median(s):.1f}  max={s.max():.1f}")
print(f"  Pct s <= 20  (w=1 zone):  {(s <= 20).mean()*100:.1f}%")
print(f"  Pct 20 < s < 110 (ramp):  {((s > 20) & (s < 110)).mean()*100:.1f}%")
print(f"  Pct s >= 110 (w=0 zone):  {(s >= 110).mean()*100:.1f}%")

# --- 3. Saturation weight distribution ---
gamma_f32 = img.astype(np.float32)  # 0-255 scale
w = compute_sat_weight(gamma_f32, s0=20.0, s1=110.0)
print(f"\n=== Saturation Weight w (default s0=20, s1=110) ===")
print(f"  min={w.min():.3f}  mean={w.mean():.3f}  max={w.max():.3f}")
print(f"  Pct w == 0 (fully suppressed): {(w == 0).mean()*100:.1f}%")
print(f"  Pct w == 1 (full effect):      {(w == 1).mean()*100:.1f}%")

# --- 4. Process with sat protection ON vs OFF ---
cfg_on = WPAConfig(wa_sel=127, sat_en=True)
cfg_off = WPAConfig(wa_sel=127, sat_en=False)
out_on = wpa_process_rgb_uint8(img, cfg_on)
out_off = wpa_process_rgb_uint8(img, cfg_off)

diff_on = np.abs(out_on.astype(np.int16) - img.astype(np.int16))
diff_off = np.abs(out_off.astype(np.int16) - img.astype(np.int16))

print(f"\n=== Output Difference (sat_en=True) ===")
print(f"  max pixel diff: {diff_on.max()}")
print(f"  mean pixel diff: {diff_on.mean():.2f}")
print(f"  R_diff mean: {diff_on[...,0].mean():.2f}  G_diff mean: {diff_on[...,1].mean():.2f}  B_diff mean: {diff_on[...,2].mean():.2f}")

print(f"\n=== Output Difference (sat_en=False) ===")
print(f"  max pixel diff: {diff_off.max()}")
print(f"  mean pixel diff: {diff_off.mean():.2f}")
print(f"  R_diff mean: {diff_off[...,0].mean():.2f}  G_diff mean: {diff_off[...,1].mean():.2f}  B_diff mean: {diff_off[...,2].mean():.2f}")

# --- 5. Show a sample pixel trace ---
print(f"\n=== Sample Pixel Trace (center pixel) ===")
h, w_img = img.shape[:2]
cy, cx = h // 2, w_img // 2
px = img[cy, cx]
print(f"  Input:  R={px[0]}, G={px[1]}, B={px[2]}")
s_px = abs(int(px[0])-int(px[1])) + abs(int(px[1])-int(px[2])) + abs(int(px[2])-int(px[0]))
print(f"  Sat proxy s = {s_px}")
w_px = max(0, min(1, (110 - s_px) / (110 - 20)))
print(f"  Sat weight w = {w_px:.3f}")
print(f"  Output (sat ON):  R={out_on[cy,cx,0]}, G={out_on[cy,cx,1]}, B={out_on[cy,cx,2]}")
print(f"  Output (sat OFF): R={out_off[cy,cx,0]}, G={out_off[cy,cx,1]}, B={out_off[cy,cx,2]}")

# --- 6. Gain magnitude analysis ---
print(f"\n=== Default Cool Gains ===")
print(f"  cool_gain_global = (0.94, 1.00, 1.06)")
print(f"  Max gain deviation from 1.0 = ±0.06 = ±6%")
print(f"  At mid-brightness 128: R_change = 128*0.06 = {128*0.06:.1f} codes, B_change = {128*0.06:.1f} codes")
for i, node in enumerate(cfg_on.luma_nodes):
    g = cfg_on.cool_gains_bins[i]
    print(f"  Node Y={node:3d}: R_gain={g[0]:.4f}  G_gain={g[1]:.4f}  B_gain={g[2]:.4f}")

print("\n=== ROOT CAUSE SUMMARY ===")
print("1. Saturation protection (s0=20, s1=110) suppresses the effect on MOST")
print("   pixels in a colorful image (w→0 for any pixel with s >= 110)")
print("2. Even with sat protection OFF, the gains are only ±6% which is subtle")
