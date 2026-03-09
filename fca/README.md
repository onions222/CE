# FCA — False Color Adjustment (Hue Shift)

基于 HSV 6 扇区模型的色相调整算法，**严格保持亮度 (V) 和饱和度 (S) 不变**。

## 文件结构

| 文件 | 说明 |
|------|------|
| `FCA.py` | Python 浮点版（主实现） |
| `FCA.m` | MATLAB 浮点版 |
| `FCA_fixedpoint.py` | Python 定点版（整数运算） |
| `FCA_fixedpoint.m` | MATLAB 定点版 |
| `legacy/` | 旧版 MATLAB 实现（无 cross-sector / taper） |

## 算法特性

- **跨扇区 hue shift**：不在 60° 扇区边界 clamp，消除色相跳变
- **Soft taper**：hue range 边界处 deltaH 线性渐弱至 0，防止颜色逃逸和色相断层
- **近灰过滤** (`delta_eps`)：跳过低饱和度像素，避免 hue 噪声
- **deltaH 硬限制** (`max_abs_deltaH`)：防止过大偏移导致的色彩异常
- **Auto-clamp taper**：窄 hue range 时自动缩小 taper_width

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `deltaH` | — | 色相偏移量（度），正=顺时针 |
| `use_hue_range` | `false` | 是否只对指定 hue 范围生效 |
| `hue_min` / `hue_max` | 0 / 360 | 目标 hue 范围（支持 wrap-around） |
| `taper_width` | 0 | Taper 宽度（度），建议 ≥ 2×\|deltaH\| |
| `delta_eps` | 1/255 | 最小 Delta 阈值 |
| `max_abs_deltaH` | 15 | \|deltaH\| 硬上限（度） |
| `frac_bits` | 10 | 定点版精度（仅定点版） |

## 快速使用

### Python
```python
from FCA import hue_shift_v_s_strict_ycocg
import numpy as np
from PIL import Image

img = np.asarray(Image.open("photo.jpg")).astype(np.float64) / 255.0
out, dbg = hue_shift_v_s_strict_ycocg(img, deltaH=+10, use_hue_range=True,
    hue_min=60, hue_max=180, taper_width=20)
```

### MATLAB
```matlab
I = im2double(imread('photo.jpg'));
opts.useHueRange = true;
opts.hueMin = 60; opts.hueMax = 180;
opts.taperWidth = 20;
opts.deltaEps = 1/255; opts.maxAbsDeltaH = 15; opts.epsT = 1e-6;
[J, dbg] = hue_shift_cross_sector(I, +10, opts);
```

## 定点版精度

| FRAC_BITS | MAE (per channel) | MaxErr |
|-----------|-------------------|--------|
| 10 | < 0.05 | ≤ 1 |
| 12 | < 0.02 | ≤ 1 |
| 14 | < 0.01 | ≤ 1 |

## Legacy

`legacy/` 目录保存了原始实现（无 cross-sector / taper / delta_eps），仅供参考。
