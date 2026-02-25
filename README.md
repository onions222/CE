# CE — Color Engineering Toolkit

图像色彩工程算法集，包含 **WPA（白点调节）** 和 **FCA（伪彩校正 / 色相偏移）** 两大模块，提供 MATLAB、Python 浮点及定点实现。

## 目录结构

```
CE/
├── wpa/                  # WPA MATLAB 原版（YCoCg + LMS，无 banding）
├── wpa_python_port/      # WPA Python 移植（从 MATLAB 直接翻译）
├── wpa_simple_version/   # WPA 精简重构版（含 CLI、pytest、文档）
├── fca/                  # FCA 色相偏移（浮点 & 定点，Python + MATLAB）
└── compare/              # WPA 新旧版本对比脚本
```

## 模块概览

### WPA — White Point Adjustment（白点调节 / 冷暖色调）

对图像进行冷暖色调调整，核心流程：

1. 亮度分箱 → 按 bin 插值增益
2. 饱和度保护 → 高饱和像素减弱调整
3. 亮度保护 → 暗部 / 高光平滑衰减
4. RGB headroom 溢出限制

**两种模式：**

| 模式 | 说明 |
|------|------|
| `diag_rgb` | RGB 对角增益 + 亮度分箱，速度快 |
| `kelvin_ycocg` | 基于 Kelvin 白点变化 + YCoCg 色彩空间变换 |

**快速使用（`wpa_simple_version`）：**

```bash
cd wpa_simple_version
python -m pip install -e ".[cli,test]"
python -m wpa.cli --in sky.jpg --out out.png --wa-sel 127
```

详细参数请参阅 [`wpa_simple_version/README.md`](wpa_simple_version/README.md)。

---

### FCA — False Color Adjustment（色相偏移）

基于 HSV 6 扇区模型的色相调整算法，**严格保持亮度 (V) 和饱和度 (S) 不变**。

**核心特性：**

- 跨扇区 hue shift — 不在 60° 扇区边界 clamp
- Soft taper — hue range 边界处 deltaH 线性渐弱
- 近灰过滤 — 跳过低饱和度像素
- deltaH 硬限制 — 防止过大偏移

**快速使用：**

```python
from FCA import hue_shift_v_s_strict_ycocg
import numpy as np
from PIL import Image

img = np.asarray(Image.open("photo.jpg")).astype(np.float64) / 255.0
out, dbg = hue_shift_v_s_strict_ycocg(img, deltaH=+10,
    use_hue_range=True, hue_min=60, hue_max=180, taper_width=20)
```

详细参数请参阅 [`fca/README.md`](fca/README.md)。

## 依赖

- **Python** ≥ 3.8
- `numpy`、`Pillow`
- MATLAB（可选，用于运行 `.m` 文件）

## License

Internal use only.
