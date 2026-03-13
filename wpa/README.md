# WPA — White Point Adjustment

移动显示端色温调节 Python 实现。通过 per-channel RGB gain 实现暖色/冷色调节，
支持 12-bin 亮度分段增益与 sRGB gamma 处理。

当前仓库规则：

- 本目录是唯一活跃的 WPA 开发目录。
- 历史 MATLAB/Python 版本与对比脚本都在 `../archive/` 下，只读留档。

---

## Quick Start

### Python API

```python
import numpy as np
from PIL import Image
from wpa import wpa_process_rgb_uint8, WPAConfig

# 读取图片
img = np.array(Image.open("photo.jpg").convert("RGB"), dtype=np.uint8)

# 暖色调节 (wa_sel < 64, 越小越暖)
cfg = WPAConfig(wa_sel=10)
warm_out = wpa_process_rgb_uint8(img, cfg)

# 冷色调节 (wa_sel > 64, 越大越冷)
cfg = WPAConfig(wa_sel=120)
cool_out = wpa_process_rgb_uint8(img, cfg)

# 保存
Image.fromarray(warm_out).save("warm.jpg")
Image.fromarray(cool_out).save("cool.jpg")
```

### CLI

```bash
# 暖色
python -m wpa.cli --in photo.jpg --out warm.jpg --wa-sel 10

# 冷色
python -m wpa.cli --in photo.jpg --out cool.jpg --wa-sel 120

# 关闭调节 (identity)
python -m wpa.cli --in photo.jpg --out out.jpg --wa-sel 64
```

---

## 安装

**依赖：** numpy（必须）、Pillow（仅 CLI 需要）

```bash
pip install numpy Pillow
```

本项目为纯 Python 包，无需编译：

```bash
# 从项目目录直接使用
cd wpa
python -m wpa.cli --in input.jpg --out output.jpg --wa-sel 20

# 或安装到环境
pip install -e .
```

---

## 参数说明

### 核心控制

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `wa_en` | bool | `True` | 总开关。`False` 时输出 = 输入 |
| `wa_sel` | int | `64` | 色温选择器 (0–127) |

**`wa_sel` 语义：**

```
0 ──────── 32 ──────── 64 ──────── 95 ──────── 127
最暖        暖         无变化       冷          最冷
(α=1.0)    (α=0.5)    (α=0)      (α≈0.5)    (α=1.0)
```

### Gamma 模式

| 参数 | 默认 | 说明 |
|---|---|---|
| `gamma_mode` | `"srgb"` | `"srgb"` / `"power"` / `"none"` |
| `gamma_power` | `2.2` | `"power"` 模式的指数 |
| `use_gamma_lut` | `False` | 使用 256-entry LUT 加速 sRGB（更快但精度略低） |

### 亮度分段

| 参数 | 默认 | 说明 |
|---|---|---|
| `luma_nodes` | `[15,31,...,255]` | 12 个 8-bit 亮度采样点 |
| `bin_interp` | `True` | 节点间线性插值（`False` = floor snap） |
| `luma_domain` | `"gamma"` | 计算亮度代理的域 (`"gamma"` / `"linear"`) |

### 饱和度保护

| 参数 | 默认 | 说明 |
|---|---|---|
| `sat_en` | `False` | 饱和度保护开关 |
| `sat_s0` | `100` | 灰色阈值（s ≤ s0 → 全调节） |
| `sat_s1` | `500` | 饱和阈值（s ≥ s1 → 不调节） |
| `sat_weight_domain` | `"gamma"` | 计算饱和度权重的域 |

> **注意：** 饱和度保护默认关闭。对于高饱和度图像（如草地、天空），开启后会显著减弱调节效果。
> 适用场景：需要保护已有色彩不被偏移时（如产品色彩还原），可设 `sat_en=True`。

### 增益表

| 参数 | 默认 | 说明 |
|---|---|---|
| `warm_gains_bins` | 自动生成 | (12,3) ndarray，每个亮度节点的 RGB 暖色增益 |
| `cool_gains_bins` | 自动生成 | (12,3) ndarray，每个亮度节点的 RGB 冷色增益 |

默认端点由 CCT 自动计算（`3000K -> warm`, `6500K -> neutral`, `9300K -> cool`），
运行时仍使用 RGB gain（无 3x3 矩阵运算）。

用户可自定义：

```python
import numpy as np
from wpa import WPAConfig, wpa_process_rgb_uint8

my_warm = np.tile([1.20, 1.00, 0.80], (12, 1))  # 均匀增益
cfg = WPAConfig(wa_sel=0, warm_gains_bins=my_warm)
out = wpa_process_rgb_uint8(img, cfg)
```

---

## CLI 参数

```
python -m wpa.cli --help
```

| 参数 | 说明 |
|---|---|
| `--in PATH` | 输入图片路径 |
| `--out PATH` | 输出图片路径 |
| `--wa-sel N` | WA_SEL 值 (0–127, 默认 64) |
| `--wa-en {0,1}` | 总开关 (默认 1) |
| `--gamma {srgb,power,none}` | Gamma 模式 (默认 srgb) |
| `--gamma-lut` | 启用 LUT 加速 |
| `--no-sat` | 禁用饱和度保护 |

---

## 文件结构

```
wpa/
├── __init__.py      # 公共 API: WPAConfig, wpa_process_rgb_uint8
├── __main__.py      # python -m wpa 入口
├── cli.py           # 命令行接口 (argparse + Pillow)
├── config.py        # WPAConfig dataclass, 默认增益表生成
├── gamma.py         # sRGB/power/none degamma & engamma (公式 + LUT)
├── weights.py       # 亮度代理, 12-bin 插值, 饱和度权重
└── core.py          # 主处理管线
tests/
├── test_core.py     # 核心流程测试 (identity, gamma roundtrip, 方向性)
└── test_bins.py     # 12-bin 插值、增益形状、亮度代理测试
└── test_cct_gain.py # CCT 映射与 gain LUT 测试
ALGORITHM.md         # 详细算法文档 (含公式)
pyproject.toml       # 项目配置
```

---

## 运行测试

```bash
pip install pytest
python -m pytest tests/ -v
```

测试覆盖核心算法行为，以及视觉稳定性数据集相关的 profile、synthetic chart 和 manifest 生成逻辑。

## 测试数据集

验证数据集现在以视觉稳定性为主，而不是以 Kodak 真实图为主：

- `smoke_visual`: 最敏感的灰阶、中性、UI、近黑/近白图，用于快速回归
- `core_visual`: 日常默认 profile，覆盖 neutral-stability core、bin boundary、2D neutral gradient、skin/saturation side-effect set
- `release_visual`: 发版级 profile，包含 `core_visual`、精选 real-world sanity groups、少量 Kodak legacy baseline 和 JPEG ladder

构建命令：

```bash
python scripts/build_test_set.py --profile smoke_visual
python scripts/build_test_set.py --profile core_visual
python scripts/build_test_set.py --profile release_visual
```

`release_visual` 现在会额外拉取小规模真实图 sanity layer，包括 portrait、HDR window、night neon、workspace/UI-like scene 和 mixed-light research sample。

manifest 会输出 `dataset_role`、`inspection_priority`、`failure_modes`、`recommended_wa_sel`、`pass_hint` 和 `expected_observation`，便于按失真模式而不是按文件名做诊断。

---

## MATLAB 定点版

仓库现在包含一个与 `wpa_fixed` 对齐的 MATLAB 版本，代码位于 [matlab/README.md](matlab/README.md) 旁的 `matlab/` 目录。

典型流程：

```bash
python scripts/export_matlab_wpa_fixed_golden.py
```

然后在 MATLAB 中：

```matlab
addpath('matlab');
summary = validate_wpa_fixed_against_python('matlab/golden_cases');
```

单张图片入口：

```matlab
addpath('matlab');
run_wpa_fixed_image('input.png', 'output.png', 'wa_sel', 0, 'coeff_frac_bits', 8);
```

### Python fixed / MATLAB hw_runtime 的 low-luma gate

`wpa_fixed` 与 `matlab/hw_runtime` 现在都包含同一套 low-luma bypass/blend 逻辑，
用于抑制 `Q0.8` 下低亮 shoulder 区域的 colored ring。

问题背景：

- 当输入先进入线性域，再量化到较低 bit-depth 时，某些低亮偏蓝 shoulder 像素会落到极小的 raw code
- 例如 `(16, 18, 22)` 在 `frac_bits=8` 下会量化成近似 `[1, 2, 2]`
- warm gain 作用后，容易进一步落到近似 `[1, 2, 1]`
- 再 `engamma` 后会表现成明显的 green/cyan ring

当前解决方式不是在最终显示域修补，而是在线性域 raw code 上做 gate：

- `gate_luma_code = floor((R_code + 2*G_code + B_code) / 4)`
- `gate_luma_code <= bypass_code`
  - 直接使用原始 `pixel_code`
- `bypass_code < gate_luma_code < blend_end_code`
  - 在线性域对 `pixel_code` 和 `adjusted_code` 做整数 blend
- `gate_luma_code >= blend_end_code`
  - 维持原始 `adjusted_code`

默认门限会根据 `frac_bits` 自动从视觉参考值 `31/63` 映射到线性域 code：

- `frac_bits=8`
  - `bypass_code=4`
  - `blend_end_code=13`
- `frac_bits=10`
  - `bypass_code=14`
  - `blend_end_code=51`

这套 gate 目前在 Python fixed 与 MATLAB `hw_runtime` 上已经对齐。

---

## 算法文档

详细的数学公式和处理步骤见 [ALGORITHM.md](ALGORITHM.md)。
