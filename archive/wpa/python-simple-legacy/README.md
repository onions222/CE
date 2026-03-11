# WPA（White Point Adjustment）

这是一个白点调节（冷暖色调）参考实现，包含两种模式：

- `diag_rgb`：速度快，基于 RGB 对角增益与亮度分箱
- `kelvin_ycocg`：基于 Kelvin 白点变化与 YCoCg 变换

算法细节与数学公式请看：`docs/ALGORITHM.md`。

## Quickstart

## 1）安装

```bash
python -m pip install -e ".[cli,test]"
```

## 2）基础运行（默认 `diag_rgb`）

```bash
python -m wpa.cli --in grass.jpg --out out.png --wa-sel 127
```

说明：

- CLI 默认开启 WPA。
- 建议用 PNG 做效果评估，避免 JPEG 重压缩干扰。

## 3）使用 Kelvin 模式

```bash
python -m wpa.cli --in grass.jpg --out out_kelvin.png --wa-mode kelvin_ycocg --wa-sel 127
```

## 4）输出量化报告

```bash
python -m wpa.cli --in grass.jpg --out out.png --wa-sel 127 --report
```

## 5）运行测试

```bash
python -m pytest -q
```

## CLI 参数说明

以下参数均可通过 `python -m wpa.cli` 调整。

## 输入输出与开关

- `--in`：输入图片路径（必填）
- `--out`：输出图片路径（必填）
- `--wa-en`：开启 WPA（默认开启）
- `--no-wa-en`：关闭 WPA
- `--wa-sel`：范围 `[0,127]`，`64` 为恒等
  - `<64`：偏暖
  - `>64`：偏冷

## 模式选择

- `--wa-mode`：`diag_rgb` 或 `kelvin_ycocg`
  - 默认：`diag_rgb`

## 亮度域与 gamma

- `--luma-domain`：`gamma` 或 `linear`（亮度代理计算域）
- `--sat-domain`：`gamma` 或 `linear`（饱和度权重计算域）
- `--gamma-mode`：`srgb`、`power`、`none`
- `--use-gamma-lut`：启用 LUT gamma 转换
- `--power-gamma`：`--gamma-mode power` 时使用的 gamma 值

## 饱和度保护

- `--sat-s0`：低阈值
- `--sat-s1`：高阈值

行为：

- 低饱和像素更容易被冷暖调整影响
- 高饱和像素会被更多保护

## 冷暖侧强度（两种模式通用）

- `--warm-strength`：暖侧强度系数
- `--cool-strength`：冷侧强度系数

## Kelvin 模式专用参数

- `--kelvin-strength`：Kelvin 变换全局强度
- `--kelvin-warm-side-scale`：Kelvin 暖侧额外缩放
- `--kelvin-cool-side-scale`：Kelvin 冷侧额外缩放

Kelvin 模式有效侧向强度：

- warm：`warm_strength * kelvin_warm_side_scale`
- cool：`cool_strength * kelvin_cool_side_scale`

## 报告参数

- `--report`：输出本次处理的量化指标：
  - `mean_l1`
  - `p50/p90/p99`
  - `mean_delta_rgb`
  - `highlight_spike`

## 库级参数（`WPAConfig`）

如果你直接在 Python 中调用 `wpa_process_rgb_uint8`，可用 `WPAConfig` 调整以下参数：

- 核心控制：
  - `WA_EN`, `WA_SEL`, `wa_mode`
- 亮度分箱：
  - `luma_nodes_12`, `bin_interp`
- 域与 gamma：
  - `luma_domain`, `sat_weight_domain`, `gamma_mode`, `use_gamma_lut`, `power_gamma`
- 饱和度保护：
  - `sat_s0`, `sat_s1`
- `diag_rgb` 相关：
  - `warm_gain_global`, `cool_gain_global`
  - `warm_gains_bins`, `cool_gains_bins`
  - `warm_strength`, `cool_strength`
- `kelvin_ycocg` 相关：
  - `kelvin_warm_end`, `kelvin_cool_end`
  - `kelvin_w_shadow`, `kelvin_w_mid`, `kelvin_w_highlight`, `kelvin_bin_mid`
  - `kelvin_bright_minfac`, `kelvin_y_dark2`, `kelvin_y_dark1`, `kelvin_y_bright1`, `kelvin_y_bright2`
  - `kelvin_strength`, `kelvin_warm_side_scale`, `kelvin_cool_side_scale`

公式级说明见 `docs/ALGORITHM.md`。

