# WPA 算法细节说明

本文档描述本仓库 White Point Adjustment（WPA）的完整处理流程、两种模式以及关键数学公式。

支持两种处理模式：

- `diag_rgb`：基于 RGB 对角增益（12 个亮度分箱）
- `kelvin_ycocg`：基于 Kelvin 的 YCoCg 域变换

## 1. 输入/输出与基本语义

- 输入：`uint8`，形状 `(H, W, 3)`，gamma 域 RGB
- 输出：同尺寸同类型 `uint8` RGB
- 旁路条件：
  - `WA_EN == False`，或
  - `WA_SEL == 64`

`WA_SEL` 到冷暖方向的映射：

- `WA_SEL < 64`：暖色（warm）
- `WA_SEL > 64`：冷色（cool）
- `WA_SEL == 64`：恒等（identity）

插值因子：

- 暖侧：`alpha = (64 - WA_SEL) / 64`
- 冷侧：`alpha = (WA_SEL - 64) / 63`

## 2. 颜色域转换

归一化输入：

`x_gamma = x_u8 / 255`

转换到线性域：

`x_linear = to_linear(x_gamma, gamma_mode)`

最后转换回输出：

`y_gamma = from_linear(y_linear, gamma_mode)`

`y_u8 = round(clip(y_gamma, 0, 1) * 255)`

## 3. 亮度代理与饱和度权重

亮度代理（根据配置在 gamma 或 linear 域计算）：

`Y = (R + 2G + B) / 4`

`Y8 = clip(255 * Y, 0, 255)`

饱和度代理：

`s = |R - G| + |G - B| + |B - R|`

分段线性权重：

- `s <= s0` 时：`w = 1`
- `s >= s1` 时：`w = 0`
- `s0 < s < s1` 时：`w = (s1 - s) / (s1 - s0)`

两种模式统一的最终混合：

`out_linear = in_linear + w * (gained_linear - in_linear)`

## 4. 模式 A：`diag_rgb`（12 分箱 RGB 增益）

### 4.1 12 分箱插值

给定固定亮度节点 `nodes[12]`，以及 warm/cool 的每箱增益：

- `G_warm[12, 3]`
- `G_cool[12, 3]`

按 `Y8` 线性插值获得 `Gmax(Y8)`。

### 4.2 按 `WA_SEL` 缩放

- 暖侧：`G = 1 + alpha * (G_warm(Y8) - 1)`
- 冷侧：`G = 1 + alpha * (G_cool(Y8) - 1)`

线性域逐像素增益：

`gained_linear = in_linear * G`

### 4.3 高亮安全与平滑回滚

对高亮区域（`Y8 >= 223`）：

- 增益限制到 `<= 1`（单向保护）
- 侧向通道设最小增益下限 `192/255`，避免过度压暗导致尖峰
- 在接近白点处平滑回滚：

`t = clip((Y8 - 223) / (255 - 223), 0, 1)`

`G <- G + (t * rolloff) * (1 - G)`

## 5. 模式 B：`kelvin_ycocg`（Kelvin + Bradford）

### 5.1 RGB 与 YCoCg 互转

线性域 `[0,255]` 下：

- `Y  = 0.25R + 0.50G + 0.25B`
- `Co = 0.50(R - B)`
- `Cg = -0.25R + 0.50G - 0.25B`

逆变换：

- `R = Y + Co - Cg`
- `G = Y + Cg`
- `B = Y - Co - Cg`

### 5.2 12 分箱 Kelvin 端点矩阵

每个分箱会构造目标白点并计算色适应矩阵：

1. 由 CCT 计算目标白点
2. 转到 XYZ
3. 用 Bradford（LMS）做色适应：

`A = M_L2X * diag(LMS_target / LMS_base) * M_X2L`

4. 转到 YCoCg 域：

`T = M_R2Y * M_X2R * A * M_R2X * M_Y2R`

5. 得到增量矩阵：

`D = T - I`

最终得到：

- `D_warm[12,3,3]`
- `D_cool[12,3,3]`

### 5.3 像素级插值与应用

按亮度分箱插值：

`Delta = lerp(D_lo @ y0, D_hi @ y0, t_bin)`

强度系数：

`s = alpha * strength * kelvin_strength * w_bin * brightness_factor`

随后会基于 RGB 头间距限制 `s`，确保结果仍在 `[0, rgb_max]`。

应用：

`y_out = y0 + s * Delta`

### 5.4 Kelvin 高亮处理

高亮像素（`Y8 >= 223`）：

- 在 RGB-delta 空间施加单向限制，抑制高亮色偏尖峰

转换回 RGB 后，再做高亮平滑回滚（向白点靠拢）：

- 回滚起点：`Y8 >= 191`
- `t = clip((Y8 - 191) / (255 - 191), 0, 1)`
- `rgb <- rgb + (t * rolloff) * (rgb_max - rgb)`

## 6. 强度参数与有效强度

对 `diag_rgb`：

- warm 使用 `warm_strength`
- cool 使用 `cool_strength`

对 `kelvin_ycocg`：

`effective_strength = side_strength * kelvin_side_scale`

其中：

- warm 侧缩放：`kelvin_warm_side_scale`
- cool 侧缩放：`kelvin_cool_side_scale`

且总强度还会乘 `kelvin_strength`。

## 7. `--report` 指标定义

CLI `--report` 输出：

- `mean_l1`：每像素 `|dR| + |dG| + |dB|` 的均值
- `p50/p90/p99`：上述每像素 L1 的分位数
- `mean_delta_rgb`：`[mean(dR), mean(dG), mean(dB)]`
- `highlight_spike`：灰阶高亮跳变指标
  - 构造灰阶 patch：`Y = [0, 32, ..., 255]`
  - 计算每个 patch 的 `RG = mean(R-G)`、`GB = mean(G-B)`
  - 在高亮区（`Y >= 223`）计算：
    - `max(max|diff(RG)|, max|diff(GB)|)`

`highlight_spike` 越小，表示高亮颜色过渡越平滑。

