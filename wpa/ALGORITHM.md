# WPA Algorithm Specification (Final Delivery)

White Point Adjustment (WPA) 算法文档（面向算法工程师与数字 IC 设计工程师）。

本文档以当前工程实现为准，覆盖：
- CCT 驱动的增益建模（离线）
- 浮点参考路径（`wpa`）
- 定点硬件仿真路径（`wpa_fixed`，交付主路径）
- 资源消耗、位宽、验证结论

---

## 1. 目标与边界

### 1.1 目标
- 输入控制量：`WA_SEL ∈ [0,127]`
- 中性点：`WA_SEL=64 -> 6500K (D65)`
- 冷暖端点：`3000K ~ 9300K`
- 运行时使用 RGB 对角增益（不使用 3x3 色适应矩阵）
- 保留 12-bin 亮度衰减机制，减少暗部偏色和高亮 clipping

### 1.2 非目标
- 不做全色域高精度色适应（运行时不引入矩阵）
- 不引入复杂 ISP 级多通道色彩管理

---

## 2. 架构总览

### 2.1 两阶段
1. 离线阶段（高精度）
- `WA_SEL -> CCT -> xy -> linear RGB gain` 建模
- 生成参考 `cct_gain_lut[128][3]`

2. 运行时阶段（轻量）
- 当前交付主路径（`wpa_fixed`）：
  - 仅常驻 3 个锚点 gain（warm/neutral/cool）
  - `WA_SEL` 变化时分段线性插值出 `base gain`
  - 结合 `atten` LUT 生成当前 `12x3`
  - 每像素仅做亮度插值 + 定点乘移位

### 2.2 两套实现角色
- `wpa`（浮点）：算法参考与可视化验证路径
- `wpa_fixed`（定点）：硬件仿真与资源评估路径（交付主路径）

---

## 3. 数学定义

### 3.1 WA_SEL 到 CCT

分段线性：

- 当 `wa <= 64` 时：

$$
T(wa)=6500-(64-wa)\cdot\frac{6500-3000}{64}
$$

- 当 `wa > 64` 时：

$$
T(wa)=6500+(wa-64)\cdot\frac{9300-6500}{63}
$$

锚点：
- `T(0)=3000K`
- `T(64)=6500K`
- `T(127)=9300K`

### 3.1.1 CCT 分段平滑参数化（实现约束）

`cct_to_xy_approx` 的分段点使用参数化定义，而不是写死常数：
- `T_split`：分段中心（默认 `4000K`）
- `delta_t`：平滑半窗（默认 `100K`）
- `blend_lo = T_split - delta_t`
- `blend_hi = T_split + delta_t`

在 `[blend_lo, blend_hi]` 区间使用 `smoothstep` 融合低温/高温两段公式，避免分段处出现 LUT 折点。后续若要调整分段中心，只需改参数，不改文档公式结构。

### 3.2 亮度代理

$$
Y = \frac{R+2G+B}{4}
$$

硬件对应 `(R + 2*G + B) >> 2`。

### 3.3 亮度衰减函数 `atten(Y)`

$$
atten(y)=
\begin{cases}
0.55, & y\le31 \\
0.55 + 0.45\cdot\frac{y-31}{96}, & 31<y\le127 \\
1.00 - 0.65\cdot\frac{y-127}{112}, & 127<y\le239 \\
0.35, & y>239
\end{cases}
$$

### 3.4 base gain 与 per-bin gain

`G_base` 为 RGB 三维向量。

定点主路径（3 锚点）中，先由 `WA_SEL` 分段插值得到 `G_base(wa)`：

- 当 `wa <= 64` 时：

$$
G_{base}(wa)=G_{warm}+\frac{wa}{64}\cdot\left(G_{neutral}-G_{warm}\right)
$$

- 当 `wa > 64` 时：

$$
G_{base}(wa)=G_{neutral}+\frac{wa-64}{64}\cdot\left(G_{cool}-G_{neutral}\right)
$$

再由衰减得到当前 `WA_SEL` 的 12-bin 增益：

$$
G_{bin}[i]=1+atten(y_i)\cdot\left(G_{base}-1\right)
$$

---

## 4. 浮点参考路径（`wpa`）

### 4.1 处理流程
1. Degamma（sRGB -> linear）
2. 计算 `Y`
3. 用 `Y` 在 `warm_gains_bins/cool_gains_bins` 做 12-bin 插值，得到 `G_max(Y)`
4. 用 `WA_SEL` 得到 `alpha` 并混合，计算公式为：

$$
G_{final}(Y)=1+\alpha\cdot(G_{max}(Y)-1)
$$

5. 线性域乘法 `RGB_out_linear = RGB_in_linear * G_final`
6. 可选饱和度保护
7. Engamma + 量化到 `uint8`

### 4.2 WA_SEL 到 alpha（浮点参考）
- `wa < 64`: `alpha=(64-wa)/64`
- `wa = 64`: `alpha=0`
- `wa > 64`: `alpha=(wa-64)/63`

---

## 5. 定点硬件仿真路径（`wpa_fixed`，主交付）

### 5.1 数据格式
- 系数格式：
  - 默认 `UQ1.8`（`coeff_frac_bits=8`，编码 `round(x*256)`）
  - 可选 `UQ1.10`（`coeff_frac_bits=10`，编码 `round(x*1024)`）
- 线性像素内部格式：`Q0.frac_bits`（默认 `frac_bits=10`）

### 5.2 常驻数据
- `wa_base_gain_lut_fixed[3][3]`：`[warm, neutral, cool]`
- `atten_q_lut_fixed[12]`
- `luma_nodes[12]`

### 5.3 `WA_SEL` 变化时更新（非每像素）
`WA_SEL` 更新只发生在控制量变化时，不在每像素路径内重复执行。实现对应
`wpa_fixed/config.py` 中的 `runtime_base_gain_fixed()` 与
`runtime_bin_gains_fixed()`。

#### 5.3.1 定点常量与符号

以下以默认 `UQ1.8` 为例：

- `F = coeff_frac_bits = 8`
- `COEFF_ONE = 1 << F = 256`
- `COEFF_HALF = 1 << (F-1) = 128`
- `G_warm_fix = [436, 221, 128]`
- `G_neutral_fix = [256, 256, 256]`
- `G_cool_fix = [219, 259, 339]`
- `atten_q_lut_fixed = [141, 141, 160, 179, 218, 256, 230, 205, 179, 166, 166, 166]`
- `luma_nodes = [15, 31, 47, 63, 95, 127, 159, 191, 223, 239, 247, 255]`

其中：
- `G_*_fix` 为 RGB 三通道 anchor gain 的定点编码
- `atten_q_lut_fixed[i] = round(atten(luma_nodes[i]) * 256)`
- `256` 表示实数 `1.0`

#### 5.3.2 `WA_SEL` 更新总流程

每次 `WA_SEL` 改变时，按以下步骤重建当前 `runtime_bin_gain[12][3]`：

```mermaid
flowchart TD
  A[输入 WA_SEL] --> B[Clamp 到 0..127]
  B --> C{wa <= 64 ?}
  C -->|Yes| D[Warm 分支: warm 到 neutral 插值]
  C -->|No| E[Cool 分支: neutral 到 cool 插值]
  D --> F[得到 runtime_base_gain_fixed]
  E --> F
  F --> G[计算 delta_base = base - COEFF_ONE]
  G --> H[对 12 个 luma bin 逐项乘 atten_q]
  H --> I[舍入右移 F 位]
  I --> J[得到 runtime_bin_gain[12][3]]
```

对应数学形式如下。

1. 输入裁剪：

$$
wa=\min(\max(WA\_SEL,0),127)
$$

2. 若 `wa <= 64`，走 warm 分支：

$$
num=wa
$$

$$
\Delta_{warm}=G_{neutral\_fix}-G_{warm\_fix}
$$

$$
G_{base\_fix}=G_{warm\_fix}+\left(\left(num\cdot\Delta_{warm}\right)+32\right)\gg 6
$$

3. 若 `wa > 64`，走 cool 分支：

$$
num=wa-64
$$

$$
\Delta_{cool}=G_{cool\_fix}-G_{neutral\_fix}
$$

$$
G_{base\_fix}=G_{neutral\_fix}+\left(\left(num\cdot\Delta_{cool}\right)+32\right)\gg 6
$$

4. 得到当前 `WA_SEL` 对应的 base gain 后，对 12 个亮度 bin 展开：

$$
\Delta_{base}=G_{base\_fix}-COEFF\_ONE
$$

$$
G_{bin\_fix}[i]=COEFF\_ONE+\left(atten\_q[i]\cdot\Delta_{base}+COEFF\_HALF\right)\gg F
$$

其中 `i=0...11`，每个 `G_bin_fix[i]` 都是三通道向量 `[R,G,B]`。

5. 特殊点：
- 当 `wa = 64` 时，`G_base_fix = [256,256,256]`
- 此时 `Delta_base = [0,0,0]`
- 因而全部 `runtime_bin_gain[12][3] = [256,256,256]`，即 identity

#### 5.3.3 Warm 分支逐步示例（`WA_SEL = 32`）

`WA_SEL=32` 位于 warm 半区，因此使用 `warm -> neutral` 插值。

1. 输入裁剪：
- `wa = 32`

2. 计算分段插值参数：
- `num = 32`
- `Delta_warm = G_neutral_fix - G_warm_fix`
- `Delta_warm = [256,256,256] - [436,221,128] = [-180,35,128]`

3. 计算 base gain 增量：
- `((num * Delta_warm) + 32) >> 6`
- `((32 * [-180,35,128]) + 32) >> 6`
- `([-5760,1120,4096] + 32) >> 6`
- `[-90,18,64]`

4. 得到当前 `WA_SEL` 的 base gain：
- `G_base_fix = G_warm_fix + [-90,18,64]`
- `G_base_fix = [436,221,128] + [-90,18,64]`
- `G_base_fix = [346,239,192]`
- 对应浮点约为 `[1.3516, 0.9336, 0.7500]`

5. 计算相对 `1.0` 的偏移：
- `Delta_base = G_base_fix - 256 = [90,-17,-64]`

6. 对 12 个 bin 全部展开：

| bin i | luma node | atten_q | `runtime_bin_gain[i]` (UQ1.8) | 浮点近似 |
|---:|---:|---:|---|---|
| 0 | 15 | 141 | [306, 247, 221] | [1.1953, 0.9648, 0.8633] |
| 1 | 31 | 141 | [306, 247, 221] | [1.1953, 0.9648, 0.8633] |
| 2 | 47 | 160 | [312, 245, 216] | [1.2188, 0.9570, 0.8438] |
| 3 | 63 | 179 | [319, 244, 211] | [1.2461, 0.9531, 0.8242] |
| 4 | 95 | 218 | [333, 242, 202] | [1.3008, 0.9453, 0.7891] |
| 5 | 127 | 256 | [346, 239, 192] | [1.3516, 0.9336, 0.7500] |
| 6 | 159 | 230 | [337, 241, 199] | [1.3164, 0.9414, 0.7773] |
| 7 | 191 | 205 | [328, 242, 205] | [1.2812, 0.9453, 0.8008] |
| 8 | 223 | 179 | [319, 244, 211] | [1.2461, 0.9531, 0.8242] |
| 9 | 239 | 166 | [314, 245, 215] | [1.2266, 0.9570, 0.8398] |
| 10 | 247 | 166 | [314, 245, 215] | [1.2266, 0.9570, 0.8398] |
| 11 | 255 | 166 | [314, 245, 215] | [1.2266, 0.9570, 0.8398] |

7. 以 `bin 0` 为例验证展开公式：

$$
G_{bin\_fix}[0]=256+\left([141]\cdot[90,-17,-64]+128\right)\gg 8
$$

$$
=256+\left([12690,-2397,-9024]+128\right)\gg 8
$$

$$
=256+[50,-9,-35]=[306,247,221]
$$

结论：
- 暖端时 `R` 增益大于 `1.0`
- `B` 增益小于 `1.0`
- 中亮度附近（`atten=1.0`）作用最强，暗部和高亮部衰减

#### 5.3.4 Cool 分支逐步示例（`WA_SEL = 96`）

`WA_SEL=96` 位于 cool 半区，因此使用 `neutral -> cool` 插值。

1. 输入裁剪：
- `wa = 96`

2. 计算分段插值参数：
- `num = wa - 64 = 32`
- `Delta_cool = G_cool_fix - G_neutral_fix`
- `Delta_cool = [219,259,339] - [256,256,256] = [-37,3,83]`

3. 计算 base gain 增量：
- `((num * Delta_cool) + 32) >> 6`
- `((32 * [-37,3,83]) + 32) >> 6`
- `([-1184,96,2656] + 32) >> 6`
- `[-18,2,42]`

4. 得到当前 `WA_SEL` 的 base gain：
- `G_base_fix = G_neutral_fix + [-18,2,42]`
- `G_base_fix = [256,256,256] + [-18,2,42]`
- `G_base_fix = [238,258,298]`
- 对应浮点约为 `[0.9297, 1.0078, 1.1641]`

5. 计算相对 `1.0` 的偏移：
- `Delta_base = G_base_fix - 256 = [-18,2,42]`

6. 对 12 个 bin 全部展开：

| bin i | luma node | atten_q | `runtime_bin_gain[i]` (UQ1.8) | 浮点近似 |
|---:|---:|---:|---|---|
| 0 | 15 | 141 | [246, 257, 279] | [0.9609, 1.0039, 1.0898] |
| 1 | 31 | 141 | [246, 257, 279] | [0.9609, 1.0039, 1.0898] |
| 2 | 47 | 160 | [245, 257, 282] | [0.9570, 1.0039, 1.1016] |
| 3 | 63 | 179 | [243, 257, 285] | [0.9492, 1.0039, 1.1133] |
| 4 | 95 | 218 | [241, 258, 292] | [0.9414, 1.0078, 1.1406] |
| 5 | 127 | 256 | [238, 258, 298] | [0.9297, 1.0078, 1.1641] |
| 6 | 159 | 230 | [240, 258, 294] | [0.9375, 1.0078, 1.1484] |
| 7 | 191 | 205 | [242, 258, 290] | [0.9453, 1.0078, 1.1328] |
| 8 | 223 | 179 | [243, 257, 285] | [0.9492, 1.0039, 1.1133] |
| 9 | 239 | 166 | [244, 257, 283] | [0.9531, 1.0039, 1.1055] |
| 10 | 247 | 166 | [244, 257, 283] | [0.9531, 1.0039, 1.1055] |
| 11 | 255 | 166 | [244, 257, 283] | [0.9531, 1.0039, 1.1055] |

7. 以 `bin 0` 为例验证展开公式：

$$
G_{bin\_fix}[0]=256+\left([141]\cdot[-18,2,42]+128\right)\gg 8
$$

$$
=256+\left([-2538,282,5922]+128\right)\gg 8
$$

$$
=256+[-10,1,23]=[246,257,279]
$$

结论：
- 冷端时 `R` 增益低于 `1.0`
- `B` 增益高于 `1.0`
- 仍然由同一套 `atten_q_lut_fixed` 控制强度随亮度变化

### 5.4 每像素路径
1. Degamma（允许浮点）
2. 计算 `Y`
3. 用 `Y` 在当前 `runtime_bin_gain[12][3]` 插值得 `gain`
4. 定点乘法与舍入公式为：

$$
out_{fix}=\frac{pixel_{fix}\cdot gain_{fix}+2^{F-1}}{2^F}
$$

其中 `F=coeff_frac_bits`。
5. 可选饱和度保护（定点）
6. Clip，Engamma，`uint8` 量化

---

## 6. 计算样例（参考链路）

统一示例：
- `RGB_in=[120,150,180]`
- `Y=150`
- `atten_demo=0.6`（演示固定值）

| WA_SEL | CCT(K) | g_base=[R,G,B] | g(Y)=1+0.6*(g_base-1) | RGB_out |
|---:|---:|---|---|---|
| 0 | 3000.0 | [1.705, 0.865, 0.500] | [1.423, 0.919, 0.700] | [171,138,126] |
| 32 | 4750.0 | [1.204, 0.971, 0.686] | [1.122, 0.983, 0.812] | [135,147,146] |
| 64 | 6500.0 | [1.000, 1.000, 1.000] | [1.000, 1.000, 1.000] | [120,150,180] |
| 96 | 7922.2 | [0.912, 1.007, 1.186] | [0.947, 1.004, 1.112] | [114,151,200] |
| 127 | 9300.0 | [0.857, 1.010, 1.324] | [0.914, 1.006, 1.194] | [110,151,215] |

---

## 7. 硬件资源与操作量（`frac_bits=10`）

### 7.1 静态资源

| 配置 | 系数位宽 | base LUT | atten LUT | luma_nodes | 总静态 |
|---|---:|---:|---:|---:|---:|
| UQ1.8 | 9 bit | 11 B | 14 B | 12 B | 36 B |
| UQ1.10 | 11 bit | 13 B | 17 B | 12 B | 41 B |

运行时 scratch（`WA_SEL` 更新时的 `12x3`）：
- UQ1.8: `41 B`
- UQ1.10: `50 B`

### 7.2 每像素操作（sat off）
- 比较 11
- 加法 13
- 减法 5
- 乘法 6
- 除法 0
- 移位 8
- clip 1

### 7.3 每次 `WA_SEL` 更新操作（非每像素）
- `wa_interp_mul/add/shift`: `3/3/3`
- `bin_expand_mul/add/shift`: `36/36/36`

---

## 8. 线性插值合理性验证

验证对象：
- 候选：3 锚点分段线性插值 `runtime_base_gain_fixed`
- 参考：`build_cct_gain_lut()` 128 档浮点基准

结果：
- `coeff_frac_bits=8`：`MAE=0.010408`, `MaxAbs=0.028228`
- `coeff_frac_bits=10`：`MAE=0.010742`, `MaxAbs=0.027910`

结论：
- 当前模型下，3 锚点线性插值约 `~1%` 平均误差、`~2.8%` 最坏误差，可作为低资源实现依据。

验证文件：
- `validation/linear_interp_basis/evaluate_anchor_interp.py`
- `validation/linear_interp_basis/anchor_interp_results.md`

---

## 9. 验证要求（交付门槛）

### 9.1 数值一致性
- `WA_SEL=64` 必须 identity（<=1 LSB）
- `WA_SEL` 从暖到冷时，`B/R` 趋势单调合理
- 定点输出无溢出、clip 行为可解释

### 9.2 图像观感
- 灰阶：无明显断层
- 天空/肤色/草地：冷暖方向正确，无异常偏色
- 高亮：无异常剪切与色漂

### 9.3 资源符合性
- 资源统计应与 `python -m wpa_fixed.hw_stats ...` 输出一致

---

## 10. 代码映射（便于联调）

- 浮点路径：`wpa/core.py`, `wpa/config.py`, `wpa/weights.py`
- 定点路径：`wpa_fixed/core.py`, `wpa_fixed/config.py`, `wpa_fixed/weights.py`
- 资源统计：`wpa_fixed/hw_stats.py`
- 插值验证：`validation/linear_interp_basis/`

---

## 11. 历史方案说明

历史版本曾使用 `128 x 12 x 3` 全表常驻方案。当前交付版本已切换为：
- 常驻 3 锚点 + 常驻 12 点 atten LUT
- `WA_SEL` 更新时生成当前 `12x3`
以降低常驻资源并保持可接受误差。
