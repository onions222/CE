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
- 冷暖端点：`4500K ~ 9300K`
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

- `wa <= 64`
\[
T(wa)=6500-(64-wa)\cdot\frac{6500-4500}{64}
\]

- `wa > 64`
\[
T(wa)=6500+(wa-64)\cdot\frac{9300-6500}{63}
\]

锚点：
- `T(0)=4500K`
- `T(64)=6500K`
- `T(127)=9300K`

### 3.2 亮度代理
\[
Y = \frac{R+2G+B}{4}
\]
硬件对应 `(R + 2*G + B) >> 2`。

### 3.3 亮度衰减函数 `atten(Y)`
\[
atten(y)=
\begin{cases}
0.55, & y\le31 \\
0.55 + 0.45\cdot\frac{y-31}{96}, & 31<y\le127 \\
1.00 - 0.35\cdot\frac{y-127}{112}, & 127<y\le239 \\
0.65, & y>239
\end{cases}
\]

### 3.4 base gain 与 per-bin gain

`G_base` 为 RGB 三维向量。

定点主路径（3 锚点）中，先由 `WA_SEL` 分段插值得到 `G_base(wa)`：

- `wa <= 64`
\[
G_{base}(wa)=G_{warm}+\frac{wa}{64}\cdot\left(G_{neutral}-G_{warm}\right)
\]

- `wa > 64`
\[
G_{base}(wa)=G_{neutral}+\frac{wa-64}{64}\cdot\left(G_{cool}-G_{neutral}\right)
\]

再由衰减得到当前 WA 的 12-bin 增益：
\[
G_{bin}[i]=1+atten(y_i)\cdot\left(G_{base}-1\right)
\]

---

## 4. 浮点参考路径（`wpa`）

### 4.1 处理流程
1. Degamma（sRGB -> linear）
2. 计算 `Y`
3. 用 `Y` 在 `warm_gains_bins/cool_gains_bins` 做 12-bin 插值，得到 `G_max(Y)`
4. 用 `WA_SEL` 得到 `alpha` 并混合：
\[
G_{final}(Y)=1+\alpha\cdot(G_{max}(Y)-1)
\]
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

### 5.3 WA 变化时更新（非每像素）
1. `WA_SEL` 分段线性插值得 `base gain`（warm `/64`，cool `/64`）
2. 用 `atten_q_lut_fixed` 生成当前 `runtime_bin_gain[12][3]`

### 5.4 每像素路径
1. Degamma（允许浮点）
2. 计算 `Y`
3. 用 `Y` 在当前 `runtime_bin_gain[12][3]` 插值得 `gain`
4. 定点乘法与舍入：
\[
out_{fix}=\frac{pixel_{fix}\cdot gain_{fix}+2^{F-1}}{2^F}
\]
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
| 0 | 4500.0 | [1.249, 0.963, 0.632] | [1.150, 0.978, 0.779] | [138,147,140] |
| 32 | 5500.0 | [1.097, 0.988, 0.834] | [1.058, 0.993, 0.900] | [127,149,162] |
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

运行时 scratch（WA 更新时的 `12x3`）：
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

### 7.3 每次 WA 更新操作（非每像素）
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
- WA 更新时生成当前 `12x3`
以降低常驻资源并保持可接受误差。
