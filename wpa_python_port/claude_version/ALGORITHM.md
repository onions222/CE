# WPA Algorithm Specification

White Point Adjustment (WPA) 算法文档 — 移动显示端色温调节

---

## 1. 概述

WPA 通过 **per-channel RGB gain**（等价于对角 3×3 矩阵）实现暖色/冷色调节。  
增益在 **linear-light 域** 应用，并通过 **12-bin 亮度分段** 控制不同亮度区间的调节强度，
以减少暗部偏色与高亮 clipping。

### 处理流程

```
Input (uint8 sRGB)
    │
    ├─ ① Degamma ──────────────────────→ linear [0,1]
    │
    ├─ ② Luma proxy Y ────────────────→ per-pixel Y
    │
    ├─ ③ 12-bin gain lookup ──────────→ per-pixel (Gr, Gg, Gb)
    │
    ├─ ④ WA_SEL blend ────────────────→ final gain
    │
    ├─ ⑤ Apply gain (linear domain) ─→ adjusted linear
    │
    ├─ ⑥ Saturation protection ───────→ blend back for saturated px
    │
    ├─ ⑦ Clip [0,1] ──────────────────→ safe range
    │
    ├─ ⑧ Engamma ─────────────────────→ sRGB [0,1]
    │
    └─ ⑨ Quantise ────────────────────→ uint8 output
```

---

## 2. WA_SEL → 插值系数

控制参数 `WA_SEL ∈ [0, 127]`，其中 64 为中性（无调节）：

$$
\text{if } WA\_SEL < 64: \quad \text{side} = \text{warm}, \quad \alpha = \frac{64 - WA\_SEL}{64}
$$

$$
\text{if } WA\_SEL = 64: \quad \text{identity}, \quad \alpha = 0
$$

$$
\text{if } WA\_SEL > 64: \quad \text{side} = \text{cool}, \quad \alpha = \frac{WA\_SEL - 64}{63}
$$

- `WA_SEL = 0` → 最暖 (α = 1.0)
- `WA_SEL = 127` → 最冷 (α = 1.0)

---

## 3. Gamma 变换

### 3.1 sRGB Degamma（编码域 → 线性域）

标准 IEC 61966-2-1 分段公式：

$$
L = \begin{cases}
\dfrac{V}{12.92} & \text{if } V \leq 0.04045 \\[8pt]
\left(\dfrac{V + 0.055}{1.055}\right)^{2.4} & \text{otherwise}
\end{cases}
$$

其中 $V = \text{pixel\_value} / 255$，$L$ 为线性值。

### 3.2 sRGB Engamma（线性域 → 编码域）

$$
V = \begin{cases}
12.92 \cdot L & \text{if } L \leq 0.0031308 \\[8pt]
1.055 \cdot L^{1/2.4} - 0.055 & \text{otherwise}
\end{cases}
$$

### 3.3 可选模式

| `gamma_mode` | Degamma | Engamma |
|---|---|---|
| `"srgb"` | 上述分段公式 | 上述分段公式 |
| `"power"` | $L = V^\gamma$ | $V = L^{1/\gamma}$（默认 $\gamma = 2.2$） |
| `"none"` | 恒等（不变换） | 恒等 |

### 3.4 LUT 加速

当 `use_gamma_lut = True` 且 `gamma_mode = "srgb"` 时：
- **Degamma**: 预计算 256 条目 LUT，输入 uint8 直接索引得到 float32 线性值
- **Engamma**: 预计算 4096 条目 LUT，线性值量化后索引得到 uint8 输出

---

## 4. 亮度代理 (Luma Proxy)

$$
Y = \frac{R + 2G + B}{4}
$$

- 对应硬件整数实现中的 `(R + 2*G + B) >> 2`
- 可在 gamma 域或 linear 域计算（由 `luma_domain` 配置）
- 用于 12-bin 增益表查表

---

## 5. 12-Bin 亮度分段增益

### 5.1 固定节点

12 个 8-bit 亮度采样点（单调递增）：

```
luma_nodes = [15, 31, 47, 63, 95, 127, 159, 191, 223, 239, 247, 255]
```

### 5.2 增益表结构

每个节点对应一组 RGB 增益三元组：

$$
\text{gains\_table} \in \mathbb{R}^{12 \times 3}
$$

分别有 `warm_gains_bins` 和 `cool_gains_bins` 两张表。

### 5.3 线性插值查表

对于像素亮度 $Y \in [0, 255]$，找到相邻节点 $n_i \leq Y < n_{i+1}$：

$$
t = \frac{Y - n_i}{n_{i+1} - n_i}
$$

$$
G(Y) = (1 - t) \cdot G[i] + t \cdot G[i+1]
$$

边界处理：
- $Y < n_0 \Rightarrow G(Y) = G[0]$（clamp 到首节点）
- $Y \geq n_{11} \Rightarrow G(Y) = G[11]$（clamp 到末节点）

当 `bin_interp = False` 时，不做插值，直接取 $G[i]$（floor snap）。

---

## 6. 默认增益表自动生成

### 6.1 全局增益端点

```
warm_gain_global = (1.40, 1.00, 0.60)   # R↑ G不变 B↓
cool_gain_global = (0.60, 1.00, 1.40)   # R↓ G不变 B↑
```

### 6.2 亮度衰减曲线

3 段折线 $\text{atten}(y)$ 控制不同亮度区间的增益强度：

$$
\text{atten}(y) = \begin{cases}
0.55 & y \leq 31 \\[4pt]
0.55 + 0.45 \cdot \dfrac{y - 31}{96} & 31 < y \leq 127 \\[4pt]
1.00 - 0.35 \cdot \dfrac{y - 127}{112} & 127 < y \leq 239 \\[4pt]
0.65 & y > 239
\end{cases}
$$

设计意图：
- **暗部** ($y \leq 31$)：$\text{atten} = 0.55$，减少暗部偏色
- **中间亮度** ($y = 127$)：$\text{atten} = 1.00$，最大调节强度
- **高亮** ($y \geq 239$)：$\text{atten} = 0.65$，避免高亮 clipping

### 6.3 逐节点增益生成

对每个节点 $y_i$：

$$
G_{\text{bin}}[i] = 1 + \text{atten}(y_i) \cdot (G_{\text{global}} - 1)
$$

例如 warm 在 $y = 127$（atten = 1.0）：

$$
G_{\text{warm}}[5] = 1 + 1.0 \times (1.40 - 1, \; 1.00 - 1, \; 0.60 - 1) = (1.40, \; 1.00, \; 0.60)
$$

在 $y = 31$（atten = 0.55）：

$$
G_{\text{warm}}[1] = 1 + 0.55 \times (0.40, \; 0, \; -0.40) = (1.22, \; 1.00, \; 0.78)
$$

---

## 7. 增益混合 (WA_SEL Blending)

用 α 在恒等增益 $(1,1,1)$ 和最大增益之间插值：

$$
G_{\text{final}}(Y) = 1 + \alpha \cdot \bigl(G_{\text{max}}(Y) - 1\bigr)
$$

其中 $G_{\text{max}}$ 根据 side 选择 `warm_gains_bins` 或 `cool_gains_bins`。

---

## 8. 线性域增益应用

$$
\text{out}_{\text{linear}} = \text{in}_{\text{linear}} \cdot G_{\text{final}}
$$

逐通道相乘（per-channel）。

---

## 9. 低饱和度保护 (Saturation Protection)

### 9.1 饱和度代理

$$
s = |R - G| + |G - B| + |B - R|
$$

轻量级计算，灰色时 $s = 0$，彩色时 $s$ 增大。

### 9.2 权重斜坡

$$
w = \text{clip}\!\left(\frac{s_1 - s}{s_1 - s_0}, \; 0, \; 1\right)
$$

| 条件 | $w$ | 语义 |
|---|---|---|
| $s \leq s_0$ | 1 | 灰色像素 → 全调节 |
| $s_0 < s < s_1$ | 线性过渡 | 部分调节 |
| $s \geq s_1$ | 0 | 高饱和 → 不调节 |

默认 $s_0 = 100$, $s_1 = 500$（gamma 域 0–255 尺度），默认关闭 (`sat_en = False`)。

### 9.3 加权混合

$$
\text{out} = \text{in}_{\text{linear}} + w \cdot (\text{adjusted}_{\text{linear}} - \text{in}_{\text{linear}})
$$

$w = 1$ 时完全应用增益，$w = 0$ 时保持原样。

---

## 10. 输出量化

1. Clip 到有效范围：$\text{out} = \text{clip}(\text{out}, 0, 1)$
2. Engamma：linear → sRGB
3. 量化：$\text{uint8} = \text{round}(\text{encoded} \times 255)$，再 clip 到 $[0, 255]$

所有内部计算使用 **float32**。

---

## 11. 参数汇总

| 参数 | 默认值 | 说明 |
|---|---|---|
| `wa_en` | `True` | 总开关 |
| `wa_sel` | `64` | 0=最暖, 64=无变化, 127=最冷 |
| `gamma_mode` | `"srgb"` | gamma 模式 |
| `gamma_power` | `2.2` | power 模式指数 |
| `use_gamma_lut` | `False` | 使用 LUT 加速 sRGB |
| `luma_nodes` | 12 节点 | 亮度采样点 |
| `bin_interp` | `True` | 节点间线性插值 |
| `luma_domain` | `"gamma"` | 计算 Y 所用域 |
| `sat_en` | `False` | 饱和度保护开关 |
| `sat_s0` | `100` | 灰色阈值 |
| `sat_s1` | `500` | 饱和阈值 |
| `warm_gains_bins` | 自动生成 | (12,3) 暖色增益表 |
| `cool_gains_bins` | 自动生成 | (12,3) 冷色增益表 |
