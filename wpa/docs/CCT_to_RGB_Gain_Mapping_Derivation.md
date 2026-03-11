# Correlated Color Temperature to RGB Gain Mapping
## A Matrix-Free Runtime Method for White Point Adjustment in Display Pipelines

## Abstract

本文给出一套面向显示驱动白点调节（White Point Adjustment, WPA）的完整数学框架，用于将用户控制量（`WA_SEL`）映射为可直接用于硬件流水线的 `RGB gain`。核心目标是：

1. 使色温调节具备明确的色度学依据，而非经验系数；
2. 运行时仅使用查表、定点乘加和分段插值，避免复杂矩阵运算；
3. 与现有 `12-bin` 亮度分段增益架构兼容。

该方法采用“离线高精度建模 + 运行时轻量执行”策略：离线阶段完成 `CCT -> xy -> XYZ -> linear RGB gain` 的推导与 LUT 生成；运行时阶段仅执行 LUT 读取与 per-channel 增益应用。

---

## 1. Problem Formulation

设输入图像在某一工作域（通常为线性域）中的像素为

$$
\mathbf{p}_{in} = [R, G, B]^T.
$$

白点调节目标是对每个像素应用逐通道增益：

$$
\mathbf{p}_{out} = \operatorname{diag}(g_R, g_G, g_B)\,\mathbf{p}_{in},
$$

其中 $g_R,g_G,g_B$ 由目标色温决定。

控制量定义为 `WA_SEL \in [0,127]`，锚点约束为：

- `WA_SEL=0` 对应暖端 $T_w=3000\,\text{K}$；
- `WA_SEL=64` 对应中性 $T_n=6500\,\text{K}$（D65）；
- `WA_SEL=127` 对应冷端 $T_c=9300\,\text{K}$。

问题可归纳为：构造映射

$$
WA\_SEL \longmapsto \mathbf{g}(WA\_SEL)=[g_R,g_G,g_B],
$$

并保证：

1. 单调性（从暖到冷，蓝红相对强度整体上升）；
2. 中性点恒等（$WA\_SEL=64 \Rightarrow \mathbf{g}=[1,1,1]$）；
3. 数值稳定与硬件可实现性。

---

## 2. WA_SEL to CCT Mapping

采用分段线性映射：

$$
T(w)=
\begin{cases}
T_n - \dfrac{64-w}{64}(T_n-T_w), & w\le 64 \\
T_n + \dfrac{w-64}{63}(T_c-T_n), & w>64
\end{cases}
$$

其中 $w=WA\_SEL$。

该映射满足三个锚点精确命中：

$$
T(0)=T_w,
\quad T(64)=T_n,
\quad T(127)=T_c.
$$

优点：

- 易于寄存器/固件实现；
- 便于后续替换为感知均匀映射（例如对数域或 CIELAB 近似均匀映射）。

---

## 3. CCT to CIE 1931 Chromaticity (x, y)

使用标准经验近似将 CCT 映射到 CIE 1931 色度坐标。定义 $T$ 为 Kelvin。

### 3.1 Compute x(T)

当 $1667\le T \le 4000$：

$$
x = -\frac{0.2661239\times 10^9}{T^3}
    -\frac{0.2343580\times 10^6}{T^2}
    +\frac{0.8776956\times 10^3}{T}
    +0.179910.
$$

当 $4000 < T \le 25000$：

$$
x = -\frac{3.0258469\times 10^9}{T^3}
    +\frac{2.1070379\times 10^6}{T^2}
    +\frac{0.2226347\times 10^3}{T}
    +0.240390.
$$

### 3.2 Compute y(x)

当 $1667\le T \le 2222$：

$$
y = -1.1063814x^3 -1.34811020x^2 +2.18555832x -0.20219683.
$$

当 $2222 < T \le 4000$：

$$
y = -0.9549476x^3 -1.37418593x^2 +2.09137015x -0.16748867.
$$

当 $4000 < T \le 25000$：

$$
y = 3.0817580x^3 -5.87338670x^2 +3.75112997x -0.37001483.
$$

得到目标白点 $(x_t,y_t)$ 与中性白点 $(x_n,y_n)$（$T_n=6500K$）。

工程实现建议使用参数化平滑窗口，而不是写死 4000K：

- `T_split`：分段中心（默认 `4000K`）；
- `delta_t`：平滑半窗（默认 `100K`）；
- `blend_lo=T_split-delta_t`，`blend_hi=T_split+delta_t`。

在 `[blend_lo, blend_hi]` 区间用 `smoothstep` 融合两段公式，保证分段点附近连续，并便于后续按产品需求调整中心温度。

---

## 4. Chromaticity to Tristimulus

在归一化约定 $Y=1$ 下，白点的 XYZ 可写为：

$$
X = \frac{x}{y},
\quad Y=1,
\quad Z=\frac{1-x-y}{y}.
$$

于是

$$
\mathbf{w}_{XYZ}=[X,Y,Z]^T.
$$

---

## 5. XYZ to Linear RGB White Vector

针对线性 sRGB（D65）基，采用矩阵

$$
\mathbf{M}_{XYZ\rightarrow RGB}=
\begin{bmatrix}
3.2406 & -1.5372 & -0.4986 \\
-0.9689 & 1.8758 & 0.0415 \\
0.0557 & -0.2040 & 1.0570
\end{bmatrix}.
$$

对白点向量：

$$
\mathbf{w}_{RGB}=\mathbf{M}_{XYZ\rightarrow RGB}\,\mathbf{w}_{XYZ}.
$$

分别计算中性白点与目标白点：

$$
\mathbf{w}_{RGB}^{(n)},\;\mathbf{w}_{RGB}^{(t)}.
$$

注：这里可能出现接近零或负值（近似误差、工作域边界），工程上需设下限保护：

$$
\mathbf{w}_{RGB}\leftarrow \max(\mathbf{w}_{RGB},\epsilon),\; \epsilon\approx10^{-6}.
$$

---

## 6. Deriving RGB Gains from White Vectors

定义原始增益比值：

$$
\tilde{\mathbf{g}} = \frac{\mathbf{w}_{RGB}^{(t)}}{\mathbf{w}_{RGB}^{(n)}}
= [\tilde g_R,\tilde g_G,\tilde g_B].
$$

这是“将中性白点推向目标白点”的逐通道比例系数。

### 6.1 Luminance-Preserving Normalization

为了减少整体亮度漂移，采用线性亮度权重归一：

$$
Y_g = 0.2126\tilde g_R + 0.7152\tilde g_G + 0.0722\tilde g_B,
$$

$$
\mathbf{g} = \frac{\tilde{\mathbf{g}}}{Y_g}.
$$

这样可使白点迁移主要体现为色度变化，而非明显亮度跳变。

### 6.2 Safety Clipping

根据面板与管线能力做安全范围限制：

$$
\mathbf{g} \leftarrow \operatorname{clip}(\mathbf{g}, g_{min}, g_{max}).
$$

典型值：$g_{min}=0.5,\;g_{max}=1.8$（需按硬件能力校准）。

### 6.3 Neutral Anchor Constraint

强制中性档精确恒等：

$$
\mathbf{g}(WA\_SEL=64)=[1,1,1].
$$

用于抵消近似公式与数值误差，确保“中档不偏色”。

---

## 7. LUT Construction and Runtime Execution

## 7.1 Offline LUT Build

构建 128 档 LUT：

$$
\mathbf{G}[w] = \mathbf{g}(w),\quad w\in\{0,1,\dots,127\}.
$$

离线阶段允许复杂计算；只需一次生成并固化为常量。

## 7.2 Runtime (Matrix-Free)

运行时仅执行：

1. 读取 `WA_SEL=w`；
2. 查表得到 `\mathbf{G}[w]`；
3. 与现有亮度衰减合成 per-pixel 增益；
4. 逐通道乘法输出。

即：

$$
\mathbf{p}_{out}=\operatorname{diag}(g_R^{pix},g_G^{pix},g_B^{pix})\,\mathbf{p}_{in}.
$$

无 3x3 矩阵、无浮点三色空间变换，满足硬件轻量路径。

---

## 8. Integration with 12-Bin Luma Attenuation

在已有 12-bin 架构下，取节点亮度 $y_i$ 的衰减函数 $a(y_i)$：

$$
\mathbf{g}_{bin}(i,w)=\mathbf{1}+a(y_i)\bigl(\mathbf{G}[w]-\mathbf{1}\bigr).
$$

对像素亮度 $Y$ 在线性插值后得

$$
\mathbf{g}^{pix}(Y,w)=\operatorname{lerp}\bigl(\mathbf{g}_{bin}(i,w),\mathbf{g}_{bin}(i+1,w),t\bigr).
$$

最终仍保持“亮度分段 + 冷暖调节”组合策略：

- 暗部/高亮减弱调节强度，降低偏色与 clipping 风险；
- 中间亮度保留较高白点调节效率。

---

## 9. Fixed-Point Realization (Hardware-Oriented)

设定 Q 格式 `Qm.n`（如 `Q6.10`，`ONE=2^n`）：

$$
G_f = \operatorname{round}(g\cdot ONE).
$$

像素运算：

$$
R' = \frac{R\cdot G_{R,f} + 2^{n-1}}{2^n},
$$

对 $G,B$ 同理。插值参数 $t$ 也可用 `Q0.k` 表示。

优势：

- 时序可控；
- 功耗低；
- 与 DDIC 乘加单元匹配。

---

## 10. Error Sources and Analysis

主要误差项：

1. CCT->xy 经验近似误差；
2. XYZ->RGB 线性模型与真实面板主色不完全一致；
3. 定点量化误差；
4. 剪裁与后续 gamma/clip 的非线性影响。

可通过如下机制控制：

- 单调性校验：`w` 增加时 $g_B/g_R$ 应总体上升；
- 中性点锚定：$w=64$ 强制 identity；
- 范围校验：`g_min <= g_c <= g_max`；
- Patch 验证：灰阶/肤色/天空场景进行主观+客观评估。

---

## 11. Why RGB Gain (Diagonal) Is Sufficient Here

白点调节目标是“参考白色的色度迁移”，而非完整色域严格色适应。对该目标：

$$
\mathbf{p}_{out}=\operatorname{diag}(g_R,g_G,g_B)\mathbf{p}_{in}
$$

已能稳定实现冷暖趋势与白点移动，且硬件成本显著低于完整 3x3 变换。

当需求升级为“高精度全色域感知一致性”时，才需要引入 CAT/3x3 矩阵路径；但这不属于本项目当前运行时约束。

---

## 12. Validation Protocol (Recommended)

1. 锚点一致性：
   - `T(0)=3000K`, `T(64)=6500K`, `T(127)=9300K`。
2. 中性一致性：
   - `G[64] = [1,1,1]`（容差 < 1e-6）。
3. 方向一致性：
   - 暖端 `g_R > g_B`，冷端 `g_B > g_R`。
4. 单调性：
   - `T(w)` 单调非降；`g_B/g_R` 总体单调上升。
5. 图像回归：
   - 原有身份测试（identity）、bin 插值测试、saturation 保护测试全部通过。

---

## 13. Practical Notes for This Repository

本仓库实现对应关系为：

- `wa_sel_to_cct(...)`：完成第 2 节分段映射；
- `cct_to_xy_approx(...)`：完成第 3 节近似；
- `_xy_to_linear_srgb_white(...)`：完成第 4~5 节转换；
- `build_cct_gain_lut(...)`：完成第 6~7 节并输出 `128x3` LUT；
- `WPAConfig` 默认使用 CCT 驱动端点，运行时仍走原有 RGB gain 管线。

---

## 14. Conclusion

本文方法将“经验 warm/cool 系数”替换为“可追溯的色度学映射”，并保持运行时矩阵无关（matrix-free）特性。该框架兼顾了：

- 理论可解释性（CCT/xy/XYZ/RGB 链路）；
- 工程可落地性（LUT + 定点乘加）；
- 与既有 12-bin 算法的兼容性。

这为后续面板实测标定（用真实原色与白点替换标准 sRGB 假设）提供了稳定扩展路径。
