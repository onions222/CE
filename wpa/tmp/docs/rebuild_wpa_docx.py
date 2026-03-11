from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt


OUT = Path("docs/WPA_Algorithm_Spec_For_Algo_and_DIC.docx")
ASSET_DIR = Path("tmp/docs/flowcharts")


def set_base_styles(doc: Document) -> None:
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(10.5)
    for name in ("Heading 1", "Heading 2", "Heading 3"):
        style = doc.styles[name]
        style.font.name = "Times New Roman"


def add_title(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(16)


def add_meta(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.add_run(text)


def add_bullets(doc: Document, items: list[str]) -> None:
    for item in items:
        doc.add_paragraph(item, style="List Bullet")


def add_numbered(doc: Document, items: list[str]) -> None:
    for item in items:
        doc.add_paragraph(item, style="List Number")


def add_table(doc: Document, headers: list[str], rows: list[list[str]]) -> None:
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    for i, h in enumerate(headers):
        table.rows[0].cells[i].text = h
    for row in rows:
        cells = table.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = v


def add_image(doc: Document, path: Path, width_in: float, caption: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(str(path), width=Inches(width_in))
    cap = doc.add_paragraph(caption)
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER


def _draw_box(ax, x: float, y: float, w: float, h: float, text: str, face: str = "#F7FAFC") -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.5,
        edgecolor="#334155",
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=10,
        color="#0F172A",
        wrap=True,
    )


def _draw_arrow(ax, x1: float, y1: float, x2: float, y2: float) -> None:
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=1.4,
        color="#475569",
        connectionstyle="arc3",
    )
    ax.add_patch(arrow)


def render_algorithm_flowchart(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.8))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.95, "WPA Overall Algorithm Flow", fontsize=16, weight="bold", color="#0F172A")

    nodes = [
        (0.03, 0.36, 0.12, 0.22, "WA_SEL\ninput", "#DBEAFE"),
        (0.18, 0.36, 0.14, 0.22, "WA_SEL ->\nCCT mapping", "#E0F2FE"),
        (0.35, 0.36, 0.16, 0.22, "CCT -> CIE 1931 xy\nwith smoothstep blend", "#E0F2FE"),
        (0.54, 0.36, 0.14, 0.22, "xy -> XYZ ->\nlinear RGB white", "#E0F2FE"),
        (0.71, 0.36, 0.14, 0.22, "Normalize vs\nneutral white\nget G_base", "#E0F2FE"),
        (0.88, 0.36, 0.09, 0.22, "12-bin atten\nexpand", "#E0F2FE"),
    ]
    for x, y, w, h, text, color in nodes:
        _draw_box(ax, x, y, w, h, text, color)

    for i in range(len(nodes) - 1):
        x, y, w, h, *_ = nodes[i]
        nx, ny, nw, nh, *_ = nodes[i + 1]
        _draw_arrow(ax, x + w, y + h / 2, nx, ny + nh / 2)

    _draw_box(ax, 0.33, 0.05, 0.20, 0.18, "Build current\nG_bin[i]\nfor 12 luma nodes", "#DCFCE7")
    _draw_box(ax, 0.58, 0.05, 0.18, 0.18, "Per-pixel luma\ninterpolation\nget gain(Y)", "#DCFCE7")
    _draw_box(ax, 0.81, 0.05, 0.14, 0.18, "Apply gain\nand output", "#DCFCE7")
    _draw_arrow(ax, 0.92, 0.36, 0.43, 0.23)
    _draw_arrow(ax, 0.53, 0.14, 0.58, 0.14)
    _draw_arrow(ax, 0.76, 0.14, 0.81, 0.14)

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_fixed_hw_flowchart(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6.8))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.96, "WPA Fixed-Point Hardware Flow", fontsize=16, weight="bold", color="#0F172A")

    ax.text(0.06, 0.88, "WA_SEL update path", fontsize=12, weight="bold", color="#1D4ED8")
    ax.text(0.06, 0.40, "Per-pixel path after WA_SEL is fixed", fontsize=12, weight="bold", color="#15803D")

    top_nodes = [
        (0.05, 0.62, 0.13, 0.16, "WA_SEL\nupdate event", "#DBEAFE"),
        (0.22, 0.62, 0.13, 0.16, "Clamp WA_SEL\n0..127", "#DBEAFE"),
        (0.39, 0.62, 0.15, 0.16, "Warm/cool\nbranch select", "#DBEAFE"),
        (0.58, 0.62, 0.16, 0.16, "runtime_base_gain_fixed", "#DBEAFE"),
        (0.78, 0.62, 0.16, 0.16, "Expand 12x3\nruntime_bin_gain", "#DBEAFE"),
    ]
    for x, y, w, h, text, color in top_nodes:
        _draw_box(ax, x, y, w, h, text, color)
    for i in range(len(top_nodes) - 1):
        x, y, w, h, *_ = top_nodes[i]
        nx, ny, nw, nh, *_ = top_nodes[i + 1]
        _draw_arrow(ax, x + w, y + h / 2, nx, ny + nh / 2)

    bottom_nodes = [
        (0.05, 0.15, 0.12, 0.16, "Pixel input", "#DCFCE7"),
        (0.20, 0.15, 0.12, 0.16, "Degamma /\nlinearize", "#DCFCE7"),
        (0.35, 0.15, 0.12, 0.16, "pixel_fix\nquantize", "#DCFCE7"),
        (0.50, 0.15, 0.12, 0.16, "Y_luma\ncompute", "#DCFCE7"),
        (0.65, 0.15, 0.14, 0.16, "runtime_bin_gain\nlookup + interp", "#DCFCE7"),
        (0.82, 0.15, 0.13, 0.16, "pixel * gain\n+ rounding", "#DCFCE7"),
    ]
    for x, y, w, h, text, color in bottom_nodes:
        _draw_box(ax, x, y, w, h, text, color)
    for i in range(len(bottom_nodes) - 1):
        x, y, w, h, *_ = bottom_nodes[i]
        nx, ny, nw, nh, *_ = bottom_nodes[i + 1]
        _draw_arrow(ax, x + w, y + h / 2, nx, ny + nh / 2)

    _draw_box(ax, 0.72, 0.42, 0.18, 0.12, "runtime_bin_gain\nsmall RAM", "#FDE68A")
    _draw_arrow(ax, 0.86, 0.62, 0.81, 0.42)
    _draw_arrow(ax, 0.76, 0.42, 0.72, 0.31)

    _draw_box(ax, 0.82, 0.02, 0.13, 0.10, "Optional sat\nprotect", "#DCFCE7")
    _draw_box(ax, 0.60, 0.02, 0.18, 0.10, "Clip / engamma /\nuint8 output", "#DCFCE7")
    _draw_arrow(ax, 0.89, 0.15, 0.885, 0.12)
    _draw_arrow(ax, 0.82, 0.07, 0.78, 0.07)

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def ensure_flowchart_assets() -> tuple[Path, Path]:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    overall = ASSET_DIR / "wpa_overall_flow.png"
    fixed = ASSET_DIR / "wpa_fixed_flow.png"
    render_algorithm_flowchart(overall)
    render_fixed_hw_flowchart(fixed)
    return overall, fixed


def build_doc() -> Document:
    doc = Document()
    set_base_styles(doc)
    overall_flow, fixed_flow = ensure_flowchart_assets()

    add_title(doc, "WPA 色温调整算法设计规范")
    add_meta(doc, "面向对象：算法工程师 / 数字 IC 设计工程师")
    add_meta(doc, "版本：v2.1")
    add_meta(doc, "日期：2026-03-09")
    add_meta(doc, "项目：wpa_python_port/claude_version_cct_rgb_gain_method")

    doc.add_heading("1. 文档目的与范围", level=1)
    doc.add_paragraph(
        "本文档定义 White Point Adjustment (WPA) 算法的数学模型、色彩科学基础、定点实现路径、"
        "硬件资源预算与验证准则，作为算法和数字 IC 设计协同交付的主文档。"
    )
    doc.add_paragraph(
        "本文档以当前工程实现为准，重点覆盖三部分内容："
        "1) 算法数学原理；2) 定点版本描述；3) 定点计算对应的硬件资源。"
    )
    doc.add_paragraph(
        "适用范围：显示链路中的冷暖调节，运行时以 WA_SEL 为控制量，不使用 3x3 色适应矩阵，"
        "仅使用 RGB 对角增益和 12-bin 亮度衰减模型。"
    )

    doc.add_heading("2. 术语与符号", level=1)
    add_table(
        doc,
        ["符号", "定义", "范围/备注"],
        [
            ["WA_SEL", "冷暖控制码", "0..127，64 为中性"],
            ["T", "目标相关色温 CCT", "单位 K"],
            ["x,y", "CIE 1931 色度坐标", "目标白点坐标"],
            ["X,Y,Z", "CIE XYZ 三刺激值", "由 x,y 和归一化亮度得到"],
            ["RGB_target", "目标白点对应的 linear RGB", "XYZ 经过颜色空间变换得到"],
            ["RGB_neutral", "中性白点对应的 linear RGB", "固定使用 WA_SEL=64 参考"],
            ["G_base", "基础 RGB 增益向量", "线性域，运行时由 WA_SEL 计算"],
            ["Y_luma", "亮度代理", "(R + 2G + B) / 4"],
            ["atten(Y)", "亮度衰减权重", "三段分段线性函数"],
            ["G_bin[i]", "第 i 个亮度 bin 的 RGB 增益", "12x3 运行时缓存"],
            ["F_c", "系数小数位", "默认 8，可选 10"],
            ["F_p", "像素内部小数位", "默认 10"],
        ],
    )

    doc.add_heading("3. 设计目标与约束", level=1)
    doc.add_paragraph("目标：")
    add_bullets(
        doc,
        [
            "WA_SEL=64 对应 D65(6500K) 且严格 identity。",
            "冷暖端点覆盖约 3000K 到 9300K。",
            "运行时复杂度满足数字 IC 实时预算。",
            "亮度感知尽量稳定，避免整体明暗随色温调整明显漂移。",
            "暗部和高亮端减弱调整强度，降低偏色和 clipping 风险。",
        ],
    )
    doc.add_paragraph("约束：")
    add_bullets(
        doc,
        [
            "运行时不引入 3x3 色彩适应矩阵。",
            "运行时常驻只保留 3 个基础增益锚点和 12 点亮度衰减 LUT。",
            "默认定点系数格式为 UQ1.8，可选 UQ1.10。",
            "WA_SEL 更新路径与逐像素路径必须分离；前者只在控制量改变时运行。",
            "逐像素主路径必须尽量使用整数加减乘移位实现。",
        ],
    )

    doc.add_heading("4. 算法数学原理与色彩科学基础", level=1)
    doc.add_paragraph(
        "本章给出从 WA_SEL 到目标白点 RGB 增益的完整数学链路。"
        "算法本质上是以相关色温为控制变量的白点重定标问题：先根据目标色温获得目标白点色度，"
        "再将该白点映射到面板工作颜色空间的 linear RGB，最后归一化为相对中性白点的 RGB 增益。"
    )

    doc.add_heading("4.1 WA_SEL 到 CCT 映射", level=2)
    doc.add_paragraph("WA_SEL 的设计中心为 64，对应 6500K。暖端和冷端采用分段线性映射：")
    add_numbered(
        doc,
        [
            "Eq. (1)  当 wa <= 64 时：T(wa) = 6500 - (64 - wa) * (6500 - 3000) / 64",
            "Eq. (2)  当 wa > 64 时：T(wa) = 6500 + (wa - 64) * (9300 - 6500) / 63",
        ],
    )
    doc.add_paragraph("锚点：T(0)=3000K，T(64)=6500K，T(127)=9300K。")
    doc.add_paragraph("Eq. (1) 和 Eq. (2) 为本项目的控制量映射定义，用于把 WA_SEL 映射到目标 CCT。")
    doc.add_paragraph(
        "该映射的作用不是直接决定 RGB 通道增益，而是先确定目标白点的色温位置。"
        "色温越低，目标白点越偏暖；色温越高，目标白点越偏冷。"
    )

    doc.add_heading("4.2 CCT 到 xy 的色度近似与平滑过渡", level=2)
    doc.add_paragraph(
        "离线建模阶段使用 CCT 到 CIE 1931 xy 的标准近似公式。"
        "由于低温段和高温段采用不同多项式近似，直接硬切换容易在分段点附近产生导数不连续，"
        "进而让 LUT 出现局部折线。实现以当前源码中的 `cct_to_xy_approx()` 为准。"
    )
    add_numbered(
        doc,
        [
            "Eq. (3)  t = clip(CCT, 1667, 25000)  [4]",
            "Eq. (4)  x_low(t) = -0.2661239e9 / t^3 - 0.2343580e6 / t^2 + 0.8776956e3 / t + 0.179910  [4]",
            "Eq. (5)  x_high(t) = -3.0258469e9 / t^3 + 2.1070379e6 / t^2 + 0.2226347e3 / t + 0.240390  [4]",
            "Eq. (6)  blend_lo = T_split - delta_t",
            "Eq. (7)  blend_hi = T_split + delta_t",
            "Eq. (8)  u_raw = (t - blend_lo) / (blend_hi - blend_lo)",
            "Eq. (9)  u = u_raw^2 * (3 - 2 * u_raw)",
            "Eq. (10)  x(t) = x_low,                t <= blend_lo",
            "Eq. (11)  x(t) = x_high,               t >= blend_hi",
            "Eq. (12)  x(t) = (1 - u) * x_low + u * x_high,   blend_lo < t < blend_hi",
            "Eq. (13)  y_low(x) = -1.1063814 * x^3 - 1.34811020 * x^2 + 2.18555832 * x - 0.20219683  [4]",
            "Eq. (14)  y_mid(x) = -0.9549476 * x^3 - 1.37418593 * x^2 + 2.09137015 * x - 0.16748867  [4]",
            "Eq. (15)  y_high(x) = 3.0817580 * x^3 - 5.87338670 * x^2 + 3.75112997 * x - 0.37001483  [4]",
            "Eq. (16)  y(t) = y_low(x),                              t <= 2222",
            "Eq. (17)  y(t) = y_mid(x),                              2222 < t < blend_lo",
            "Eq. (18)  y(t) = y_high(x),                             t > blend_hi",
            "Eq. (19)  y(t) = (1 - u) * y_mid(x) + u * y_high(x),    blend_lo <= t <= blend_hi",
        ],
    )
    doc.add_paragraph(
        "因此实现中不把分段点写死，而是引入参数：T_split、delta_t、blend_lo、blend_hi。"
        "在 [blend_lo, blend_hi] 区间内使用 smoothstep 在低温公式和高温公式之间平滑融合，其中 "
        "u 由 Eq. (8) 和 Eq. (9) 定义。"
    )
    add_bullets(
        doc,
        [
            "T_split：默认 4000K，表示低温/高温公式的中心切换点。",
            "delta_t：默认 100K，表示平滑半窗宽度。",
            "blend_lo = T_split - delta_t。",
            "blend_hi = T_split + delta_t。",
        ],
    )
    doc.add_paragraph(
        "该处理的工程意义是避免 WA_SEL 改变时，CCT 增益表在 4000K 附近出现可见折点，"
        "从而减少冷暖连续滑动过程中的突变。"
    )

    doc.add_heading("4.3 xy 到 XYZ，再到 linear RGB 目标白点", level=2)
    doc.add_paragraph(
        "得到目标色度坐标 (x, y) 后，可在归一化亮度条件下恢复 XYZ 三刺激值："
    )
    add_numbered(
        doc,
        [
            "Eq. (20)  y_safe = max(y, 1e-8)  [2][3]",
            "Eq. (21)  XYZ_target = [x / y_safe, 1, (1 - x - y_safe) / y_safe]^T  [2][3]",
        ],
    )
    doc.add_paragraph(
        "实现中使用 y_safe = max(y, 1e-8) 防止分母为零。随后，使用固定的 XYZ 到 linear RGB"
        " 变换矩阵将目标白点转换到工作 RGB 空间："
    )
    add_numbered(
        doc,
        [
            "Eq. (22)  [R, G, B]^T = M_XYZ_to_RGB * XYZ_target  [1][2]",
            "Eq. (23)  M_XYZ_to_RGB = [[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]]  [1][2]",
            "Eq. (24)  RGB_target = clip(M_XYZ_to_RGB * XYZ_target, 1e-6, +inf)",
        ],
    )
    doc.add_paragraph(
        "这里的下限裁剪不是经验值，而是实现约束。"
        "由于多项式近似和颜色空间边界可能导致极小或负通道值，工程实现会对 linear RGB 通道设置下限"
        " 1e-6，以避免后续取比值时出现数值发散。"
    )

    doc.add_heading("4.4 相对中性白点的基础增益建模", level=2)
    doc.add_paragraph(
        "算法不是直接使用 RGB_target 作为运行时白点，而是先求相对中性参考白点的通道比值。"
        "中性参考固定为 WA_SEL=64 所对应的白点 RGB_neutral。"
    )
    add_numbered(
        doc,
        [
            "Eq. (25)  G_base_float = RGB_target / RGB_neutral",
            "Eq. (26)  Y_norm = 0.2126 * G_R + 0.7152 * G_G + 0.0722 * G_B  [5]",
            "Eq. (27)  G_base = clip(G_base_float / max(Y_norm, 1e-6), g_min, g_max)",
        ],
    )
    doc.add_paragraph(
        "其中亮度归一化是必要步骤，不是可选项。它的作用是约束整体明暗稳定，防止色温变化时因为某一通道被"
        "大幅提升导致图像整体亮度漂移。归一化后再裁剪到 [g_min, g_max]，默认 g_min=0.5，g_max=1.8。"
    )
    add_bullets(
        doc,
        [
            "CCT 先钳位到 [1667K, 25000K]，避免色度多项式外推失真。",
            "RGB_target 的各通道设置下限，避免除零或极端比值。",
            "LUT 构建完成后，强制 WA_SEL=64 对应 [1,1,1]，确保严格 identity。",
        ],
    )

    doc.add_heading("4.5 三锚点压缩与运行时基础增益插值", level=2)
    doc.add_paragraph(
        "离线高精度模型可生成 128 档 cct_gain_lut[128][3]。"
        "但运行时主路径为了节省常驻资源，只保留三个锚点：warm、neutral、cool。"
    )
    add_numbered(
        doc,
        [
            "Eq. (28)  当 wa <= 64 时：G_base(wa) = G_warm + (wa / 64) * (G_neutral - G_warm)",
            "Eq. (29)  当 wa > 64 时：G_base(wa) = G_neutral + ((wa - 64) / 64) * (G_cool - G_neutral)",
        ],
    )
    doc.add_paragraph(
        "其中定点主路径的 cool 侧采用 /64，而不是浮点参考路径中的 /63。"
        "原因是 /64 可由右移直接实现，更适合硬件。代价是 wa=127 时略弱于理论冷端，"
        "但误差经验证处于可接受范围。"
    )

    doc.add_heading("4.6 亮度代理、衰减模型与 12-bin 展开", level=2)
    doc.add_paragraph(
        "为了避免在暗部和高亮端施加过强色温调整，算法引入基于亮度代理的衰减模型。"
    )
    add_numbered(
        doc,
        [
            "Eq. (30) Y_luma = (R + 2G + B) / 4",
            "Eq. (31) atten(y) = 0.55, y <= 31",
            "Eq. (32) atten(y) = 0.55 + 0.45 * (y - 31) / 96, 31 < y <= 127",
            "Eq. (33) atten(y) = 1.00 - 0.35 * (y - 127) / 112, 127 < y <= 239",
            "Eq. (34) atten(y) = 0.65, y > 239",
            "Eq. (35) G_bin[i] = 1 + atten(y_i) * (G_base - 1)",
        ],
    )
    doc.add_paragraph(
        "其中 y_i 为 12 个亮度节点。运行时先根据当前 WA_SEL 生成当前的 12x3 bin 增益缓存，"
        "再根据每个像素的 Y_luma 在相邻节点间线性插值得到实际像素增益。"
    )
    doc.add_paragraph(
        "这一设计的好处是把复杂度从逐像素重计算，转移为 WA_SEL 更新时的一次性表展开。"
    )

    doc.add_heading("4.7 浮点参考路径与定点主路径的角色差异", level=2)
    doc.add_paragraph(
        "浮点参考路径用于算法验证、图像观感分析和高精度参考输出；"
        "定点主路径用于资源评估和硬件交付。两者在数学目标上保持一致，但在实现细节上允许采用不同的"
        "工程近似，如 cool 侧 /64 取代 /63。"
    )
    doc.add_heading("4.8 总体算法流程图", level=2)
    add_image(doc, overall_flow, 6.7, "流程图 1. 总体算法与建模流程")

    doc.add_heading("5. 定点版本描述", level=1)
    doc.add_paragraph(
        "本章描述硬件交付主路径 `wpa_fixed`。其核心思想是："
        "把与 WA_SEL 有关的计算集中在控制量更新时完成，把逐像素阶段压缩为亮度插值和乘移位。"
    )

    doc.add_heading("5.1 定点数据格式与量化规则", level=2)
    add_bullets(
        doc,
        [
            "系数格式默认 UQ1.8，对应 coeff_frac_bits = 8，编码 scale = 256。",
            "可选 UQ1.10，对应 coeff_frac_bits = 10，编码 scale = 1024。",
            "线性像素内部格式默认 Q0.10，对应 frac_bits = 10。",
            "系数量化：gain_fix = round(gain_float * 2^F_c)。",
            "像数量化：pixel_fix = round(pixel_linear * 2^F_p)。",
            "乘法舍入：out_fix = (pixel_fix * gain_fix + 2^(F_c-1)) >> F_c。",
            "engamma 前必须先 clip 到 [0,1]，防止逆伽马输入越界。",
        ],
    )

    doc.add_heading("5.2 常驻数据", level=2)
    add_bullets(
        doc,
        [
            "wa_base_gain_lut_fixed[3][3]：仅存 warm / neutral / cool 三个 anchor。",
            "atten_q_lut_fixed[12]：12 点亮度衰减 LUT。",
            "luma_nodes[12]：亮度节点位置，默认 [15,31,47,63,95,127,159,191,223,239,247,255]。",
        ],
    )

    doc.add_heading("5.3 WA_SEL 更新路径（非每像素）", level=2)
    doc.add_paragraph(
        "WA_SEL 更新只发生在控制量改变时，不在逐像素路径内重复执行。该路径的输出是当前"
        " runtime_bin_gain[12][3]。"
    )
    doc.add_paragraph("运行流程：")
    add_numbered(
        doc,
        [
            "对 WA_SEL 做 0..127 钳位，得到 wa。",
            "若 wa <= 64，则在 warm 与 neutral 两锚点之间插值，得到 G_base_fix。",
            "若 wa > 64，则在 neutral 与 cool 两锚点之间插值，得到 G_base_fix。",
            "计算 Delta_base = G_base_fix - COEFF_ONE。",
            "将 Delta_base 与 12 点 atten_q_lut_fixed 相乘并右移，生成 runtime_bin_gain[12][3]。",
            "若 wa = 64，则全部 runtime_bin_gain 强制为 [COEFF_ONE, COEFF_ONE, COEFF_ONE]。",
        ],
    )
    doc.add_paragraph("默认 UQ1.8 下的关键常量如下：")
    add_table(
        doc,
        ["项目", "数值", "说明"],
        [
            ["F", "8", "系数小数位"],
            ["COEFF_ONE", "256", "表示实数 1.0"],
            ["COEFF_HALF", "128", "右移前四舍五入偏置"],
            ["G_warm_fix", "[436, 221, 128]", "warm 锚点"],
            ["G_neutral_fix", "[256, 256, 256]", "neutral 锚点"],
            ["G_cool_fix", "[219, 259, 339]", "cool 锚点"],
            ["atten_q_lut_fixed", "[141,141,160,179,218,256,230,205,179,166,166,166]", "12 点衰减 LUT"],
        ],
    )
    doc.add_paragraph("对应定点公式：")
    add_numbered(
        doc,
        [
            "Eq. (36) wa = min(max(WA_SEL, 0), 127)",
            "Eq. (37) 若 wa <= 64：Delta_warm = G_neutral_fix - G_warm_fix",
            "Eq. (38) 若 wa <= 64：G_base_fix = G_warm_fix + ((wa * Delta_warm + 32) >> 6)",
            "Eq. (39) 若 wa > 64：Delta_cool = G_cool_fix - G_neutral_fix",
            "Eq. (40) 若 wa > 64：G_base_fix = G_neutral_fix + (((wa - 64) * Delta_cool + 32) >> 6)",
            "Eq. (41) Delta_base = G_base_fix - COEFF_ONE",
            "Eq. (42) G_bin_fix[i] = COEFF_ONE + ((atten_q[i] * Delta_base + COEFF_HALF) >> F)",
        ],
    )

    doc.add_heading("5.4 Warm 分支逐步定点示例（WA_SEL = 32）", level=2)
    add_numbered(
        doc,
        [
            "输入裁剪：wa = 32。",
            "Delta_warm = [256,256,256] - [436,221,128] = [-180,35,128]。",
            "增量 = ((32 * Delta_warm) + 32) >> 6 = [-90,18,64]。",
            "G_base_fix = [436,221,128] + [-90,18,64] = [346,239,192]。",
            "对应浮点约为 [1.3516, 0.9336, 0.7500]。",
            "Delta_base = [346,239,192] - 256 = [90,-17,-64]。",
        ],
    )
    add_table(
        doc,
        ["bin", "luma node", "atten_q", "runtime_bin_gain[i] (UQ1.8)", "浮点近似"],
        [
            ["0", "15", "141", "[306, 247, 221]", "[1.1953, 0.9648, 0.8633]"],
            ["1", "31", "141", "[306, 247, 221]", "[1.1953, 0.9648, 0.8633]"],
            ["2", "47", "160", "[312, 245, 216]", "[1.2188, 0.9570, 0.8438]"],
            ["3", "63", "179", "[319, 244, 211]", "[1.2461, 0.9531, 0.8242]"],
            ["4", "95", "218", "[333, 242, 202]", "[1.3008, 0.9453, 0.7891]"],
            ["5", "127", "256", "[346, 239, 192]", "[1.3516, 0.9336, 0.7500]"],
            ["6", "159", "230", "[337, 241, 199]", "[1.3164, 0.9414, 0.7773]"],
            ["7", "191", "205", "[328, 242, 205]", "[1.2812, 0.9453, 0.8008]"],
            ["8", "223", "179", "[319, 244, 211]", "[1.2461, 0.9531, 0.8242]"],
            ["9", "239", "166", "[314, 245, 215]", "[1.2266, 0.9570, 0.8398]"],
            ["10", "247", "166", "[314, 245, 215]", "[1.2266, 0.9570, 0.8398]"],
            ["11", "255", "166", "[314, 245, 215]", "[1.2266, 0.9570, 0.8398]"],
        ],
    )
    doc.add_paragraph(
        "结论：暖端时 R 通道增益高于 1.0，B 通道低于 1.0；中亮度附近衰减最弱，因此暖色效应最强。"
    )

    doc.add_heading("5.5 Cool 分支逐步定点示例（WA_SEL = 96）", level=2)
    add_numbered(
        doc,
        [
            "输入裁剪：wa = 96。",
            "Delta_cool = [219,259,339] - [256,256,256] = [-37,3,83]。",
            "增量 = ((32 * Delta_cool) + 32) >> 6 = [-18,2,42]。",
            "G_base_fix = [256,256,256] + [-18,2,42] = [238,258,298]。",
            "对应浮点约为 [0.9297, 1.0078, 1.1641]。",
            "Delta_base = [238,258,298] - 256 = [-18,2,42]。",
        ],
    )
    add_table(
        doc,
        ["bin", "luma node", "atten_q", "runtime_bin_gain[i] (UQ1.8)", "浮点近似"],
        [
            ["0", "15", "141", "[246, 257, 279]", "[0.9609, 1.0039, 1.0898]"],
            ["1", "31", "141", "[246, 257, 279]", "[0.9609, 1.0039, 1.0898]"],
            ["2", "47", "160", "[245, 257, 282]", "[0.9570, 1.0039, 1.1016]"],
            ["3", "63", "179", "[243, 257, 285]", "[0.9492, 1.0039, 1.1133]"],
            ["4", "95", "218", "[241, 258, 292]", "[0.9414, 1.0078, 1.1406]"],
            ["5", "127", "256", "[238, 258, 298]", "[0.9297, 1.0078, 1.1641]"],
            ["6", "159", "230", "[240, 258, 294]", "[0.9375, 1.0078, 1.1484]"],
            ["7", "191", "205", "[242, 258, 290]", "[0.9453, 1.0078, 1.1328]"],
            ["8", "223", "179", "[243, 257, 285]", "[0.9492, 1.0039, 1.1133]"],
            ["9", "239", "166", "[244, 257, 283]", "[0.9531, 1.0039, 1.1055]"],
            ["10", "247", "166", "[244, 257, 283]", "[0.9531, 1.0039, 1.1055]"],
            ["11", "255", "166", "[244, 257, 283]", "[0.9531, 1.0039, 1.1055]"],
        ],
    )
    doc.add_paragraph(
        "结论：冷端时 R 通道低于 1.0，B 通道高于 1.0；亮度衰减模型与 warm 分支共用同一套 LUT。"
    )

    doc.add_heading("5.6 WA_SEL 固定后的逐像素路径", level=2)
    add_numbered(
        doc,
        [
            "Step 1. 对输入 sRGB 图像做 degamma，得到 linear_f。",
            "Step 2. 量化为 pixel_fix = round(linear_f * 2^F_p)。",
            "Step 3. 计算亮度代理 Y_luma；可工作于 gamma 域或 linear 域。",
            "Step 4. 以 Y_luma 在 runtime_bin_gain[12][3] 上做线性插值，得到 gain_fix。",
            "Step 5. 逐通道执行 out_fix = (pixel_fix * gain_fix + 2^(F_c-1)) >> F_c。",
            "Step 6. 可选执行饱和度保护：out = in + w * (adjusted - in)。",
            "Step 7. 将结果 clip 到 [0, 2^F_p]，再除以 2^F_p、engamma 并量化到 uint8。",
        ],
    )
    doc.add_paragraph("浮点参考路径中的节点插值公式如下：")
    add_numbered(
        doc,
        [
            "Eq. (43)  y_255 = 255 * Y_luma",
            "Eq. (44)  idx_hi = searchsorted(luma_nodes, y_255, side='right')",
            "Eq. (45)  idx_lo = idx_hi - 1",
            "Eq. (46)  t = clip((y_255 - node_lo) / (node_hi - node_lo), 0, 1)",
            "Eq. (47)  gain(Y) = (1 - t) * gain_lo + t * gain_hi",
        ],
    )
    doc.add_paragraph("定点硬件路径中的节点插值公式如下：")
    add_numbered(
        doc,
        [
            "Eq. (48)  numer = ((y - node_lo) << interp_bits) + (span >> 1)",
            "Eq. (49)  若 span 为 2 的幂：t_fix = clip(numer >> log2(span), 0, 2^interp_bits)",
            "Eq. (50)  若 span 不是 2 的幂：t_fix = clip(numer // span, 0, 2^interp_bits)",
            "Eq. (51)  gain_fix = gain_lo + ((t_fix * (gain_hi - gain_lo) + 2^(interp_bits-1)) >> interp_bits)",
            "Eq. (52)  out_fix = (pixel_fix * gain_fix + 2^(F_c-1)) >> F_c",
        ],
    )
    add_table(
        doc,
        ["变量", "格式/位宽", "说明"],
        [
            ["y", "8 bit", "uint8 亮度代理，范围 0..255"],
            ["node_lo / node_hi", "8 bit", "相邻亮度节点"],
            ["span", "最多 6 bit", "node_hi - node_lo"],
            ["interp_bits", "默认 10", "插值小数位"],
            ["t", "float", "浮点节点插值系数"],
            ["t_fix", "Q0.interp_bits", "定点节点插值系数"],
            ["pixel_fix", "Q0.F_p", "定点线性像素"],
            ["gain_fix", "UQ1.F_c", "定点增益"],
        ],
    )
    doc.add_paragraph(
        "硬件重点是：WA_SEL 固定后，逐像素路径不再需要重新计算 CCT、xy 或基础增益，"
        "只依赖当前 runtime_bin_gain 和像素亮度位置。"
    )
    doc.add_heading("5.8 定点硬件流程图", level=2)
    add_image(doc, fixed_flow, 6.7, "流程图 2. 定点硬件路径")

    doc.add_heading("5.7 饱和度保护（可选）", level=2)
    add_numbered(
        doc,
        [
            "Eq. (53) s = |R-G| + |G-B| + |B-R|",
            "Eq. (54) w = clip((s1 - s) / (s1 - s0), 0, 1)",
            "Eq. (55) out = in + w * (adjusted - in)",
        ],
    )
    doc.add_paragraph(
        "当 sat_weight_domain=linear 时，阈值 s0/s1 需与域定义一致，不能简单沿用 gamma 域阈值。"
    )

    doc.add_heading("6. 定点计算所需硬件资源整体描述", level=1)
    doc.add_paragraph(
        "硬件资源应按两个阶段理解："
        "第一阶段是 WA_SEL 更新时的控制路径；第二阶段是 WA_SEL 固定后的逐像素数据路径。"
    )

    doc.add_heading("6.1 常驻存储与 scratch 存储", level=2)
    add_table(
        doc,
        ["配置", "系数位宽", "base LUT", "atten LUT", "luma_nodes", "总静态", "runtime scratch"],
        [
            ["UQ1.8", "9 bit", "11 B", "14 B", "12 B", "36 B", "41 B"],
            ["UQ1.10", "11 bit", "13 B", "17 B", "12 B", "41 B", "50 B"],
        ],
    )
    doc.add_paragraph(
        "其中 runtime scratch 指 WA_SEL 更新阶段生成的 runtime_bin_gain[12][3] 临时或工作 RAM。"
    )

    doc.add_heading("6.2 关键运行时变量位宽", level=2)
    add_table(
        doc,
        ["变量", "UQ1.8 / 默认路径", "UQ1.10 / 默认路径", "说明"],
        [
            ["WA_SEL", "7 bit", "7 bit", "0..127"],
            ["luma_u8", "8 bit", "8 bit", "0..255"],
            ["luma idx", "4 bit", "4 bit", "12 节点索引"],
            ["interp t", "11 bit", "11 bit", "Q0.10 插值参数"],
            ["pixel_fix_ch", "11 bit", "11 bit", "Q0.10"],
            ["gain_fix_ch", "9 bit", "11 bit", "UQ1.8 或 UQ1.10"],
            ["pixel * gain", "20+ bit", "22+ bit", "建议 int32 或更宽乘法器"],
            ["runtime_bin_gain entry", "9 bit", "11 bit", "每通道一项"],
        ],
    )

    doc.add_heading("6.3 WA_SEL 更新阶段资源", level=2)
    doc.add_paragraph(
        "该阶段在控制寄存器变化时运行，不属于逐像素高吞吐路径，但其结果会决定之后所有像素的增益表。"
    )
    add_table(
        doc,
        ["资源估算前提", "默认值/假设", "说明"],
        [
            ["WA_SEL update latency", "允许多周期", "控制路径不要求每周期更新一次"],
            ["复用策略", "允许时分复用乘法器", "降低面积，增加 WA_SEL 更新延迟"],
            ["统计口径", "以标量操作数计", "不绑定具体 SIMD 或流水复制数"],
        ],
    )
    add_bullets(
        doc,
        [
            "输入：WA_SEL、3 个 anchor gain、12 点 atten_q_lut_fixed。",
            "输出：runtime_bin_gain[12][3]。",
            "核心操作：三通道基础插值 + 12x3 bin 展开。",
            "推荐实现：一个小型控制状态机驱动可复用乘加器。",
        ],
    )
    add_table(
        doc,
        ["阶段", "比较", "加法", "减法", "乘法", "移位", "备注"],
        [
            ["WA_SEL -> base gain 插值", "1", "3", "3", "3", "3", "三通道并行或复用"],
            ["12x3 bin 展开", "0", "36", "0", "36", "36", "每个 bin 三通道各一次"],
            ["合计", "1", "39", "3", "39", "39", "不含 RAM 读写控制"],
        ],
    )
    doc.add_paragraph(
        "若采用乘法器时分复用，可显著降低面积，但 WA_SEL 更新延迟会增加；"
        "该延迟一般可接受，因为 WA_SEL 变化频率远低于像素时钟。"
    )

    doc.add_heading("6.4 WA_SEL 固定后的逐像素资源", level=2)
    doc.add_paragraph(
        "该阶段是主数据通路，应按每像素吞吐进行预算。"
        "资源重点包括亮度代理计算、节点定位、bin 插值、增益乘法与可选饱和度保护。"
    )
    add_table(
        doc,
        ["资源估算前提", "默认值/假设", "说明"],
        [
            ["luma_nodes", "[15,31,47,63,95,127,159,191,223,239,247,255]", "当前默认节点集合"],
            ["相邻节点跨度", "{16,16,16,32,32,32,32,32,16,8,8}", "全部为 2 的幂"],
            ["pixel path initiation interval", "II = 1", "按 1 pixel / cycle 理解吞吐压力"],
            ["pixel path latency", "与流水深度相关", "本文统计操作数，不绑定固定级数"],
            ["interp divider", "默认不需要", "默认节点集合下可由移位实现；若节点改为非 2 的幂间距则需要除法器"],
        ],
    )
    add_table(
        doc,
        ["配置", "比较", "加法", "减法", "乘法", "除法", "移位", "clip"],
        [
            ["sat_en=False", "11", "13", "5", "6", "0", "8", "1"],
            ["sat_en=True 额外开销", "+0", "+7", "+5", "+3", "+1", "+4", "+0"],
        ],
    )
    add_bullets(
        doc,
        [
            "Y_luma = (R + 2G + B) >> 2，可由 2 个加法器和 1 个右移实现。",
            "节点定位可由比较链、查找表或小型编码器实现。",
            "默认 luma_nodes 下，相邻节点跨度全部为 2 的幂，因此插值分母可由移位实现。",
            "若后续更改节点集合且跨度出现非 2 的幂，则需要增加通用整数除法器或小型除法单元。",
            "bin 插值通常需要 1 个插值系数 t_fix 和 3 个通道的乘加。",
            "逐通道增益乘法后需加 COEFF_HALF，再右移 F_c 位实现四舍五入。",
            "若启用 sat_en，还需增加饱和度差分路径和一组混合乘法。",
        ],
    )

    doc.add_heading("6.5 数据流划分建议", level=2)
    add_numbered(
        doc,
        [
            "控制路径：WA_SEL 更新 -> runtime_base_gain_fixed -> runtime_bin_gain[12][3]。",
            "像素路径：degamma/linear 输入 -> luma -> gain 插值 -> gain 乘法 -> 可选 sat -> clip -> engamma。",
            "存储建议：anchor LUT 与 atten LUT 放常驻 ROM 或小 RAM；runtime_bin_gain 放可写小 RAM。",
        ],
    )

    doc.add_heading("7. 数值验证与可视验证结论", level=1)
    add_bullets(
        doc,
        [
            "线性插值相对 128 点参考的误差：coeff_frac_bits=8 时 MAE=0.010408, MaxAbs=0.028228。",
            "coeff_frac_bits=10 时 MAE=0.010742, MaxAbs=0.027910。",
            "WA_SEL=64 时必须严格 identity。",
            "暖端趋势为 R 上升、B 下降；冷端趋势为 R 下降、B 上升。",
            "在低资源约束下，3 锚点 + cool 侧 /64 插值的误差可接受。",
        ],
    )
    add_table(
        doc,
        ["coeff_frac_bits", "MAE", "P99 Abs", "Max Abs"],
        [
            ["8", "0.010408", "0.028085", "0.028228"],
            ["10", "0.010742", "0.027249", "0.027910"],
        ],
    )

    doc.add_heading("8. 测试集建议（分层）", level=1)
    add_bullets(
        doc,
        [
            "smoke：最小快测，覆盖灰阶、色卡、高光、UI 文本和混光代表样本。",
            "core：日常回归，覆盖 13 合成 + 15 Kodak + JPEG 退化梯度。",
            "full：发布门禁，覆盖 13 合成 + 全 24 Kodak + JPEG 退化梯度。",
            "构建命令：python scripts/build_test_set.py --profile smoke|core|full",
        ],
    )

    doc.add_heading("9. 交付与联调检查单", level=1)
    add_bullets(
        doc,
        [
            "WA_SEL=64 输出与输入逐像素一致（<=1 LSB）。",
            "冷暖方向正确：暖端 R↑ B↓，冷端 R↓ B↑。",
            "灰阶与天空渐变无明显新增 banding。",
            "高亮和暗部无异常偏色或剪切突变。",
            "WA_SEL 更新阶段和逐像素阶段的资源统计与文档一致。",
            "hw_stats 输出与资源表一致。",
        ],
    )

    doc.add_heading("10. 参考文献与标准", level=1)
    refs = [
        "[1] IEC 61966-2-1:1999, Colour management - Default RGB colour space - sRGB.",
        "[2] CIE 15:2018, Colorimetry, 4th Edition.",
        "[3] Wyszecki, G. and Stiles, W. S., Color Science, 2nd ed., Wiley, 2000.",
        "[4] Hernandez-Andres, J., Lee, R. L., and Romero, J., Calculating correlated color temperatures across the entire gamut of daylight and skylight chromaticities, Applied Optics, 38(27), 5703-5709, 1999.",
        "[5] ITU-R BT.709-6, Parameter values for the HDTV standards.",
    ]
    for ref in refs:
        doc.add_paragraph(ref)

    return doc


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = build_doc()
    doc.save(OUT)


if __name__ == "__main__":
    main()
