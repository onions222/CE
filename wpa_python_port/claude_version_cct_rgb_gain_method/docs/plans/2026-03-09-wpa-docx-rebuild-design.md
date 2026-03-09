# WPA DOCX Rebuild Design

**Context**

现有 `docs/WPA_Algorithm_Spec_For_Algo_and_DIC.docx` 具备基础章节结构，但在以下方面明显不足：

- 第 4 章数学原理描述不够完整，缺少从 CCT 到白点、再到 RGB 增益的完整数学链路解释
- 第 5 章定点实现描述过于概括，缺少 `WA_SEL` 更新阶段的逐步定点推导
- 第 6 章硬件资源描述未按“`WA_SEL` 更新”和“`WA_SEL` 固定后逐像素计算”分拆
- 与当前 [ALGORITHM.md](/Users/onion/Desktop/code/CE/wpa_python_port/claude_version_cct_rgb_gain_method/ALGORITHM.md) 的详细程度不一致

**Decision**

采用整份重构方案，而不是局部增补。保留原 `docx` 文件路径，重写其中的主体内容，使其成为正式交付版本。

**Document Goals**

新 `docx` 需要覆盖三部分核心内容：

1. 算法数学原理
   - 说明 `WA_SEL -> CCT -> xy -> XYZ -> linear RGB gain` 的完整数学链路
   - 补充亮度归一化、三锚点压缩、亮度衰减与 12-bin 展开
   - 加入必要公式、变量定义和工程解释

2. 定点版本描述
   - 说明数据格式、量化规则、舍入规则、饱和/裁剪策略
   - 明确 `WA_SEL` 更新阶段与逐像素阶段的边界
   - 纳入 `WA_SEL=32` 和 `WA_SEL=96` 的逐步示例

3. 定点硬件资源整体描述
   - 单列 `WA_SEL` 更新阶段所需资源
   - 单列 `WA_SEL` 固定后的逐像素计算资源
   - 给出常驻存储、scratch 存储、关键位宽与操作量拆分

**Structure**

新文档仍沿用现有章节编号，重构重点如下：

- 第 2 章补强术语与符号表
- 第 4 章扩展为完整数学原理章节
- 第 5 章扩展为完整定点实现章节
- 第 6 章重构为分阶段硬件资源章节
- 保留第 7-10 章作为验证、测试、交付和参考章节，但必要时补入与前文一致的新内容

**Formatting Approach**

- 使用 `python-docx` 直接重建文档内容
- 保留标题、章节层级、表格等 Word 结构
- 公式以工程可读文本方式录入，避免依赖复杂的 Word 公式对象
- 表格用于术语、位宽、静态资源、操作量和逐步示例结果

**Acceptance Criteria**

- `docx` 中的硬件描述详细程度与 `ALGORITHM.md` 对齐或更完整
- `WA_SEL` 更新路径和逐像素路径均有明确分阶段说明
- warm/cool 两侧至少各有一个完整逐步示例
- 资源章节明确区分 `WA_SEL` 更新与 `WA_SEL` 固定后计算
- 文档可被 `python-docx` 正常读取，结构完整
