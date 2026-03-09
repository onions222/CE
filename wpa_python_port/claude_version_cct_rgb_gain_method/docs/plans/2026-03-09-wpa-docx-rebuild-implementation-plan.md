# WPA DOCX Rebuild Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 重建 `docs/WPA_Algorithm_Spec_For_Algo_and_DIC.docx`，使其数学原理、定点实现和硬件资源描述与 `ALGORITHM.md` 的详细程度对齐。

**Architecture:** 以现有 `docx` 为目标文件路径，但采用程序化重写正文内容的方式统一结构和表述。以 `ALGORITHM.md`、`wpa_fixed` 默认实现和当前 `docx` 的章节组织为事实来源，输出一份完整可交付的 Word 文档。

**Tech Stack:** `python-docx`, repo local Markdown/docs, `wpa_fixed` Python implementation

---

### Task 1: Gather Source Structure

**Files:**
- Read: `docs/WPA_Algorithm_Spec_For_Algo_and_DIC.docx`
- Read: `ALGORITHM.md`
- Read: `wpa_fixed/config.py`
- Read: `wpa_fixed/core.py`

**Step 1: Extract current DOCX structure**

Run:

```bash
python - <<'PY'
from docx import Document
doc = Document('docs/WPA_Algorithm_Spec_For_Algo_and_DIC.docx')
for p in doc.paragraphs:
    if p.text.strip():
        print(p.style.name, p.text.strip())
PY
```

**Step 2: Map missing detail from ALGORITHM.md**

Identify content gaps in:
- mathematical principle
- fixed-point runtime update path
- hardware resource split

### Task 2: Rebuild DOCX Content Model

**Files:**
- Modify: `docs/WPA_Algorithm_Spec_For_Algo_and_DIC.docx`

**Step 1: Define rebuilt section content**

Prepare rewritten content for:
- Chapter 1-3 overview sections
- Chapter 4 detailed math principles
- Chapter 5 fixed-point implementation
- Chapter 6 hardware resources
- Chapter 7-10 validation and references

**Step 2: Define tables**

Prepare tables for:
- symbols
- static storage
- key bit widths
- per-stage operation counts
- warm/cool examples

### Task 3: Implement DOCX Rewrite

**Files:**
- Modify: `docs/WPA_Algorithm_Spec_For_Algo_and_DIC.docx`

**Step 1: Write rewrite script inline and execute**

Use `python-docx` to:
- create a fresh `Document`
- write headings, paragraphs, and tables
- save back to the target `docx`

**Step 2: Preserve target file path**

Output must overwrite:

```bash
docs/WPA_Algorithm_Spec_For_Algo_and_DIC.docx
```

### Task 4: Verify Document Structure

**Files:**
- Verify: `docs/WPA_Algorithm_Spec_For_Algo_and_DIC.docx`

**Step 1: Re-read DOCX**

Run:

```bash
python - <<'PY'
from docx import Document
doc = Document('docs/WPA_Algorithm_Spec_For_Algo_and_DIC.docx')
print(len(doc.paragraphs), len(doc.tables))
PY
```

Expected:
- headings present
- tables present
- no parse failure

**Step 2: Inspect key chapter headings**

Confirm chapters 4, 5, 6 contain the rebuilt detailed content.

### Task 5: Optional Visual Render Check

**Files:**
- Verify: `docs/WPA_Algorithm_Spec_For_Algo_and_DIC.docx`

**Step 1: Try render if tooling exists**

Run one of:

```bash
python3 scripts/render_docx.py docs/WPA_Algorithm_Spec_For_Algo_and_DIC.docx --output_dir tmp/docs/wpa_docx_pages
```

or

```bash
soffice --headless --convert-to pdf --outdir tmp/docs docs/WPA_Algorithm_Spec_For_Algo_and_DIC.docx
```

**Step 2: If rendering unavailable**

Report that structural verification passed but visual review was not possible in this environment.
