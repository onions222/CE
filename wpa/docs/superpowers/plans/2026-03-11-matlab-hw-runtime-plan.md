# MATLAB HW Runtime Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `matlab/hw_runtime/` 下新增一套完全独立的 MATLAB 硬件仿真实现，只保留 3 个 anchor gain、12 个 luma node 与 runtime `12x3` bin gain 展开结构。

**Architecture:** 新目录不依赖现有 `matlab/wpa_fixed_*` 文件，独立提供 config、runtime gain 展开、像素处理和单图/文件夹入口。数值规则与当前 Python fixed 对齐，但表达口径保持 raw code + scale factor，不强调宿主整型位宽。

**Tech Stack:** MATLAB `.m` 文件、Python `pytest` 静态约束测试、MATLAB batch 运行验证

---

## Chunk 1: 目录与测试边界

### Task 1: 为独立目录写失败测试

**Files:**
- Modify: `tests/test_matlab_hw_style_runtime.py`
- Create: `matlab/hw_runtime/README.md`
- Create: `matlab/hw_runtime/hw_fixed_config.m`
- Create: `matlab/hw_runtime/hw_fixed_runtime_bin_gains.m`
- Create: `matlab/hw_runtime/hw_fixed_process_image.m`
- Create: `matlab/hw_runtime/run_hw_fixed_image.m`
- Create: `matlab/hw_runtime/run_hw_fixed_folder.m`

- [ ] **Step 1: 写失败测试**
- [ ] **Step 2: 跑 `python -m pytest tests/test_matlab_hw_style_runtime.py -q`，确认因目录/文件不存在而失败**

### Task 2: 限定独立实现边界

**Files:**
- Modify: `tests/test_matlab_hw_style_runtime.py`

- [ ] **Step 1: 在测试中要求新目录不引用 `wpa_fixed_` 旧实现**
- [ ] **Step 2: 在测试中要求新目录 README 明确写出 `3 anchor`、`12 luma nodes`、`runtime 12x3`**
- [ ] **Step 3: 跑测试确认失败点正确**

## Chunk 2: 独立硬件仿真实现

### Task 3: 新增独立 config/runtime/process

**Files:**
- Create: `matlab/hw_runtime/hw_fixed_config.m`
- Create: `matlab/hw_runtime/hw_fixed_runtime_bin_gains.m`
- Create: `matlab/hw_runtime/hw_fixed_process_image.m`

- [ ] **Step 1: 复制并改写现有 MATLAB fixed 逻辑，改成 `hw_fixed_*` 命名**
- [ ] **Step 2: 保持 raw code 语义，不在计算中显式写 `uint16/uint32/int32`**
- [ ] **Step 3: 只常驻 3 个 anchor gain + 12 点 atten/node + highlight tail 表**
- [ ] **Step 4: 跑测试，确认静态约束通过**

### Task 4: 新增独立入口与说明

**Files:**
- Create: `matlab/hw_runtime/run_hw_fixed_image.m`
- Create: `matlab/hw_runtime/run_hw_fixed_folder.m`
- Create: `matlab/hw_runtime/README.md`

- [ ] **Step 1: 提供单图入口**
- [ ] **Step 2: 提供脚本式文件夹入口**
- [ ] **Step 3: README 说明输入输出、WA_SEL、3 anchor/12 node 结构**
- [ ] **Step 4: 跑测试，确认 README/入口约束通过**

## Chunk 3: 验证

### Task 5: Python 测试与 MATLAB 冒烟

**Files:**
- Modify: `tests/test_matlab_hw_style_runtime.py`
- Create: `matlab/hw_runtime/*`

- [ ] **Step 1: 跑 `python -m pytest tests/test_matlab_hw_style_runtime.py -q`**
- [ ] **Step 2: 跑 `python -m pytest tests/test_fixed_coeff_lut.py tests/test_matlab_hw_style_runtime.py -q`**
- [ ] **Step 3: 用 MATLAB batch 跑 `run('matlab/hw_runtime/run_hw_fixed_folder.m')` 做冒烟**
- [ ] **Step 4: 确认输出目录生成图像**
