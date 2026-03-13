# MATLAB HW Runtime

这是一套独立的 MATLAB 硬件仿真版本，目录内不依赖旧的 MATLAB fixed 实现。

核心结构严格按硬件交付口径组织：

- 只常驻 `3 anchor` gain
- 只常驻 `12 luma nodes`
- `WA_SEL` 更新时展开当前 `runtime 12x3` bin gain
- 每像素阶段只做亮度定位、12-bin 插值和 raw-code 乘法

## 文件

- `hw_fixed_config.m`
  - 构建 raw code 配置、3 个 anchor gain、12 个 atten/node 与 highlight tail 表
- `hw_fixed_runtime_bin_gains.m`
  - 由 `WA_SEL` 生成当前 `runtime 12x3` 增益表
- `hw_fixed_process_image.m`
  - 主像素路径：degamma、luma、gain 插值、乘法、engamma
- `run_hw_fixed_image.m`
  - 单张图片入口
- `run_hw_fixed_folder.m`
  - 文件夹批处理脚本，只扫描当前目录，不递归
- `validate_hw_fixed_against_python.m`
  - 用 Python golden cases 验证独立 hw_runtime 输出

## 数据口径

- 增益码值：`round(real_gain * 2^coeff_frac_bits)`
- 像素码值：`round(real_linear * 2^frac_bits)`
- 真实值理解为：
  - `gain = gain_code / 2^coeff_frac_bits`
  - `pixel = pixel_code / 2^frac_bits`

默认使用：

- `coeff_frac_bits = 8`
- `frac_bits = 8`
- `luma_nodes = [15, 31, 47, 63, 95, 127, 159, 191, 223, 239, 247, 255]`

对应位宽：

- `coeff_bits = 9 bit`
  - 因为 `UQ1.8`，即 `1` 个整数位 + `8` 个小数位
- `pixel_bits = 9 bit`
  - 因为 `Q0.8`，内部像素码值按 `frac_bits + 1`
- `mul_bits = 18 bit`
  - 因为乘法累加位宽按 `pixel_bits + coeff_bits`
- `luma_bits = 8 bit`
  - 亮度代理使用 `Y = (R + 2G + B) / 4` 后映射到 `0..255`

如果换成别的小数位数，对应关系也是一样的：

- `coeff_bits = coeff_frac_bits + 1`
- `pixel_bits = frac_bits + 1`
- `mul_bits = pixel_bits + coeff_bits`

小数编码示例：

- 当 `coeff_frac_bits = 8` 时，`scale factor = 256`
- `0.1 -> round(0.1 * 256) = 26`
- `0.2 -> round(0.2 * 256) = 51`
- 实际理解为：
  - `26 / 256 = 0.1015625`
  - `51 / 256 = 0.19921875`
- 例如某一路增益乘法可以理解成：
  - `out_code = round(in_code * 26 / 256)`

## 用法

先在 MATLAB 中执行：

```matlab
addpath('matlab/hw_runtime');
```

单图：

```matlab
run_hw_fixed_image('input.png', 'output.png', 'wa_sel', 0);
```

批处理：

- 打开 `run_hw_fixed_folder.m`
- 修改顶部 `input_dir`、`output_dir`、`wa_sel`
- 点击 MATLAB Run

## Visual Gates

`hw_runtime` 的正式测试目标是看算法效果，不是看和 Python 的逐像素数值一致。
推荐采用“人工主判 + 自动辅助”的三套 gate：

- `smoke_hw_visual`
  - 本地快速回归，只跑最敏感的中性 / 节点 / 暗部 / 高亮图
- `core_hw_visual`
  - 日常完整 synthetic 评估，覆盖 neutral / UI / highlight / saturation / skin / mixed-light
- `release_hw_visual`
  - `core_hw_visual` + 真实场景 sanity

常用 `WA_SEL` 检查档位：

- `0`
- `64`
- `127`

如果需要补看中间档位，再增加 `32` 和 `96`。

## Review Workflow

推荐工作流：

1. 在 MATLAB 中用 `run_hw_fixed_folder.m` 批量生成 `wa=0/64/127` 输出
2. 用 Python 脚本生成 comparison panels 和 review 模板
3. 人工填写 `review_sheet.csv`
4. 只有在人工判定 `Fail` 时，才运行 `validate_hw_fixed_against_python.m` 做定位

生成 review pack：

```bash
python scripts/generate_matlab_hw_review_pack.py \
  --profile smoke_hw_visual \
  --input-dir test_images/synthetic \
  --result-dir outputs/matlab_hw_runtime_smoke \
  --output-dir outputs/matlab_hw_runtime_smoke_review
```

输出内容：

- `comparison_panels/`
- `review_sheet.csv`
- `summary.md`

配套人工检查说明见：

- `validation/matlab_hw_runtime_review_checklist.md`

对齐验证：

```matlab
addpath('matlab/hw_runtime');
summary = validate_hw_fixed_against_python('matlab/golden_cases');
```

这里的 `validate_hw_fixed_against_python` 只作为调试工具，不作为正式放行 gate。
