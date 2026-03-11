# MATLAB Fixed WPA

MATLAB 版本用于对齐 Python `wpa_fixed`，并且显式采用“raw code + scale factor”的定点表达方式，方便 debug 和硬件资源统计。

## 文件结构

- `wpa_fixed_config.m`
  - 生成配置、anchor gain raw code、atten raw code、warm/cool highlight tail raw code
- `wpa_fixed_process_matlab.m`
  - 主处理流程：degamma、像素 raw code、luma、gain 插值、乘法、饱和度保护、engamma
- `wpa_fixed_runtime_bin_gains.m`
  - 根据 `WA_SEL` 展开当前 `12x3` 增益码值表
- `run_wpa_fixed_image.m`
  - 单张图片入口
- `run_wpa_fixed_folder.m`
  - 文件夹批处理入口，只扫描当前文件夹，不递归子目录
- `validate_wpa_fixed_against_python.m`
  - 与 Python golden outputs 做逐图对齐验证

## 定点语义

MATLAB 代码里的 `uint16` / `uint32` / `int32` 只是存储容器，不代表硬件最终位宽。

真正的定点语义按 raw code 理解：

- 像素码值：`Q0.frac_bits`
- 增益码值：`UQ1.coeff_frac_bits`
- 真实数值 = `raw_code / scale_factor`

例如当 `coeff_frac_bits = 8` 时：

- `COEFF_ONE = 2^8 = 256`
- `0.1 -> round(0.1 * 256) = 26`
- `0.2 -> round(0.2 * 256) = 51`

实际 debug 时可以直接理解为：

- `26 / 256 ~= 0.1015625`
- `51 / 256 ~= 0.19921875`

这和 Python `wpa_fixed` 的表达方式是一致的。

## 关键位宽

默认配置下：

- `frac_bits = 10`
- `coeff_frac_bits = 8`
- `pixel_bits = 11`
- `coeff_bits = 9`
- `mul_bits = 20`

这些位宽和 Python 侧 `wpa_fixed.hw_stats` 的资源统计口径一致。

## 最常用流程

先进入项目根目录，在 MATLAB 中执行：

```matlab
addpath('matlab');
```

### 单张图片

```matlab
run_wpa_fixed_image('input.png', 'output.png', 'wa_sel', 0, 'coeff_frac_bits', 8);
```

只返回输出数组，不写文件：

```matlab
img = imread('input.png');
cfg = wpa_fixed_config('wa_sel', 127, 'coeff_frac_bits', 8);
out = wpa_fixed_process_matlab(img, cfg);
imshow(out);
```

### 文件夹批处理

批处理入口只读取一个文件夹下的所有图像文件，不递归子目录。
`run_wpa_fixed_folder.m` 现在是脚本，不是函数。推荐直接打开文件后修改顶部配置，再点击 MATLAB Run。

脚本顶部可以直接调整：

```matlab
input_dir = 'test_images/synthetic';
output_dir = 'outputs/matlab_batch_smoke';
wa_sel = 0;
coeff_frac_bits = 8;
frac_bits = 10;
```

脚本顶部还带了一段参数说明，建议直接按那一段修改。重点参数含义：

- `input_dir`
  - 输入图像文件夹，只扫描当前目录，不递归
- `output_dir`
  - 输出文件夹，不存在时自动创建
- `wa_sel`
  - 色温调节档位，范围 `0~127`
- `coeff_frac_bits`
  - 增益 raw code 小数位数，默认 `8`
- `frac_bits`
  - 像素 raw code 小数位数，默认 `10`

WA_SEL 常用预设：

- `0 = 最暖`
- `32 = 偏暖`
- `64 = 关闭调节`
- `96 = 偏冷`
- `127 = 最冷`

支持的扩展名：

- `.png`
- `.jpg`
- `.jpeg`
- `.bmp`
- `.tif`
- `.tiff`

输出文件名会附带 `wa` 信息，例如：

- `photo.png -> photo_wa0.png`
- `scene.jpg -> scene_wa127.jpg`

## 配置示例

默认配置：

```matlab
cfg = wpa_fixed_config();
```

指定 `wa_sel` 和定点精度：

```matlab
cfg = wpa_fixed_config('wa_sel', 20, 'coeff_frac_bits', 8);
```

`wa_sel = 64` 为 identity。

## 与 Python 对齐验证

先导出 Python golden cases：

```bash
python scripts/export_matlab_wpa_fixed_golden.py
```

然后在 MATLAB 中执行：

```matlab
addpath('matlab');
summary = validate_wpa_fixed_against_python('matlab/golden_cases');
```

当前建议始终以 Python fixed 版本作为行为基准，MATLAB 侧负责保持对齐。
