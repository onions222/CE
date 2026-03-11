# MATLAB Fixed WPA

MATLAB 版本用于对齐 Python `wpa_fixed`。

## 入口文件

- `wpa_fixed_config.m`
- `wpa_fixed_process_matlab.m`
- `run_wpa_fixed_image.m`
- `validate_wpa_fixed_against_python.m`

## 最常用流程

先进入项目根目录，在 MATLAB 中执行：

```matlab
addpath('matlab');
```

处理单张图片：

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

## 说明

- MATLAB 版本语义对齐 Python `wpa_fixed`
- gamma 路径使用浮点
- gain / bin / interpolation 主路径使用定点风格实现
- 当前建议以 Python fixed 版本作为基准，MATLAB 侧做行为对齐
