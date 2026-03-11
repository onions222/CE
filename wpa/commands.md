# Commands

## 1. 创建测试图

生成全部 synthetic 测试图（输出到 `tests/images/`，主要用于单独查看生成器，不是四联图脚本的默认输入目录）：

```bash
python scripts/generate_test_images.py
```

按 profile 构建测试集（四联图脚本默认读取这里的输入）：

```bash
python scripts/build_test_set.py --profile core_visual
python scripts/build_test_set.py --profile release_visual
```

## 2. 单张/多张测试图输出

说明：

- 仓库默认不提交四联图产物；`test_images/visualizations/` 只有在你运行下面命令后才会出现。
- 如果你只运行了 `python scripts/generate_test_images.py`，图会在 `tests/images/` 下，`generate_comparison_visuals.py` 不会自动读取它们。
- 先执行 `python scripts/build_test_set.py --profile core_visual`，再执行四联图生成命令。

批量生成四联图：

```bash
python scripts/build_test_set.py --profile core_visual
python scripts/generate_comparison_visuals.py
```

输出位置：

```bash
test_images/visualizations/
```

对单张图片生成四联图：

```bash
python - <<'PY'
from pathlib import Path
from PIL import Image
from scripts.generate_comparison_visuals import render_comparison_panel

root = Path('.')
img_path = root / 'test_images' / 'synthetic' / '12_specular_clip_chart.png'
out_path = root / 'test_images' / 'visualizations' / 'synthetic' / '12_specular_clip_chart_comparison.png'

image = Image.open(img_path).convert('RGB')
panel, _ = render_comparison_panel(image, image_name=img_path.name)
out_path.parent.mkdir(parents=True, exist_ok=True)
panel.save(out_path)
print(out_path)
PY
```

## 3. 如何运行 WPA 算法

浮点版：

```bash
python -m wpa.cli --in input.jpg --out output.jpg --wa-sel 10
python -m wpa.cli --in input.jpg --out output.jpg --wa-sel 64
python -m wpa.cli --in input.jpg --out output.jpg --wa-sel 120
```

定点版：

```bash
python -m wpa_fixed.cli --in input.jpg --out output.jpg --wa-sel 20
python -m wpa_fixed.cli --in input.jpg --out output.jpg --wa-sel 20 --frac-bits 16
```

验证定点 anchor interpolation：

```bash
python -m wpa_fixed.anchor_eval --coeff-frac-bits 8
python -m wpa_fixed.anchor_eval --coeff-frac-bits 10
```
