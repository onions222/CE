# Commands

## 1. 创建测试图

生成全部 synthetic 测试图：

```bash
python scripts/generate_test_images.py
```

按 profile 构建测试集：

```bash
python scripts/build_test_set.py --profile core_visual
python scripts/build_test_set.py --profile release_visual
```

## 2. 单张/多张测试图输出

批量生成四联图：

```bash
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
img_path = root / 'test_images' / 'synthetic' / 'test_highlight.png'
out_path = root / 'test_images' / 'visualizations' / 'synthetic' / 'test_highlight_comparison.png'

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
