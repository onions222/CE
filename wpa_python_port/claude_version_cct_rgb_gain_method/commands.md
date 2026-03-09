
#### 浮点版本
python -m wpa.cli --in ../sky.jpg --out b.jpg --wa-sel 10

#### 默认精度 (frac_bits=10)
python -m wpa_fixed.cli --in ../sky.jpg --out out.jpg --wa-sel 20

#### 高精度
python -m wpa_fixed.cli --in ../sky.jpg --out out.jpg --wa-sel 20 --frac-bits 16


---
#### 验证线性插值有效性：
python -m wpa_fixed.anchor_eval --coeff-frac-bits 8
python -m wpa_fixed.anchor_eval --coeff-frac-bits 10
