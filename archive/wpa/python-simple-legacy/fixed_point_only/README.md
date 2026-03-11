# Fixed-Point Only WPA

这个目录是独立的“纯定点/整数”版本，不依赖原有浮点流程。

## 特点

- 仅使用整数与定点（Q12）运算
- 仅使用整数与定点运算（支持可配置 `Q`，默认 `Q12`）
- 仅实现 `diag_rgb` 风格的冷暖调节
- 输入/输出均为 `uint8 RGB`

## 运行

```bash
python -m fixed_point_only.wpa_fixed.cli --in grass.jpg --out out_int.png --wa-sel 127
python -m fixed_point_only.wpa_fixed.cli --in grass.jpg --out out_q8.png --wa-sel 127 --q-bits 8
```

## 主要参数

- `--wa-en` / `--no-wa-en`：开关
- `--wa-sel`：`[0,127]`，64 恒等
- `--sat-s0-255`、`--sat-s1-255`：饱和保护阈值（整数，0~510）
- `--q-bits`：定点小数位数（建议 8 或 12）
- `--warm-strength-q`、`--cool-strength-q`：强度（当前 Q 域，`qone=1<<q_bits`）
- `--report`：输出量化统计

## 说明

- 这是独立实现，不会改动 `wpa/` 原有代码路径。
- 该版本为了“纯整数”目标，采用 gamma 域整数近似处理，不使用浮点 gamma 变换。
