function gains = hw_fixed_runtime_bin_gains(cfg, wa_sel)
%HW_FIXED_RUNTIME_BIN_GAINS 由 WA_SEL 展开当前 runtime 12x3 增益表。
%
% 输出：
% - gains(i, c) 是 UQ1.coeff_frac_bits raw code
% - 真实增益 = gain_code / cfg.COEFF_ONE
% - 增益码值位宽 = cfg.coeff_bits
%
% 这个函数对应硬件里的“控制量更新路径”，不是每像素路径：
% 1. 先由 WA_SEL 在 3 个 anchor 上插值得到 base gain raw code
% 2. 再结合 12 个 atten_q 展开当前 runtime 12x3
% 3. 最后对高亮末端应用 warm/cool tail 修正

wa = min(max(round(wa_sel), 0), 127);
one = cfg.COEFF_ONE;
base = local_runtime_base_gain(cfg, wa);
atten_q = cfg.atten_q_lut_fixed(:);
delta = base - one;
gains = zeros(numel(atten_q), 3);

for i = 1:numel(atten_q)
    % 12 个亮度节点逐项展开：
    % gain_bin = 1 + atten * (base - 1)
    gains(i, :) = one + floor((atten_q(i) .* delta + cfg.COEFF_HALF) / (2 ^ cfg.coeff_frac_bits));
end

if wa < 64
    % warm 半区：
    % alpha_num 越大表示越接近最暖端，tail 修正越强。
    alpha_num = 64 - wa;

    red_targets = cfg.warm_highlight_red_caps_fixed(:);
    red_delta = red_targets - one;
    red_caps = one + floor((alpha_num .* red_delta + 32) / 64);
    red_idx = red_targets > one;
    gains(red_idx, 1) = min(gains(red_idx, 1), red_caps(red_idx));

    green_targets = cfg.warm_highlight_green_caps_fixed(:);
    green_delta = one - green_targets;
    green_caps = one - floor((alpha_num .* green_delta + 32) / 64);
    green_idx = green_targets < one;
    gains(green_idx, 2) = min(gains(green_idx, 2), green_caps(green_idx));

    blue_targets = cfg.warm_highlight_blue_floors_fixed(:);
    blue_delta = one - blue_targets;
    blue_floors = one - floor((alpha_num .* blue_delta + 32) / 64);
    blue_idx = blue_targets < one;
    gains(blue_idx, 3) = max(gains(blue_idx, 3), blue_floors(blue_idx));
end

if wa > 64
    % cool 半区：
    % alpha_num 越大表示越接近最冷端，tail 修正越强。
    alpha_num = wa - 64;

    green_targets = cfg.cool_highlight_green_caps_fixed(:);
    green_delta = one - green_targets;
    green_caps = one - floor((alpha_num .* green_delta + 31) / 63);
    green_idx = green_targets < one;
    gains(green_idx, 2) = min(gains(green_idx, 2), green_caps(green_idx));

    blue_targets = cfg.cool_highlight_blue_caps_fixed(:);
    blue_delta = blue_targets - one;
    blue_caps = one + floor((alpha_num .* blue_delta + 31) / 63);
    blue_idx = blue_targets > one;
    gains(blue_idx, 3) = min(gains(blue_idx, 3), blue_caps(blue_idx));
end

if wa == 64
    gains(:, :) = one;
end
end

function base = local_runtime_base_gain(cfg, wa_sel)
% base 是当前 WA_SEL 对应的 1x3 anchor 插值结果，单位仍然是增益 raw code。
% 位宽按 cfg.coeff_bits 理解。
wa = min(max(round(wa_sel), 0), 127);
warm = cfg.wa_base_gain_lut_fixed(1, :);
neutral = cfg.wa_base_gain_lut_fixed(2, :);
cool = cfg.wa_base_gain_lut_fixed(3, :);

if wa <= 64
    % warm -> neutral，除数固定为 64，方便硬件实现为固定移位/常数除法。
    base = warm + floor((wa .* (neutral - warm) + 32) / 64);
else
    % neutral -> cool，同样保持一段固定结构。
    base = neutral + floor(((wa - 64) .* (cool - neutral) + 32) / 64);
end
end
