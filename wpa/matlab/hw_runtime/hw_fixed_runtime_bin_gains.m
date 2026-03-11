function gains = hw_fixed_runtime_bin_gains(cfg, wa_sel)
%HW_FIXED_RUNTIME_BIN_GAINS 由 WA_SEL 展开当前 runtime 12x3 增益表。
%
% 输出：
% - gains(i, c) 是 UQ1.coeff_frac_bits raw code
% - 真实增益 = gain_code / cfg.COEFF_ONE

wa = min(max(round(wa_sel), 0), 127);
one = cfg.COEFF_ONE;
base = local_runtime_base_gain(cfg, wa);
atten_q = cfg.atten_q_lut_fixed(:);
delta = base - one;
gains = zeros(numel(atten_q), 3);

for i = 1:numel(atten_q)
    gains(i, :) = one + floor((atten_q(i) .* delta + cfg.COEFF_HALF) / (2 ^ cfg.coeff_frac_bits));
end

if wa < 64
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
wa = min(max(round(wa_sel), 0), 127);
warm = cfg.wa_base_gain_lut_fixed(1, :);
neutral = cfg.wa_base_gain_lut_fixed(2, :);
cool = cfg.wa_base_gain_lut_fixed(3, :);

if wa <= 64
    base = warm + floor((wa .* (neutral - warm) + 32) / 64);
else
    base = neutral + floor(((wa - 64) .* (cool - neutral) + 32) / 64);
end
end
