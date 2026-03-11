function gains = wpa_fixed_runtime_bin_gains(cfg, wa_sel)
%WPA_FIXED_RUNTIME_BIN_GAINS Expand runtime 12x3 bin gains.
% Output format: Nx3 uint16 in UQ1.coeff_frac_bits.

wa = min(max(uint16(wa_sel), uint16(0)), uint16(127));
one = uint16(cfg.COEFF_ONE);
base = wpa_fixed_runtime_base_gain(cfg, wa);
atten_q = uint16(cfg.atten_q_lut_fixed(:));
delta = int32(base) - int32(one);
gains = zeros(numel(atten_q), 3, 'uint16');
one_s32 = int32(one);
for i = 1:numel(atten_q)
    row = one_s32 + bitshift(int32(int64(atten_q(i)) .* int64(delta) + int64(cfg.COEFF_HALF)), -cfg.coeff_frac_bits);
    gains(i, :) = uint16(row);
end
if wa < 64
    alpha_num = uint32(64 - wa);
    target_caps = uint16(cfg.warm_highlight_green_caps_fixed(:));
    delta_caps = uint32(one) - uint32(target_caps);
    caps = uint16(uint32(one) - bitshift(alpha_num .* delta_caps + uint32(32), -6));
    high_idx = target_caps < one;
    gains(high_idx, 2) = min(gains(high_idx, 2), caps(high_idx));
end
if wa > 64
    alpha_num = uint32(wa - 64);
    green_targets = uint16(cfg.cool_highlight_green_caps_fixed(:));
    green_delta = uint32(one) - uint32(green_targets);
    green_caps = uint16(uint32(one) - idivide(alpha_num .* green_delta + uint32(31), uint32(63), 'floor'));
    green_idx = green_targets < one;
    gains(green_idx, 2) = min(gains(green_idx, 2), green_caps(green_idx));

    blue_targets = uint16(cfg.cool_highlight_blue_caps_fixed(:));
    blue_delta = uint32(blue_targets) - uint32(one);
    blue_caps = uint16(uint32(one) + idivide(alpha_num .* blue_delta + uint32(31), uint32(63), 'floor'));
    blue_idx = blue_targets > one;
    gains(blue_idx, 3) = min(gains(blue_idx, 3), blue_caps(blue_idx));
end
if wa == 64
    gains(:, :) = one;
end
end
