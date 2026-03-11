function gains = wpa_fixed_runtime_bin_gains(cfg, wa_sel)
%WPA_FIXED_RUNTIME_BIN_GAINS Expand runtime 12x3 bin gains.

wa = min(max(int32(wa_sel), int32(0)), int32(127));
one = int32(cfg.COEFF_ONE);
base = wpa_fixed_runtime_base_gain(cfg, wa);
atten_q = int32(cfg.atten_q_lut_fixed(:));
delta = int32(base - one);
gains = zeros(numel(atten_q), 3, 'int32');
for i = 1:numel(atten_q)
    row = one + bitshift(int32(int64(atten_q(i)) .* int64(delta) + int64(cfg.COEFF_HALF)), -cfg.coeff_frac_bits);
    gains(i, :) = int32(row);
end
if wa < 64
    green_cap = one - int32(round((double(64 - wa) / 64.0) * 0.06 * double(one)));
    high_idx = cfg.luma_nodes(:) >= 239;
    gains(high_idx, 2) = min(gains(high_idx, 2), green_cap);
end
if wa > 64
    alpha = double(wa - 64) / 63.0;
    green_deltas = [0.02; 0.025; 0.03; 0.035];
    blue_deltas = [0.15; 0.12; 0.08; 0.06];
    green_caps = one - int32(round(alpha * green_deltas * double(one)));
    blue_caps = one + int32(round(alpha * blue_deltas * double(one)));
    gains(end-3:end, 2) = min(gains(end-3:end, 2), green_caps);
    gains(end-3:end, 3) = min(gains(end-3:end, 3), blue_caps);
end
if wa == 64
    gains(:, :) = one;
end
end
