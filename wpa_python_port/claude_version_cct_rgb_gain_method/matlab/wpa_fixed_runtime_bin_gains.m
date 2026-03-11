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
if wa == 64
    gains(:, :) = one;
end
end
