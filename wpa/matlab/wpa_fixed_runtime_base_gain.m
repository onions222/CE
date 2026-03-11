function base = wpa_fixed_runtime_base_gain(cfg, wa_sel)
%WPA_FIXED_RUNTIME_BASE_GAIN Compute runtime 3-channel base gain.
% Output format: 1x3 uint16 in UQ1.coeff_frac_bits.

wa = min(max(uint16(wa_sel), uint16(0)), uint16(127));
warm = uint16(cfg.wa_base_gain_lut_fixed(1, :));
neutral = uint16(cfg.wa_base_gain_lut_fixed(2, :));
cool = uint16(cfg.wa_base_gain_lut_fixed(3, :));

if wa <= 64
    num = uint32(wa);
    delta = uint32(neutral) - uint32(warm);
    base = uint16(uint32(warm) + bitshift(num .* delta + uint32(32), -6));
else
    num = int32(wa - 64);
    delta = int32(cool) - int32(neutral);
    base = uint16(int32(neutral) + bitshift(num .* delta + int32(32), -6));
end
end
