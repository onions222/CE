function base = wpa_fixed_runtime_base_gain(cfg, wa_sel)
%WPA_FIXED_RUNTIME_BASE_GAIN Compute runtime 3-channel base gain.

wa = min(max(int32(wa_sel), int32(0)), int32(127));
warm = int32(cfg.wa_base_gain_lut_fixed(1, :));
neutral = int32(cfg.wa_base_gain_lut_fixed(2, :));
cool = int32(cfg.wa_base_gain_lut_fixed(3, :));

if wa <= 64
    num = wa;
    delta = neutral - warm;
    base = warm + bitshift(int32(num .* delta + 32), -6);
else
    num = wa - 64;
    delta = cool - neutral;
    base = neutral + bitshift(int32(num .* delta + 32), -6);
end
base = int32(base);
end
