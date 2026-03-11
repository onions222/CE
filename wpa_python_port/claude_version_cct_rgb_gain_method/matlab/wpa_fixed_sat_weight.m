function w = wpa_fixed_sat_weight(rgb_u8, s0, s1, frac_bits)
%WPA_FIXED_SAT_WEIGHT Fixed-point saturation protection weight.

r = int32(rgb_u8(:, :, 1));
g = int32(rgb_u8(:, :, 2));
b = int32(rgb_u8(:, :, 3));
s = abs(r - g) + abs(g - b) + abs(b - r);

one = bitshift(int32(1), frac_bits);
denom = max(int32(s1 - s0), int32(1));
numer = bitshift(int32(s1) - s, frac_bits) + bitshift(denom, -1);
w = min(max(idivide(numer, denom, 'floor'), int32(0)), one);
end
