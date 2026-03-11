function w = wpa_fixed_sat_weight(rgb_u8, s0, s1, frac_bits)
%WPA_FIXED_SAT_WEIGHT Fixed-point saturation protection weight.
% Runtime storage: input uint8, signed diffs int16, output weight uint16.

r = int16(rgb_u8(:, :, 1));
g = int16(rgb_u8(:, :, 2));
b = int16(rgb_u8(:, :, 3));
s = uint16(abs(r - g) + abs(g - b) + abs(b - r));

one = uint16(bitshift(uint16(1), frac_bits));
denom = max(uint16(s1 - s0), uint16(1));
numer = bitshift(uint32(max(int32(s1) - int32(s), 0)), frac_bits) + bitshift(uint32(denom), -1);
w = uint16(min(idivide(numer, uint32(denom), 'floor'), uint32(one)));
end
