function y = wpa_fixed_luma_proxy_u8(rgb_u8)
%WPA_FIXED_LUMA_PROXY_U8 Compute Y = (R + 2G + B) >> 2 in uint8 scale.
% Runtime storage: uint8 input, uint16 accumulator, uint8 output.

r = uint16(rgb_u8(:, :, 1));
g = uint16(rgb_u8(:, :, 2));
b = uint16(rgb_u8(:, :, 3));
acc = r + bitshift(g, 1) + b;
y = uint8(bitshift(acc, -2));
end
