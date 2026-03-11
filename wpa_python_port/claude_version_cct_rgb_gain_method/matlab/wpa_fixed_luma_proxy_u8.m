function y = wpa_fixed_luma_proxy_u8(rgb_u8)
%WPA_FIXED_LUMA_PROXY_U8 Compute Y = (R + 2G + B) >> 2 in uint8 scale.

r = int16(rgb_u8(:, :, 1));
g = int16(rgb_u8(:, :, 2));
b = int16(rgb_u8(:, :, 3));
y = uint8(bitshift(r + 2 .* g + b, -2));
end
