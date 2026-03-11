function out = wpa_fixed_process_matlab(img, cfg)
%WPA_FIXED_PROCESS_MATLAB MATLAB implementation aligned with Python wpa_fixed.
% Gamma stages stay float. Main fixed path uses:
%   pixel_uq0f      : uint16  (Q0.frac_bits)
%   gain_uq1f       : uint16  (UQ1.coeff_frac_bits)
%   mul_acc_u32     : uint32  (pixel * gain accumulation)
%   sat_delta_s32   : int32   (only where signed difference is required)

if nargin < 2 || isempty(cfg)
    cfg = wpa_fixed_config();
end

if ~isa(img, 'uint8') || ndims(img) ~= 3 || size(img, 3) ~= 3
    error('Input must be HxWx3 uint8.');
end

if ~cfg.wa_en || cfg.wa_sel == 64
    out = img;
    return;
end

linear_f = wpa_fixed_degamma(img, cfg.gamma_mode, cfg.gamma_power);
pixel_uq0f = uint16(min(max(round(double(linear_f) * double(cfg.ONE)), 0), double(cfg.ONE)));

if strcmp(char(cfg.luma_domain), 'gamma')
    luma_u8 = wpa_fixed_luma_proxy_u8(img);
else
    r_uq0f = uint32(pixel_uq0f(:, :, 1));
    g_uq0f = uint32(pixel_uq0f(:, :, 2));
    b_uq0f = uint32(pixel_uq0f(:, :, 3));
    luma_uq0f = bitshift(r_uq0f + bitshift(g_uq0f, 1) + b_uq0f, -2);
    luma_u8 = uint8(min(bitshift(luma_uq0f .* uint32(255) + uint32(cfg.HALF), -cfg.frac_bits), uint32(255)));
end

gains_wa = wpa_fixed_runtime_bin_gains(cfg, cfg.wa_sel);
gain_uq1f = wpa_fixed_interpolate_gains(luma_u8, gains_wa, cfg.luma_nodes, cfg.bin_interp, cfg.frac_bits);

mul_acc_u32 = uint32(pixel_uq0f) .* uint32(gain_uq1f) + uint32(cfg.COEFF_HALF);
adjusted_uq0f = uint16(bitshift(mul_acc_u32, -cfg.coeff_frac_bits));

if cfg.sat_en
    sat_w_uq0f = wpa_fixed_sat_weight(img, cfg.sat_s0, cfg.sat_s1, cfg.frac_bits);
    sat_w3_uq0f = repmat(sat_w_uq0f, [1, 1, 3]);
    sat_delta_s32 = int32(adjusted_uq0f) - int32(pixel_uq0f);
    adjusted_uq0f = uint16(int32(pixel_uq0f) + bitshift(int32(int64(sat_w3_uq0f) .* int64(sat_delta_s32) + int64(cfg.HALF)), -cfg.frac_bits));
end

adjusted_uq0f = uint16(min(adjusted_uq0f, uint16(cfg.ONE)));
linear_out = single(double(adjusted_uq0f) ./ double(cfg.ONE));
encoded = wpa_fixed_engamma(linear_out, cfg.gamma_mode, cfg.gamma_power);
out = uint8(min(max(round(double(encoded) * 255.0), 0), 255));
end
