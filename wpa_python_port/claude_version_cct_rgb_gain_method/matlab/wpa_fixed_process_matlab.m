function out = wpa_fixed_process_matlab(img, cfg)
%WPA_FIXED_PROCESS_MATLAB MATLAB implementation aligned with Python wpa_fixed.

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
pixel_fix = int32(min(max(round(double(linear_f) * double(cfg.ONE)), 0), double(cfg.ONE)));

if strcmp(char(cfg.luma_domain), 'gamma')
    luma_u8 = wpa_fixed_luma_proxy_u8(img);
else
    r_fix = int32(pixel_fix(:, :, 1));
    g_fix = int32(pixel_fix(:, :, 2));
    b_fix = int32(pixel_fix(:, :, 3));
    luma_fix = bitshift(r_fix + 2 .* g_fix + b_fix, -2);
    luma_u8 = uint8(min(max(bitshift(luma_fix .* 255 + cfg.HALF, -cfg.frac_bits), 0), 255));
end

gains_wa = wpa_fixed_runtime_bin_gains(cfg, cfg.wa_sel);
gain = wpa_fixed_interpolate_gains(luma_u8, gains_wa, cfg.luma_nodes, cfg.bin_interp, cfg.frac_bits);

adjusted = int32(bitshift(int64(pixel_fix) .* int64(gain) + int64(cfg.COEFF_HALF), -cfg.coeff_frac_bits));

if cfg.sat_en
    w = wpa_fixed_sat_weight(img, cfg.sat_s0, cfg.sat_s1, cfg.frac_bits);
    w3 = repmat(w, [1, 1, 3]);
    delta = int32(adjusted - pixel_fix);
    adjusted = pixel_fix + int32(bitshift(int64(w3) .* int64(delta) + int64(cfg.HALF), -cfg.frac_bits));
end

adjusted = int32(min(max(adjusted, 0), cfg.ONE));
linear_out = single(double(adjusted) ./ double(cfg.ONE));
encoded = wpa_fixed_engamma(linear_out, cfg.gamma_mode, cfg.gamma_power);
out = uint8(min(max(round(double(encoded) * 255.0), 0), 255));
end
