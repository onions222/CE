function out = hw_fixed_process_image(img, cfg)
%HW_FIXED_PROCESS_IMAGE 独立硬件仿真像素路径。
%
% 关键中间量：
% - pixel_code      : Q0.frac_bits 像素码值，位宽 = cfg.pixel_bits
% - gain_code       : UQ1.coeff_frac_bits 增益码值，位宽 = cfg.coeff_bits
% - mul_acc_code    : 乘法累加码值，位宽 = cfg.mul_bits
% - sat_delta_code  : 饱和度保护差值码值
%
% 这一路径只使用 3 anchor + 12 luma nodes + runtime 12x3 结构。

if nargin < 2 || isempty(cfg)
    cfg = hw_fixed_config();
end

if ~isa(img, 'uint8') || ndims(img) ~= 3 || size(img, 3) ~= 3
    error('Input must be HxWx3 uint8.');
end

if ~cfg.wa_en || cfg.wa_sel == 64
    out = img;
    return;
end

linear_f = local_degamma(img, cfg.gamma_mode, cfg.gamma_power);
pixel_code = min(max(round(double(linear_f) * cfg.ONE), 0), cfg.ONE);

if strcmp(char(cfg.luma_domain), 'gamma')
    luma_u8 = local_luma_proxy_u8(img);
else
    r_code = pixel_code(:, :, 1);
    g_code = pixel_code(:, :, 2);
    b_code = pixel_code(:, :, 3);
    luma_code = floor((r_code + 2 .* g_code + b_code) / 4);
    luma_u8 = min(floor((luma_code .* 255 + cfg.HALF) / (2 ^ cfg.frac_bits)), 255);
end

gain_table_codes = hw_fixed_runtime_bin_gains(cfg, cfg.wa_sel);
gain_code = local_interpolate_gains(luma_u8, gain_table_codes, cfg.luma_nodes, cfg.bin_interp, cfg.frac_bits);

mul_acc_code = pixel_code .* gain_code + cfg.COEFF_HALF;
adjusted_code = floor(mul_acc_code / (2 ^ cfg.coeff_frac_bits));

if cfg.sat_en
    sat_weight_code = local_sat_weight(img, cfg.sat_s0, cfg.sat_s1, cfg.frac_bits);
    sat_weight_code3 = repmat(sat_weight_code, [1, 1, 3]);
    sat_delta_code = adjusted_code - pixel_code;
    adjusted_code = pixel_code + floor((sat_weight_code3 .* sat_delta_code + cfg.HALF) / (2 ^ cfg.frac_bits));
end

adjusted_code = min(max(adjusted_code, 0), cfg.ONE);
linear_out = single(adjusted_code ./ cfg.ONE);
encoded = local_engamma(linear_out, cfg.gamma_mode, cfg.gamma_power);
out = uint8(min(max(round(double(encoded) * 255.0), 0), 255));
end

function linear = local_degamma(img, mode, gamma_power)
if nargin < 3
    gamma_power = 2.2;
end

if isa(img, 'uint8')
    x = single(img) ./ 255.0;
else
    x = single(img);
end

switch char(mode)
    case 'none'
        linear = x;
    case 'srgb'
        linear = single(zeros(size(x), 'single'));
        mask = x <= 0.04045;
        linear(mask) = x(mask) ./ 12.92;
        linear(~mask) = ((x(~mask) + 0.055) ./ 1.055) .^ 2.4;
    case 'power'
        linear = max(min(x, 1.0), 0.0) .^ gamma_power;
    otherwise
        error('Unknown gamma mode: %s', char(mode));
end
end

function encoded = local_engamma(img_linear, mode, gamma_power)
if nargin < 3
    gamma_power = 2.2;
end

x = single(img_linear);

switch char(mode)
    case 'none'
        encoded = x;
    case 'srgb'
        encoded = single(zeros(size(x), 'single'));
        mask = x <= 0.0031308;
        encoded(mask) = 12.92 .* x(mask);
        encoded(~mask) = 1.055 .* (max(x(~mask), 0.0) .^ (1.0 / 2.4)) - 0.055;
    case 'power'
        encoded = max(min(x, 1.0), 0.0) .^ (1.0 / gamma_power);
    otherwise
        error('Unknown gamma mode: %s', char(mode));
end
end

function y = local_luma_proxy_u8(rgb_u8)
r = double(rgb_u8(:, :, 1));
g = double(rgb_u8(:, :, 2));
b = double(rgb_u8(:, :, 3));
y = floor((r + 2 .* g + b) / 4);
end

function gain = local_interpolate_gains(luma_u8, gains_table, luma_nodes, interp, interp_bits)
if nargin < 4
    interp = true;
end
if nargin < 5
    interp_bits = 10;
end

nodes = luma_nodes(:)';
y = luma_u8;
[h, w] = size(y);
gain = zeros(h, w, 3);
half_interp = 2 ^ (interp_bits - 1);

for row = 1:h
    for col = 1:w
        yc = y(row, col);
        idx_hi = find(nodes >= yc, 1, 'first');
        if isempty(idx_hi)
            idx_hi = numel(nodes);
        end
        if idx_hi == 1
            idx_lo = 1;
            idx_hi = min(2, numel(nodes));
        else
            idx_lo = idx_hi - 1;
        end

        if ~interp || idx_lo == idx_hi
            gain(row, col, :) = gains_table(idx_lo, :);
            continue;
        end

        node_lo = nodes(idx_lo);
        node_hi = nodes(idx_hi);
        span = max(node_hi - node_lo, 1);
        numer = (yc - node_lo) * (2 ^ interp_bits) + floor(span / 2);
        t_code = min(floor(numer / span), 2 ^ interp_bits);

        g_lo = gains_table(idx_lo, :);
        g_hi = gains_table(idx_hi, :);
        gain(row, col, :) = g_lo + floor((t_code .* (g_hi - g_lo) + half_interp) / (2 ^ interp_bits));
    end
end
end

function w = local_sat_weight(rgb_u8, s0, s1, frac_bits)
r = double(rgb_u8(:, :, 1));
g = double(rgb_u8(:, :, 2));
b = double(rgb_u8(:, :, 3));
s = abs(r - g) + abs(g - b) + abs(b - r);

one = 2 ^ frac_bits;
denom = max(s1 - s0, 1);
numer = max(s1 - s, 0) * one + floor(denom / 2);
w = min(floor(numer / denom), one);
end
