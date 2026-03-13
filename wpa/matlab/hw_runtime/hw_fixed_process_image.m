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
% 运行顺序对应硬件理解：
% 1. gamma 域输入先转线性域浮点
% 2. 量化成 Q0.8 pixel raw code
% 3. 计算 8bit 亮度代理 Y
% 4. 用 Y 在 runtime 12x3 上做节点插值，得到 gain raw code
% 5. 做 pixel_code * gain_code / COEFF_ONE
% 6. 回到浮点线性域，再做 engamma 输出到 uint8

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
% pixel_code 是像素内部 raw code。
% 默认 frac_bits = 8，因此 1.0 <-> 256，位宽按 9 bit 理解。
pixel_code = min(max(round(double(linear_f) * cfg.ONE), 0), cfg.ONE);
gate_luma_code = floor((pixel_code(:, :, 1) + 2 .* pixel_code(:, :, 2) + pixel_code(:, :, 3)) / 4);

if strcmp(char(cfg.luma_domain), 'gamma')
    % gamma 域亮度代理直接基于输入 8bit RGB 计算。
    luma_u8 = local_luma_proxy_u8(img);
else
    r_code = pixel_code(:, :, 1);
    g_code = pixel_code(:, :, 2);
    b_code = pixel_code(:, :, 3);
    % luma_code 仍在 pixel raw code 域，位宽近似为 cfg.pixel_bits。
    luma_code = gate_luma_code;
    luma_u8 = min(floor((luma_code .* 255 + cfg.HALF) / (2 ^ cfg.frac_bits)), 255);
end

gain_table_codes = hw_fixed_runtime_bin_gains(cfg, cfg.wa_sel);
% gain_code 是每像素插值得到的增益 raw code，位宽按 cfg.coeff_bits 理解。
gain_code = local_interpolate_gains(luma_u8, gain_table_codes, cfg.luma_nodes, cfg.bin_interp, cfg.frac_bits);

% mul_acc_code 是像素码值与增益码值的乘法累加结果，位宽按 cfg.mul_bits 理解。
mul_acc_code = pixel_code .* gain_code + cfg.COEFF_HALF;
adjusted_code = floor(mul_acc_code / (2 ^ cfg.coeff_frac_bits));

if cfg.sat_en
    % sat_weight_code 与 pixel_code 处于同一小数位域。
    sat_weight_code = local_sat_weight(img, cfg.sat_s0, cfg.sat_s1, cfg.frac_bits);
    sat_weight_code3 = repmat(sat_weight_code, [1, 1, 3]);
    sat_delta_code = adjusted_code - pixel_code;
    adjusted_code = pixel_code + floor((sat_weight_code3 .* sat_delta_code + cfg.HALF) / (2 ^ cfg.frac_bits));
end

adjusted_code = min(max(adjusted_code, 0), cfg.ONE);
gated_code = adjusted_code;
if cfg.low_luma_gate_en
    bypass_code = round(cfg.low_luma_bypass_code);
    blend_end_code = round(cfg.low_luma_blend_end_code);
    bypass_mask = gate_luma_code <= bypass_code;
    if any(bypass_mask(:))
        for c = 1:3
            plane = gated_code(:, :, c);
            src = pixel_code(:, :, c);
            plane(bypass_mask) = src(bypass_mask);
            gated_code(:, :, c) = plane;
        end
    end

    blend_mask = gate_luma_code > bypass_code & gate_luma_code < blend_end_code;
    if any(blend_mask(:))
        numer = double(gate_luma_code(blend_mask)) - double(bypass_code);
        denom = double(blend_end_code - bypass_code);
        for c = 1:3
            src = double(pixel_code(:, :, c));
            dst = double(adjusted_code(:, :, c));
            blended = floor((src(blend_mask) .* (denom - numer) + dst(blend_mask) .* numer + floor(denom / 2)) ./ denom);
            plane = gated_code(:, :, c);
            plane(blend_mask) = min(max(blended, 0), cfg.ONE);
            gated_code(:, :, c) = plane;
        end
    end
end

linear_out = single(gated_code ./ cfg.ONE);
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
% 亮度代理直接使用硬件友好的公式：
% Y = floor((R + 2G + B) / 4)
% 输出位宽固定为 8 bit。
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
        % t_code 是节点间插值系数的 raw code，位宽约为 interp_bits + 1。
        span = max(node_hi - node_lo, 1);
        numer = (yc - node_lo) * (2 ^ interp_bits) + floor(span / 2);
        t_code = min(floor(numer / span), 2 ^ interp_bits);

        g_lo = gains_table(idx_lo, :);
        g_hi = gains_table(idx_hi, :);
        % 对 12 个亮度节点之间的 gain 做逐像素线性插值。
        gain(row, col, :) = g_lo + floor((t_code .* (g_hi - g_lo) + half_interp) / (2 ^ interp_bits));
    end
end
end

function w = local_sat_weight(rgb_u8, s0, s1, frac_bits)
% 饱和度保护权重：
% - w 的真实范围是 [0, 1]
% - raw code 位宽按 pixel_bits 理解
% - 当 sat_en = false 时，主路径不会进入这里
r = double(rgb_u8(:, :, 1));
g = double(rgb_u8(:, :, 2));
b = double(rgb_u8(:, :, 3));
s = abs(r - g) + abs(g - b) + abs(b - r);

one = 2 ^ frac_bits;
denom = max(s1 - s0, 1);
numer = max(s1 - s, 0) * one + floor(denom / 2);
w = min(floor(numer / denom), one);
end
