function cfg = hw_fixed_config(varargin)
%HW_FIXED_CONFIG 构建独立硬件仿真配置。
%
% 设计口径：
% 1. 全部定点量按 raw code 整数码值理解；
% 2. 真实数值 = raw_code / scale_factor；
% 3. 常驻数据只保留 3 个 anchor gain、12 个 luma node、12 点 atten 与 tail 表。

cfg = struct( ...
    'frac_bits', 8, ...
    'coeff_frac_bits', 8, ...
    'wa_en', true, ...
    'wa_sel', 64, ...
    'gamma_mode', 'srgb', ...
    'gamma_power', 2.2, ...
    'luma_nodes', [15 31 47 63 95 127 159 191 223 239 247 255], ...
    'bin_interp', true, ...
    'luma_domain', 'gamma', ...
    'sat_en', false, ...
    'sat_s0', 100, ...
    'sat_s1', 500, ...
    'cct_warm_k', 3000.0, ...
    'cct_neutral_k', 6500.0, ...
    'cct_cool_k', 9300.0, ...
    'cct_xy_split_k', 4000.0, ...
    'cct_xy_blend_half_width_k', 100.0, ...
    'wa_base_gain_lut_fixed', [], ...
    'atten_q_lut_fixed', [], ...
    'warm_highlight_red_caps_fixed', [], ...
    'warm_highlight_green_caps_fixed', [], ...
    'warm_highlight_blue_floors_fixed', [], ...
    'cool_highlight_green_caps_fixed', [], ...
    'cool_highlight_blue_caps_fixed', [], ...
    'pixel_bits', [], ...
    'coeff_bits', [], ...
    'luma_bits', 8, ...
    'sat_bits', 10, ...
    'mul_bits', []);

cfg = local_apply_overrides(cfg, varargin{:});

if ~ismember(cfg.coeff_frac_bits, [8 10])
    error('coeff_frac_bits must be 8 or 10');
end

cfg.ONE = 2 ^ cfg.frac_bits;
cfg.HALF = 2 ^ (cfg.frac_bits - 1);
cfg.COEFF_ONE = 2 ^ cfg.coeff_frac_bits;
cfg.COEFF_HALF = 2 ^ (cfg.coeff_frac_bits - 1);
cfg.pixel_bits = cfg.frac_bits + 1;
cfg.coeff_bits = cfg.coeff_frac_bits + 1;
cfg.mul_bits = cfg.pixel_bits + cfg.coeff_bits;

if isempty(cfg.wa_base_gain_lut_fixed)
    cfg.wa_base_gain_lut_fixed = local_build_wa_base_gain_lut_fixed(cfg);
else
    cfg.wa_base_gain_lut_fixed = round(cfg.wa_base_gain_lut_fixed);
end

if isempty(cfg.atten_q_lut_fixed)
    atten = zeros(numel(cfg.luma_nodes), 1);
    for i = 1:numel(cfg.luma_nodes)
        atten(i) = local_atten_curve(cfg.luma_nodes(i));
    end
    cfg.atten_q_lut_fixed = round(atten * cfg.COEFF_ONE);
else
    cfg.atten_q_lut_fixed = round(cfg.atten_q_lut_fixed);
end

if isempty(cfg.warm_highlight_red_caps_fixed)
    cfg.warm_highlight_red_caps_fixed = local_build_warm_highlight_red_caps_fixed(cfg);
else
    cfg.warm_highlight_red_caps_fixed = round(cfg.warm_highlight_red_caps_fixed);
end

if isempty(cfg.warm_highlight_green_caps_fixed)
    cfg.warm_highlight_green_caps_fixed = local_build_warm_highlight_green_caps_fixed(cfg);
else
    cfg.warm_highlight_green_caps_fixed = round(cfg.warm_highlight_green_caps_fixed);
end

if isempty(cfg.warm_highlight_blue_floors_fixed)
    cfg.warm_highlight_blue_floors_fixed = local_build_warm_highlight_blue_floors_fixed(cfg);
else
    cfg.warm_highlight_blue_floors_fixed = round(cfg.warm_highlight_blue_floors_fixed);
end

if isempty(cfg.cool_highlight_green_caps_fixed)
    cfg.cool_highlight_green_caps_fixed = local_build_cool_highlight_green_caps_fixed(cfg);
else
    cfg.cool_highlight_green_caps_fixed = round(cfg.cool_highlight_green_caps_fixed);
end

if isempty(cfg.cool_highlight_blue_caps_fixed)
    cfg.cool_highlight_blue_caps_fixed = local_build_cool_highlight_blue_caps_fixed(cfg);
else
    cfg.cool_highlight_blue_caps_fixed = round(cfg.cool_highlight_blue_caps_fixed);
end
end

function cfg = local_apply_overrides(cfg, varargin)
if mod(numel(varargin), 2) ~= 0
    error('Name-value arguments must come in pairs.');
end
for i = 1:2:numel(varargin)
    name = char(varargin{i});
    if ~isfield(cfg, name)
        error('Unknown config field: %s', name);
    end
    cfg.(name) = varargin{i + 1};
end
end

function table = local_build_wa_base_gain_lut_fixed(cfg)
one = cfg.COEFF_ONE;
lut = local_build_cct_gain_lut(cfg);
anchors = lut([1 65 128], :);
gain_q = round(anchors * one);
table = min(max(gain_q, 0), 65535);
table(2, :) = one;
end

function lut = local_build_cct_gain_lut(cfg)
lut = zeros(128, 3);
[x_n, y_n] = local_cct_to_xy_approx(cfg.cct_neutral_k, cfg.cct_xy_split_k, cfg.cct_xy_blend_half_width_k);
neutral_rgb = local_xy_to_linear_srgb_white(x_n, y_n);
for wa_sel = 0:127
    cct = local_wa_sel_to_cct(wa_sel, cfg.cct_warm_k, cfg.cct_neutral_k, cfg.cct_cool_k);
    [x_t, y_t] = local_cct_to_xy_approx(cct, cfg.cct_xy_split_k, cfg.cct_xy_blend_half_width_k);
    target_rgb = local_xy_to_linear_srgb_white(x_t, y_t);
    gain = target_rgb ./ neutral_rgb;
    y_gain = 0.2126 * gain(1) + 0.7152 * gain(2) + 0.0722 * gain(3);
    gain = gain ./ max(y_gain, 1e-6);
    if wa_sel > 64
        gain(2) = min(gain(2), 1.0);
    end
    lut(wa_sel + 1, :) = min(max(gain, 0.5), 1.8);
end
lut(65, :) = [1.0 1.0 1.0];
end

function cct = local_wa_sel_to_cct(wa_sel, warm_k, neutral_k, cool_k)
w = min(max(double(wa_sel), 0.0), 127.0);
if w <= 64
    cct = neutral_k - (64.0 - w) * (neutral_k - warm_k) / 64.0;
else
    cct = neutral_k + (w - 64.0) * (cool_k - neutral_k) / 63.0;
end
end

function [x, y] = local_cct_to_xy_approx(cct, split_k, blend_half_width_k)
t = min(max(double(cct), 1667.0), 25000.0);

x_low = -0.2661239e9 / (t ^ 3) - 0.2343580e6 / (t ^ 2) + 0.8776956e3 / t + 0.179910;
x_high = -3.0258469e9 / (t ^ 3) + 2.1070379e6 / (t ^ 2) + 0.2226347e3 / t + 0.240390;

split = double(split_k);
half = max(double(blend_half_width_k), 0.0);
blend_lo = split - half;
blend_hi = split + half;

if half <= 0.0
    if t <= split
        x = x_low;
    else
        x = x_high;
    end
    u = 0.0;
elseif t <= blend_lo
    x = x_low;
    u = 0.0;
elseif t >= blend_hi
    x = x_high;
    u = 1.0;
else
    u = (t - blend_lo) / (blend_hi - blend_lo);
    u = u * u * (3.0 - 2.0 * u);
    x = (1.0 - u) * x_low + u * x_high;
end

if t <= 2222.0
    y = -1.1063814 * (x ^ 3) - 1.34811020 * (x ^ 2) + 2.18555832 * x - 0.20219683;
elseif t < blend_lo
    y = -0.9549476 * (x ^ 3) - 1.37418593 * (x ^ 2) + 2.09137015 * x - 0.16748867;
elseif t > blend_hi
    y = 3.0817580 * (x ^ 3) - 5.87338670 * (x ^ 2) + 3.75112997 * x - 0.37001483;
else
    y_mid = -0.9549476 * (x ^ 3) - 1.37418593 * (x ^ 2) + 2.09137015 * x - 0.16748867;
    y_high = 3.0817580 * (x ^ 3) - 5.87338670 * (x ^ 2) + 3.75112997 * x - 0.37001483;
    y = (1.0 - u) * y_mid + u * y_high;
end
end

function rgb = local_xy_to_linear_srgb_white(x, y)
y_safe = max(y, 1e-8);
xyz = [x / y_safe, 1.0, (1.0 - x - y_safe) / y_safe]';
m_xyz_to_srgb = [ ...
    3.2406, -1.5372, -0.4986; ...
   -0.9689,  1.8758,  0.0415; ...
    0.0557, -0.2040,  1.0570];
rgb = max(m_xyz_to_srgb * xyz, 1e-6)';
end

function a = local_atten_curve(y)
if y <= 31
    a = 0.55;
elseif y <= 127
    a = 0.55 + 0.45 * (double(y) - 31.0) / (127.0 - 31.0);
elseif y <= 239
    a = 1.00 + (0.35 - 1.00) * (double(y) - 127.0) / (239.0 - 127.0);
else
    a = 0.35;
end
end

function caps = local_build_warm_highlight_red_caps_fixed(cfg)
one = cfg.COEFF_ONE;
caps = repmat(one, numel(cfg.luma_nodes), 1);
caps(end-3:end) = round([1.30; 1.22; 1.16; 1.10] * one);
end

function caps = local_build_warm_highlight_green_caps_fixed(cfg)
one = cfg.COEFF_ONE;
caps = repmat(one, numel(cfg.luma_nodes), 1);
caps(end-3:end) = round([0.94; 0.94; 0.94; 0.94] * one);
end

function caps = local_build_warm_highlight_blue_floors_fixed(cfg)
one = cfg.COEFF_ONE;
caps = repmat(one, numel(cfg.luma_nodes), 1);
caps(end-3:end) = round([0.80; 0.84; 0.87; 0.90] * one);
end

function caps = local_build_cool_highlight_green_caps_fixed(cfg)
one = cfg.COEFF_ONE;
caps = repmat(one, numel(cfg.luma_nodes), 1);
caps(end-3:end) = round([0.98; 0.975; 0.97; 0.965] * one);
end

function caps = local_build_cool_highlight_blue_caps_fixed(cfg)
one = cfg.COEFF_ONE;
caps = repmat(one, numel(cfg.luma_nodes), 1);
caps(end-3:end) = round([1.15; 1.12; 1.08; 1.06] * one);
end
