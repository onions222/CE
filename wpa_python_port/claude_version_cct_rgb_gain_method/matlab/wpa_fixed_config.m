function cfg = wpa_fixed_config(varargin)
%WPA_FIXED_CONFIG Build MATLAB config aligned with Python FixedWPAConfig.

cfg = struct( ...
    'frac_bits', 10, ...
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
    'atten_q_lut_fixed', []);

cfg = local_apply_overrides(cfg, varargin{:});

if ~ismember(cfg.coeff_frac_bits, [8 10])
    error('coeff_frac_bits must be 8 or 10');
end

cfg.ONE = bitshift(int32(1), cfg.frac_bits);
cfg.HALF = bitshift(int32(1), cfg.frac_bits - 1);
cfg.COEFF_ONE = bitshift(int32(1), cfg.coeff_frac_bits);
cfg.COEFF_HALF = bitshift(int32(1), cfg.coeff_frac_bits - 1);

if isempty(cfg.wa_base_gain_lut_fixed)
    cfg.wa_base_gain_lut_fixed = local_build_wa_base_gain_lut_fixed(cfg);
else
    cfg.wa_base_gain_lut_fixed = int32(cfg.wa_base_gain_lut_fixed);
end

if isempty(cfg.atten_q_lut_fixed)
    atten = zeros(numel(cfg.luma_nodes), 1);
    for i = 1:numel(cfg.luma_nodes)
        atten(i) = wpa_fixed_atten_curve(cfg.luma_nodes(i));
    end
    cfg.atten_q_lut_fixed = int32(round(atten * double(cfg.COEFF_ONE)));
else
    cfg.atten_q_lut_fixed = int32(cfg.atten_q_lut_fixed);
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
one = double(bitshift(int32(1), cfg.coeff_frac_bits));
lut = local_build_cct_gain_lut(cfg);
anchors = lut([1 65 128], :);
gain_q = round(anchors * one);
table = int32(min(max(gain_q, 0), 65535));
table(2, :) = int32(one);
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
