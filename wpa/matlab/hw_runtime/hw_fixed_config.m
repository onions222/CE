function cfg = hw_fixed_config(varargin)
%HW_FIXED_CONFIG 构建独立硬件仿真配置。
%
% 设计口径：
% 1. 全部定点量按 raw code 整数码值理解；
% 2. 真实数值 = raw_code / scale_factor；
% 3. 常驻数据只保留 3 个 anchor gain、12 个 luma node、12 点 atten 与 tail 表。
%
% 默认位宽口径：
% - coeff_frac_bits = 8，对应 UQ1.8，增益 raw code 位宽 = 9 bit
% - frac_bits = 8，对应 Q0.8，像素 raw code 位宽 = 9 bit
% - mul_bits = 18 bit，对应像素码值与增益码值乘法累加位宽
%
% 常驻表说明：
% - wa_base_gain_lut_fixed[3][3]：只保存 warm / fair(neutral) / cool 三个 anchor
% - atten_q_lut_fixed[12]：12 个亮度节点的衰减 raw code
% - luma_nodes[12]：亮度节点 [15, 31, ..., 255]
% - warm/cool highlight tail：高亮末端单独修正表
%
% 当前实现是纯表驱动：
% - 3 个端点增益值已经提前算好，直接写入 MATLAB
% - 12 个亮度衰减 raw code 也已经提前算好，直接写入 MATLAB
% - 运行时不再做 CCT -> xy -> RGB gain 推导
% - 运行时也不再做 atten_curve 在线生成

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
    'low_luma_gate_en', true, ...
    'low_luma_bypass_code', [], ...
    'low_luma_blend_end_code', [], ...
    'sat_en', false, ...
    'sat_s0', 100, ...
    'sat_s1', 500, ...
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
% 位宽换算：
% - ONE / HALF        是 Q0.frac_bits 域的 scale factor 与舍入常量
% - COEFF_ONE / HALF  是 UQ1.coeff_frac_bits 域的 scale factor 与舍入常量
cfg.pixel_bits = cfg.frac_bits + 1;
cfg.coeff_bits = cfg.coeff_frac_bits + 1;
cfg.mul_bits = cfg.pixel_bits + cfg.coeff_bits;

if isempty(cfg.wa_base_gain_lut_fixed)
    cfg.wa_base_gain_lut_fixed = local_default_wa_base_gain_lut_fixed(cfg);
else
    cfg.wa_base_gain_lut_fixed = round(cfg.wa_base_gain_lut_fixed);
end

if isempty(cfg.atten_q_lut_fixed)
    cfg.atten_q_lut_fixed = local_default_atten_q_lut_fixed(cfg);
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

if isempty(cfg.low_luma_bypass_code)
    cfg.low_luma_bypass_code = local_srgb_u8_to_linear_code(31, cfg.frac_bits);
else
    cfg.low_luma_bypass_code = round(cfg.low_luma_bypass_code);
end

if isempty(cfg.low_luma_blend_end_code)
    cfg.low_luma_blend_end_code = local_srgb_u8_to_linear_code(63, cfg.frac_bits);
else
    cfg.low_luma_blend_end_code = round(cfg.low_luma_blend_end_code);
end

if cfg.low_luma_blend_end_code <= cfg.low_luma_bypass_code
    error('low_luma_blend_end_code must be greater than low_luma_bypass_code');
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

function table = local_default_wa_base_gain_lut_fixed(cfg)
% 3 个端点增益值已经在 Python fixed 版本中提前算好，这里直接写死。
% 行顺序固定为：
% - row 1: warm
% - row 2: fair(neutral)
% - row 3: cool
if cfg.coeff_frac_bits == 8
    table = [
        436, 221, 128;
        256, 256, 256;
        219, 256, 339;
    ];
elseif cfg.coeff_frac_bits == 10
    table = [
        1746, 885, 512;
        1024, 1024, 1024;
        878, 1024, 1355;
    ];
else
    error('coeff_frac_bits must be 8 or 10');
end
end

function table = local_default_atten_q_lut_fixed(cfg)
% 12 个亮度节点衰减值也提前算好，直接写成 raw code 常量表。
% 顺序对应 luma_nodes = [15, 31, 47, 63, 95, 127, 159, 191, 223, 239, 247, 255]
if cfg.coeff_frac_bits == 8
    table = [141, 141, 160, 179, 218, 256, 208, 161, 113, 90, 90, 90];
elseif cfg.coeff_frac_bits == 10
    table = [563, 563, 640, 717, 870, 1024, 834, 644, 453, 358, 358, 358];
else
    error('coeff_frac_bits must be 8 or 10');
end
end

function caps = local_build_warm_highlight_red_caps_fixed(cfg)
% 高亮末 4 个节点单独设置 R cap，避免 warm 端高亮区过早裁剪后留下脏黄尾巴。
one = cfg.COEFF_ONE;
caps = repmat(one, numel(cfg.luma_nodes), 1);
caps(end-3:end) = round([1.30; 1.22; 1.16; 1.10] * one);
end

function caps = local_build_warm_highlight_green_caps_fixed(cfg)
% 高亮末 4 个节点单独设置 G cap，抑制 warm 端偏黄绿。
one = cfg.COEFF_ONE;
caps = repmat(one, numel(cfg.luma_nodes), 1);
caps(end-3:end) = round([0.94; 0.94; 0.94; 0.94] * one);
end

function caps = local_build_warm_highlight_blue_floors_fixed(cfg)
% 高亮末 4 个节点单独设置 B floor，让最亮区回到低色度暖白。
one = cfg.COEFF_ONE;
caps = repmat(one, numel(cfg.luma_nodes), 1);
caps(end-3:end) = round([0.80; 0.84; 0.87; 0.90] * one);
end

function caps = local_build_cool_highlight_green_caps_fixed(cfg)
% cool 端高亮末 4 个节点单独设置 G cap，防止蓝端往 cyan 方向走。
one = cfg.COEFF_ONE;
caps = repmat(one, numel(cfg.luma_nodes), 1);
caps(end-3:end) = round([0.98; 0.975; 0.97; 0.965] * one);
end

function caps = local_build_cool_highlight_blue_caps_fixed(cfg)
% cool 端高亮末 4 个节点单独设置 B cap，目标是低色度淡蓝，不是高饱和 cyan。
one = cfg.COEFF_ONE;
caps = repmat(one, numel(cfg.luma_nodes), 1);
caps(end-3:end) = round([1.15; 1.12; 1.08; 1.06] * one);
end

function code = local_srgb_u8_to_linear_code(v, frac_bits)
x = min(max(double(v), 0), 255) / 255.0;
if x <= 0.04045
    linear = x / 12.92;
else
    linear = ((x + 0.055) / 1.055) ^ 2.4;
end
code = round(linear * (2 ^ frac_bits));
code = min(max(code, 0), 2 ^ frac_bits);
end
