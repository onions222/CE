function summary = validate_hw_fixed_against_python(golden_dir)
%VALIDATE_HW_FIXED_AGAINST_PYTHON 比较独立 hw_runtime 输出与 Python golden。
%
% 作用：
% - 读取 matlab/golden_cases/manifest.json
% - 对 manifest 中的每个 case 重新运行 hw_fixed_process_image
% - 与 Python 导出的 golden 图像做逐像素比较
%
% 位宽与 raw code 口径：
% - 这里的 frac_bits / coeff_frac_bits 直接继承 manifest
% - 默认会对应 Q0.8 像素 raw code 与 UQ1.8 增益 raw code
% - 统计结果中的 max_abs / mean_abs / p99_abs 都是输出图像 8bit 域的差值
%
% 这个脚本只负责验证：
% - 不改算法参数
% - 不重建 Python golden
% - 不依赖旧的 wpa_fixed MATLAB 实现

if nargin < 1 || isempty(golden_dir)
    golden_dir = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'golden_cases');
end

manifest_path = fullfile(golden_dir, 'manifest.json');
manifest = jsondecode(fileread(manifest_path));
summary = struct();
summary.cases = cell(numel(manifest.cases), 1);

for i = 1:numel(manifest.cases)
    case_info = manifest.cases(i);
    % 每个 case 都按 manifest 指定的 WA_SEL 和小数位宽重新运行一次。
    img = imread(case_info.source_image);
    cfg = hw_fixed_config( ...
        'wa_sel', case_info.wa_sel, ...
        'frac_bits', manifest.frac_bits, ...
        'coeff_frac_bits', manifest.coeff_frac_bits);
    out = hw_fixed_process_image(img, cfg);
    ref = imread(fullfile(golden_dir, case_info.output_image));

    diff = abs(double(out) - double(ref));
    case_summary = struct();
    case_summary.source_image = case_info.source_image;
    case_summary.output_image = case_info.output_image;
    case_summary.wa_sel = case_info.wa_sel;
    case_summary.max_abs = max(diff(:));
    case_summary.mean_abs = mean(diff(:));
    case_summary.p99_abs = local_percentile(diff(:), 99);
    summary.cases{i} = case_summary;
end

for i = 1:numel(summary.cases)
    c = summary.cases{i};
    % 输出的是 8bit 图像域统计，便于快速看是否与 Python golden 对齐。
    fprintf('wa=%d max_abs=%.3f mean_abs=%.3f p99_abs=%.3f %s\n', ...
        c.wa_sel, c.max_abs, c.mean_abs, c.p99_abs, c.output_image);
end
end

function p = local_percentile(values, pct)
% 简单百分位函数，用于输出 p99_abs。
values = sort(values(:));
if isempty(values)
    p = 0.0;
    return;
end
idx = max(1, ceil((pct / 100.0) * numel(values)));
p = values(idx);
end
