function summary = validate_hw_fixed_against_python(golden_dir)
%VALIDATE_HW_FIXED_AGAINST_PYTHON 比较独立 hw_runtime 输出与 Python golden。

if nargin < 1 || isempty(golden_dir)
    golden_dir = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'golden_cases');
end

manifest_path = fullfile(golden_dir, 'manifest.json');
manifest = jsondecode(fileread(manifest_path));
summary = struct();
summary.cases = cell(numel(manifest.cases), 1);

for i = 1:numel(manifest.cases)
    case_info = manifest.cases(i);
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
    fprintf('wa=%d max_abs=%.3f mean_abs=%.3f p99_abs=%.3f %s\n', ...
        c.wa_sel, c.max_abs, c.mean_abs, c.p99_abs, c.output_image);
end
end

function p = local_percentile(values, pct)
values = sort(values(:));
if isempty(values)
    p = 0.0;
    return;
end
idx = max(1, ceil((pct / 100.0) * numel(values)));
p = values(idx);
end
