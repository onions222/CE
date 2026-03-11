% RUN_HW_FIXED_FOLDER MATLAB 脚本入口。
%
% 用法：
% 1. 打开本文件
% 2. 修改下面“用户配置区”的输入目录、输出目录、WA_SEL 等参数
% 3. 直接点击 MATLAB Run 运行
%
% 结构说明：
% - 只常驻 3 anchor gain
% - 只常驻 12 luma nodes
% - WA_SEL 更新时展开当前 runtime 12x3
% - 只扫描 input_dir 当前目录，不递归子目录
%
% 默认位宽：
% - coeff_frac_bits = 8，对应 UQ1.8，增益 raw code 位宽 = 9 bit
% - frac_bits = 8，对应 Q0.8，像素 raw code 位宽 = 9 bit
% - mul_bits = 18 bit，对应乘法累加位宽

%% 用户配置区
input_dir = 'test_images/synthetic';
output_dir = 'outputs/matlab_hw_runtime_smoke';
wa_sel = 0;
coeff_frac_bits = 8;
frac_bits = 8;
gamma_mode = 'srgb';
gamma_power = 2.2;
file_patterns = {'*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff'};

%% 主流程
script_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(fileparts(script_dir));
input_dir = local_resolve_path(input_dir, repo_root);
output_dir = local_resolve_path(output_dir, repo_root);

if ~isfolder(input_dir)
    error('Input directory does not exist: %s', input_dir);
end

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

cfg = hw_fixed_config( ...
    'wa_sel', wa_sel, ...
    'coeff_frac_bits', coeff_frac_bits, ...
    'frac_bits', frac_bits, ...
    'gamma_mode', gamma_mode, ...
    'gamma_power', gamma_power);

files = local_collect_files(input_dir, file_patterns);

fprintf('Processing folder: %s\n', input_dir);
fprintf('Output folder    : %s\n', output_dir);
fprintf('wa_sel=%d coeff_frac_bits=%d frac_bits=%d\n', cfg.wa_sel, cfg.coeff_frac_bits, cfg.frac_bits);
fprintf('pixel_bits=%d coeff_bits=%d mul_bits=%d\n', cfg.pixel_bits, cfg.coeff_bits, cfg.mul_bits);

for i = 1:numel(files)
    in_name = files(i).name;
    in_path = fullfile(files(i).folder, in_name);
    [~, stem, ext] = fileparts(in_name);
    out_name = sprintf('%s_wa%d%s', stem, cfg.wa_sel, ext);
    out_path = fullfile(output_dir, out_name);

    img = imread(in_path);
    if size(img, 3) ~= 3
        fprintf('skip non-rgb: %s\n', in_name);
        continue;
    end

    out = hw_fixed_process_image(img, cfg);
    imwrite(out, out_path);
    fprintf('  [%d/%d] %s -> %s\n', i, numel(files), in_name, out_name);
end

%% 本地辅助函数
function files = local_collect_files(input_dir, patterns)
% 收集当前目录下一层图像文件，不递归子目录。
files = struct('name', {}, 'folder', {}, 'date', {}, 'bytes', {}, 'isdir', {}, 'datenum', {});
seen = containers.Map('KeyType', 'char', 'ValueType', 'logical');

for i = 1:numel(patterns)
    matched = dir(fullfile(input_dir, patterns{i}));
    for j = 1:numel(matched)
        key = fullfile(matched(j).folder, matched(j).name);
        if ~isKey(seen, key)
            seen(key) = true;
            files(end + 1, 1) = matched(j); %#ok<AGROW>
        end
    end
end

if isempty(files)
    fprintf('No image files found in %s\n', input_dir);
    return;
end

[~, order] = sort(lower({files.name}));
files = files(order);
end

function resolved = local_resolve_path(path_value, repo_root)
if isstring(path_value)
    path_value = char(path_value);
end

if isfolder(path_value) || local_is_absolute_path(path_value)
    resolved = path_value;
else
    resolved = fullfile(repo_root, path_value);
end
end

function tf = local_is_absolute_path(path_value)
tf = startsWith(path_value, filesep) || (~isempty(regexp(path_value, '^[A-Za-z]:[\\/]', 'once')));
end
