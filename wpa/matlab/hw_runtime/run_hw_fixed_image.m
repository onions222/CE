function out = run_hw_fixed_image(input_path, output_path, varargin)
%RUN_HW_FIXED_IMAGE 独立硬件仿真单图入口。
%
% 用法：
% - 读取一张输入图片
% - 用 hw_fixed_config 构建配置
% - 调用 hw_fixed_process_image 完成硬件风格处理
% - 如果给了 output_path，就把结果写回磁盘
%
% 默认位宽口径：
% - coeff_frac_bits = 8，对应 UQ1.8，增益 raw code 位宽 = 9 bit
% - frac_bits = 8，对应 Q0.8，像素 raw code 位宽 = 9 bit
% - mul_bits = 18 bit，对应乘法累加位宽
%
% 输入输出说明：
% - 输入图像仍然是 uint8 gamma 域图像
% - 中间计算会转到 raw code 域
% - 输出仍然回到 uint8 图像，便于直接观察效果

img = imread(input_path);
cfg = hw_fixed_config(varargin{:});
out = hw_fixed_process_image(img, cfg);

if nargin >= 2 && ~isempty(output_path)
    imwrite(out, output_path);
end
end
