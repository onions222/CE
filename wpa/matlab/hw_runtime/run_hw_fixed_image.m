function out = run_hw_fixed_image(input_path, output_path, varargin)
%RUN_HW_FIXED_IMAGE Load, process, and optionally save an image.

img = imread(input_path);
cfg = hw_fixed_config(varargin{:});
out = hw_fixed_process_image(img, cfg);

if nargin >= 2 && ~isempty(output_path)
    imwrite(out, output_path);
end
end
