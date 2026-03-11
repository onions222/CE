function out = run_wpa_fixed_image(input_path, output_path, varargin)
%RUN_WPA_FIXED_IMAGE Load, process, and optionally save an image with MATLAB fixed WPA.

img = imread(input_path);
if size(img, 3) ~= 3
    error('Input image must be RGB.');
end
cfg = wpa_fixed_config(varargin{:});
out = wpa_fixed_process_matlab(img, cfg);
if nargin >= 2 && ~isempty(output_path)
    imwrite(out, output_path);
end
end
