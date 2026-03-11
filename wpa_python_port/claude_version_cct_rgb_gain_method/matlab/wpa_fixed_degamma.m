function linear = wpa_fixed_degamma(img, mode, gamma_power)
%WPA_FIXED_DEGAMMA Degamma uint8 or float image to linear float.

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
