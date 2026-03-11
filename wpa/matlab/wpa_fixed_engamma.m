function encoded = wpa_fixed_engamma(img_linear, mode, gamma_power)
%WPA_FIXED_ENGAMMA Engamma linear float image to encoded float.

if nargin < 3
    gamma_power = 2.2;
end

x = single(img_linear);

switch char(mode)
    case 'none'
        encoded = x;
    case 'srgb'
        encoded = single(zeros(size(x), 'single'));
        mask = x <= 0.0031308;
        encoded(mask) = 12.92 .* x(mask);
        encoded(~mask) = 1.055 .* (max(x(~mask), 0.0) .^ (1.0 / 2.4)) - 0.055;
    case 'power'
        encoded = max(min(x, 1.0), 0.0) .^ (1.0 / gamma_power);
    otherwise
        error('Unknown gamma mode: %s', char(mode));
end
end
