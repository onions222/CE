function a = wpa_fixed_atten_curve(y)
%WPA_FIXED_ATTEN_CURVE Highlight-aware attenuation curve for default gains.

if y <= 31
    a = 0.55;
elseif y <= 127
    a = 0.55 + 0.45 * (double(y) - 31.0) / (127.0 - 31.0);
elseif y <= 239
    a = 1.00 + (0.35 - 1.00) * (double(y) - 127.0) / (239.0 - 127.0);
else
    a = 0.35;
end
end
