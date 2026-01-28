function [Y, Co, Cg] = rgb2ycocg(I)
% RGB -> YCoCg (your definition)
%
% Input:
%   I: HxWx3, uint8 or double, expected RGB in [0,255]
% Output:
%   Y : HxW double, range [0,255]
%   Co: HxW double, range [-127.5, 127.5]
%   Cg: HxW double, range [-127.5, 127.5]
%
% Notes:
%   This is a linear transform. If I is uint8, MATLAB promotes to double.

    R = double(I(:,:,1));
    G = double(I(:,:,2));
    B = double(I(:,:,3));

    Y  = 0.25*R + 0.50*G + 0.25*B;
    Co = 0.50*(R - B);
    Cg = -0.25*R + 0.50*G - 0.25*B;
end
