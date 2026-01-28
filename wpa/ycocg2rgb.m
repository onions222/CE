function I = ycocg2rgb(Y, Co, Cg)
% YCoCg -> RGB (inverse of your definition)
%
% Input:
%   Y : HxW double
%   Co: HxW double
%   Cg: HxW double
% Output:
%   I: HxWx3 double (NOT clipped). You usually clip to [0,255] and cast to uint8.

    R = Y + Co - Cg;
    G = Y + Cg;
    B = Y - Co - Cg;

    I = cat(3, R, G, B);
end
