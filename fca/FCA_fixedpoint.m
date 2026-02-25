%% FCA_fixedpoint.m — Fixed-point hue shift (integer-only core)
% MATLAB port of FCA_fixedpoint.py. Configurable FRAC_BITS precision.
% Compares against floating-point version for verification.
clear;

imgPath = 'soap-bubbles-nature.jpg';
img = imread(imgPath);

deltaH_deg = +10;
% Hue offset(degrees)

        % -- -Parameters-- -
    useHueRange = true;
hueMinDeg = 50;
hueMaxDeg = 170;
taperWidthDeg = 15;
deltaEps = 1;
% integer threshold(1 = one 8 - bit LSB) maxAbsDeltaHDeg = 15;
FRAC_BITS = 10;

out_fp = hue_shift_fp(img, deltaH_deg, useHueRange, hueMinDeg, hueMaxDeg,
                      ... taperWidthDeg, deltaEps, maxAbsDeltaHDeg, FRAC_BITS);

imwrite(out_fp, 'out_fixedpoint.png');
fprintf('Saved: out_fixedpoint.png\n');

% -- -Compare against floating - point version-- - I = im2double(img);
opts.useHueRange = useHueRange;
opts.hueMin = hueMinDeg;
opts.hueMax = hueMaxDeg;
opts.taperWidth = taperWidthDeg;
opts.deltaEps = 1 / 255;
opts.maxAbsDeltaH = maxAbsDeltaHDeg;
opts.epsT = 1e-6;
[ J_float, ~] = hue_shift_cross_sector_ref(I, deltaH_deg, opts);
ref_u8 = uint8(round(min(max(J_float, 0), 1) * 255));

diff = abs(int16(out_fp) - int16(ref_u8));
fprintf('=== Fixed-point vs Float (FRAC_BITS=%d) ===\n', FRAC_BITS);
fprintf('  MAE  : R=%.4f  G=%.4f  B=%.4f\n', ... mean(diff( :, :, 1), 'all'),
        mean(diff( :, :, 2), 'all'), mean(diff( :, :, 3), 'all'));
fprintf('  MaxE : R=%d  G=%d  B=%d\n', ... max(diff( :, :, 1), [], 'all'),
        max(diff( :, :, 2), [], 'all'), max(diff( :, :, 3), [], 'all'));

figure('Name', 'Fixed-point vs Floating-point');
subplot(1, 3, 1);
imshow(img);
title('Original');
subplot(1, 3, 2);
imshow(out_fp);
title('Fixed-point');
subplot(1, 3, 3);
imshow(ref_u8);
title('Float ref');

% % == == == == == == == == == ==
    = Fixed - point core == == == == == == == == == ==
    = function out_u8 =
        hue_shift_fp(img_u8, deltaH_deg, useHueRange, ... hueMinDeg, hueMaxDeg,
                     taperWidthDeg, deltaEps, maxAbsDeltaH, FRAC_BITS)

            ONE = int64(bitshift(int64(1), FRAC_BITS));
HUE_FULL = int64(6) * ONE;
HALF = int64(bitshift(int64(1), FRAC_BITS - 1));

    % --- 0) Clamp and convert params to hue units ---
    deltaH_deg = max(-maxAbsDeltaH, min(deltaH_deg, maxAbsDeltaH));
    deg2hu = @(d) int64(round(double(d) * double(HUE_FULL) / 360));

    dH_hu = deg2hu(deltaH_deg);
    hmin_hu = deg2hu(hueMinDeg);
    hmax_hu = deg2hu(hueMaxDeg);
    taper_hu = deg2hu(taperWidthDeg);

    % --- 1) Channels as int64 ---
    R = int64(img_u8(:,:,1));
    G = int64(img_u8( :, :, 2));
    B = int64(img_u8( :, :, 3));

    V = max(max(R, G), B);
    m_val = min(min(R, G), B);
    Delta = V - m_val;

    active = (Delta >= deltaEps);

    % -- -Sector classification-- - isRmax = active & (R >= G) & (R >= B);
    isGmax = active & (G > R) & (G >= B);
    isBmax = active & ~(isRmax | isGmax);

    s0 = isRmax & (G >= B);
    s5 = isRmax & ~(G >= B);
    s1 = isGmax & (B <= R);
    s2 = isGmax & ~(B <= R);
    s3 = isBmax & (R <= G);
    s4 = isBmax & ~(R <= G);

    % --- 3) t_fp = round((diff << FRAC_BITS) / Delta) ---
    t_fp = zeros(size(V), 'int64');
    secs = {s0, s1, s2, s3, s4, s5};
    diffs = {G - B, G - R, B - R, B - G, R - G, R - B};
    for
      k = 1 : 6 mask = secs{k};
    if any (mask( :))
      num = bitshift(diffs{k}(mask), FRAC_BITS);
    den = Delta(mask);
    t_fp(mask) = idivide(num + bitshift(den, -1), den, 'floor');
    end end t_fp = max(min(t_fp, ONE), int64(0));

    % -- -Hue in hue units-- - hue_hu = zeros(size(V), 'int64');
    for
      k = 1 : 6 mask = secs{k};
    hue_hu(mask) = int64(k - 1) * ONE + t_fp(mask);
    end

                % -- -Hue range filter-- -
            if useHueRange if hmin_hu <=
        hmax_hu enable = (hue_hu >= hmin_hu) & (hue_hu <= hmax_hu);
    else enable = (hue_hu >= hmin_hu) | (hue_hu <= hmax_hu);
    end active = active & enable;
    end

    % --- 4) Effective deltaH with taper ---
    eff_dH = repmat(dH_hu, size(V));

    if useHueRange
      &&taper_hu > 0 if hmin_hu <= hmax_hu rw = hmax_hu - hmin_hu;
    else
      rw = (HUE_FULL - hmin_hu) + hmax_hu;
    end effTaper = min(taper_hu, idivide(rw, int64(2), 'floor'));

    if effTaper
      > 0 edgeDist = zeros(size(V), 'int64');
    h = hue_hu(active);
    if hmin_hu
      <= hmax_hu dLo = h - hmin_hu;
    dHi = hmax_hu - h;
    else dLo = h;
    dLo(h >= hmin_hu) = h(h >= hmin_hu) - hmin_hu;
    dLo(h < hmin_hu) = h(h < hmin_hu) + HUE_FULL - hmin_hu;
    dHi = h;
    dHi(h <= hmax_hu) = hmax_hu - h(h <= hmax_hu);
    dHi(h > hmax_hu) = hmax_hu + HUE_FULL - h(h > hmax_hu);
    end edgeDist(active) = min(dLo, dHi);

    scale = zeros(size(V), 'int64');
    scale(active) = min(max(... idivide(bitshift(edgeDist(active), FRAC_BITS),
                                        effTaper, 'floor'),
                            ... int64(0)),
                        ONE);

    eff_dH(active) = bitshift(dH_hu * scale(active), -FRAC_BITS);
    end end

            % -- -New hue-- -
        newHue = zeros(size(V), 'int64');
    newHue(active) = mod(hue_hu(active) + eff_dH(active), HUE_FULL);

    newSec = zeros(size(V), 'int64');
    newT = zeros(size(V), 'int64');
    newSec(active) = bitshift(newHue(active), -FRAC_BITS);
    newT(active) = bitand(newHue(active), ONE - 1);
    newSec = max(min(newSec, int64(5)), int64(0));

    ns = cell(1, 6);
    for
      k = 0 : 5 ns{k + 1} = active & (newSec == k);
    end

    % --- 5) Reconstruct ---
    R2 = R;
    G2 = G;
    B2 = B;

    dt_mul = @(mask) bitshift(Delta(mask).*newT(mask) + HALF, -FRAC_BITS);

    % ns0 : V = R, min = B, G = m + D * t R2(ns{1}) = V(ns{1});
    B2(ns{1}) = m_val(ns{1});
    if any (ns{1}( :))
      ;
    G2(ns{1}) = m_val(ns{1}) + dt_mul(ns{1});
    end

        % ns1 : V = G,
                min = B, R = V - D * t G2(ns{2}) = V(ns{2});
    B2(ns{2}) = m_val(ns{2});
    if any (ns{2}( :))
      ;
    R2(ns{2}) = V(ns{2}) - dt_mul(ns{2});
    end

        % ns2 : V = G,
                min = R, B = m + D * t G2(ns{3}) = V(ns{3});
    R2(ns{3}) = m_val(ns{3});
    if any (ns{3}( :))
      ;
    B2(ns{3}) = m_val(ns{3}) + dt_mul(ns{3});
    end

        % ns3 : V = B,
                min = R, G = V - D * t B2(ns{4}) = V(ns{4});
    R2(ns{4}) = m_val(ns{4});
    if any (ns{4}( :))
      ;
    G2(ns{4}) = V(ns{4}) - dt_mul(ns{4});
    end

        % ns4 : V = B,
                min = G, R = m + D * t B2(ns{5}) = V(ns{5});
    G2(ns{5}) = m_val(ns{5});
    if any (ns{5}( :))
      ;
    R2(ns{5}) = m_val(ns{5}) + dt_mul(ns{5});
    end

        % ns5 : V = R,
                min = G, B = V - D * t R2(ns{6}) = V(ns{6});
    G2(ns{6}) = m_val(ns{6});
    if any (ns{6}( :))
      ;
    B2(ns{6}) = V(ns{6}) - dt_mul(ns{6});
    end

        R2 = max(min(R2, int64(255)), int64(0));
    G2 = max(min(G2, int64(255)), int64(0));
    B2 = max(min(B2, int64(255)), int64(0));

    out_u8 = uint8(cat(3, R2, G2, B2));
end

%% ===================== Float reference (for comparison) =====================
function [Iout, dbg] = hue_shift_cross_sector_ref(Iin, deltaH, opts)
    deltaH = max(-opts.maxAbsDeltaH, min(deltaH, opts.maxAbsDeltaH));
R = Iin( :, :, 1);
G = Iin( :, :, 2);
B = Iin( :, :, 3);
V = max(max(R, G), B);
m = min(min(R, G), B);
Delta = V - m;
S = zeros(size(V));
nzV = (V > 0);
S(nzV) = Delta(nzV)./ V(nzV);
active = (Delta >= opts.deltaEps);
isRmax = active & (R >= G) & (R >= B);
isGmax = active & (G > R) & (G >= B);
isBmax = active & ~(isRmax | isGmax);
s0 = isRmax & (G >= B);
s5 = isRmax & ~(G >= B);
s1 = isGmax & (B <= R);
s2 = isGmax & ~(B <= R);
s3 = isBmax & (R <= G);
s4 = isBmax & ~(R <= G);
t = zeros(size(V));
t(s0) = (G(s0) - B(s0))./ Delta(s0);
t(s1) = (G(s1) - R(s1))./ Delta(s1);
t(s2) = (B(s2) - R(s2))./ Delta(s2);
t(s3) = (B(s3) - G(s3))./ Delta(s3);
t(s4) = (R(s4) - G(s4))./ Delta(s4);
t(s5) = (R(s5) - B(s5))./ Delta(s5);
t = min(max(t, 0), 1);
Hue = zeros(size(V));
Hue(s0) = 60 * (0 + t(s0));
Hue(s1) = 60 * (1 + t(s1));
Hue(s2) = 60 * (2 + t(s2));
Hue(s3) = 60 * (3 + t(s3));
Hue(s4) = 60 * (4 + t(s4));
Hue(s5) = 60 * (5 + t(s5));
Hue = mod(Hue, 360);
if opts
  .useHueRange if opts.hueMin <= opts.hueMax;
enable = (Hue >= opts.hueMin) & (Hue <= opts.hueMax);
else;
enable = (Hue >= opts.hueMin) | (Hue <= opts.hueMax);
end active = active & enable;
end eff_dH = deltaH * ones(size(V));
if opts
  .useHueRange &&opts.taperWidth > 0 if opts.hueMin <= opts.hueMax;
rw = opts.hueMax - opts.hueMin;
else;
rw = (360 - opts.hueMin) + opts.hueMax;
end effT = min(opts.taperWidth, rw / 2);
if effT
  > 0 ed = zeros(size(V));
h = Hue(active);
if opts
  .hueMin <= opts.hueMax;
dL = h - opts.hueMin;
dH_ = opts.hueMax - h;
else dL = h;
dL(h >= opts.hueMin) = h(h >= opts.hueMin) - opts.hueMin;
dL(h < opts.hueMin) = h(h < opts.hueMin) + 360 - opts.hueMin;
dH_ = h;
dH_(h <= opts.hueMax) = opts.hueMax - h(h <= opts.hueMax);
dH_(h > opts.hueMax) = opts.hueMax + 360 - h(h > opts.hueMax);
end ed(active) = min(dL, dH_);
sc = min(max(ed / effT, 0), 1);
eff_dH(active) = deltaH * sc(active);
end end newHue = zeros(size(V));
newHue(active) = mod(Hue(active) + eff_dH(active), 360);
newSec = zeros(size(V));
newT = zeros(size(V));
newSec(active) = floor(newHue(active) / 60);
newSec(active) = min(max(newSec(active), 0), 5);
newT(active) = (newHue(active) - newSec(active) * 60) / 60;
newT(active) = min(max(newT(active), 0), 1 - opts.epsT);
ns0 = active & (newSec == 0);
ns1 = active & (newSec == 1);
ns2 = active & (newSec == 2);
ns3 = active & (newSec == 3);
ns4 = active & (newSec == 4);
ns5 = active & (newSec == 5);
R2 = R;
G2 = G;
B2 = B;
R2(ns0) = V(ns0);
B2(ns0) = V(ns0) - Delta(ns0);
G2(ns0) = B2(ns0) + Delta(ns0).*newT(ns0);
G2(ns1) = V(ns1);
B2(ns1) = V(ns1) - Delta(ns1);
R2(ns1) = V(ns1) - Delta(ns1).*newT(ns1);
G2(ns2) = V(ns2);
R2(ns2) = V(ns2) - Delta(ns2);
B2(ns2) = R2(ns2) + Delta(ns2).*newT(ns2);
B2(ns3) = V(ns3);
R2(ns3) = V(ns3) - Delta(ns3);
G2(ns3) = V(ns3) - Delta(ns3).*newT(ns3);
B2(ns4) = V(ns4);
G2(ns4) = V(ns4) - Delta(ns4);
R2(ns4) = G2(ns4) + Delta(ns4).*newT(ns4);
R2(ns5) = V(ns5);
G2(ns5) = V(ns5) - Delta(ns5);
B2(ns5) = V(ns5) - Delta(ns5).*newT(ns5);
Iout = cat(3, R2, G2, B2);
[ Y2, Co2, Cg2 ] = rgb2ycocg_ref(Iout);
Iout = ycocg2rgb_ref(Y2, Co2, Cg2);
Iout = min(max(Iout, 0), 1);
Rf = Iout( :, :, 1);
Gf = Iout( :, :, 2);
Bf = Iout( :, :, 3);
Vf = max(max(Rf, Gf), Bf);
mf = min(min(Rf, Gf), Bf);
Df = Vf - mf;
Sf = zeros(size(Vf));
nzVf = (Vf > 0);
Sf(nzVf) = Df(nzVf)./ Vf(nzVf);
dbg.maxErrV = max(abs(Vf( :) - V( :)));
dbg.maxErrS = max(abs(Sf( :) - S( :)));
end

    function[Y, Co, Cg] = rgb2ycocg_ref(I) R = I( :, :, 1);
G = I( :, :, 2);
B = I( :, :, 3);
Y = 0.25 * R + 0.50 * G + 0.25 * B;
Co = 0.50 * (R - B);
Cg = -0.25 * R + 0.50 * G - 0.25 * B;
end function I = ycocg2rgb_ref(Y, Co, Cg) I =
    cat(3, Y + Co - Cg, Y + Cg, Y - Co - Cg);
end
