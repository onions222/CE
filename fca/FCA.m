% % FCA.m — Floating -
    point hue shift(improved : cross - sector, taper, delta_eps) %
        MATLAB port of FCA.py.Preserves V and S strictly.clear;

imgPath = 'soap-bubbles-nature.jpg';
I = im2double(imread(imgPath));

deltaH = +10;
% Hue offset(degrees)

        % -- -Parameters-- -
    opts.useHueRange = true;
opts.hueMin = 50;
% Only affect hue in[50, 170] opts.hueMax = 170;
opts.taperWidth = 15;
% Soft taper at range edges(degrees) opts.deltaEps = 1 / 255;
% Near - gray threshold opts.maxAbsDeltaH = 15;
% Hard limit on | deltaH | opts.epsT = 1e-6;  % Small epsilon for t clamp

[J, dbg] = hue_shift_cross_sector(I, deltaH, opts);

figure('Name', 'Cross-sector Hue shift (taper + delta_eps)');
subplot(1, 2, 1);
imshow(I);
title('Original');
subplot(1, 2, 2);
imshow(J);
title(sprintf('Hue %+g deg', deltaH));

imwrite(J, 'out_hue_strict_vs.png');
fprintf('Saved: out_hue_strict_vs.png\n');
fprintf('Max abs err V: %.3g\n', dbg.maxErrV);
fprintf('Max abs err S: %.3g\n', dbg.maxErrS);

%% ===================== Core function =====================
function [Iout, dbg] = hue_shift_cross_sector(Iin, deltaH, opts)
    % --- 0) Hard-clamp deltaH ---
    deltaH = max(-opts.maxAbsDeltaH, min(deltaH, opts.maxAbsDeltaH));

    % --- 1) RGB -> YCoCg (interface only) ---
    [~, ~, ~] = rgb2ycocg(Iin);

    % --- 2) V, Delta, S and sector classification ---
    R = Iin(:,:,1);
    G = Iin( :, :, 2);
    B = Iin( :, :, 3);

    V = max(max(R, G), B);
    m = min(min(R, G), B);
    Delta = V - m;

    S = zeros(size(V));
    nzV = (V > 0);
    S(nzV) = Delta(nzV)./ V(nzV);

    active = (Delta >= opts.deltaEps);

    % Sector masks isRmax = active & (R >= G) & (R >= B);
    isGmax = active & (G > R) & (G >= B);
    isBmax = active & ~(isRmax | isGmax);

    s0 = isRmax & (G >= B);
    % 0..60 s5 = isRmax & ~(G >= B);
    % 300..360 s1 = isGmax & (B <= R);
    % 60..120 s2 = isGmax & ~(B <= R);
    % 120..180 s3 = isBmax & (R <= G);
    % 180..240 s4 = isBmax & ~(R <= G);   % 240..300

    % --- 3) Compute t and Hue ---
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

    % -- -Hue range filter-- - if opts.useHueRange if opts.hueMin <=
        opts.hueMax enable = (Hue >= opts.hueMin) & (Hue <= opts.hueMax);
    else enable = (Hue >= opts.hueMin) | (Hue <= opts.hueMax);
    end active = active & enable;
    end

    % --- 4) Effective deltaH with soft taper ---
    eff_dH = deltaH * ones(size(V));

    if opts
      .useHueRange &&opts.taperWidth > 0 if opts.hueMin <=
          opts.hueMax rangeWidth = opts.hueMax - opts.hueMin;
    else
      rangeWidth = (360 - opts.hueMin) + opts.hueMax;
    end effTaper = min(opts.taperWidth, rangeWidth / 2);

    if effTaper
      > 0 edgeDist = hue_dist_to_edge(Hue, opts.hueMin, opts.hueMax, active);
    scale = min(max(edgeDist / effTaper, 0), 1);
    eff_dH(active) = deltaH * scale(active);
    end end

            % -- -New hue(cross - sector)-- -
        newHue = zeros(size(V));
    newHue(active) = mod(Hue(active) + eff_dH(active), 360);

    newSec = zeros(size(V));
    newT = zeros(size(V));
    newSec(active) = floor(newHue(active) / 60);
    newSec(active) = min(max(newSec(active), 0), 5);
    newT(active) = (newHue(active) - newSec(active) * 60) / 60;
    newT(active) = min(max(newT(active), 0), 1 - opts.epsT);

    % New sector masks ns0 = active & (newSec == 0);
    ns1 = active & (newSec == 1);
    ns2 = active & (newSec == 2);
    ns3 = active & (newSec == 3);
    ns4 = active & (newSec == 4);
    ns5 = active & (newSec == 5);

    % --- 5) Reconstruct RGB preserving V, Delta ---
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

    % --- 6) YCoCg round-trip ---
    [Y2, Co2, Cg2] = rgb2ycocg(Iout);
    Iout = ycocg2rgb(Y2, Co2, Cg2);
    Iout = min(max(Iout, 0), 1);

    % --- 7) Verify V, S ---
    Rf = Iout(:,:,1);
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

            % %
        == == == == == == == == == ==
        = Helper : edge distance == == == == == == == == == ==
        = function dist = hue_dist_to_edge(Hue, hueMin, hueMax, mask) dist =
            zeros(size(Hue));
    if
      ~any(mask( :));
    return;
    end h = Hue(mask);
    if hueMin
      <= hueMax dLo = h - hueMin;
    dHi = hueMax - h;
    else dLo = h;
    dLo(h >= hueMin) = h(h >= hueMin) - hueMin;
    dLo(h < hueMin) = h(h < hueMin) + 360 - hueMin;
    dHi = h;
    dHi(h <= hueMax) = hueMax - h(h <= hueMax);
    dHi(h > hueMax) = hueMax + 360 - h(h > hueMax);
    end dist(mask) = min(dLo, dHi);
    end

            % %
        == == == == == == == == == ==
        = YCoCg == == == == == == == == == == = function[Y, Co, Cg] =
                                                  rgb2ycocg(I) R = I( :, :, 1);
    G = I( :, :, 2);
    B = I( :, :, 3);
    Y = 0.25 * R + 0.50 * G + 0.25 * B;
    Co = 0.50 * (R - B);
    Cg = -0.25 * R + 0.50 * G - 0.25 * B;
    end

        function I = ycocg2rgb(Y, Co, Cg) R = Y + Co - Cg;
    G = Y + Cg;
    B = Y - Co - Cg;
    I = cat(3, R, G, B);
    end
