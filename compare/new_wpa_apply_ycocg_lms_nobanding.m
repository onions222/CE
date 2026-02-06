function [Y2, Co2, Cg2] = wpa_apply_ycocg_lms_nobanding(Y, Co, Cg, wa_sel, wa_en, tbl, p)
% WPA on YCoCg with NO visible banding:
%   1) Bin interpolation (continuous across bin boundaries)
%   2) Smooth brightness protection (piecewise linear ramps)
%   3) Overflow limiting by analytic s_max (continuous shrink), NOT /2 steps
%
% Input:
%   Y,Co,Cg : HxW double, your YCoCg definition (RGB in [0,255])
%   wa_sel  : 0..127, 64 is neutral
%   wa_en   : 0/1 enable
%   tbl     : from wpa_build_tables_kelvin_12bin(), must have:
%              tbl.w(1x12), tbl.DmatWarm{1..12}, tbl.DmatCool{1..12}
%   p       : struct for protections, optional fields:
%              p.RGB_MAX (default 255)
%              p.keepY (default 0)
%              p.Y_dark2, p.Y_dark1, p.Y_bright1, p.Y_bright2 (defaults given)
%              p.bright_minfac (default 0.25)
%              p.s_quant_en (default 0), p.s_quant_Q (default 64)
%
% Output:
%   Y2,Co2,Cg2 : adjusted YCoCg (double). Convert back by ycocg2rgb() and clip.

    if wa_en == 0
        Y2 = Y; Co2 = Co; Cg2 = Cg;
        return;
    end

    % ---------------- Defaults ----------------
    if nargin < 7, p = struct(); end
    if ~isfield(p,'RGB_MAX'),        p.RGB_MAX = 255; end
    if ~isfield(p,'keepY'),          p.keepY = 0; end

    if ~isfield(p,'Y_dark2'),        p.Y_dark2 = 8; end
    if ~isfield(p,'Y_dark1'),        p.Y_dark1 = 16; end
    if ~isfield(p,'Y_bright1'),      p.Y_bright1 = 240; end
    if ~isfield(p,'Y_bright2'),      p.Y_bright2 = 248; end
    if ~isfield(p,'bright_minfac'),  p.bright_minfac = 0.25; end

    if ~isfield(p,'s_quant_en'),     p.s_quant_en = 0; end
    if ~isfield(p,'s_quant_Q'),      p.s_quant_Q = 64; end

    % ---------------- WA_SEL mapping (HW-friendly /64) ----------------
    sel = double(wa_sel);
    sel = min(max(sel,0),127);

    if sel < 64
        side = 1; % warm
        t_global = (64 - sel) / 64.0; % in [0,1]
    else
        side = 2; % cool
        t_global = (sel - 64) / 64.0; % in [0, ~1]
    end

    % ---------------- Output init ----------------
    Y2  = Y;
    Co2 = Co;
    Cg2 = Cg;

    % ---------------- Continuous bin position ----------------
    % u = Y * 12 / 256 in [0,12)
    [b_clamp, a_clamp] = map_luma_to_bin_alpha(double(Y), p);

    mask_last  = (b_clamp == 11);   % last bin: no blending
    mask_blend = (b_clamp <= 10);   % bins 0..10 can blend with b+1 (if alpha>0)


    % % Clamp:
    % % - For b>=11: force bin=11 and a=0 (no blending beyond last bin)
    % b_clamp = b;
    % a_clamp = a;
    % 
    % mask_last = (b_clamp >= 11);
    % b_clamp(mask_last) = 11;
    % a_clamp(mask_last) = 0;
    % 
    % mask_blend = (b_clamp <= 10); % bins 0..10 blend with b+1

    % ==========================================================
    % Case 1: bins 0..10 (blend between bb and bb+1)
    % ==========================================================
    for bb = 0:10
        mask = mask_blend & (b_clamp == bb);
        if ~any(mask(:)), continue; end

        % ---- Gather pixels -> enforce column vectors then build 3xN ----
        Yv  = double(Y(mask));  Yv  = Yv(:);
        Cov = double(Co(mask)); Cov = Cov(:);
        Cgv = double(Cg(mask)); Cgv = Cgv(:);

        y0 = [Yv.'; Cov.'; Cgv.'];   % 3xN
        N  = size(y0,2);

        % ---- a as 1xN row vector ----
        aa = double(a_clamp(mask));
        aa = aa(:).';                % 1xN
        assert(numel(aa) == N);

        % ---- wbin as 1xN row vector ----
        w1 = tbl.w(bb+1);
        w2 = tbl.w(bb+2);
        wbin = w1 + aa .* (w2 - w1); % 1xN

        % ---- Dmat blend: delta = ( (1-a)*D1 + a*D2 ) * y ----
        if side == 1
            D1 = tbl.DmatWarm{bb+1};
            D2 = tbl.DmatWarm{bb+2};
        else
            D1 = tbl.DmatCool{bb+1};
            D2 = tbl.DmatCool{bb+2};
        end

        delta1 = D1 * y0;                 % 3xN
        deltaD = (D2 - D1) * y0;          % 3xN
        delta  = delta1 + bsxfun(@times, deltaD, aa); % 3xN
        assert(all(size(delta) == [3, N]));

        % ---- base strength s (1xN) ----
        s = t_global * wbin;              % 1xN
        s = s(:).';                       % force 1xN
        assert(numel(s) == N);

        % ---- smooth brightness factor (force 1xN) ----
        bfac = brightness_factor_smooth(Yv, p); % returns Nx1
        bfac = bfac(:).';                      % 1xN
        assert(numel(bfac) == N);

        % IMPORTANT: now s and bfac are both 1xN => no NxN outer product
        s = s .* bfac;                         % 1xN

        % ---- continuous overflow limiting (returns 1xN) ----
        s = limit_strength_by_rgb_headroom(y0, delta, s, p.RGB_MAX);
        s = s(:).';                            % force 1xN
        assert(numel(s) == N);

        % ---- optional quantization (still 1xN) ----
        if p.s_quant_en
            Q = p.s_quant_Q;
            s = round(s * Q) / Q;
        end
        s = min(max(s,0),1);

        % ---- apply update (delta .* s per-column) ----
        y_cur = y0 + bsxfun(@times, delta, s);  % 3xN
        assert(all(size(y_cur) == [3, N]));

        if p.keepY
            y_cur(1,:) = y0(1,:);
        end

        % ---- scatter back ----
        Y2(mask)  = reshape(y_cur(1,:), size(Y(mask)));
        Co2(mask) = reshape(y_cur(2,:), size(Co(mask)));
        Cg2(mask) = reshape(y_cur(3,:), size(Cg(mask)));
    end

    % ==========================================================
    % Case 2: last bin 11 (index 12), no blending
    % ==========================================================
    mask = (b_clamp == 11);
    if any(mask(:))
        Yv  = double(Y(mask));  Yv  = Yv(:);
        Cov = double(Co(mask)); Cov = Cov(:);
        Cgv = double(Cg(mask)); Cgv = Cgv(:);

        y0 = [Yv.'; Cov.'; Cgv.'];   % 3xN
        N  = size(y0,2);

        wbin = tbl.w(12) * ones(1, N);  % 1xN

        if side == 1
            D = tbl.DmatWarm{12};
        else
            D = tbl.DmatCool{12};
        end
        delta = D * y0;               % 3xN
        assert(all(size(delta) == [3, N]));

        s = t_global * wbin;          % 1xN
        s = s(:).';                   % force 1xN
        assert(numel(s) == N);

        bfac = brightness_factor_smooth(Yv, p); % Nx1
        bfac = bfac(:).';                      % 1xN
        assert(numel(bfac) == N);

        s = s .* bfac;               % 1xN (safe)

        s = limit_strength_by_rgb_headroom(y0, delta, s, p.RGB_MAX);
        s = s(:).';                  % 1xN
        assert(numel(s) == N);

        if p.s_quant_en
            Q = p.s_quant_Q;
            s = round(s * Q) / Q;
        end
        s = min(max(s,0),1);

        y_cur = y0 + bsxfun(@times, delta, s); % 3xN

        if p.keepY
            y_cur(1,:) = y0(1,:);
        end

        Y2(mask)  = reshape(y_cur(1,:), size(Y(mask)));
        Co2(mask) = reshape(y_cur(2,:), size(Co(mask)));
        Cg2(mask) = reshape(y_cur(3,:), size(Cg(mask)));
    end
end

% =====================================================================
% Helper: smooth brightness factor (continuous piecewise-linear ramps)
% Input Yv is Nx1 (column). Output bfac is Nx1.
% =====================================================================
function bfac = brightness_factor_smooth(Yv, p)
    minfac = p.bright_minfac;
    bfac = ones(size(Yv));

    % Dark side: <=dark2 => minfac; between dark2..dark1 => ramp to 1
    if p.Y_dark1 > p.Y_dark2
        bfac(Yv <= p.Y_dark2) = minfac;
        m = (Yv > p.Y_dark2) & (Yv < p.Y_dark1);
        bfac(m) = minfac + (Yv(m) - p.Y_dark2) * (1 - minfac) / (p.Y_dark1 - p.Y_dark2);
    end

    % Bright side: >=bright2 => minfac; between bright1..bright2 => ramp 1->minfac
    if p.Y_bright2 > p.Y_bright1
        bfac(Yv >= p.Y_bright2) = minfac;
        m = (Yv > p.Y_bright1) & (Yv < p.Y_bright2);
        bfac(m) = 1 - (Yv(m) - p.Y_bright1) * (1 - minfac) / (p.Y_bright2 - p.Y_bright1);
    end

    bfac = min(max(bfac, minfac), 1);
end

% =====================================================================
% Helper: continuous overflow limiting by analytic s_max
% y0, delta are 3xN. s is 1xN (row).
% Output s2 is 1xN (row).
% =====================================================================
function s2 = limit_strength_by_rgb_headroom(y0, delta, s, RGB_MAX)
    s = s(:).'; % force row

    [R0,G0,B0] = ycocg2rgb_sep(y0(1,:), y0(2,:), y0(3,:));
    [dR,dG,dB] = ycocg2rgb_sep(delta(1,:), delta(2,:), delta(3,:));

    smax = inf(size(s));  % 1xN

    smax = min(smax, bound_channel(R0, dR, RGB_MAX));
    smax = min(smax, bound_channel(G0, dG, RGB_MAX));
    smax = min(smax, bound_channel(B0, dB, RGB_MAX));

    s2 = min(s, smax);
    s2 = max(s2, 0);
    s2 = s2(:).'; % force row
end

% =====================================================================
% Helper: bound one channel so that C0 + s*dC stays in [0, RGB_MAX]
% Inputs C0,dC are 1xN. Output is 1xN.
% =====================================================================
function bnd = bound_channel(C0, dC, RGB_MAX)
    C0 = C0(:).';
    dC = dC(:).';

    bnd = inf(size(C0));
    epsv = 1e-12;

    pos = dC > epsv;
    neg = dC < -epsv;

    % If dC>0: s <= (RGB_MAX - C0)/dC
    bnd(pos) = (RGB_MAX - C0(pos)) ./ dC(pos);

    % If dC<0: s <= (0 - C0)/dC (dC<0 => ratio is positive when C0>=0)
    bnd(neg) = (0 - C0(neg)) ./ dC(neg);

    % If starting already outside range, force bound 0
    bad0 = (C0 < 0) | (C0 > RGB_MAX);
    bnd(bad0) = 0;

    bnd = max(bnd, 0);
    bnd = bnd(:).';
end

% =====================================================================
% Helper: YCoCg -> RGB (vector form, all 1xN)
% =====================================================================
function [R,G,B] = ycocg2rgb_sep(Y, Co, Cg)
    R = Y + Co - Cg;
    G = Y + Cg;
    B = Y - Co - Cg;
end

function [b_clamp, a_clamp] = map_luma_to_bin_alpha(Y, p)
% Map luma Y (HxW double) to:
%   b_clamp: integer bin index in [0..11]   (0-based)
%   a_clamp: alpha in [0..1] for interpolation between bin b and b+1
%
% Modes:
%   'uniform_linear' : u=12Y/256, b=floor(u), alpha=u-b (blend adjacent bins)
%   'doc_step'       : use doc nodes as thresholds, choose ONE bin (alpha=0)
%   'doc_linear'     : use doc nodes, find segment and alpha=(Y-Yk)/(Yk1-Yk)
%
% doc nodes default:
%   [15,31,47,63,95,127,159,191,223,239,247,255]

    if ~isfield(p, 'bin_mode')
        p.bin_mode = 'uniform_linear';
    end

    mode = lower(string(p.bin_mode));

    if ~isfield(p,'docY')
        p.docY = [15,31,47,63,95,127,159,191,223,239,247,255];
    end
    Ynode = double(p.docY(:).'); % 1x12
    assert(numel(Ynode)==12, 'p.docY must have 12 nodes.');
    assert(all(diff(Ynode) > 0), 'p.docY must be strictly increasing.');

    switch mode
        % =============================================================
        % Mode 1: uniform 12-bin + linear interpolation
        % =============================================================
        case "uniform_linear"
            u = Y * 12 / 256;        % continuous bin coordinate in [0,12)
            b = floor(u);            % nominal 0..11
            a = u - b;               % alpha 0..1

            b_clamp = b;
            a_clamp = a;

            % clamp to [0..11], last bin no blending
            mlow  = (b_clamp < 0);
            b_clamp(mlow) = 0;  a_clamp(mlow) = 0;

            mlast = (b_clamp >= 11);
            b_clamp(mlast) = 11; a_clamp(mlast) = 0;

            a_clamp = min(max(a_clamp,0),1);

        % =============================================================
        % Mode 2: doc nodes + step (strict doc, no interpolation)
        %
        % Interpretation (very common in ISP docs):
        %   pick the first node index j such that Y <= Ynode(j)
        %   use bin = j-1 (0-based), alpha=0.
        %
        % So boundaries are: (-inf..15]->bin0, (15..31]->bin1, ..., (247..255]->bin11
        % =============================================================
        case "doc_step"
            b_clamp = zeros(size(Y));
            a_clamp = zeros(size(Y));  % no interpolation

            % For each node j=1..12, assign pixels with Y <= Ynode(j) and not assigned yet.
            assigned = false(size(Y));

            for j = 1:12
                mj = (~assigned) & (Y <= Ynode(j));
                if any(mj(:))
                    b_clamp(mj) = j-1; % 0-based
                    assigned(mj) = true;
                end
            end

            % Any remaining (Y > last node) => last bin
            mrem = ~assigned;
            if any(mrem(:))
                b_clamp(mrem) = 11;
            end

        % =============================================================
        % Mode 3: doc nodes + linear interpolation (smooth)
        %
        % Segment rule:
        %   if Y <= Y1  -> bin0, alpha=0
        %   if Y >= Y12 -> bin11, alpha=0
        %   else find k in [1..11] s.t. Ynode(k) < = Y < Ynode(k+1)
        %        bin = k-1, alpha = (Y - Yk)/(Yk1 - Yk)
        % =============================================================
        case "doc_linear"
            b_clamp = zeros(size(Y));
            a_clamp = zeros(size(Y));

            mlow = (Y <= Ynode(1));
            mhi  = (Y >= Ynode(12));

            b_clamp(mlow) = 0;  a_clamp(mlow) = 0;
            b_clamp(mhi)  = 11; a_clamp(mhi)  = 0;

            mmid = ~(mlow | mhi);
            if any(mmid(:))
                Ymid = Y(mmid);

                btmp = zeros(size(Ymid));
                atmp = zeros(size(Ymid));

                % find segment
                for k = 1:11
                    Yk  = Ynode(k);
                    Yk1 = Ynode(k+1);
                    mk = (Ymid >= Yk) & (Ymid < Yk1);
                    if any(mk)
                        btmp(mk) = k-1; % 0-based bin
                        atmp(mk) = (Ymid(mk) - Yk) / (Yk1 - Yk); % [0,1)
                    end
                end

                b_clamp(mmid) = btmp;
                a_clamp(mmid) = atmp;
            end

            a_clamp = min(max(a_clamp,0),1);

        otherwise
            error('Unknown p.bin_mode: %s (use uniform_linear / doc_step / doc_linear)', p.bin_mode);
    end

    % Ensure integer bins and valid range
    b_clamp = floor(b_clamp);
    b_clamp = min(max(b_clamp,0),11);
end
