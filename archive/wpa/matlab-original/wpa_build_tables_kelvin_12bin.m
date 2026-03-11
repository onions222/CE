function tbl = wpa_build_tables_kelvin_12bin(opts)
% Build 12-bin endpoint tables for WPA in YCoCg domain,
% where the endpoint is defined by LMS white-point adjustment.
%
% The runtime will do:
%   y' = y + s * (T_end - I) * y
% where s depends on WA_SEL, bin weight, and protections.
%
% Required opts (you can omit and use defaults):
%   opts.T_warm_end  (default 3500)
%   opts.T_cool_end  (default 9000)
%   opts.xy_base     (default D65 = [0.3127 0.3290])
%   opts.w_shadow, opts.w_mid, opts.w_highlight, opts.binMid
%
% Output tbl:
%   tbl.w(1x12)
%   tbl.DmatWarm{1..12}  (3x3)
%   tbl.DmatCool{1..12}  (3x3)
%   tbl.Twarm{1..12}     (3x3) full endpoint transform
%   tbl.Tcool{1..12}     (3x3)

    % ---------- defaults ----------
    if ~isfield(opts,'T_warm_end'),    opts.T_warm_end = 3500; end
    if ~isfield(opts,'T_cool_end'),    opts.T_cool_end = 9000; end
    if ~isfield(opts,'xy_base'),       opts.xy_base = [0.3127 0.3290]; end % D65
    if ~isfield(opts,'w_shadow'),      opts.w_shadow = 0.60; end
    if ~isfield(opts,'w_mid'),         opts.w_mid = 1.00; end
    if ~isfield(opts,'w_highlight'),   opts.w_highlight = 0.70; end
    if ~isfield(opts,'binMid'),        opts.binMid = 6; end

    % ---------- YCoCg matrices matching your rgb2ycocg/ycocg2rgb ----------
    % y = M_R2Y * rgb, rgb = M_Y2R * y
    M_R2Y = [ 0.25, 0.50, 0.25;
              0.50, 0.00,-0.50;
             -0.25, 0.50,-0.25 ];
    M_Y2R = [ 1,  1, -1;
              1,  0,  1;
              1, -1, -1 ];

    % ---------- Rec.709/sRGB linear RGB <-> XYZ ----------
    % If you want "custom" later: we can replace these with primaries+white computation.
    M_R2X = [0.4124564 0.3575761 0.1804375;
             0.2126729 0.7151522 0.0721750;
             0.0193339 0.1191920 0.9503041];
    M_X2R = inv(M_R2X);

    % ---------- Bradford CAT XYZ->LMS ----------
    M_X2L = [ 0.8951  0.2664 -0.1614;
             -0.7502  1.7135  0.0367;
              0.0389 -0.0685  1.0296 ];
    M_L2X = inv(M_X2L);

    % ---------- base white XYZ (Y=1) ----------
    XYZ_base = xy_to_XYZ_Y1(opts.xy_base(1), opts.xy_base(2));
    LMS_base = M_X2L * XYZ_base;

    % ---------- 12-bin weights ----------
    w = build_weights_12(opts.w_shadow, opts.w_mid, opts.w_highlight, opts.binMid);

    % ---------- base CCT estimate for interpolation anchor ----------
    T_base = xy_to_cct_mccamy(opts.xy_base(1), opts.xy_base(2));

    % ---------- per-bin CCT by mired interpolation ----------
    Tw = interp_cct_mired(T_base, opts.T_warm_end, w);
    Tc = interp_cct_mired(T_base, opts.T_cool_end, w);

    % ---------- allocate output ----------
    tbl = struct();
    tbl.opts = opts;
    tbl.w = w;
    tbl.T_base = T_base;
    tbl.Tw = Tw;
    tbl.Tc = Tc;

    tbl.xy_warm = zeros(12,2);
    tbl.xy_cool = zeros(12,2);

    tbl.Twarm = cell(1,12);
    tbl.Tcool = cell(1,12);
    tbl.DmatWarm = cell(1,12);
    tbl.DmatCool = cell(1,12);

    % ---------- build endpoint transforms per bin ----------
    for i = 1:12
        xy_w = cct_to_xy_kang2002(Tw(i));
        xy_c = cct_to_xy_kang2002(Tc(i));
        tbl.xy_warm(i,:) = xy_w;
        tbl.xy_cool(i,:) = xy_c;

        XYZ_w = xy_to_XYZ_Y1(xy_w(1), xy_w(2));
        XYZ_c = xy_to_XYZ_Y1(xy_c(1), xy_c(2));

        LMS_w = M_X2L * XYZ_w;
        LMS_c = M_X2L * XYZ_c;

        r_w = LMS_w ./ LMS_base; % 3x1 gains
        r_c = LMS_c ./ LMS_base;

        % XYZ adaptation matrices
        A_w = M_L2X * diag(r_w) * M_X2L;
        A_c = M_L2X * diag(r_c) * M_X2L;

        % Final YCoCg-domain endpoint transforms:
        % y' = M_R2Y * M_X2R * A * M_R2X * M_Y2R * y
        T_w = M_R2Y * M_X2R * A_w * M_R2X * M_Y2R;
        T_c = M_R2Y * M_X2R * A_c * M_R2X * M_Y2R;

        tbl.Twarm{i} = T_w;
        tbl.Tcool{i} = T_c;

        tbl.DmatWarm{i} = T_w - eye(3);
        tbl.DmatCool{i} = T_c - eye(3);
    end
end

% ================= helper functions =================

function w = build_weights_12(w_shadow, w_mid, w_highlight, binMid)
% Build 12 weights in [0,1] with piecewise-linear curve:
%   bin1 -> binMid: w_shadow -> w_mid
%   binMid -> bin12: w_mid -> w_highlight
    w = zeros(1,12);
    for i=1:12
        if i <= binMid
            t = (i-1) / max(1,(binMid-1));
            w(i) = w_shadow + t*(w_mid - w_shadow);
        else
            t = (i-binMid) / max(1,(12-binMid));
            w(i) = w_mid + t*(w_highlight - w_mid);
        end
    end
    w = min(max(w,0),1);
end

function T = interp_cct_mired(T_base, T_end, w)
% Interpolate in mired domain: m = 1e6 / K
    m_base = 1e6 / T_base;
    m_end  = 1e6 / T_end;
    m = (1 - w).*m_base + w.*m_end;
    T = 1e6 ./ m;
end

function XYZ = xy_to_XYZ_Y1(x, y)
% Convert xy to XYZ with Y normalized to 1
    Y = 1.0;
    X = (x / y) * Y;
    Z = ((1 - x - y) / y) * Y;
    XYZ = [X; Y; Z];
end

function T = xy_to_cct_mccamy(x, y)
% McCamy approximation (xy -> CCT). Used only to get a reasonable T_base.
    n = (x - 0.3320) / (0.1858 - y);
    T = 449*n^3 + 3525*n^2 + 6823.3*n + 5520.33;
end

function xy = cct_to_xy_kang2002(T)
% Kang 2002 approximation, valid 1667..25000K
    if T < 1667 || T > 25000
        error('Kang2002 valid range is 1667..25000K.');
    end

    if T <= 4000
        x = -0.2661239e9 / T^3 -0.2343589e6 / T^2 +0.8776956e3 / T + 0.179910;
    else
        x = -3.0258469e9 / T^3 +2.1070379e6 / T^2 +0.2226347e3 / T + 0.240390;
    end

    if T <= 2222
        y = -1.1063814*x^3 -1.34811020*x^2 +2.18555832*x -0.20219683;
    elseif T <= 4000
        y = -0.9549476*x^3 -1.37418593*x^2 +2.09137015*x -0.16748867;
    else
        y =  3.0817580*x^3 -5.8733867*x^2 +3.75112997*x -0.37001483;
    end

    xy = [x, y];
end
