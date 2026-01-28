I = imread('grass.jpg'); % uint8 RGB, [0..255]

% 1) RGB -> YCoCg
[Y,Co,Cg] = rgb2ycocg(I);

% 2) Build 12-bin Kelvin tables (offline step)
opts = struct();
opts.T_warm_end  = 3500;  % can change later
opts.T_cool_end  = 9000;  % can change later
opts.xy_base     = [0.3127 0.3290]; % D65
opts.w_shadow    = 0.60;
opts.w_mid       = 1.00;
opts.w_highlight = 0.70;
opts.binMid      = 6;

% 离线表仍用你原来的 build 函数
tbl = wpa_build_tables_kelvin_12bin(opts);

% 在线改用无断层版
p = struct();
p.RGB_MAX = 255;
p.Y_dark2 = 8; p.Y_dark1 = 16;
p.Y_bright1 = 240; p.Y_bright2 = 248;
p.bright_minfac = 0.25;
p.keepY = 0;

% 建议先关闭 s 量化验证最平滑
p.s_quant_en = 0;     % 如果要模拟硬件步进，改为 1
p.s_quant_Q  = 64;

wa_sel = 30; wa_en = 1;
[Y2,Co2,Cg2] = wpa_apply_ycocg_lms_nobanding(Y,Co,Cg, wa_sel, wa_en, tbl, p);

I2 = ycocg2rgb(Y2,Co2,Cg2);
I2 = uint8(min(max(I2,0),255));

figure;
subplot(121);
imshow(I);title('Original');
subplot(122);
imshow(I2);title('Processed');
