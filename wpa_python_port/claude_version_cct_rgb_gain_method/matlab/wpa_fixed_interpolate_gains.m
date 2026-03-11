function gain = wpa_fixed_interpolate_gains(luma_u8, gains_table, luma_nodes, interp, interp_bits)
%WPA_FIXED_INTERPOLATE_GAINS Integer 12-bin interpolation for per-pixel gains.

if nargin < 4
    interp = true;
end
if nargin < 5
    interp_bits = 10;
end

nodes = int32(luma_nodes(:)');
gains_table = int32(gains_table);
y = int32(luma_u8);
[h, w] = size(y);
gain = zeros(h, w, 3, 'int32');
half_interp = bitshift(int32(1), interp_bits - 1);

for row = 1:h
    for col = 1:w
        yc = y(row, col);
        idx_hi = find(nodes >= yc, 1, 'first');
        if isempty(idx_hi)
            idx_hi = numel(nodes);
        end
        if idx_hi == 1
            idx_lo = 1;
            idx_hi = min(2, numel(nodes));
        else
            idx_lo = idx_hi - 1;
        end

        if ~interp || idx_lo == idx_hi
            gain(row, col, :) = gains_table(idx_lo, :);
            continue;
        end

        node_lo = nodes(idx_lo);
        node_hi = nodes(idx_hi);
        span = max(node_hi - node_lo, 1);
        numer = bitshift(yc - node_lo, interp_bits) + bitshift(span, -1);
        t = min(max(idivide(numer, span, 'floor'), int32(0)), bitshift(int32(1), interp_bits));

        g_lo = int32(gains_table(idx_lo, :));
        g_hi = int32(gains_table(idx_hi, :));
        delta = g_hi - g_lo;
        gain(row, col, :) = g_lo + bitshift(int32(int64(t) .* int64(delta) + int64(half_interp)), -interp_bits);
    end
end
end
