function gain = wpa_fixed_interpolate_gains(luma_u8, gains_table, luma_nodes, interp, interp_bits)
%WPA_FIXED_INTERPOLATE_GAINS Integer 12-bin interpolation for per-pixel gains.
% Runtime storage: luma_u8/nodes uint8, gains uint16, delta int32.

if nargin < 4
    interp = true;
end
if nargin < 5
    interp_bits = 10;
end

nodes = uint8(luma_nodes(:)');
gains_table = uint16(gains_table);
y = uint8(luma_u8);
[h, w] = size(y);
gain = zeros(h, w, 3, 'uint16');
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

        node_lo = uint16(nodes(idx_lo));
        node_hi = uint16(nodes(idx_hi));
        span = max(node_hi - node_lo, uint16(1));
        numer = bitshift(uint32(uint16(yc) - node_lo), interp_bits) + bitshift(uint32(span), -1);
        t = min(idivide(numer, uint32(span), 'floor'), bitshift(uint32(1), interp_bits));

        g_lo = int32(gains_table(idx_lo, :));
        g_hi = int32(gains_table(idx_hi, :));
        delta = g_hi - g_lo;
        gain(row, col, :) = uint16(g_lo + bitshift(int32(int64(t) .* int64(delta) + int64(half_interp)), -interp_bits));
    end
end
end
