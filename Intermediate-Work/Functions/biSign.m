function [vec] = biSign(vec)
    vec = sign(vec);
    vec(vec == 0) = -1;
end
