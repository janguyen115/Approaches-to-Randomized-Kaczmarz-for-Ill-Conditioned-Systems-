function mat_norm = stdMatrix(mat)
        row_norms = sqrt(sum(mat.^2, 2));
        mat_norm = mat ./ row_norms;
end