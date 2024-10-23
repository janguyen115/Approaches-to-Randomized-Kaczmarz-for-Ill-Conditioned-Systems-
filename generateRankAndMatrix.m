function [ranking, comparisonMatrix] = generateRankAndMatrix(n)
    % Generate a random ranking vector (a permutation of 1:n)
    ranking = randperm(n);
    
    % Preallocate the pairwise comparison matrix
    numComparisons = nchoosek(n, 2);  % Number of pairwise comparisons
    comparisonMatrix = zeros(numComparisons, n);
    
    % Generate pairwise comparisons based on the ranking
    row = 1;
    for i = 1:n
        for j = i+1:n
            if ranking(i) < ranking(j)
                comparisonMatrix(row, i) = -1;  % i loses to j
                comparisonMatrix(row, j) = 1; % j beats i
            else
                comparisonMatrix(row, i) = 1; % i > j
                comparisonMatrix(row, j) = -1;  % j < i
            end
            row = row + 1;
        end
    end
end