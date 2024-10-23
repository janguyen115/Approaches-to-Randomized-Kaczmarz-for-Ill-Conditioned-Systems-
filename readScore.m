function [rank] = readScore(score)
    n = length(score);
    rank = zeros(1, n); % initialize
    [~, index] = sort(score, 'descend'); % sort in descending order
    for i = 1:n
        rank(index(i)) = i;
    end
end
