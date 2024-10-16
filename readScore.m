function [rank] = readScore(score)
    % Extract indices in descending order
    [~, sorted_indices] = sort(score, 'descend');
    
    % Initialize rank vector
    rank = zeros(size(score));
    
    % Populate vector
    rank(sorted_indices) = 1:length(score);
end