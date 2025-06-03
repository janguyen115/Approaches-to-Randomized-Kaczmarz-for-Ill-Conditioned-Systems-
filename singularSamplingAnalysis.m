function [outputErrWeighted, outputErr, appErr_weighted, appErr] = main(Mat, m, n, maxIter)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Generate Ill-Conditioned Matrix

    A = Mat;

    [U,S,V] = svd(A);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    [m,n] = size(A);

    %sprintf('K(A): %s', cond(A))

    xstar = eye(n); xstar = xstar(:,end); % set solution vector to e_n
    b = A * xstar;


    C = singularCoefficients(A);
    sampleWeights = abs(C(:,n)); % last column of C denotes coefficients for the last singular vector
    sampleWeights = sampleWeights ./ sum(sampleWeights);

    %sprintf("sum weights: %.2f", sum(sampleWeights)) % verify weights sum to 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% weighted by singualar
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% coefficients

    appErr_weighted = zeros(maxIter, 1); 
    err_weighted = zeros(maxIter, n); 
    k = 1;
    xk = zeros(n,1);

    while k <= maxIter
        i = randsample(1:m, true, 1, sampleWeights);
        row = A(i, :);
        residual = dot(row, xk) - b(i);

        xk = xk - residual / norm(row)^2 * row';
        appErr_weighted(k) = norm(xk - xstar);
        err_weighted(k,:) = (xk-xstar);
        k = k+1;
    end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% uniform randomized kaczmarz

    appErr = zeros(1, maxIter);
    err = zeros(maxIter, n);
    xk = zeros(n,1);  
    k = 1;

    while k <= maxIter
        i = randperm(m, 1);
        row = A(i, :);
        residual = dot(row, xk) - b(i);
        xk = xk - residual / norm(row)^2 * row';
        appErr(k) = norm(xk - xstar);
        err(k,:) = (xk - xstar);
        k = k+1;
    end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% subset by quantile greatest
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% singular coefficients
    
    % sortedWeights = sort(sampleWeights, 'descend');
    % mask = sampleWeights >= sortedWeights(2);
    % Subset = A(mask,:);
    % appErr_subset = zeros(1, maxIter);
    % xk = zeros(n,1);

    
    % k = 1;
    % while k <= maxIter
    %     i = randperm(2, 1);
    %     row = Subset(i, :);
    %     residual = dot(row, xk) - b(i);
    %     xk = xk - residual * row';
    %     appErr_subset(k) = norm(xk - xstar);
    %     k = k+1;
    % end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

    % sprintf("max dot product: %.4f", abs(dot(A(sampleWeights == max(sortedWeights),:), xstar)))
    % sprintf("min dot product: %.4f", abs(dot(A(sampleWeights == min(sortedWeights),:), xstar)))

    % 
    % figure;
    % subplot(1,4,1);
    % hold on;
    % plot(1:maxIter, appErr_weighted, 'DisplayName', 'Weighted by Coefficient of nth Singular Vector')
    % plot(1:maxIter, appErr, 'DisplayName', 'Uniform Sampling Randomized Kaczmarz', 'LineWidth', 2)
    % % plot(1:maxIter, appErr_subset, 'DisplayName', '100 Largest Weights Subset')
    % xlabel('Iteration (k)')
    % ylabel('Approximation Error: norm(xk - xstar)')
    % legend()
    % grid on;
    % 
    % subplot(1,4,2);
    % plot(1:n, svd(A))
    % ylabel('singular values')
    % 
    % subplot(1,4,3);
    % plot(1:m, sampleWeights)
    % ylabel('weights')
    % 
    % subplot(1,4,4);
    % for k = 1
    %     semilogy(1:maxIter, abs(err * V(k,:)'), 'DisplayName', 'RK');
    %     hold on;
    %     semilogy(1:maxIter, abs(err_weighted * V(k,:)'), 'DisplayName', 'Weighted');
    % 
    % end
    % grid on;

    
    outputErrWeighted = abs(V'*err_weighted')'; % singular error, dotted with each singular vector
    outputErr = abs(V'*err')';  % singular error, dotted with each singular vector

    appErr = appErr';

end

function [C] = singularCoefficients(A) % considering SVD is given
    [~, ~, V] = svd(A);
    C = A * V;
end