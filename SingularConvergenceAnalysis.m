function [] = main()
    m = 100; n = 5;
    x0 = ones(n, 1);
    maxIter = 1e2;
    numTrials = 100;
    resultsMatList = {};

    for trial = 1:numTrials
        [A, xstar, b] = setupProblem(m, n);
        [~, ~, V] = svd(A);
        X = randomizedKaczmarz(A, b, x0, maxIter);
        resMat = X - repmat(xstar, 1, maxIter);
        resultsMatList{end+1} = abs(V * resMat); % n x maxIter
    end

    sprintf('Final Approximation Error:%0.3f', norm(X(:,end) - xstar))

    avgMat = averageMatricesInList(resultsMatList);
    Iters = 1:maxIter;

    figure;
    for k = 1:n
        semilogy(Iters, avgMat(k, :), 'DisplayName', ['\Sigma_', [num2str(k)]])
        hold on;
    end
    xlabel('Iterations')
    ylabel('|\langle (x_k-x*), v_k \rangle|')
    title('Singular Covergence Analysis')
    grid on;
    legend;
end

function [A, xstar, b] = setupProblem(m, n)
    [U, ~] = qr(randn(m,m));
    [V, ~] = qr(randn(n,n));
    s = linspace(1, 10, n);           % Or some desired spectrum
    S = zeros(m,n);
    singular_values = 1:n ;
    singular_values = sort(singular_values, 'descend'); % ensure last singular value is <1

    S(1:n, :) = diag(singular_values);

    xstar = unifrnd(-1,1, [n,1]); xstar = xstar / norm(xstar);

    A = U * S * V';
    b = A * xstar;
end

function [X] = randomizedKaczmarz(A, b, x0, maxIter, dist)
    [m, n] = size(A);
    if nargin < 5
        dist = ones([m,1]) / m;
    end

    X = zeros(n, maxIter);
    X(:, 1) = x0;

    for k = 1:(maxIter-1)
        i = randsample(m, 1, true, dist);
        row = A(i,:);
        residual = row * X(:,k) - b(i);
        X(:, k+1) = X(:, k) - residual / norm(row)^2 * row';
        X(:, k+1) = X(:, k+1) / norm(X(:, k+1));
    end
end

function [avgMat] = averageMatricesInList(matrixList)
    sumMatrix = zeros(size(matrixList{1}));
    
    for i = 1:length(matrixList)
        sumMatrix = sumMatrix + matrixList{i};
    end
    
    avgMat = sumMatrix / length(matrixList);
end
