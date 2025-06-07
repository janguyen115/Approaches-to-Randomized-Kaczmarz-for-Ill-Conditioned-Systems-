function [outputErrWeighted, outputErr, appErr_weighted, appErr] = singularSamplingAnalysis(A, maxIter)

    [~,~,V] = svd(A);
    [m, n] = size(A);

    xstar = eye(n); xstar = xstar(:,end); % set solution vector to e_n
    b = A * xstar;

    %%%%%% given SVD, extract singular coefficients c_j (j = 1:n) from a_i = c_1(v_1) + ... + c_n(v_n)

    C = A * V;                      % C := singular coefficient matrix
    sampleWeights = abs(C(:,n));    % last column of C denotes coefficients for the last singular vector
    sampleWeights = sampleWeights ./ sum(sampleWeights);

    %%%%%% RK (singularly-weighted sampling)

    appErr_weighted = zeros(maxIter, 1);    % |x_k-x^*|
    err_weighted = zeros(maxIter, n);       % |<x_k-x^*, v_k>|
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

    %%%%%% RK (uniform)

    appErr = zeros(maxIter, 1);             % |x_k-x^*|
    err = zeros(maxIter, n);                % |<x_k-x^*, v_k>|        
    k = 1;
    xk = zeros(n,1);

    while k <= maxIter
        i = randperm(m, 1);
        row = A(i, :);
        residual = dot(row, xk) - b(i);
        xk = xk - residual / norm(row)^2 * row';
        appErr(k) = norm(xk - xstar);
        err(k,:) = (xk - xstar);
        k = k+1;
    end

    %%%%%% transform error matrix to |<x_k-x^*, v_k>|
    
    outputErrWeighted = abs(V' * err_weighted')'; 
    outputErr = abs(V' * err')';  


end
