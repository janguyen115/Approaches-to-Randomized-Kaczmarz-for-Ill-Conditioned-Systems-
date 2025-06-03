function [] = avgOverTrials()
    m = 160; n = 12; numTrials = 100; maxIter = 1e4;

    errMat = zeros(maxIter, n);
    errWeightedMat = zeros(maxIter, n);
    appErrVec = zeros(maxIter, 1);
    appErrWeightedVec = zeros(maxIter, 1);
    
    for i = 1:numTrials
        %%% Redefine new Matrix
        [U, ~] = qr(randn(m,m));
        [V, ~] = qr(randn(n,n));
        s = (1:n).^2;%linspace(1, n, n);           % Or some desired spectrum
        S = zeros(m,n);    
        S(1:n, :) = diag(s);   
        xstar = unifrnd(-1,1, [n,1]); xstar = xstar / norm(xstar);
        A = U * S * V';
        Mat = A;
        %%% Results
        [errWeightedMat_temp, errMat_temp, appErrWeighted_temp, appErr_temp] = singularSamplingAnalysis(Mat, m, n, maxIter); 
        errWeightedMat = errWeightedMat + errWeightedMat_temp;
        errMat = errMat + errMat_temp;

        appErrWeightedVec = appErrWeightedVec + appErrWeighted_temp;
        appErrVec = appErrVec + appErr_temp;
    end

    err = errMat / numTrials;
    errWeighted = errWeightedMat / numTrials;

    appErrWeightedVec = appErrWeightedVec / numTrials;
    appErrVec = appErrVec / numTrials;

    figure;
    subplot(1,3,1);
    for i = [1:3, n-2:n]
        semilogy(1:maxIter, err(:,i), 'DisplayName', ['RK: \Sigma', num2str(i)]);
        hold on; 
    end
    ylabel('|\langle (x_k-x*), v_k \rangle|')
    xlabel('Iterations')
    grid on;
    legend;

    subplot(1,3,2);
    for i = [1:n/2]
        semilogy(1:maxIter, errWeighted(:,i), 'DisplayName', ['Weighted: \Sigma', num2str(i)]);
        hold on;
    end
    ylabel('|\langle (x_k-x*), v_k \rangle|')
    xlabel('Iterations')
    grid on;
    legend;

    subplot(1,3,3);
    semilogy(1:maxIter, appErrWeightedVec, 'DisplayName', 'Weighted')
    hold on;
    semilogy(1:maxIter, appErrVec, 'DisplayName', 'RK')
    grid on;
    ylabel('|x-x^*|')
    title('Approximation Error')
    legend;
end