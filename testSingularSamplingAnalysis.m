function [] = main()

    %%%%%% test parameters
    m = 240; n = 12; 
    numTrials = 50; maxIter = 1e5;

    %%%%%% initialize data arrays
    errMat = zeros(maxIter, n);                 % |<x_k-x*, v_k>| for each column k in 1:n
    errWeightedMat = zeros(maxIter, n);
    appErrVec = zeros(maxIter, 1);              % approximation error |x_k-x*| at each iteration
    appErrWeightedVec = zeros(maxIter, 1);
    
    %%%%%% loop over numTrials
    for i = 1:numTrials

        A = generateRandomMatrix(m,n);  
        
        %%%%%% results
        [errWeightedMat_temp, errMat_temp, appErrWeighted_temp, appErr_temp] = singularSamplingAnalysis(A, maxIter); 
        errWeightedMat = errWeightedMat + errWeightedMat_temp;
        errMat = errMat + errMat_temp;

        %%%%%% sum trial data
        appErrWeightedVec = appErrWeightedVec + appErrWeighted_temp;
        appErrVec = appErrVec + appErr_temp;
    end

    %%%%%% divide by numTrials to average trial data
    err = errMat / numTrials;
    errWeighted = errWeightedMat / numTrials;

    appErrWeightedVec = appErrWeightedVec / numTrials;
    appErrVec = appErrVec / numTrials;
    
    %%%%%% plot
    figure;
    for i = [1:3, n-2:n]
        semilogy(1:maxIter, err(:,i), 'DisplayName', ['j=', num2str(i)]);
        hold on; 
    end
    ylabel('|\langle (x_k-x*), v_j \rangle|')
    xlabel('Iterations (k)')
    title('Directional Convergence: RK - Uniform Sampling')
    grid on;
    legend;

    %%%%%% compare approximation error between RK and singularSamplingKaczmarz
    

    %%%%%% 
    figure;
    subplot(1,2,1);
    semilogy(1:maxIter, errWeighted(:,n), 'DisplayName', 'Weighted RK');
    hold on; 
    semilogy(1:maxIter, err(:,n), 'DisplayName', 'Uniform RK');
    grid on;
    xlabel('Iterations (k)')
    ylabel('|\langle x_k-x^*, v_n \rangle|')
    title('Directional Convergence - v_n');
    legend;

    subplot(1,2,2);
    semilogy(1:maxIter, appErrWeightedVec.^2, 'DisplayName', 'Weighted RK')
    hold on;
    semilogy(1:maxIter, appErrVec.^2, 'DisplayName', 'Uniform RK')
    grid on;
    xlabel('Iterations (k)')
    ylabel('||x_k-x^*||^2')
    title('Approximation Error by Algorithm')
    legend;
end

function [A] = generateRandomMatrix(m,n)
    [U, ~] = qr(randn(m,m));
    [V, ~] = qr(randn(n,n));
    s = (1:n).^2;            % Or some desired spectrum
    S = zeros(m,n);    
    S(1:n, :) = diag(s);   

    A = U * S * V';
end