function [] = avgOverTrials(Mat, maxIter, numTrials)
    [m,n] = size(Mat);

    errMat = zeros(maxIter, numTrials);
    errWeightedMat = zeros(maxIter, numTrials);

    for i = 1:numTrials
        [errWeightedMat(:,i), errMat(:,i)] = singularSamplingAnalysis(Mat, m, n, maxIter);
    end

    err = sum(errMat, 2) / numTrials;
    errWeighted = sum(errWeightedMat, 2) / numTrials;
    figure;
    semilogy(1:maxIter, err, 'DisplayName', 'RK');
    hold on;
    semilogy(1:maxIter, errWeighted, 'DisplayName', 'Weighted');
    grid on;
end