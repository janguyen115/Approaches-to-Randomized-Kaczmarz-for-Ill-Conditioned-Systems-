m = 240;
n = 12;

A = normalizeRows(generateRandomMatrix(m, n));
x_star = unifrnd(-1,1,[1,n])';
x_star = x_star/norm(x_star);
b = biSign(A * x_star);

A2 = matrixPlusPairComps(m,n,A);
A2 = normalizeRows(A2);
b2 = biSign(A2 * x_star);

x0 = randn(n,1);
x0 = x0/norm(x0);
beta = 3;
lambda = 1;
K = 1000;

numTrials = 20;

err_gP = zeros(numTrials, K); acc_gP = zeros(numTrials, K);
err_g = zeros(numTrials, K); acc_g = zeros(numTrials, K);

for i = 1:numTrials
    [~, err_gP(i, :), acc_gP(i, :)] = SamplingKaczmarzMotzkins(A2,b2,x0,beta,lambda,K,x_star);
    [~, err_g(i, :), acc_g(i, :)] = SamplingKaczmarzMotzkins(A,b,x0,beta,lambda,K,x_star);
end

mean_err_gP = mean(err_gP, 1); mean_acc_gP = mean(acc_gP, 1);
mean_err_g = mean(err_g, 1); mean_acc_g = mean(acc_g, 1);

k = 1:K;

figure('Position', [100, 100, 1200, 400]);

subplot(1,2,2);
plot(k, mean_err_gP, k, mean_err_g)
title('Average Error (Beta = 3.00, Lambda = 1.00)')
xlabel('Iteration')
ylabel('Average Error')
legend('G U P', 'G', 'Location', 'east')

subplot(1,2,1);
plot(k, mean_acc_gP, k, mean_acc_g)
title('Average Accuracy (Beta = 3.00, Lambda = 1.00)')
xlabel('Iteration')
ylabel('Average Accuracy')
legend('G U P', 'G', 'Location', 'east')

function output = matrixPlusPairComps(m,n,A)
    comp_index = nchoosek(1:m, 2);
    diff_matrix = zeros(length(comp_index), n);
    for k = 1:length(comp_index)
        i = comp_index(k, 1);
        j = comp_index(k, 2);
        diff_matrix(k, :) = A(i, :) - A(j, :);
    end
    output = [A;diff_matrix];
end

function [xk, err, acc] = SamplingKaczmarzMotzkins(A, b, x0, beta, lambda, k, x_star)
    b_mat = repmat(b,1,size(A,2)); A = -(b_mat.*A); b = zeros(size(A,1),1); %hadamard
    xk = x0; err = zeros(k, 1); acc = zeros(k, 1);
    for iter = 1:k 
        indices = randperm(size(A, 1), beta);
        rows_A = A(indices, :);
        residuals_A = rows_A * xk - b(indices);
        max_diff = max(residuals_A);
        row_max = rows_A(residuals_A == max_diff,:);
        max_diff(max_diff <= 0) = 0;
        xk = xk - lambda * max_diff * row_max';
        xk = xk / norm(xk);
        err(iter) = norm(xk - x_star)^2;
        acc(iter) = mean(biSign(A * xk) == biSign(A * x_star));
    end
end

function A_normalized = normalizeRows(A)
    row_norms = sqrt(sum(A.^2, 2));
    A_normalized = A ./ row_norms;
end

function return_vec = biSign(vec)
        return_vec = sign(vec);
        return_vec(return_vec == 0) = 1;
end

function [A] = generateRandomMatrix(m,n)
    [U, ~] = qr(randn(m,m));
    [V, ~] = qr(randn(n,n));
    s = (1:n).^2;            % Or some desired spectrum
    S = zeros(m,n);
    S(1:n, :) = diag(s);

    A = U * S * V';
end
