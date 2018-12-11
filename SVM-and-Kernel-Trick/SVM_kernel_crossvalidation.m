% this function implement the N-1 crossvalidation of SVM to select the
% 'regularization' parameter
% This main program groups the handwritten data into two groups: digit 1 and
% not digit 1 by using SVM with 8th order polynomial kernel.

%Provided by: Ren Wang
%most recently updated time 11/24/2018

clear;clc;

%[training_data, training_label, test_data, test_label, feature, n_row, n_column] = dataprocess();
load training_data
load training_label
[n_row, n_col] = size(training_data);
C = [0.001 0.002 0.005 0.01 0.05 0.1 0.5 1 5 10 100];
test_error_list = zeros(1, length(C));
for kk = 1:length(C)
test_error = 0;
for ii = 1:n_col
    training_data_new = [training_data(:,1:ii-1) training_data(:,ii+1:end)];
    training_label_new = [training_label(1:ii-1) training_label(ii+1:end)];
    G = zeros(n_col-1, n_col-1);
    kernel = zeros(n_col-1, n_col-1);
    for i = 1:n_col-1
        for j = 1:n_col-1
            kernel(i, j) = (1 + training_data_new(:, i)'*training_data_new(:, j)).^8;
            G(i, j) = kernel(i, j)*training_label_new(i)*training_label_new(j);
        end
    end
    f = -ones(n_col-1, 1);
    %A = [-1*eye(n_col-1, n_col-1); eye(n_col-1, n_col-1)];
    %b = [zeros(n_col-1, 1); C(kk)*ones(n_col-1, 1)];
    A_eq = training_label_new;
    b_eq = 0;
    lb = zeros(n_col-1, 1);
    ub = C(kk)*ones(n_col-1, 1);
    alpha = quadprog(G, f, [], [], A_eq, b_eq, lb, ub);
    sig_test = SVM_Decision(alpha, training_data(:,ii), kernel, training_data_new, training_label_new);
    test_error = test_error + abs(sig_test - training_label(ii))/2;
end
test_error_list(kk) = test_error/n_col;
end
plot(C, test_error_list)