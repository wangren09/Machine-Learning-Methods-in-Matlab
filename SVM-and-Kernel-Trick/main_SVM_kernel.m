%This main program groups the handwritten data into two groups: digit 1 and
% not digit 1 by using SVM with 8th order polynomial kernel.

%Provided by: Ren Wang
%most recently updated time 11/24/2018

clear;clc;

%[training_data, training_label, test_data, test_label, feature, n_row, n_column] = dataprocess();
load training_data
load training_label
[n_row, n_col] = size(training_data);

% transfor to a duel problem and use the quadratic programing
G = zeros(n_col, n_col);
kernel = zeros(n_col, n_col);
for i = 1:n_col
    for j = 1:n_col
        kernel(i, j) = (1 + training_data(:, i)'*training_data(:, j)).^8;
        G(i, j) = kernel(i, j)*training_label(i)*training_label(j);
    end
end
f = -ones(n_col, 1);
%A = [-1*eye(n_col, n_col); eye(n_col, n_col)];
%b = [zeros(n_col, 1); 10*ones(n_col, 1)];
A_eq = training_label;
b_eq = 0;
lb = zeros(n_col, 1);
ub = 100*ones(n_col, 1);
alpha = quadprog(G, f, [], [], A_eq, b_eq, lb, ub);


% plot the decision boundary
x=-1:0.001:1;
y=-1:0.001:1;
[X,Y]=meshgrid(x,y);
[n1, n2] = size(X);
ZZ = zeros(n1, n2);
for k=1:size(Y,1)
    for j=1:size(Y,1)
        ZZ(k, j) = SVM_Decision(alpha, [X(k,j); Y(k,j)], kernel, training_data, training_label);
    end
end

figure;
plot(training_data(1,training_label == 1),training_data(2,training_label == 1),'bo');
hold on
plot(training_data(1,training_label == -1),training_data(2,training_label == -1),'rx');
contour(X,Y,ZZ,[0 0],'lineColor','m','linewidth',2);
xlabel('Intensity');
ylabel('Asymmetry');
