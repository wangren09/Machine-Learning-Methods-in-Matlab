%This main program groups the handwritten data into two groups: digit 1 and
% not digit 1 by using RBF-network method.
%N-1 crossvalidation is leveraged to select number of nearest neighbors
%We measure the test error at last

%Provided by: Ren Wang
%most recently updated time 11/24/2018

clear;clc;

[training_data, training_label, test_data, test_label, feature, n_row, n_column] = dataprocess();
[errorlist, best_k] = crossvalidation_rbf(training_data, training_label);

plot(1:15, errorlist, 'r');
xlabel('Number of centers')
ylabel('Error of cross validation (%)')
set(gcf,'unit','centimeters','position',[6 6 16 12]);
set(gca,'Position',[.125 .14 .77 .8]);


%plot the decision boundary
x=-1:0.001:1;
y=-1:0.001:1;
[X,Y]=meshgrid(x,y);
[n1, n2] = size(X);
ZZ = zeros(n1, n2);
[center, W] = RBF(training_data, training_label, best_k);
for i=1:size(Y,1)
    est_label = estimate_rbf(center, W, [X(i, :); Y(i, :)]);
    ZZ(i, :) = est_label;
end
figure;
plot(training_data(1,training_label == 1),training_data(2,training_label == 1),'bo');
hold on
plot(training_data(1,training_label == -1),training_data(2,training_label == -1),'rx');
contour(X,Y,ZZ,[0 0],'lineColor','m','linewidth',2);
xlabel('Intensity');
ylabel('Asymmetry');


%test error
est_label = estimate_rbf(center, W, test_data);
test_error = sum(abs(est_label' - test_label))/2;