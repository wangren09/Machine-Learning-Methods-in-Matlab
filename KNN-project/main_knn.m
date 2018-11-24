%This main program groups the handwritten data into two groups: digit 1 and
% not digit 1 by using KNN method.
%We use branch and bound method to reduce the complexity
%N-1 crossvalidation is leveraged to select number of nearest neighbors
%We measure the test error at last

%Provided by: Ren Wang
%most recently updated time 11/24/2018

clear;clc;

[training_data, training_label, test_data, test_label, feature, n_row, n_column] = dataprocess();


%Use branch and bound method to reduce the complexity
%initialize centers
n_part=2;
center=zeros(2,n_part);
for i = 1:n_part
    if rand > 0.5
        center(1,i) = rand;
    else
        center(1,i) = -rand;
    end
    if rand > 0.5
        center(2,i) = rand;
    else
        center(2,i) = -rand;
    end
end



figure;
plot(training_data(1,training_label == 1),training_data(2,training_label == 1),'b.');
hold on;
plot(training_data(1,training_label == -1),training_data(2,training_label == -1),'r.');
hold on;
plot(center(1,1:n_part),center(2,1:n_part),'ro','markersize',4,'linewidth',2.5);
title('Initially selected center points');

%% initialized the centers with a greedy method
[center, data_index, radius] = ini_brandandbound(center, training_data);


 figure;
plot(training_data(1,training_label == 1),training_data(2,training_label == 1),'b.');
hold on;
plot(training_data(1,training_label == -1),training_data(2,training_label == -1),'r.');
hold on;
 plot(center(1,1:n_part),center(2,1:n_part),'ro','markersize',8,'linewidth',2.5);
 title('Updated center points');

%Use branch and bound method to reduce the complexity


error = crossvalidation(center, training_data, training_label);
[min_err, best_k] = min(error);

plot(1:150, error/300, 'ro');
xlabel('Number of nearest neighbors')
ylabel('Error of cross validation (%)')
set(gcf,'unit','centimeters','position',[6 6 16 12]);
set(gca,'Position',[.125 .14 .77 .8]);

%plot the decision boundary
x=-1:0.001:1;
y=-1:0.001:1;
[X,Y]=meshgrid(x,y);
[n1, n2] = size(X);
ZZ = zeros(n1, n2);

for i=1:size(Y,1)
    neighbor = branchandbound_knn(center, data_index, radius, training_data, [X(i, :); Y(i, :)], best_k);
    ZZ(i, :) = sign(sum(training_label(neighbor),1));
end
figure;
plot(training_data(1,training_label == 1),training_data(2,training_label == 1),'bo');
hold on
plot(training_data(1,training_label == -1),training_data(2,training_label == -1),'rx');
contour(X,Y,ZZ,[0 0],'lineColor','m','linewidth',2);
xlabel('Intensity');
ylabel('Asymmetry');


%test error
neighbor = branchandbound_knn(center, data_index, radius, training_data, test_data, best_k);
test_sign = sign(sum(training_label(neighbor),1));
test_error = sum(abs(test_sign - test_label))/2;
