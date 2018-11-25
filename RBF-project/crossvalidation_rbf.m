function [errorlist, bestk] = crossvalidation_rbf(data, label)
% this function implement the N-1 crossvalidation of RBF to select the
% number of centers k(n_part)
% input: training data, labels
% output: errors for each k, best choice of k
%Provided by: Ren Wang
%most recently updated time 11/24/2018

[dim, N_T] = size(data);
errorlist = zeros(1, 15);
for n_part = 1:15
    center=zeros(dim,n_part);
    for i = 1:n_part
        for j = 1:dim
            if rand > 0.5
                center(j,i) = rand;
            else
                center(j,i) = -rand;
            end
        end
    end
    
    [center, data_index, radius] = Kmeans(center, data);
    % compute the weights of each training data to the centers
    r = 2/sqrt(n_part);
    dist = zeros(n_part, N_T);
    for i = 1:n_part
        for k = 1:dim
            dist(i, :) = (center(k,i)- data(k,:)).^2 + dist(i, :);
        end
        dist(i, :) = sqrt(dist(i, :));
    end
    gw = exp(-0.5*(dist./r).^2);
    gw = gw';
    error = 0;
    for i = 1:N_T
        gwn = [gw(1:i-1,:); gw(i+1:N_T,:)];
        label_n = [label(:,1:i-1) label(:,i+1:N_T)]';
        w = inv(gwn'*gwn)*gwn'*label_n;
        error = error + abs(sign(gw(i,:)*w) - label(i))/2;
    end
    errorlist(n_part) = error / N_T;
end
[~, bestk] = min(errorlist); 