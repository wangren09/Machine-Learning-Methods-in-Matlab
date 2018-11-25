function [center, W] = RFB(data, label, k)

% this function implement RBF to select centers and coefficients
% input: training data, labels, number of centers
% output: centers and coefficients vector
%Provided by: Ren Wang
%most recently updated time 11/24/2018

[dim, N_T] = size(data);
center=zeros(dim, k);
for i = 1:k
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
r = 2/sqrt(k);
dist = zeros(k, N_T);
for i = 1:k
    for j = 1:dim
        dist(i, :) = (center(j,i)- data(j,:)).^2 + dist(i, :);
    end
    dist(i, :) = sqrt(dist(i, :));
end
gw = exp(-0.5*(dist./r).^2);
gw = gw';
W = inv(gw'*gw)*gw'*label';