function est_label = estimate_rbf(center, W, data)
% this function implement RBF to estimate the data labels
% input: center, coefficients, data to estimate
% output: data label estimation
%Provided by: Ren Wang
%most recently updated time 11/24/2018


[dim, N_T] = size(data);
[~, k] = size(center);
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
est_label = sign(gw*W);
