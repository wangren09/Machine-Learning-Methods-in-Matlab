function error_k = crossvalidation(center, data, label)
% this function implement the N-1 crossvalidation of knn to select the
% number of neighbors k
% input: data belonging to each center (group), training data, labels
% output: errors for each k
%Provided by: Ren Wang
%most recently updated time 11/23/2018

[n1, n2] = size(data);
error_k = zeros(1, 1);
for k = 1:150
    error = 0;
    for i = 1:n2
        data_val = [data(:,1:i-1) data(:, i+1:n2)];
        label_val = [label(1:i-1) label(i+1:n2)];
        %% initialized the centers with a greedy method
        [center, data_index, radius] = ini_brandandbound(center, data_val);
        neighbor = branchandbound_knn(center, data_index, radius, data_val, data(:, i), k);
        lab = sign(sum(label_val(neighbor)));
        if lab ~= label(i)
            error = error + 1;
        end
    end
    error_k(k) = error;
end

