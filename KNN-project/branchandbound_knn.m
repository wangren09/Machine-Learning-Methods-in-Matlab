function test_neighbor = branchandbound_knn(center, data_index, radius, data, testdata, neib_n)
%this function leverages brand and bound method to implement knn
% input: initialized centers, data belonging to each center (group), radius of
% each center, olddata, newdata, number of neighbors
% output: neighbors of test data
% the final label of test data can be calculated by labels of data
%Provided by: Ren Wang
%most recently updated time 11/23/2018


[dim, n_part] = size(center);
N_T = size(testdata, 2);


test_neighbor = zeros(neib_n, N_T);
dist = zeros(1, n_part);
for i = 1:N_T
    for k = 1:dim
        dist = (center(k,:)- testdata(k,i)).^2 + dist;
    end
    data_part = 1:n_part;
    dist_min = min(dist);
    data_part(dist - radius >= dist_min)=[];

    Idx_T = [];
    delta_T = [];
    for jj = 1:length(data_part)
        num = size(data_index{data_part(jj)}, 2);
        delta = zeros(1, num);
        for k = 1:dim
            delta = (data(k,data_index{data_part(jj)}) - testdata(k,i)).^2 + delta;
        end
        delta = sqrt(delta);
        delta_T = [delta_T delta];
        Idx_T = [Idx_T data_index{data_part(jj)}];
    end
    [min_delta, index_delta] = mink(delta_T, neib_n);
    test_neighbor(:,i) = Idx_T(index_delta);
end
