function [center, data_index, radius] = ini_brandandbound(center, data)
%this function initializes the centers
% input: initialized centers, data
% output: final centers, data belonging to each center (group), radius of
% each center
%Provided by: Ren Wang
%most recently updated time 11/23/2018

[dim, n_part] = size(center);
N_T = size(data, 2);

% initialized the centers with a greedy method
for i = 2:n_part
    dis = zeros(N_T, i-1);
    for j = 1:i-1
        cur = 0;
        for k = 1:dim
            cur = (center(k, j) - data(k, :)).^2 +  cur;
        end
        dis(:,j) = cur;
    end
    [val, idx] = max(min(dis, [], 2));
    center(:, i) = data(:,idx);
end


%% update the centers
distance=zeros(n_part,N_T);
for i=1:n_part
    for j = 1:dim
        distance(i,:) = distance(i,:) + (data(j,:)-center(j,i)).^2;
    end
    distance(i,:) = sqrt(distance(i,:));
end
times=10;
while(times>=0)
    [~,index]=min(distance);
    for i=1:n_part
        center(:,i)=mean(data(:,index==i),2);
    end
    
    for i=1:n_part
        for j = 1:dim
            distance(i,:) = distance(i,:) + (data(j,:)-center(j,i)).^2;
        end
        distance(i,:) = sqrt(distance(i,:));
    end   
    times=times-1;
end

[~,index]=min(distance);
data_index=cell(1,n_part);      % record the indices of data belong to each cluster
for i=1:n_part
    data_index{i}=find(index==i);
end

radius=zeros(1,n_part);
for i=1:n_part
    radius(i)=max(distance(i,index==i));
end
