function [center, data_index, radius] = Kmeans(center, data)
% input: initialized centers, data
% output: final centers, data belonging to each center (group), radius of
% each center
%Provided by: Ren Wang
%most recently updated time 11/23/2018

[dim, n_part] = size(center);
N_T = size(data, 1);

%distance of data to centers
distance=zeros(n_part,N_T);
for i=1:n_part
    for j = 1:dim
        distance(i,:) = distance(i,:) + (data(j,:)-center(j,i)).^2;
    end
    distance(i,:) = sqrt(distance(i,:));
end

[mindist,index]=min(distance);
E_in = sum(mindist.^2);
E_in_new = 0;
times=1;
while(E_in_new < E_in || times>0)
    E_in = E_in_new;
    for i=1:n_part
        center(:,i)=mean(data(:,index==i),2);
    end
    
    % update the decision centers and compute the radius
    for i=1:n_part
        for j = 1:dim
            distance(i,:) = distance(i,:) + (data(j,:)-center(j,i)).^2;
        end
        distance(i,:) = sqrt(distance(i,:));
    end
    
    [mindist,index]=min(distance);
    E_in_new = sum(mindist.^2);
    times=times-1;
end

% record the indices of data belong to each cluster
data_index=cell(1,n_part);
for i=1:n_part
    data_index{i}=find(index==i);
end

radius=zeros(1,n_part);
for i=1:n_part
    radius(i)=max(distance(i,index==i));
end