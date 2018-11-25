function [training_data, training_label, test_data, test_label, feature, n_row, n_column] = dataprocess()
%Provided by: Ren Wang
%most recently updated time 11/22/2018
%combine Train.mat and Test.mat together and generate training and testing
%data

%combine all the data to one data set
load('Train.mat','ZipDigits');
A=ZipDigits;
load('Test.mat','ZipDigits');
ZipDigits=[A;ZipDigits];
[n_row,n_column]=size(ZipDigits);

%% extract the features: 1. density 2. symmetry
index1=find(ZipDigits(:,1)==1); %indices of digit "1" images
index2=setdiff(1:n_row,index1)'; %indices of other images
feature=zeros(2,n_row);

grayscale=ZipDigits(:,2:end);
d=size(grayscale,2);
w=floor(sqrt(d));

for i=1:n_row
    curimage=reshape(grayscale(i,:),w,w);
    curimage=curimage';
    feature(1,i)=sum(grayscale(i,:))/d;
    flipped=curimage(w:-1:1,:);
    feature(2,i)=-sum(sum(abs(curimage-flipped)))/d/2;
end

%% rescale the features into [-1,1]
feature_min=min(feature,[],2);
feature_max=max(feature,[],2);
feature(1,:)=-1+(feature(1,:)-feature_min(1))/(feature_max(1)-feature_min(1))*2;
feature(2,:)=-1+(feature(2,:)-feature_min(2))/(feature_max(2)-feature_min(2))*2;

%reorder features and give labels: 1 for digit 1, -1 for other digits
feature=feature(:,[index1;index2]);
labels=[ones(1,length(index1)) -1*ones(1,length(index2))];

%randomly pick 300 data as training set
N_T = 300;
training=randperm(n_row,N_T);
training_data=feature(:,training);
training_label=labels(training);

%take the rest data as testing set
test=setdiff(1:n_row, training);
test_data=feature(:,test);
test_label=labels(test);