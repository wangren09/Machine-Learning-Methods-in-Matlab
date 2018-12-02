%This main program groups the handwritten data into two groups: digit 1 and
% not digit 1 by using Neural Network.

%Provided by: Ren Wang
%most recently updated time 11/24/2018

clear;clc;

%[training_data, training_label, test_data, test_label, feature, n_row, n_column] = dataprocess();
load training_data
load training_label
[n_row, n_col] = size(training_data);

%total number of layers
L = 2;

%initialization of layers
W = cell(1, L);
W{1, 1} = normrnd(0,0.1,[3,2]);
W{1, 2} = normrnd(0,0.1,[3,1]);

%input
x0 = [ones(1, n_col); training_data];
y = training_label;

%initialization parameters
s_out = cell(1, L);
x_out = cell(1, L);
delta = cell(1, L);
G = cell(1, L);

%maximum iterations
iter_max = 2*10^6;

%stepsize
eta = 1/10;

%parameters for variable learning rate gradient descent
beta = 0.75;

%iterate E_in
E_in = zeros(1, iter_max);

%regularization parameter
lambda = 0.01/n_col;

for iter = 1:iter_max
    
    [E_in(iter), s_out, x_out] = NN_error(x0, y, W, L);
    for i = L:-1:1
        if i == L
            delta{1,i} = 2.*(x_out{1,i} - y).*1;%(1 - x_out{1,i}.^2);
        else
            delta{1,i} = (1 - x_out{1,i}(2:end,:).^2).*(W{1, i+1}(2:end,:)*delta{1,i+1});
        end
    end
    for i = 1:L
        if i == 1
            G{1,i} = x0*delta{1,i}'/4 + 2*lambda*W{1, i};
            eta = 1/10;
            v = G{1,i};
            [error_ori, ~, ~] = NN_error(x0, y, W, L);
            W_updata = W;
            W_updata{1, i} = W{1, i} - eta*v;
            [error, ~, ~] = NN_error(x0, y, W_updata, L);
            while error > error_ori
                eta = eta*beta;
                W_updata{1, i} = W{1, i} - eta*v;
                [error, ~, ~] = NN_error(x0, y, W_updata, L);
            end
            W{1, i} = W{1, i} - eta*v;
        else
            G{1,i} = x_out{1,i-1}*delta{1,i}'/4 + 2*lambda*W{1, i};
            eta = 1/10;
            v = G{1,i};
            [error_ori, ~, ~] = NN_error(x0, y, W, L);
            W_updata = W;
            W_updata{1, i} = W{1, i} - eta*v;
            [error, ~, ~] = NN_error(x0, y, W_updata, L);
            while error > error_ori% - eta*0.8*norm(v)^2
                eta = eta*beta;
                W_updata{1, i} = W{1, i} - eta*v;
                [error, ~, ~] = NN_error(x0, y, W_updata, L);
            end
            W{1, i} = W{1, i} - eta*v;
        end
    end
    
end

figure;
plot(1:iter_max, E_in, 'r');
xlabel('Iterations')
ylabel('In-sample error')
set(gcf,'unit','centimeters','position',[6 6 16 12]);
set(gca,'Position',[.125 .14 .77 .8]);


x=-1:0.001:1;
y=-1:0.001:1;
[X,Y]=meshgrid(x,y);
[n1, n2] = size(X);
ZZ = zeros(n1, n2);
for k=1:size(Y,1)
    x0 = [ones(1, n2); [X(k, :); Y(k, :)]];
%     for i = 1:L
%         if i == 1
%             s_out{1,i} = W{1, i}'*x0;
%             x_out{1,i} = [ones(1, n2);tanh(s_out{1,i})];
%         elseif i == L
%             s_out{1,i} = W{1, i}'*x_out{1,i-1};
%             x_out{1,i} = s_out{1,i};
%         else
%             s_out{1,i} = W{1, i}'*x_out{1,i-1};
%             x_out{1,i} = [1;tanh(s_out{1,i})];
%         end
%     end
    [~, s_out, x_out] = NN_error(x0, y, W, L);
    ZZ(k, :) = sign(x_out{1,L});
end
figure;
plot(training_data(1,training_label == 1),training_data(2,training_label == 1),'bo');
hold on
plot(training_data(1,training_label == -1),training_data(2,training_label == -1),'rx');
contour(X,Y,ZZ,[0 0],'lineColor','m','linewidth',2);
xlabel('Intensity');
ylabel('Asymmetry');