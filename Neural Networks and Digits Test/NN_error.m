function [E_in, s_out, x_out] = NN_error(x0, y, W, L)
%this function is used to calculate the forward propagation and the erro
%input: x0:input data y: labels W:weights of layers  L: number of layers
%output: insample error

[n_row, n_col] = size(x0);
%initialization parameters
lambda = 0.01/n_col;
s_out = cell(1, L);
x_out = cell(1, L);
for i = 1:L
    if i == 1
        s_out{1,i} = W{1, i}'*x0;
        x_out{1,i} = [ones(1, n_col);tanh(s_out{1,i})];
    elseif i == L
        s_out{1,i} = W{1, i}'*x_out{1,i-1};
        x_out{1,i} = s_out{1,i};
        E_in = sum((x_out{1,i} - y).^2)/4/n_col + lambda*W{1, i}'*W{1, i};
    else
        s_out{1,i} = W{1, i}'*x_out{1,i-1};
        x_out{1,i} = [1;tanh(s_out{1,i})];
    end
end