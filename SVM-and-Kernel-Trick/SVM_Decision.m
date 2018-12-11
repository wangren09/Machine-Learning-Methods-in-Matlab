function data_lab = SVM_Decision(alpha_in, data, ker_in, x_data, y_lab)

% using SVM to decide the label of a new data

% input: the parameter we got from SVM, new data, kernel matrix, training
% data, training data label

%output: label of the new data

loc = find(alpha_in > 10^(-6));
loc2 = find(alpha_in > 10^(-4));
ys = y_lab(loc2(1));
b_star = ys;
for i = 1:length(loc)
    b_star = b_star  - alpha_in(loc(i))*y_lab(loc(i))*ker_in(loc(i), loc2(1));        
end

kernel = zeros(1, length(loc));
for i = 1:length(loc)
    kernel(i) = (1 + x_data(:, loc(i))'*data).^8;
end

sum = 0;
for i = 1:length(loc)
    sum = sum + alpha_in(loc(i))*y_lab(loc(i))*kernel(i);
end
data_lab = sign(sum + b_star);
end