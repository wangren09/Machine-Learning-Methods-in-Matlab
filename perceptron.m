function [w,iteration] = perceptron(X,Y,w_init)
w = w_init;
iteration = 0;
Jud = sign(Y.*(X'*w));
while sum(Jud) < length(Jud)
    iteration = iteration +1;
    f_p = find(Jud ~= 1);
    w = w + X(:,f_p(1)) * Y(f_p(1));
    Jud = sign(Y.*(X'*w));
end
end
