%ACTIVATION FUNCTION
function y = sigmoid(x)
% implementation of the sigmoid function: 
    y = 1./(1+exp(-x));
end