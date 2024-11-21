%
%METHOD TO UPDATE EVERY WEIGHT (one at a time NOT BATCH)
%

%   weight - output | (weight,input,expect_output) - inputs
function Weight = SGD_method(Weight,inputs, expected_output)
learning_rate = 0.9;

N = 4;

for k = 1:N
    % transposed means -> [1,0,0] becomes [1]
    %                                     [0]
    %                                     [0]
    transposed_input = inputs(k, :)';
    d = expected_output(k);%get the correct output

weighted_sum = Weight * transposed_input;%weghted sum calculated
output = Sigmoid(weighted_sum);%process the  weighted sum

error = d-output%correct output - output

delta = output*(1-output)*error;


dWeight = learning_rate*delta*transposed_input;%calculating the weight update

%Updating the weights:
Weight(1) = Weight(1) + dWeight(1);
Weight(2) = Weight(2) + dWeight(2);
Weight(3) = Weight(3) + dWeight(3);
end
end