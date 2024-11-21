load('Trained_Network.mat')

inputs = [0 0 1;
    0 1 1;
    1 0 1;
    1 1 1;];

N = 4;

for k = 1:N
    transposed_input = inputs(k, :)';
    weighted_sum = Weight*transposed_input
    output = Sigmoid(weighted_sum);
end