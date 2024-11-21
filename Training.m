inputs = [0 0 1;0 1 1;1 0 1;1 1 1;];

expected_output = [0 0 1 1];

Weight = 2*rand(1,3)-1;

for epoch = 1:10000
    Weight = SGD_method(Weight,inputs,expected_output);
end

save('Trained_Network.mat');


