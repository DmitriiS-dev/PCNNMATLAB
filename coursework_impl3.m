clearvars%clear variables
clc%clear commandline

Weights = zeros(1,9);%Weights are zero

w_to_insert = [0.1 0.15 0.3 -0.1 0.1 0.1 -0.2 0.15 -0.15];
for i=1: length(w_to_insert)
    Weights(i) = w_to_insert(i);
end
inputs = [0 1; 1 0];
outputs = [1 1];

% ---------------
% (Forward Pass):
% ---------------

% Forward Propagation
function node_output = forward_propagate(input_pattern, weight, first_hidden_index)
    number_of_inputs = length(input_pattern);
    number_of_nodes = size(weight, 1);

    node_output = zeros(1, number_of_nodes);

    % Copy pattern to input nodes
    node_output(2:number_of_inputs + 1) = input_pattern;
    node_output(1) = 1; % Set bias to 1

    % Compute outputs of the remaining nodes
    for i = first_hidden_index:number_of_nodes
        sum_value = sum(weight(i, 1:i - 1) .* node_output(1:i - 1));
        node_output(i) = sigmoid(sum_value);
    end
end

% Sigmoid function
function result = sigmoid(x)
    result = 1 / (1 + exp(-x));
end

% Backward Error Propagation
function delta = backprop_node_deltas(targets, node_value, weight, first_output_node, last_hidden_node)
    number_of_nodes = length(node_value);
    delta = zeros(1, number_of_nodes);

    % Calculate node deltas for output nodes
    for i = length(targets):-1:first_output_node
        err = targets(i) - node_value(i);
        delta(i) = err * node_value(i) * (1 - node_value(i)); % Sigmoid slope term
    end

    % Calculate deltas for hidden nodes, working backward
    for i = last_hidden_node:-1:first_output_node + 1
        delta(i) = 0;
        for k = i + 1:number_of_nodes
            delta(i) = delta(i) + weight(k, i) * delta(k);
        end
        delta(i) = delta(i) * node_value(i) * (1 - node_value(i)); % Sigmoid slope term
    end
end

% On-Line Weight Update
function weights = backprop_online_one_epoch(patterns, targets, weights, learning_rate, first_hidden_index)
    number_of_patterns = size(patterns, 1);
    number_of_nodes = size(weights, 1);

    % For each pattern
    for ip = 1:number_of_patterns
        index = randi([1, number_of_patterns]);
        input_pattern = patterns(index, :);
        target = targets(index, :);

        % Forward propagation
        node_output = forward_propagate(input_pattern, weights, first_hidden_index);

        % Backward propagation
        delta = backprop_node_deltas(target, node_output, weights, size(target, 2), first_hidden_index);

        % Change the weights
        for i = 1:number_of_nodes
            for j = 1:i - 1
                weights(i, j) = weights(i, j) + learning_rate * delta(i) * node_output(j);
            end
        end
    end
end
