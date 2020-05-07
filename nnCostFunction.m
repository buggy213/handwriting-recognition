function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

X = [ones(size(X, 1), 1) X];

hidden_layer_values = sigmoid(X * Theta1');

hidden_layer_values = [ones(size(hidden_layer_values), 1) hidden_layer_values];

output_layer_values = sigmoid(hidden_layer_values * Theta2');

labels = zeros(size(y, 1), num_labels);

% convert y into one hot encode
for i=1:size(y,1),
  labels(i, y(i,1)) = 1;
endfor

errors = labels .* log(output_layer_values) + (1-labels) .* log(1-output_layer_values);

theta1_excluding_bias = Theta1(:, 2:end);
theta2_excluding_bias = Theta2(:, 2:end);

regularization_term = sum(sum(theta1_excluding_bias .^ 2)) + sum(sum(theta2_excluding_bias .^ 2));

J = (-1/m) * sum(sum(errors)) + ((lambda) / (2 * m)) * regularization_term;

accumulated_delta_one = 0;
accumulated_delta_two = 0;

% backpropagation
for t = 1:m,
  a_1 = X(t,:);
  % feedforward pass
  z_2 = a_1 * transpose(Theta1);
  a_2 = [1 sigmoid(z_2)];
  z_3 = a_2 * transpose(Theta2);
  a_3 = sigmoid(z_3);
  
  % calculate 3rd and 2nd layer deltas
  third_layer_delta = a_3 - labels(t,:);
  second_layer_delta = (Theta2' * third_layer_delta') .* sigmoidGradient([0; z_2']);
  
  % ignore bias term
  second_layer_delta = second_layer_delta(2:end,:);
  third_layer_delta = [third_layer_delta'];
  
  accumulated_delta_two = accumulated_delta_two + third_layer_delta * a_2;
  accumulated_delta_one = accumulated_delta_one + second_layer_delta * a_1;
    
endfor

theta1_grad_regularization = (lambda / m) * [zeros(size(Theta1, 1),1) Theta1(:,2:end)];
theta2_grad_regularization = (lambda / m) * [zeros(size(Theta2, 1),1) Theta2(:,2:end)];


% with regularization
Theta1_grad = accumulated_delta_one / m + theta1_grad_regularization;
Theta2_grad = accumulated_delta_two / m + theta2_grad_regularization;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
