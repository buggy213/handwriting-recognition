%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this exercise
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10

% load dataset
load('ex4data1.mat');
m = size(X, 1);

% random initialization to break symmetry
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

% learning rate and regularization constant

lambda = 1;
num_iterations = input('# of iterations: ');

alpha = 0.03;
J_history = zeros(num_iterations, 1);

using_builtin_optimize = 1;

if ~using_builtin_optimize
  % run gradient descent
  for i = 1:num_iterations
    [J grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, 10, X, y, lambda);
    J_history(i) = J;
    nn_params = nn_params - alpha * grad;
    printf('iteration %d | cost: %d\n', i, J);
  endfor

  % plot J vs number of iterations
  figure;
  plot(J_history)
  xlabel('# of iterations');
  ylabel('Cost');
  title('Cost vs # of iterations');
  pause;
else
  options = optimset('MaxIter', num_iterations);

  % Create "short hand" for the cost function to be minimized
  costFunction = @(p) nnCostFunction(p, ...
                                     input_layer_size, ...
                                     hidden_layer_size, ...
                                     num_labels, X, y, lambda);
  % Now, costFunction is a function that takes in only one argument (the
  % neural network parameters)
  [nn_params, cost] = fmincg(costFunction, nn_params, options);
endif
% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

% visualize results
rp = randperm(m);

for i = 1:m
    % Display 
    fprintf('\nDisplaying Example Image\n');
    displayData(X(rp(i), :));

    pred = predict(Theta1, Theta2, X(rp(i),:));
    fprintf('\nNeural Network Prediction: %d (digit %d)\n', pred, mod(pred, 10));
    fprintf('Actual vallue: %d\n', y(rp(i)));
    % Pause with quit option
    s = input('Paused - press enter to continue, q to exit:','s');
    if s == 'q'
      break
    end
end

