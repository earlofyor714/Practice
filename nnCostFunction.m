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

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

a1 = [ones(m,1) X];
z2 = a1 * Theta1';
a2 = sigmoid(z2);
a2 = [ones(size(a2,1),1) a2];
z3 = a2 * Theta2';
a3 = sigmoid(z3);

%convert y into a matrix
yTransf = eye(num_labels);
yVec = yTransf(y, :);

J_expanded = (1/m) .* ((-1 .* yVec) .* log(a3) - (1 - yVec) .* log(1 - a3));
J = sum(sum(J_expanded));


% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

delta_3 = a3 - yVec;
Theta2_unbiased = Theta2(:,2:end);
Theta1_unbiased = Theta1(:,2:end);
delta_2 = (delta_3 * Theta2_unbiased) .* sigmoidGradient(z2);
Delta_2 = delta_3' * a2;
Delta_1 = delta_2' * a1;
Theta2_grad = (1/m).*Delta_2;
Theta1_grad = (1/m).*Delta_1;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

regularized = (lambda/(2*m))*(sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));
J = J + regularized;

Theta2_grad = Theta2_grad + (lambda/m).* [zeros(size(Theta2_unbiased,1),1) Theta2_unbiased];
Theta1_grad = Theta1_grad + (lambda/m).* [zeros(size(Theta1_unbiased,1),1) Theta1_unbiased];



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

%!test
%! input_layer_size = 2;
%! hidden_layer_size = 2;
%! num_labels = 4;
%! nn_params = [1:18]/10;
%! X = cos([1 2;3 4;5 6]);
%! y = [4; 2; 3];
%! lambda = 0;
%! assert (disp(nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)), disp(7.4070))

%!test
%! input_layer_size = 2;
%! hidden_layer_size = 2;
%! num_labels = 4;
%! nn_params = [1:18]/10;
%! X = cos([1 2;3 4;5 6]);
%! y = [4; 2; 3];
%! lambda = 3;
%! assert (disp(nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)), disp(16.457))
%! lambda = 4;
%! assert (disp(nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)), disp(19.474))

%!test
%! input_layer_size = 2;
%! hidden_layer_size = 2;
%! num_labels = 4;
%! nn_params = [1:18]/10;
%! X = cos([1 2;3 4;5 6]);
%! y = [4; 2; 3];
%! lambda = 0;
%! [J grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);
%! assert(disp(J), disp(7.4070))
%! assert(disp(grad), disp([0.766138;0.979897;-0.027540;-0.035844;-0.024929;-0.053862;0.883417;0.568762;0.584668;0.598139;0.459314;0.344618;0.256313;0.311885;0.478337;0.368920;0.259771;0.322331]))

%!test
%! input_layer_size = 2;
%! hidden_layer_size = 2;
%! num_labels = 4;
%! nn_params = [1:18]/10;
%! X = cos([1 2;3 4;5 6]);
%! y = [4; 2; 3];
%! lambda = 3;
%! [J grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);
%! assert(disp(J), disp(16.457))
%! assert(disp(grad), disp([0.76614;0.97990;0.27246;0.36416;0.47507;0.54614;0.88342;0.56876;0.58467;0.59814;1.55931;1.54462;1.55631;1.71189;1.97834;1.96892;1.95977;2.12233]))

%!test
%! input_layer_size = 2;
%! hidden_layer_size = 2;
%! num_labels = 4;
%! nn_params = [1:18]/10;
%! X = cos([1 2;3 4;5 6]);
%! y = [4; 2; 3];
%! lambda = 4;
%! [J grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);
%! assert(disp(J), disp(19.474))
%! assert(disp(grad), disp([0.76614;0.97990;0.37246;0.49749;0.64174;0.74614;0.88342;0.56876;0.58467;0.59814;1.92598;1.94462;1.98965;2.17855;2.47834;2.50225;2.52644;2.72233]))

%!test
%! input_layer_size = 2;
%! hidden_layer_size = 2;
%! num_labels = 4;
%! nn_params = [1:18]/10;
%! X = [1 2;3 4;5 6;0 1;1 2];
%! y = [4; 2; 3; 1; 2];
%! lambda = 4;
%! [J grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);
%! assert(disp(J), disp(17.441))
%! assert(disp(grad), disp([0.42849;0.45184;0.59785;0.62285;1.18634;1.23469;0.74763;0.55926;0.76838;0.77550;1.54737;1.41502;1.65535;1.77867;1.89409;1.75478;2.01039;2.12420]))
