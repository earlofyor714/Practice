function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% X dimensions = #trainers x input_layer nodes

% add X0
X = [ones(m,1) X];
hidden_layer = sigmoid(X*Theta1');					%Theta1 dimensions = hidden_layer x (input_layer+1)
num_labels = size(hidden_layer, 1);
hidden_layer = [ones(num_labels,1) hidden_layer];
hypothesis = sigmoid(hidden_layer*Theta2')';
[index, prediction] = max(hypothesis);
p = prediction';


end

%!test
%! T1 = [1 7; -5 -42];
%! T2 = [-1 3 2; 2 -10 4];
%! X = [1; -4];
%! assert (predict(T1, T2, X), [1; 2])
%! T1 = reshape(sin(0 : 0.5 : 5.9), 4, 3);
%! T2 = reshape(sin(0 : 0.3 : 5.9), 4, 5);
%! X = reshape(sin(1:16), 8, 2);
%! assert (predict(T1, T2, X), [4;1;1;4;4;4;4;2])