function g = sigmoidGradient(z)
%SIGMOIDGRADIENT returns the gradient of the sigmoid function
%evaluated at z
%   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
%   evaluated at z. This should work regardless if z is a matrix or a
%   vector. In particular, if z is a vector or matrix, you should return
%   the gradient for each element.

g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the gradient of the sigmoid function evaluated at
%               each value of z (z can be a matrix, vector or scalar).


%hint: what is the derivative of f(x)?

g = sigmoid(z) .* (1 - sigmoid(z));		



% =============================================================


end

%!test
%! assert(disp(sigmoidGradient(1)), disp(0.19661))

%!test
%! assert(disp(sigmoidGradient([2 3])), disp([0.104994 0.045177]))

%!test
%! assert(disp(sigmoidGradient([-2 0;4 999999;-1 1])), disp([0.10499 0.25000; 0.01766  0.00000; 0.19661 0.19661]))