function g = sigmoid(z)
g = 1.0 ./ (1.0 + exp(-z));
end

%!test
%!assert (sigmoid(0),0.5)
%!assert (sigmoid(1), 1/(1+exp(-1)))