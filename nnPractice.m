function [predictedMove] = nnPractice(unrolledInput,unrolledOutput)

%need to write out g-function as another method
%need to figure out general-case inputs and outputs (create file with inputs and expected outputs?)
%need to unroll input and output parameters
%need to convert strings to numbers?
%need to determine initial number of nodes and layers (use method described in book?)
%need to initialize weights (randomly?)
%need to write up cost function
%need to choose minimization function

numRows = 2;
numCols = 3;
Input = reshape(unrolledInput(1:6),numRows,numCols);

predictedMove = 2;

%!test
 %!assert (nnPractice([1],1), 2)