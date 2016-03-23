% Reading the test matrix in the same way we read the training matrix
N = dlmread('/home/jigyasa/Desktop/Machine-Learning-Algorithms/Naive_Bayes/test-features.txt', ' ');
spmatrix = sparse(N(:,1), N(:,2), N(:,3));
test_matrix = full(spmatrix);

% Number of test documents and the size of the dictionary
numTestDocs = size(test_matrix, 1);
numTokens = size(test_matrix, 2);

% The output vector is a vector storing the spam/nonspam prediction
output = zeros(numTestDocs, 1);

% Classify an email as spam if : `log p(x|y=1)+log p(y=1) > log p(x|y=0)+log p(y=0)`
% (vectorized implementation) 
log_a = test_matrix*(log(prob_tokens_spam))' + log(prob_spam);
log_b = test_matrix*(log(prob_tokens_nonspam))'+ log(1 - prob_spam);  
output = log_a > log_b;

% Reading the correct labels of the test set
test_labels = dlmread('/home/jigyasa/Desktop/Machine-Learning-Algorithms/Naive_Bayes/test-labels.txt');

% Computing the error on the test set
numdocs_wrong = sum(xor(output, test_labels))

%Printing error statistics on the test set
fraction_wrong = numdocs_wrong/numTestDocs


