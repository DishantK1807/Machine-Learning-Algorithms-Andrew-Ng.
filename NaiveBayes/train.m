% Number of training examples : m
numTrainDocs = 700;

% Dictionary size : |V|
numTokens = 2500;

% Reading the features matrix
M = dlmread('/home/jigyasa/Desktop/Machine-Learning-Algorithms/Naive_Bayes/train-features.txt', ' ');
spmatrix = sparse(M(:,1), M(:,2), M(:,3), numTrainDocs, numTokens);
train_matrix = full(spmatrix);

% train_matrix contains information about the words within the emails
% the i-th row of train_matrix represents the i-th training email
% the entry in the j-th column tells how many times the j-th dictionary word appears in that email

% Reading the training labels
train_labels = dlmread('/home/jigyasa/Desktop/Machine-Learning-Algorithms/Naive_Bayes/train-labels.txt');
% the i-th entry of train_labels now indicates whether document i is spam


% Indices for the spam and nonspam labels
spam_indices = find(train_labels);
nonspam_indices = find(train_labels == 0);

% Probability of spam
prob_spam = length(spam_indices) / numTrainDocs;

% Sum the number of words in each email
email_lengths = sum(train_matrix, 2);

% Total word counts of all the spam emails and nonspam emails
spam_wc = sum(email_lengths(spam_indices));
nonspam_wc = sum(email_lengths(nonspam_indices));

% Probability of the tokens in spam emails
prob_tokens_spam = (sum(train_matrix(spam_indices, :)) + 1) ./(spam_wc + numTokens);
% Now the k-th entry of prob_tokens_spam represents phi_(k|y=1)

% Probability of the tokens in non-spam emails
prob_tokens_nonspam = (sum(train_matrix(nonspam_indices, :)) + 1)./(nonspam_wc + numTokens);
% Now the k-th entry of prob_tokens_nonspam represents phi_(k|y=0)

