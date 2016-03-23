%% SVM Email text classification

% Load training features and labels
[train_y, train_x] = libsvmread('/home/jigyasa/Desktop/Machine-Learning-Algorithms/SVM-Linear-Classification/email_train-all.txt');

% Train the model and get the primal variables w, b from the model

% Libsvm options
% -t 0 : linear kernel
% Leaving other options as their defaults 
model = svmtrain(train_y, train_x, '-t 0');

w = model.SVs' * model.sv_coef;
b = -model.rho;

if (model.Label(1) == -1)
    w = -w; b = -b;
end

% Loading testing features and labels
[test_y, test_x] = libsvmread('/home/jigyasa/Desktop/Machine-Learning-Algorithms/SVM-Linear-Classification/email_test.txt');

[predicted_label, accuracy, decision_values] = svmpredict(test_y, test_x, model);
% After running svmpredict, the accuracy is printed in console