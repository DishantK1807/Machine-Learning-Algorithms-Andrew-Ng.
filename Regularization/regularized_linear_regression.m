% Loading data
x = load('/home/jigyasa/Desktop/Machine-Learning-Algorithms/Regularization/xLin.txt');
y = load('/home/jigyasa/Desktop/Machine-Learning-Algorithms/Regularization/yLin.txt');

% Number of training examples
m = length(y); 

% Plotting the training data
figure;
plot(x, y, '.', 'MarkerFacecolor', 'r', 'MarkerSize', 8);

% Features are all powers of x from x^0 to x^5
x = [ones(m, 1), x, x.^2, x.^3, x.^4, x.^5];

% Initializing fitting parameters
theta = zeros(size(x(1,:)))'; 

% The regularization parameter
lambda = 0;

L = lambda.*eye(6); 
% the extra regularization terms

L(1) = 0;
theta = (x' * x + L)\x' * y
norm_theta = norm(theta)

% Plot the linear fit
hold on;

% Our training data was only a few points, so creating a denser array of x-values for plotting
x_vals = (-1:0.05:1)';

features = [ones(size(x_vals)), x_vals, x_vals.^2, x_vals.^3, x_vals.^4, x_vals.^5];
plot(x_vals, features*theta, 'r', '--', 'LineWidth', 2)
legend('Training data', '5th order fit')
hold off

