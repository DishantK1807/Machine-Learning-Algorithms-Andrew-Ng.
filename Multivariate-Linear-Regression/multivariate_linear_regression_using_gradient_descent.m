% Loading and preprocessing data
x = load('/home/jigyasa/Desktop/Machine-Learning-Algorithms/Multivariate-Linear-Regression/x.txt');
y = load('/home/jigyasa/Desktop/Machine-Learning-Algorithms/Multivariate-Linear-Regression/y.txt');

m = length(y);

% Adding add x(0) = 1 intercept term into x matrix
x = [ones(m, 1), x];

% Saving a copy of the unscaled features for future
x_unscaled = x;

% Scaling both types of inputs by their standard deviations and setting their means to zero
sigma = std(x);
mu = mean(x);
x(:,2) = (x(:,2) - mu(2))./ sigma(2);
x(:,3) = (x(:,3) - mu(3))./ sigma(3);

% Prepare for plotting
figure;
% Plotting each alpha's data points in a different style
 plotstyle = {'b', 'r', 'g', 'k', 'b--', 'r--'};

% Gradient Descent 
% The goal of this part is to pick a good learning rate in the range of [0.001,10]
alpha = [0.01, 0.03, 0.1, 0.3, 1, 1.3];
MAX_ITR = 100;

% Final values of theta sfter finding the best learning rate
theta_grad_descent = zeros(size(x(1,:))); 

for i = 1:length(alpha)
    % initializing fitting parameters
    theta = zeros(size(x(1,:)))'; 
    J = zeros(MAX_ITR, 1);
    
    for num_iterations = 1:MAX_ITR
        % Calculating the J term
        J(num_iterations) = (0.5/m) .* (x * theta - y)' * (x * theta - y);
        
        % The gradient
        grad = (1/m) .* x' * ((x * theta) - y);
        
        % Updation
        theta = theta - alpha(i) .* grad;
    end
    
    % Plotting the first 50 J terms
    plot(0:49, J(1:50), char(plotstyle(i)), 'LineWidth', 2)
    hold on
    
    % After trial and error, I find alpha=1 is the best learning rate and converges before 
    %the 100th iteration; hence save the theta for alpha=1 as the result of gradient descent
    if (alpha(i) == 1)
        theta_grad_descent = theta;
    end
    
end

legend('0.01','0.03','0.1', '0.3', '1', '1.3')
xlabel('Number of iterations')
ylabel('Cost J')

% Display gradient descent's result in command window
theta_grad_descent

% Estimating the price of a 1650 sq-ft, 3 br house
price_grad_desc = dot(theta_grad_descent, [1, (1650 - mu(2))/sigma(2),(3 - mu(3))/sigma(3)]);

% Calculating the parameters from the normal equation
theta_normal = (x_unscaled' * x_unscaled)\x_unscaled' * y

%Estimating the house price again
price_normal = dot(theta_normal, [1, 1650, 3])
