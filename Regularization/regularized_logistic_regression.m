% Loading data
x = load('/home/jigyasa/Desktop/Machine-Learning-Algorithms/Regularization/xLog.txt');
y = load('/home/jigyasa/Desktop/Machine-Learning-Algorithms/Regularization/yLog.txt');
  

% Plotting the training data
% Use different markers for positives and negatives
figure
pos = find(y); neg = find(y == 0);
plot(x(pos, 1), x(pos, 2), 'kx','LineWidth', 2, 'MarkerSize', 7)
hold on
plot(x(neg, 1), x(neg, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7)


% Adding polynomial features to x by calling the feature mapping function
%% ** added via the pre-existing map_feature.m file**
x = map_feature(x(:,1), x(:,2));

[m, n] = size(x);

% Initializing fitting parameters
theta = zeros(n, 1);

% Defining the sigmoid function
g = inline('1.0 ./ (1.0 + exp(-z))'); 

% Setting up for Newton's method
MAX_ITR = 15;
J = zeros(MAX_ITR, 1);

% Lambda is the regularization parameter
lambda = 0;

%%% Newton's Method
for i = 1:MAX_ITR

    % Calculate the hypothesis function
    z = x * theta;
    h = g(z);
    
    % Calculate J (for testing convergence)
    J(i) =(1/m)*sum(-y.*log(h) - (1-y).*log(1-h))+ ...
    (lambda/(2*m))*norm(theta([2:end]))^2;
    
    % Calculate gradient and hessian.
    G = (lambda/m).*theta; G(1) = 0; % extra term for gradient
    L = (lambda/m).*eye(n); L(1) = 0;% extra term for Hessian
    grad = ((1/m).*x' * (h-y)) + G;
    H = ((1/m).*x' * diag(h) * diag(1-h) * x) + L;
    
    % Here is the actual update
    theta = theta - H\grad;
  
end

% Showing J to determine if algorithm has converged
J

% Displaying the norm of our parameters
norm_theta = norm(theta) 

% Plotting the results 

% Evaluating theta*x over a grid of features and plot the contour where theta*x equals zero

% Grid range
u = linspace(-1, 1.5, 200);
v = linspace(-1, 1.5, 200);

z = zeros(length(u), length(v));

% Evaluating z = theta*x over the grid
for i = 1:length(u)
    for j = 1:length(v)
        z(i,j) = map_feature(u(i), v(j))*theta;
    end
end

z = z'; % important to transpose z before calling contour

% Plot z = 0
% Notice you need to specify the range [0, 0]
contour(u, v, z, [0, 0], 'LineWidth', 2)
legend('y = 1', 'y = 0', 'Decision boundary')
title(sprintf('\\lambda = %g', lambda), 'FontSize', 14)


hold off

% PLotting J
figure
plot(0:MAX_ITR-1, J, 'o--', 'MarkerFaceColor', 'r', 'MarkerSize', 8)
xlabel('Iteration'); ylabel('J')