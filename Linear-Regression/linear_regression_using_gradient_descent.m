% Loading training set for a supervised learning problem with n=1 features
% The y-values are the heights measured in meters, and the x-values are the corresponding ages of the boys
x = load('/home/jigyasa/Desktop/Machine-Learning-Algorithms/Linear-Regression/ex2x.txt');
y = load('/home/jigyasa/Desktop/Machine-Learning-Algorithms/Linear-Regression/ex2y.txt');

figure; % open a new figure window
plot(x, y, '.');
ylabel('Height in meters');
xlabel('Age in years');

% Total number of training examples
m = length(y); 
% Add a column of '1's to x
x = [ones(m, 1), x]; 

% Initialize fitting parameters
theta = zeros(size(x(1,:)))'; 
MAX_ITR = 1500;
alpha = 0.07;

for num_iterations = 1:MAX_ITR
    % This is a vectorized version of the 
    % gradient descent update formula
    
    % Gradient Descent Vector format
    grad = (1/m).* x' * ((x * theta) - y);
    
    % Updating theta
    theta = theta - alpha .* grad;
end
% print theta to screen
theta

% Plot new data without clearing old plot
hold on 
% X is now a 2*m matrix with one coulmn set as '1'
plot(x(:,2), x*theta, '-')
legend('Training data', 'Linear regression')

% Example:
%% Predicting values for age 3.5 and 7
predict1 = [1, 3.5] *theta
predict2 = [1, 7] * theta

% Initializing Jvals to 100x100 matrix of 0's
J_vals = zeros(100, 100);   
% Grid over which we will calculate J
theta0_vals = linspace(-3, 3, 100);
theta1_vals = linspace(-1, 1, 100);
for i = 1:length(theta0_vals)
	  for j = 1:length(theta1_vals)
	  t = [theta0_vals(i); theta1_vals(j)];
	  J_vals(i,j) = (0.5/m) .* (x * t - y)' * (x * t - y);
    %% as J = (1/2M)*(Theta*X-Y)'*(Theta*X-Y);
    end
end

% Because of the way meshgrids work in the surf command, we need to 
% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals'
figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel('\theta_1')
