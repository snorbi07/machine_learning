function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    theta0 = theta(1);
    theta1 = theta(2);
    
    hTheta = X * theta;
    
    newTheta0 = theta0 - alpha * (m ^ -1) * sum(hTheta - y);
    newTheta1 = theta1 - alpha * (m ^ -1) * sum((hTheta - y)' * X(:,2));

    theta(1) = newTheta0;
    theta(2) = newTheta1;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
