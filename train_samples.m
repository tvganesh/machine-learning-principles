function [Jlowest Jcvlowest alpha_lowest lambda_lowest] = train_samples(randidx, training, cross_validation,test_data,degree)

[a b] = size(training);
[m n] =size(cross_validation);


% Set the xtrain and ytrain
xtrain = training(:,1:b-1); 
ytrain = training(:,b);

xtrain = poly(xtrain,degree);

% Normalize the features
[xtrain mu sigma] = featureNormalize(xtrain);

% Add intercept term to X
xtrain= [ones(size(xtrain)(1,1), 1) xtrain];
[c d] = size(xtrain);

alpha_arr = {0.001, 0.003, 0.01, 0.03,0.1};
lambda_arr = {0.001, .005, 0.01, 0.05,0.1,0.5,1,5};

figure;
% Create a matrix to store the J, Jcv, Jtest for different alpha & lambda
len= length(alpha_arr) * length(lambda_arr);
theta_cost = zeros(len, 5);

% Se the first row
row =1;

Jlowest = 100;
Jcvlowest = -1;
Jtestlowest = -1;
theta_lowest = zeros(d, 1);
alpha_lowest = -1;
lambda_lowest = -1;
Jcvhistory_lowest = zeros(m,1);

% Check different combinations of alpha and lambda to get the lowest cost
for i = 1:length(alpha_arr),
   for j = 1:length(lambda_arr)
     alpha = alpha_arr{i};
	 lambda= lambda_arr{j};
	 printf("alpha = %5.5f    lambda = %5.5f \n",alpha,lambda);
     num_iters = 400;
    
   % Init Theta and Run Gradient Descent 
	% Perform Gradient descent
    J = computeCost(xtrain,ytrain,theta);
       
	% Compute the cost for the cross validation samples with the computed theta	from the training samples
	Jcv  = cross_validate(cross_validation, randidx, theta,degree);
	% Compute the cost on the test samples with the computed theta from training samples
    Jtest = test_samples(test_data, randidx, theta,degree);
	
	 if J < Jlowest
	     Jlowest = J;
		 Jcvlowest = Jcv;
		 Jtestlowest = Jtest;
		 theta_lowest = theta;
		 alpha_lowest = alpha;
		 lambda_lowest = lambda;
    endif;
	
	printf("J = %5.5f Jcv = %5.5f  Jtest = %5.5f \n", J, Jcv,Jtest);
    size(J_history);
	theta_cost(row,1) = alpha;
	theta_cost(row,2) = lambda;
	theta_cost(row,3) = J;
	theta_cost(row,4) = Jcv;
	theta_cost(row,5) = Jtest;
	% Plot the convergence graph
     row = row + 1;
     hold on;
    plot(1:numel(J_history), J_history, '-k', 'LineWidth', 2);
		 
   end;
  end;
 val = 48 + degree;
name = strcat("convergence",val);
xlabel('Number of iterations');
ylabel('Cost J');
if degree == 1
    title('Convergence function - Cost vs Number of iterations - Degree 1');
	print -dpng convergence-1.png
elseif degree == 2
    title('Convergence function - Cost vs Number of iterations - Degree 2');
	print -dpng convergence-2.png
elseif degree == 3
    title('Convergence function - Cost vs Number of iterations - Degree 3');
	print -dpng convergence-3.png
elseif degree == 4
    title('Convergence function - Cost vs Number of iterations - Degree 4');
	print -dpng convergence-4.png
end;
;
hold off;

row =1;
for i = 1:length(alpha_arr),
   for j = 1:length(lambda_arr)
     printf("%5.5f %5.5f %5.5f %5.5f %5.5f\n", theta_cost(row,1),theta_cost(row,2),theta_cost(row,3), ...
	              theta_cost(row,4),theta_cost(row,5));
	 row = row +1;
   end;
end;

 %Print the lowest values
alpha_lowest
lambda_lowest
Jlowest
Jcvlowest
Jtestlowest 
%theta_lowest

%Plot the learning curve
learning_curve(training, cross_validation, randidx, alpha_lowest, lambda_lowest,degree);
end;