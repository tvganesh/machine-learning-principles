
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 
%% This is Octave project runs through the principles of multivariate regression and checks the optimal solution
%% Designed and developed by Tinniam V Ganesh
%% Date 25 Dec 2014
%% File: housing_compute.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Clear and Close Figures
clear ; close all; clc

fprintf('Loading data ...\n');

%% Load Housing Data
data  = load('housing.csv');

d1 = data(:,1);
d2 = data(:,3);
d3 = data(:,5:14);
data = [d1 d2 d3];
[s t] = size(data);

% Create a random index
randidx = randperm(s);

training = data(randidx(1:306), :);
[a b] = size(training);
cross_validation = data(randidx(307:406), :);
[m n] =size(cross_validation);
test_data = data(randidx(407:506), :);
J_lowest = 10000;
Jcv_lowest = 0;
alpha_lowest =0;
lambda_lowest = 0;
degree_lowest =0;

max_degrees =4;
J_history = zeros(max_degrees, 1);
Jcv_history = zeros(max_degrees, 1);
for degree = 1:max_degrees; 
    [J Jcv alpha lambda] = train_samples(randidx, training,cross_validation,test_data,degree);
	J_history(degree) = J;
	Jcv_history(degree) = Jcv;
	if J < J_lowest
	   J_lowest= J;
	   Jcv_lowest = Jcv;
	   alpha_lowest = alpha;
	   lambda_lowest = lambda;
	   degree_lowest = degree;
	 end;
end;
figure;
plot(1:numel(J_history), J_history, '-k', 'LineWidth', 2);
hold on;
plot(1:numel(Jcv_history), Jcv_history, '-m', 'LineWidth', 2);
xlabel('Degree of polynomial');
ylabel('Cost J');
title('Degree of Polynomial vs Training/Validation cost');
legend('Training cost', 'Cross Validation cost');

print -dpng degree-cost.png;
hold off;

J_lowest
J_lowest
Jcv_lowest 
alpha_lowest
lambda_lowest


% Set the xtrain and ytrain
xtrain = training(:,1:b-1); 
ytrain = training(:,b);

xtrain = poly(xtrain,degree);

% Normalize the features
[xtrain mu sigma] = featureNormalize(xtrain);

% Add intercept term to X
xtrain= [ones(size(xtrain)(1,1), 1) xtrain];
[c d] = size(xtrain);
%lambda_arr = {0,0.01,0.02,0.04,0.08,0.16,0.32,0.64,1.28,2.56,5.12,10.24};
lambda_arr = {0.0001,0.005, 0.001, .005, 0.01, 0.05,0.1,0.5,1,5,10,50};
j_plot = zeros(12,2);
jcv_plot = zeros(12,2);
for j = 1:length(lambda_arr)
	 lambda= lambda_arr{j};
	 printf("alpha = %5.5f    lambda = %5.5f \n",alpha,lambda);
     %num_iters = 400;
    
   % Init Theta and Run Gradient Descent
   
    % Minimize using fmincg
	
    J = computeCost(xtrain,ytrain,theta)
    j_plot(j,1) = lambda;
    j_plot(j,2) = J;	
	% Compute the cost for the cross validation samples with the computed theta	from the training samples
	Jcv  = cross_validate(cross_validation, randidx, theta,degree)
	jcv_plot(j,1) = lambda;
    jcv_plot(j,2) = Jcv;	
end;
j_plot
jcv_plot
figure;
plot(j_plot(:,1),j_plot(:,2), 'k');
hold on;
plot(jcv_plot(:,1),jcv_plot(:,2), 'm');
xlabel('Lambda');
ylabel('Cost J');
axis([8 12 0 20]);
title(' Lambda vs Training/Validation cost');
legend('Training cost', 'Cross Validation cost');
print -dpng lambda-cost.png;

