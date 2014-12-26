%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 
%% This is Octave project runs through the principles of multivariate regression and checks the optimal solution
%% Designed and developed by Tinniam V Ganesh
%% Date 25 Dec 2014
%% File: test_samples.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Check the test set
function [J] = test_samples(test_data, randidx, theta, degree)

%test = test_data(randidx(407:506), :);
[mtest n] = size(test_data);
xtest = test_data(:, 1:n-1); 
ytest = test_data(:, n); 
xtest = poly(xtest,degree);
% Scale features and set them to zero mean
%fprintf('Normalizing Features - CV ...\n');
[xtest mu sigma] = featureNormalize(xtest);
% Add intercept term to X
printf("1\n");
xtest = [ones(mtest, 1) xtest];
size(xtest);
size(ytest);

J = computeCost(xtest, ytest, theta);
end;
