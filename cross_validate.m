%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 
%% This is Octave project runs through the principles of multivariate regression and checks the optimal solution
%% Designed and developed by Tinniam V Ganesh
%% Date 25 Dec 2014
%% File: cross_validate.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Now check the cross-validation error
% Calculate the cost at the theta values to check whether it is a global minima
%% Load Housing Data
function [J J_history] = cross_validate(cross_validation, randidx, theta, degree)
%data  = load('housing.csv');
%cross_validation = data(randidx(307:406), :);
[mcv n] = size(cross_validation);
xcv = cross_validation(:, 1:n-1);
ycv = cross_validation(:, n); 
xcv = poly(xcv,degree);

ycv = cross_validation(:, n);   

[xcv mu sigma] = featureNormalize(xcv);

% Add intercept term to X
xcv = [ones(mcv, 1) xcv];
size(xcv);
size(ycv);

J = computeCost(xcv,ycv,theta);
