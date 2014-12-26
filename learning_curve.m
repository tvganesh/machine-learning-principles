
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 
%% This is Octave project runs through the principles of multivariate regression and checks the optimal solution
%% Designed and developed by Tinniam V Ganesh
%% Date 25 Dec 2014
%% File: learning_curve.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [J J_history] = learning_curve(training, cross_validation,randidx, alpha, lambda, degree)
alpha =0.01;
lambda = 0.1;

[a b] = size(training);
xtrain = training(:,1:b-1); 
ytrain = training(:,b);
[xtrain mu sigma] = featureNormalize(xtrain);

[m n] = size(cross_validation);
xcv = cross_validation(:, 1:n-1);
ycv = cross_validation(:, n); 
[xcv mu sigma] = featureNormalize(xcv);

num_iters = 100;


printf("1\n");
[c d] = size(xtrain);
%figure;
Jtrain = zeros(100, 1);
Jcv = zeros(100,1);
for i = 1: 100
     xsample = xtrain(1:i,:);
	 ysample = ytrain(1:i,:);
	 size(xsample);
	 size(ysample);
	 [xsample] = poly(xsample,degree);
	 xsample= [ones(i, 1) xsample];
	 [c d] = size(xsample);
	 theta = zeros(d, 1);

    % Minimize using fmincg
	 J = computeCost(xsample, ysample, theta);
     Jtrain(i) = J;
	 xsample_cv = xcv(1:i,:);
	 ysample_cv = ycv(1:i,:);
	 [xsample_cv] = poly(xsample_cv,degree);
	 xsample_cv= [ones(i, 1) xsample_cv];
	 J_cv = computeCost(xsample_cv, ysample_cv,theta)
	 Jcv(i) = J_cv;
end;

plot(1:numel(Jtrain), Jtrain, '-k', 'LineWidth', 2);
hold on;
plot(1:numel(Jcv), Jcv, '-m', 'LineWidth', 2);
xlabel('Number of samples');
ylabel('Cost J');
axis([0 100 0 40]);
legend('Training cost', 'Cross Validation cost');
if degree == 1
    title('Learning curve  - Degree 1');
	print -dpng learning-curve-1.png;
elseif degree == 2
   title('Learning curve  - Degree 2');
   print -dpng learning-curve-2.png;
elseif degree == 3
   title('Learning curve  - Degree 3');
   print -dpng learning-curve-3.png;
elseif degree == 4
   title('Learning curve  - Degree 4');
   print -dpng learning-curve-4.png;
end;


end;