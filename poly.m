%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 
%% This is Octave project runs through the principles of multivariate regression and checks the optimal solution
%% Designed and developed by Tinniam V Ganesh
%% Date 25 Dec 2014
%% File: poly.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [x] = poly(xinput, n)
x = [];
for i= 1:n
   xtemp = xinput .^i;
   x = [x xtemp];
end;

