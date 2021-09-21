% This program solves a recusive household decision problem on consumption
% and saving. Log utility function is used and uncertainty of income is
% assumed away.
% Switch on the parfor loop for parallel computing in MatLab 

clear;
close;
clc;
cd 'E:\Dropbox\Econ 899 Lectures\Fall 2018';
%% Preparation for value function iteration
% Parameters
beta=.98;      %discount factor
r=0.03;        % interest rate
y=4;           % income
tol = 1e-3;    % Tolerance level

% construct asset grid
alb=0.01; %lower bound of grid points
aub=16;   %upper bound of grid points
n=600;    %size of the asset grid
inc=(aub-alb)/(n-1); %increments
a=(alb:inc:aub)';    %asset grid
ap=repmat(a,1,n);    %creat a grid for next period asset choice

% initial guess of the value functions and decision rules
v0=zeros(n,1);
v1=zeros(n,1);
optindex=ones(n,1);

% counter of iterations
iter=0;

% initialize supnorm of values function in two consecutive iterations
supnorm=1;

%% Value function iteration
tic;
%parpool(4); % start the parallel pool in matlab. The number in parenthesis is the number of cores in CPU
while supnorm>=tol  
    for i=1:n % sequential loop
        %parfor i=1:n % parallel loop
        ap_temp=ap(:,i);
        c=y+(1+r)*a(i)-ap_temp;
        c=(c>0).*c+(c<=0).*0;
        [v1(i),optindex(i)]=max(log(c)+beta*v0);
    end
supnorm=norm(v0-v1); % update supnorm
v0=v1;
iter=iter+1;
fprintf('Iteration # %2d \tSupnorm: %g \n',iter,supnorm);
end 
toc;
%% Extract value functions and decision rules
opta=a(optindex);
optc=y+(1+r)*a-opta;

