 %
% Elad Gilboa, Yunus Saatci 2013
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;
close all;
%set seeds
%rand('state',22);
%randn('state',20);

stfunc = 'st_matern7'; 

global Dim;

%D = Dim; 
D=8;
M = 1000;
Ns = 8000; %[1000, 2000:2000:10000, 20000:10000:50000]';
ADDITIVE_VB = zeros(size(Ns));
PPGPR = zeros(size(Ns));
FULL_GP = zeros(size(Ns));
SPGP = zeros(size(Ns));

n = 1;
for N = Ns'
    
    fprintf('Gathering timing results for N = %i...\n', N);
    
    %%% GENERATE DATA FROM MODEL
    logtheta = [repmat(log(1.0), D, 1); log(1.0); log(0.1)]; %true hypers
    
    X = zeros(N,D);
    Xstar = zeros(M,D);
    Z = zeros(N,D);
    Z_test = zeros(M,D);
  
    for d = 1:D
        
        %fprintf('%i\n', d);
        t = rand(N,1)*10 - 5;
        tstar = linspace(min(t) - 1, max(t) + 1, M)';
        X(:,d) = t;
        Xstar(:,d) = tstar;
        sample = gpr_ffbs_prior(logtheta([d, D+1, D+2]), stfunc, [t; tstar]);
        Z(:,d) = sample(1:N)';
        Z_test(:,d) = sample(N+1:end)';
        
    end
    y = sum(Z,2) + randn(N,1)*exp(logtheta(end));
    mu_y = mean(y);
    y = y - mu_y;
    
    ystar = sum(Z_test,2) + randn(M,1)*exp(logtheta(end));
    ystar = ystar - mu_y;
    
    %%% RUNTIMEs
    stfunc = 'st_matern7'; %TODO: make this st_se
    run_params.numSubset = 1000; %subset of data for projection inference
    run_params.dproj = D; %number of projection dimensions
    run_params.numPseudo = 500; %number of pseudo inputs for SPGP
    run_params.numMCMC = 10; %number of full MCMC iterations
    run_params.rand_init = false; %initialize proj pursuit weight randomly or with linear model?

    results(n) = run_comparisons(stfunc, X, y, Xstar, ystar, run_params);

    
    
    n = n+1;
    
end

%save(sprintf('~/PhD/src/matlab/additive_gp/reg_runtimes_N_%s.mat', datestr(now, 'yyyymmdd')), 'Ns', 'PPGPR', 'ADDITIVE_VB', 'FULL_GP', 'SPGP');
    


