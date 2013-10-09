%%% ELEVATORS DATA

clear;
close all;
%set seeds
%rand('state',22);
%randn('state',20);
global folderPath
data_id = 'elevators';
load([folderPath.datasets,'/mat-files/elevators.mat']);

for d = 1:size(X_tr,2)
    muxd = mean(X_tr(:,d));
    sdxd = std(X_tr(:,d));
    X_tr(:,d) = (X_tr(:,d) - muxd)/sdxd;
    X_tst(:,d) = (X_tst(:,d) - muxd)/sdxd;
end

Xall = [X_tr; X_tst];
yall = [T_tr; T_tst];
Ntrain = 8752;
X = Xall(1:Ntrain, :);
y = yall(1:Ntrain);
Xstar = Xall((Ntrain+1):end, :);
ystar = yall((Ntrain+1):end);
mu_y = mean(y);
std_y = 1;
y = (y - mu_y)/std_y;
ystar = (ystar - mu_y)/std_y;
[N,D] = size(X);
M = size(Xstar,1);

stfunc = 'st_matern7'; 
run_params.numSubset = 1000; %subset of data for projection inference
run_params.dproj = D/2; %number of projection dimensions
run_params.numPseudo = 250; %number of pseudo inputs for SPGP
run_params.numMCMC = 5; %number of full MCMC iterations
run_params.rand_init = true; %initialize proj pursuit weight randomly or with linear model?

Ns= N;
results = run_comparisons_temp(stfunc, X, y, Xstar, ystar, run_params,data_id);

%print_results;

%save(sprintf('%s_experiment_%i_%i_%i.mat', data_id, N, D, M));