% RUN COMPARIONS
%
% comaprion of regression algorithms
%
% Elad Gilboa, Yunus Saatci 2013
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function results = run_comparisons(stfunc, X, y, Xstar, ystar, run_params)
results = 0;
[N, D] = size(X);
M = size(Xstar,1);
assert(length(y) == N);
assert(length(ystar) == M);
assert(mean(y) < 1e-8);
assert(size(Xstar, 2) == D);

%setup run config
rand_init = run_params.rand_init;
numSubset = min(N,run_params.numSubset); %subset of data for projection inference
maxNumProj = run_params.dproj; %number of projection dimensions
numPseudo = run_params.numPseudo; %number of pseudo inputs for SPGP
num_mcmc_iter = run_params.numMCMC; %number of full MCMC iterations
additive_mcmc = [];
ppr_mcmc = [];
ppgpr = [];
additive_vb = [];
full_gp_add = [];
full_gp = [];
spgp = [];

%hyperparameter initialization (for routines which need it)
ells = (max(X) - min(X))/10;
s0 = var(y);
logtheta_init = log([ells'; ones(D,1)*sqrt(s0/D); sqrt(s0)/4]); 

%% Projection Pursuit GP regression 
tic;
% maxNumProj = 3*D;
[fstar, phis, nmses] = pp_gpr(stfunc, X, y, Xstar, ystar, maxNumProj, rand_init);
% [fstar, phis, nmses] = pp_gpr(stfunc, X, y, Xstar, ystar, maxNumProj);
Dproj = length(nmses);
W = phis(1:D, :);
logtheta = zeros(2*Dproj+1, 1);
for d = 1:Dproj
    logtheta(d) = phis(end-2, d);
    logtheta(Dproj + d) = phis(end-1, d);
end
logtheta(end) = phis(end, end);
[mu, v, mustar, vstar] = gpr_additive_ssm(stfunc, logtheta, X*W, y, 30, 50, Xstar*W);
%[mu, v, mustar, vstar] = gpr_additive_ssm(stfunc, logtheta, X*W, y, 5, 5, Xstar*W);
mnlp = 0.5*(log(2*pi) + mean(log(vstar)) + mean(((ystar - mustar).^2)./vstar));
exec_time = toc;
fprintf('Testing NMSE (pp-gpr) = %3.5f\n', nmses(end));
fprintf('Testing MNLP (pp-gpr) = %3.5f\n', mnlp);
ppgpr.phis = phis;
ppgpr.nmses = nmses;
ppgpr.mnlp = mnlp;


ppgpr.exec_time = exec_time;
fprintf('Execution time (pp-gpr) = %5.1f\n', exec_time);

%% Variational Bayes
%TRAINING via VBEM
tic;
logtheta = gpr_additive_vbem(stfunc, logtheta_init, X, y, 3);
[mus, vars] = gpr_additive_vb_pred(stfunc, logtheta, X, y, Xstar);
additive_vb.mnlp = 0.5*(log(2*pi) + mean(log(vars)) + mean(((ystar - mus).^2)./vars));
additive_vb.nmse = mean(((ystar - mus).^2))/mean(ystar.^2);
fprintf('Testing NMSE (additive-VB) = %3.5f\n', additive_vb.nmse);
fprintf('Testing MNLP (additive-VB) = %3.5f\n', additive_vb.mnlp);
additive_vb.logtheta = logtheta;
exec_time = toc;
additive_vb.exec_time = exec_time;
fprintf('Execution time (additive GP -- VB) = %5.1f\n', exec_time);


% %%% MCMC
%setup hyper-priors (in log space)
alpha.mu = 0; alpha.std = 3;
gamma.a_noise = 1; gamma.b_noise = 0.1; %1/noise_var will have mean gamma.a_noise/gamma.b_noise and variance gamma.a_noise/gamma.b_noise^2

tic;
[sample_logthetas, S] = gpr_additive_mcmc(stfunc, alpha, gamma, X, y, Xstar, ystar, num_mcmc_iter);
additive_mcmc.logthetas = sample_logthetas;
for i = 1:length(S)
    additive_mcmc.nmse(i) = S(i).nmse_test_mcmc;
    additive_mcmc.mnlp(i) = S(i).mnlp_test_mcmc;
end
exec_time = toc;
additive_mcmc.exec_time = exec_time;
fprintf('Execution time (additive-mcmc) = %5.1f\n', exec_time);


%% Rotated MCMC 
tic;
%[d, dy, dh] = jf_checkgrad({'gp_additive_sldr', X(1:1000,:), y(1:1000), 2}, [randn(2*D,1); zeros(2*2+1,1)], 1e-8);
[dum,I] = sort(rand(N,1)); clear dum;
I = I(1:numSubset);
theta = minimize([randn(d*D,1); zeros(d*2+1,1)], 'gp_additive_sldr', -50, X(I,:), y(I), d);
W = reshape(theta(1:d*D), d, D);
Xnew = X*W'; XstarNew = Xstar*W';

[sample_logthetas2, S2] = gpr_additive_mcmc(stfunc, alpha, gamma, Xnew, y, XstarNew, ystar, num_mcmc_iter);
ppr_mcmc.logthetas = sample_logthetas2;
for i = 1:length(S2)
    ppr_mcmc.nmse(i) = S2(i).nmse_test_mcmc;
    ppr_mcmc.mnlp(i) = S2(i).mnlp_test_mcmc;
end
exec_time = toc;
ppr_mcmc.exec_time = exec_time;
fprintf('Execution time (ppr-mcmc) = %5.1f\n', exec_time);



% %% GP (additive) naive (used more as a sanity check for the above -- only run for small datasets)
% 
% if  N*D <= 72000 && N <= 9000
%     tic;
%     
%     logtheta = minimize(logtheta_init, 'gp_covSEard_additive', -50, X, y);
%     
%     %obtain test predictions in batches of 1000 in order to save memory
%     bb = unique([0:1000:M, M]);
%     mus = zeros(M,1);
%     vars = zeros(M,1);
%     for i = 1:length(bb)-1
%         [nlml, dnlml, musb, varsb] = gp_covSEard_additive(logtheta, X, y, Xstar(bb(i)+1:bb(i+1), :));
%         mus(bb(i)+1:bb(i+1)) = musb;
%         vars(bb(i)+1:bb(i+1)) = varsb;
%     end
%     mnlp_test_add = 0.5*(log(2*pi) + mean(log(vars)) + mean(((ystar - mus).^2)./vars));
%     nmse_test_add = mean(((ystar - mus).^2))/mean(ystar.^2);
%     fprintf('Testing NMSE (additive-naive) = %3.5f\n', nmse_test_add);
%     fprintf('Testing MNLP (additive-naive) = %3.5f\n', mnlp_test_add);
%     logtheta_add = logtheta;
%     exec_time = toc;
%     fprintf('Execution time (additive GP -- naive) = %5.1f\n', exec_time);
%    
%     full_gp_add.mnlp = mnlp_test_add;
%     full_gp_add.nmse = nmse_test_add;
%     full_gp_add.logtheta = logtheta_add;
%     full_gp_add.exec_time = exec_time;
% end
% 
% %% Full GP (tensor)
% 
% if  N*D <= 72000 && N <= 9000 %this is the limit for full GP models if you want to continue using your box
%     tic;
%     
%     logtheta = minimize([logtheta_init(1:D); log(sqrt(s0)); logtheta_init(end)], 'gp_covSEard', -50, X, y);
% %    obtain test predictions in batches of 1000 in order to save memory
%     bb = unique([0:1000:M, M]);
%     mus = zeros(M,1);
%     vars = zeros(M,1);
%     for i = 1:length(bb)-1
%         [nlml, dnlml, musb, varsb] = gp_covSEard(logtheta, X, y, Xstar(bb(i)+1:bb(i+1), :));
%         mus(bb(i)+1:bb(i+1)) = musb;
%         vars(bb(i)+1:bb(i+1)) = varsb;
%     end
%     mnlp_test = 0.5*(log(2*pi) + mean(log(vars)) + mean(((ystar - mus).^2)./vars));
%     nmse_test = mean(((ystar - mus).^2))/mean(ystar.^2);
%     fprintf('Testing NMSE (joint GP) = %3.5f\n', nmse_test);
%     fprintf('Testing MNLP (joint GP) = %3.5f\n', mnlp_test);
%     exec_time = toc;
%     fprintf('Execution time (joint GP) = %5.1f\n', exec_time);
%     
%     full_gp.mnlp = mnlp_test;
%     full_gp.nmse = nmse_test;
%     full_gp.logtheta = logtheta;
%     full_gp.exec_time = exec_time;
% 
% end

%% SPGP (tensor)
tic;

[N,dim] = size(X);
M = min(N,numPseudo);

% initialize pseudo-inputs to a random subset of training inputs
[dum,I] = sort(rand(N,1)); clear dum;
I = I(1:M);
xb_init = X(I,:);
%xb_init = X(1:10,:);

%%% Matern
%covfunc = {@covMaterniso, 5}; hyp.cov = [0; 0];

%%% Standrad Exp
covfunc = {@covSEard};
ell = zeros(1,D);
for i=1:D
    ell(i)=1.0;
end
sf = 1.0; hyp.cov = log([ell sf]);
% % initialize hyperparameters sensibly (see spgp_lik for how
% % the hyperparameters are encoded)
% dim = D;
% hyp.cov(1:dim,1) = -2*log((max(X)-min(X))'/2); % log 1/(lengthscales)^2
% hyp.cov(dim+1,1) = log(var(y,1)); % log size 
% hyp.lik = log(var(y,1)/4); % log noise

%nu = fix(n/2); u = linspace(-1.3,1.3,nu)';

covfuncF = {@covFITC, {covfunc}, xb_init};

likfunc = @likGauss; sn = 0.1; hyp.lik = log(sn);

hyp = minimize(hyp, @gp, -50, @infFITC, [], covfuncF, likfunc, X, y);
%nlml = gp(hyp, @infExact, [], covfunc, likfunc, xb_init, y0)
[mus vars] = gp(hyp, @infFITC, [], covfuncF, likfunc, X, y,Xstar);

mnlp_test = 0.5*(log(2*pi) + mean(log(vars)) + mean(((ystar - mus).^2)./vars));
nmse_test = mean(((ystar - mus).^2))/mean(ystar.^2);
fprintf('Testing NMSE (SPGP) = %3.5f\n', nmse_test);
fprintf('Testing MNLP (SPGP) = %3.5f\n', mnlp_test);
exec_time = toc;
fprintf('Execution time (SPGP) = %5.1f\n', exec_time);
spgp.mnlp = mnlp_test;
spgp.nmse = nmse_test;
spgp.hyp = hyp;
spgp.exec_time = exec_time;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
results.additive_mcmc = additive_mcmc;
results.ppr_mcmc = ppr_mcmc;
results.pp_gpr = ppgpr;
results.additive_vb = additive_vb;
results.full_gp_add = full_gp_add;
results.full_gp = full_gp;
results.spgp = spgp;
    
