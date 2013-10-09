function [mu, v, mustar, vstar, F, f_sums, f_sums_test] = gpr_additive_ssm(stfunc, logtheta, X, y, numGaussSeidel, numSamples, Xstar)

%compute posterior means and optionally variances at training and test
%inputs using backfitting & gibbs sampling
%INPUTS
%stfunc: state transition function
%logtheta: D+2 hyperparameters -- D lengthscales, sig_var and sig_noise
%X: training inputs
%y: training targets
%numGaussSeidel: number of backfitting/Gauss-Seidel iterations
%numSamples : number of samples for variance computations
%xstar: test inputs
%OUTPUTS
%F: individual univariate function posterior means
%mu: posterior means at training inputs
%v: posterior marginal variances at training inputs
%mustar: posterior means at test inputs
%vstar: posterior marginal variances at test inputs

%call checks
%valid call : [F, mu, v, mustar, vstar] = gpr_additive_ssm(stfunc, logtheta, X, y, numGaussSeidel, numGibbs, xstar)

assert((nargin == 7) && (nargout >= 4));

%argument size checks
[N,D] = size(X);
assert(length(y) == N);
assert(size(Xstar,2) == D);
[M,D] = size(Xstar);

%map hypers into SSM parameters
switch stfunc
    case 'st_exp'
        nu = 1;
    case 'st_matern3'
        nu = 3;
    case 'st_matern7'
        nu = 7;
    otherwise
        error('Invalid stfunc, quitting...');
end
p = (nu + 1)/2; %latent space dim
loghyper = 2*logtheta;
loghyper(1:D) = -(loghyper(1:D)/2) + log(sqrt(nu));

noise_var = exp(loghyper(end));

%DEBUG
%test gpr_ffbs
%[nlml, fsample, fstar_sample] = gpr_ffbs_fast(loghyper([1, D+1, D+2]), stfunc, X(:,1), y, Xstar(:,1));

%cache data before run -- essential!
for d = 1:D
    lh = loghyper([d, D+d, 2*D+1]);
    lambda = exp(lh(1));
    sigvar = exp(lh(2));
    [t, sort_idx_train] = sort(X(:,d));
    [tstar, sort_idx_test] = sort(Xstar(:,d));
    [tt, sort_idx] = sort([t; tstar]);
    is_train = (sort_idx < (length(t)+1));
    [zz, V0] = feval(stfunc, lambda, sigvar, -1);
    assert(size(V0,1) == p);
    delta_t = diff(tt);
    
    %[Phis, Qs] = arrayfun(@(x)feval(stfunc,lambda,sigvar,x), delta_t, 'UniformOutput', false);
    Phis = {};
    Qs = {};
    parfor x = 1:length(delta_t)
        [Phis_i, Qs_i] = feval(stfunc,lambda,sigvar,delta_t(x));
        Phis{x} = Phis_i;
        Qs{x} = Qs_i;
    end
    
    ffbs_data(d).loghyper = lh;
    ffbs_data(d).t = t;
    ffbs_data(d).tstar = tstar;
    ffbs_data(d).V0 = V0;
    ffbs_data(d).Phis = Phis;
    ffbs_data(d).Qs = Qs;
    ffbs_data(d).sort_idx_train = sort_idx_train;
    ffbs_data(d).sort_idx_test = sort_idx_test;
    ffbs_data(d).is_train = is_train;
end 

%BACKFITTING iterations
F = zeros(N,D);
Ftest = zeros(M,D);
ds = 1:D;
for i = 1:numGaussSeidel
    Fprev = F;
    for d = ds
        yd = y - sum(F(:, ds(ds ~= d)), 2);
        [nlml, Ex, Vx, Exx] = ...
        gpr_ssm_fb(ffbs_data(d).loghyper, ffbs_data(d).t, yd(ffbs_data(d).sort_idx_train), ffbs_data(d).tstar, ffbs_data(d).V0, ffbs_data(d).Phis, ffbs_data(d).Qs);
        %read out the training and test means
        Ex_train = Ex(ffbs_data(d).is_train);
        train_means = cell2mat(Ex_train);
        train_means = train_means(1:p:end);
        train_means(ffbs_data(d).sort_idx_train) = train_means;
        Ex_test = Ex(~ffbs_data(d).is_train);
        test_means = cell2mat(Ex_test);
        test_means = test_means(1:p:end);
        test_means(ffbs_data(d).sort_idx_test) = test_means;
        F(:, d) = train_means;
        Ftest(:, d) = test_means;
        
%         %%% test EG
%         Exxs{d}=Exx;
    end
    delta = sum((Fprev(:) - F(:)).^2);
    fprintf('delta F = %5.8f\n', delta);
    if delta < 1e-5
        break;
    end
end

%  %%% test EG
%  varExx = zeros(length(Exxs{1}),1);
%  for dim = 1:length(Exxs)
%      for cellidx = 1:length(Exxs{1})
%          temp = Exxs{dim}{cellidx};
%         varExx(cellidx)=varExx(cellidx)+temp(1);
%      end
%  end

%Now generate samples from the posterior
%initialize at mode!
F_sample = F;
F_sample_test = Ftest;

inter_sample_dist = 2; 
numGibbs = numSamples*inter_sample_dist;
NS = 0; 
f_sums = zeros(N, numSamples);
f_sums_test = zeros(M, numSamples);
  
%tic
for i = 1:numGibbs
    for d = ds
        yd = y - sum(F_sample(:, ds(ds ~= d)), 2);
        [nlml, train_sample, test_sample] = ...
            gpr_ssm_ffbs(ffbs_data(d).loghyper, ffbs_data(d).t, yd(ffbs_data(d).sort_idx_train), ffbs_data(d).tstar, ffbs_data(d).V0, ffbs_data(d).Phis, ffbs_data(d).Qs, 1, round(rand*1e9));
        F_sample(ffbs_data(d).sort_idx_train, d) = train_sample';
        F_sample_test(ffbs_data(d).sort_idx_test, d) = test_sample';
    end
    if (mod(i,inter_sample_dist) == 0)
        %take snapshot
        NS = NS + 1;
        fprintf('%i\n', NS);
       % toc
       % tic
        f_sums(:, NS) = sum(F_sample, 2);
        f_sums_test(:, NS) = sum(F_sample_test, 2);
    end
end
%toc
mu = sum(F, 2);
v = var(f_sums, 0, 2) + noise_var;
%xx = f_sums - repmat(mu, 1, size(f_sums, 2));
%v = mean(xx.*xx, 2);
mustar = sum(Ftest, 2);
%xx = f_sums_test - repmat(mustar, 1, size(f_sums_test, 2));
%vstar = mean(xx.*xx, 2);
vstar = var(f_sums_test, 0, 2) + noise_var;
%alpha = exp(-loghyper(D+2))*(y - mu);
