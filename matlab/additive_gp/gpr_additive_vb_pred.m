function [mustar, vstar, Ftest, Vtest] = gpr_additive_vb_pred(stfunc, logtheta, X, y, Xstar)

%argument size checks
[N,D] = size(X);
assert(length(y) == N);
assert(size(Xstar,2) == D);
assert(length(logtheta) == 2*D+1);
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

%cache data before run -- essential!
for d = 1:D
    lh = loghyper([d, D+d, end]);
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
    ssm_data(d).loghyper = lh;
    ssm_data(d).t = t;
    ssm_data(d).tstar = tstar;
    ssm_data(d).V0 = V0;
    ssm_data(d).Phis = Phis;
    ssm_data(d).Qs = Qs;
    ssm_data(d).sort_idx_train = sort_idx_train;
    ssm_data(d).sort_idx_test = sort_idx_test;
    ssm_data(d).is_train = is_train;
end 

%BACKFITTING iterations
F = zeros(N,D); V = F;
Ftest = zeros(M,D); Vtest = Ftest;
ds = 1:D;
numGaussSeidel = 5;%20;
for i = 1:numGaussSeidel
    Fprev = F;
    Vprev = V;
    for d = ds
        yd = y - sum(F(:, ds(ds ~= d)), 2);
        [nlml, Ex, Vx, Exx] = ...
            gpr_ssm_fb(ssm_data(d).loghyper, ssm_data(d).t, yd(ssm_data(d).sort_idx_train), ssm_data(d).tstar, ssm_data(d).V0, ssm_data(d).Phis, ssm_data(d).Qs);
        %read out the training and test means and vars
        Ex_train = Ex(ssm_data(d).is_train);
        train_means = cell2mat(Ex_train);
        train_means = train_means(1:p:end);
        train_means(ssm_data(d).sort_idx_train) = train_means;
        Ex_test = Ex(~ssm_data(d).is_train);
        test_means = cell2mat(Ex_test);
        test_means = test_means(1:p:end);
        test_means(ssm_data(d).sort_idx_test) = test_means;
        F(:, d) = train_means;
        Ftest(:, d) = test_means;
        Vx_train = Vx(ssm_data(d).is_train);
        train_vars = cell2mat(Vx_train);
        train_vars = train_vars(1:p:end, 1);
        train_vars(ssm_data(d).sort_idx_train) = train_vars;
        Vx_test = Vx(~ssm_data(d).is_train);
        test_vars = cell2mat(Vx_test);
        test_vars = test_vars(1:p:end, 1);
        test_vars(ssm_data(d).sort_idx_test) = test_vars;
        V(:, d) = train_vars;
        Vtest(:, d) = test_vars;
    end
    delta_F = sum((Fprev(:) - F(:)).^2);
    delta_V = sum((Vprev(:) - V(:)).^2);
    fprintf('delta F = %5.8f\n', delta_F);
    fprintf('delta V = %5.8f\n', delta_V);
    if delta_F < 1e-5
        break;
    end
end

mustar = sum(Ftest,2);
vstar = sum(Vtest,2) + noise_var;