function [nmses,mnlps,testnmses,W] = pp_gpr_testrand(stfunc, X, y, Xstar, ystar, maxNumProj, random_init)

N = size(X, 1);
assert(length(y) == N);
assert(size(X,2) == size(Xstar,2));
[M, D] = size(Xstar);
fstar = zeros(M,1);
y_orig = y;

for i = 1:maxNumProj

    %Initialization: random or linear?
    if nargin > 6 && random_init
        w = randn(D,1);
    else
        w = (X'*X + 1e-4*eye(D))\(X'*y);
    end
    logtheta = log([1; std(y); 0.1]);
    phi = [w; logtheta];
%     phi_learned = minimize(phi, 'pp_gpr_1d', -50, stfunc, X, y); 
%     phi_learned = minimize(phi, 'pp_gpr_1d_wrap', -50, stfunc, X, y); 
    phi_learned = minimize(phi, 'pp_gpr_1d_wrap_parallel', -40, stfunc, X, y);
%      phi_learned = minimize(phi, 'pp_gpr_1d_wrap', -40, stfunc, X, y);
    phis(:,i) = phi_learned;
    
    w = phi_learned(1:D);
    
%     % trial - normalize the weights
%     w = w/norm(w); 

%     w = (abs(w) == max(abs(w)))
    logtheta = phi_learned(D+1:D+3);
    
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
    loghyper = 2*logtheta;
    loghyper(1) = -(loghyper(1)/2) + log(sqrt(nu));
    
    lambda = exp(loghyper(1));
    sigvar = exp(loghyper(2));
    noise_var = exp(loghyper(3));
 
    Xw = X*w; %1D
    Xstarw = Xstar*w;
    [t, sort_idx_train] = sort(Xw);
    [tstar, sort_idx_test] = sort(Xstarw);
    [tt, sort_idx] = sort([Xw; Xstarw]);
    is_train = (sort_idx < (length(t)+1));
    [zz, V0] = feval(stfunc, lambda, sigvar, -1);
    p = size(V0,1); %ssm latent dim
    delta_t = diff(tt);
    %[Phis, Qs] = arrayfun(@(x)feval(stfunc,lambda,sigvar,x), delta_t, 'UniformOutput', false);
    Phis = {};
    Qs = {};
    parfor x = 1:length(delta_t)
        [Phis_i, Qs_i] = feval(stfunc,lambda,sigvar,delta_t(x));
        Phis{x} = Phis_i;
        Qs{x} = Qs_i;
    end
    [nlml, Ex, Vx, Exx] = ...
                gpr_ssm_fb(loghyper, t, y(sort_idx_train), tstar, V0, Phis, Qs);
    %read out the training and test means
    Ex_train = Ex(is_train);
    train_means = cell2mat(Ex_train);
    train_means = train_means(1:p:end);
    train_means(sort_idx_train) = train_means;
    Ex_test = Ex(~is_train);
    test_means = cell2mat(Ex_test);
    test_means = test_means(1:p:end);
    test_means(sort_idx_test) = test_means;
    
    y = y - train_means;
    fstar = fstar + test_means;
    
    nmse_test = mean(((ystar - fstar).^2))/mean(ystar.^2);
    fprintf('\n NMSE at level %i = %5.5f\n', i, nmse_test);
    nmses(i) = nmse_test;
    
%     if (i > 1) && ((nmses(i-1) - nmses(i)) < 1e-4)
    if (i > 1) && (abs(nmses(i-1) - nmses(i)) < 1e-14)
        fprintf('Sufficient number of projections = %i\n', i);
        break;
    end
    
    %% get test error for this
    Dproj = length(nmses);
    W = phis(1:D, :);
    logtheta1 = zeros(2*Dproj+1, 1);
    for d = 1:Dproj
        logtheta1(d) = phis(end-2, d);
        logtheta1(Dproj + d) = phis(end-1, d);
    end
    logtheta1(end) = phis(end, end);
    [mu, v, mustar, vstar] = gpr_additive_ssm(stfunc, logtheta1, X*W, y_orig, 30, 50, Xstar*W);
    %[mu, v, mustar, vstar] = gpr_additive_ssm(stfunc, logtheta, X*W, y, 5, 5, Xstar*W);
    mnlps(i) = 0.5*(log(2*pi) + mean(log(vstar)) + mean(((ystar - mustar).^2)./vstar));
    testnmses(i) = mean((ystar - mustar).^2)/mean(ystar.^2);
    
end
