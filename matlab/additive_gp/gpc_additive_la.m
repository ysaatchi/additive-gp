function [p_hat, p_hat_star, F, Ftest, nlml] = gpc_additive_la(stfunc, likfunc, logtheta, X, y, Xstar, numNewton, numGS)

%INPUTS
%stfunc: state transition function corresponding to GP kernel
%likfunc: likelihood function 
%logtheta: hyperparameters: [log(ell); log(sqrt(sigvars))]
%X : input locations (N,D)
%y : N vector OR N-cell (if multiple targets per input X_{i,d})
%Xstar : test input locations (M,D) --> MUST BE UNIQUE
%numNewton : number of Newton iterations

assert((nargin == 8 && nargout > 1));

%argument size checks
[N,D] = size(X);
assert(length(y) == N);
assert(size(Xstar,2) == D);
assert(length(logtheta) == 2*D);
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

%cache data before run -- essential!
for d = 1:D
    fprintf('d = %i\n', d);
    lh = loghyper([d, D+d]);
    lambda = exp(lh(1));
    sigvar = exp(lh(2));
    [t, sort_idx_train] = sort(X(:,d));
    [tstar, sort_idx_test] = sort(Xstar(:,d));
    [tt, sort_idx] = sort([t; tstar]);
    is_train = (sort_idx < (length(t)+1));
    [zz, V0] = feval(stfunc, lambda, sigvar, -1);
    assert(size(V0,1) == p);
    delta_t = diff(tt);
    Phis = {};
    Qs = {};
    parfor x = 1:length(delta_t)
        [Phis_i, Qs_i] = feval(stfunc,lambda,sigvar,delta_t(x));
        Phis{x} = Phis_i;
        Qs{x} = Qs_i;
    end
    %[Phis, Qs] = arrayfun(@(x)feval(stfunc,lambda,sigvar,x), delta_t, 'UniformOutput', false);
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

%NEWTON Iterations
F = zeros(N,D);
pp = zeros(N,1);
Ftest = zeros(M,D);
ds = 1:D;

for i = 1:numNewton
    
    Fprev = F;
    pprev = pp;
    [ll, nabla, W] = feval(likfunc, F, y);
    z = sum(F,2) + (nabla ./ W);
    %Run backfitting...
    for j = 1:numGS
        Fprev2 = F;
        for d = ds
            
            zd = z - sum(F(:, ds(ds ~= d)), 2);
            noises = 1./W;
            noises = noises(ssm_data(d).sort_idx_train);
            [nlml, Ex, Vx, Exx, ns, nld] = ...
                gpr_ssm_fb_diag(ssm_data(d).loghyper, noises, ssm_data(d).t, zd(ssm_data(d).sort_idx_train), ssm_data(d).tstar, ssm_data(d).V0, ssm_data(d).Phis, ssm_data(d).Qs);
            %read out the training and test means
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

%             [nlml, train_pred, test_pred, E] = gpr_ssm_diag(ssm_data(d).loghyper, stfunc, 1./W, ssm_data(d).t, zd(ssm_data(d).sort_idx_train), ssm_data(d).tstar);
%             train_means = train_pred(:,1);
%             train_means(ssm_data(d).sort_idx_train) = train_means;
%             test_means = test_pred(:,1);
%             test_means(ssm_data(d).sort_idx_test) = test_means;
%             F(:, d) = train_means;
%             Ftest(:, d) = test_means;
            
        end
        delta = mean(abs(Fprev2(:) - F(:)));
        fprintf('delta F (backfit) = %5.5f\n', delta);
        if delta < 1e-2
            break;
        end
        
    end
    pp = feval(likfunc, F);
    deltap = mean(abs(pprev - pp));
    %fprintf('delta F = %7.2f\n', delta);
    fprintf('delta p = %5.5f\n', deltap);
    if isnan(delta) || isnan(deltap)
        error('NaNs!!');
    end
    if deltap < 5e-3
        break;
    end
    
end

%compute (approximate) neg log marginal likelihood
[ll, nabla, W] = feval(likfunc, F, y);
ldw = D*sum(log(W));
nsse = 0; nlogdet = 0;
for d = ds
    [nlml, Ex, Vx, Exx, nssed, nld] = ...
            gpr_ssm_fb_diag(ssm_data(d).loghyper, ones(N,1)*1e-8, ssm_data(d).t, F(ssm_data(d).sort_idx_train, d), ssm_data(d).tstar, ssm_data(d).V0, ssm_data(d).Phis, ssm_data(d).Qs);
    [nlml, Ex, Vx, Exx, ns, nlogdetd] = ...
            gpr_ssm_fb_diag(ssm_data(d).loghyper, 1./W, ssm_data(d).t, zeros(N,1), ssm_data(d).tstar, ssm_data(d).V0, ssm_data(d).Phis, ssm_data(d).Qs);
    %[nlml, train_pred, test_pred, E, nssed] = gpr_ssm_diag(ssm_data(d).loghyper, stfunc, ones(N,1)*1e-8, ssm_data(d).t, F(ssm_data(d).sort_idx_train, d), ssm_data(d).tstar);
    %[nlml, train_pred, test_pred, E, ns, nlogdetd] = gpr_ssm_diag(ssm_data(d).loghyper, stfunc, 1./W, ssm_data(d).t, zeros(N,1), ssm_data(d).tstar);
    nsse = nsse + nssed;
    nlogdet = nlogdet + nlogdetd;
end
nlml = 0.5*(ldw + nsse + nlogdet) - ll;

% %sanity check -- PASSED!
% nsse2 = 0;
% nlogdet2 = 0;
% K = zeros(N);
% for d = ds
%     fd = F(:,d);
%     Kd = covMatern3iso(logtheta([d, D+d]), X(:,d)) + 1e-8*eye(N);
%     KdW = covMatern3iso(logtheta([d, D+d]), X(:,d)) + diag(1./W);
%     L = chol(Kd, 'lower');
%     LW = chol(KdW, 'lower');
%     alpha = L'\(L\fd);
%     nsse2 = nsse2 + fd'*alpha;
%     nlogdet2 = nlogdet2 + 2*sum(log(diag(LW)));
%     K = K + Kd;
% end
% K2 = covMatern3_additive(logtheta, X);

p_hat = feval(likfunc, F);
p_hat_star = feval(likfunc, Ftest);
