function [lmls, logthetas, nlmls, p_hat, p_hat_star] = gpc_additive_mcmc(stfunc, likfunc, X, y, Xstar, numMCMC, numNewton, mh_std) 

[Ntrn, D] = size(X);
[Ntst, D] = size(Xstar);

%hyperparameter initialization 
ells = (max(X) - min(X))/2;
logtheta = log([ells'; ones(D,1)]); 

if numMCMC == 0 %run once for fixed hypers
    
    fprintf('\n Running with fixed hyperparameters...\n');
    [p_hat, p_hat_star, F, Ftest, nlml] = gpc_additive_la(stfunc, likfunc, logtheta, X, y, Xstar, numNewton);
    lmls = 0;
    logthetas = logtheta;
    nlmls = nlml;
    
else
    
    num_accept = 3;
    logthetas = zeros(2*D, numMCMC);
    lmls = zeros(numMCMC, 1);
    nlmls = zeros(numMCMC, 1);
    p_hat = zeros(Ntrn, numMCMC);
    p_hat_star = zeros(Ntst, numMCMC);
    
    for i = 1:numMCMC
        fprintf('\n MCMC iteration %i...\n', i);
        %don't iterate over test inputs during learning
        [logtheta, lml] = mh_sampler('gpc_additive_la_wrap', logtheta, mh_std, num_accept, {stfunc, likfunc, X, y, Xstar(1,:), numNewton});
        logthetas(:,i) = logtheta;
        fprintf('Hyper sample returned = \n');
        for d = 1:(2*D)
            fprintf('%5.5f ', exp(2*logtheta(d)));
        end
        fprintf('\n');
        lmls(i) = lml;
        %run full model on sample
        [p, pstar, F, Fstar, nlml] = gpc_additive_la(stfunc, likfunc, logtheta, X, y, Xstar, numNewton);
        p_hat(:, i) = p;
        p_hat_star(:, i) = pstar;
        nlmls(i) = nlml;
    end
    
end
