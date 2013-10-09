function logp = gpr_ffbs_wrap(log_lambda, stfunc, t, f, log_sigvar, alpha)

% Sample log_lambda (i.e. lengthscale) given samples of everything else
% 1D problem! TODO: better alternatives than slice sampling?

%CAUTION: Have to recompute Phi and Qs etc. due to changing lambda!! i.e. costly!

lambda = exp(log_lambda);
sigvar = exp(log_sigvar);

[zz, V0] = feval(stfunc, lambda, sigvar, -1);
[Phis, Qs] = arrayfun(@(x)feval(stfunc,lambda,sigvar,x), diff(t), 'UniformOutput', false);

[nlml, nsse] = gpr_ssm_lik([log_lambda; log_sigvar; log(1e-10)], t, f, V0, Phis, Qs);

nlml = nlml + log(alpha.std_log_lambda) + ((log_lambda - alpha.mu_log_lambda)^2)/(2*alpha.std_log_lambda^2);
logp = -nlml;
