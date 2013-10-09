function logp = gp_lik_wrap(lh, stfunc, t, y, log_noise_var, alpha)

% Sample log_lambda (i.e. lengthscale) given samples of everything else
% 1D problem! TODO: better alternatives than slice sampling?

%CAUTION: Have to recompute Phi and Qs etc. due to changing lambda!! i.e. costly!

lambda = exp(lh(1));
sigvar = exp(lh(2));

[zz, V0] = feval(stfunc, lambda, sigvar, -1);
%[Phis, Qs] = arrayfun(@(x)feval(stfunc,lambda,sigvar,x), diff(t), 'UniformOutput', false);

delta_t =  diff(t);
Phis = {};
Qs = {};
parfor x = 1:length(delta_t)
    [Phis_i, Qs_i] = feval(stfunc,lambda,sigvar,delta_t(x));
    Phis{x} = Phis_i;
    Qs{x} = Qs_i;
end

[nlml, nsse] = gpr_ssm_lik([lh; log_noise_var], t, y, V0, Phis, Qs);

nlml = nlml + log(alpha.std) + ((lh(1) - alpha.mu)^2)/(2*alpha.std^2);
nlml = nlml + log(alpha.std) + ((lh(2) - alpha.mu)^2)/(2*alpha.std^2);
logp = -nlml;
