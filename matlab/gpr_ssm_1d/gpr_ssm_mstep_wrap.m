function [ecdll, decdll] = gpr_ssm_mstep_wrap(loghyper, stfunc, x, y, Ex, Vx, Exx)

delta_t = diff(x);
lambda = exp(loghyper(1));
sigvar = exp(loghyper(2));
[zz, V0, deriv] = feval(stfunc, lambda, sigvar, -1);
[Phis, Qs, derivs] = arrayfun(@(xx)feval(stfunc,lambda,sigvar,xx), delta_t, 'UniformOutput', false);
derivs_all(1) = {deriv};
derivs_all(2:length(derivs)+1) = derivs;
derivs = derivs_all;
[ecdll, decdll] = gpr_ssm_mstep(loghyper, x, y, Ex, Vx, Exx, V0, Phis, Qs, derivs);