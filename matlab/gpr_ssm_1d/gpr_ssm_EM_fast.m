function loghyper = gpr_ssm_EM_fast(loghyper, stfunc, x, y)

[x, sort_idx_train] = sort(x);
y = y(sort_idx_train);

nlmle = Inf;
enter = true;
while (enter || (nlmle_prev - nlmle) > 0.01*length(y)) %nlml improvement is greater than 100th of a nat per observation
    
    nlmle_prev = nlmle; enter = false;
    
    %E-step
    delta_t = diff(x);
    lambda = exp(loghyper(1));
    sigvar = exp(loghyper(2));
    [zz, V0] = feval(stfunc, lambda, sigvar, -1);
    [Phis, Qs] = arrayfun(@(xx)feval(stfunc,lambda,sigvar,xx), delta_t, 'UniformOutput', false);
    [nlmle, Ex, Vx, Exx] = gpr_ssm_estep(loghyper, x, y, V0, Phis, Qs);
    fprintf('nlml = %5.5f; %5.5f\n', nlmle, nlml);
    
    %M-step
    [dd, dy, dh] = jf_checkgrad({'gpr_ssm_mstep_wrap', stfunc, x, y, Ex, Vx, Exx}, loghyper, 1e-8);
    if dd > 1e-3
        warning(sprintf('derivs are not very accurate, dd = %5.5f!\n', dd)); dd
    end
    loghyper = minimize(loghyper, 'gpr_ssm_mstep_wrap', -30, stfunc, x, y, Ex, Vx, Exx);
    
    fprintf('loghyper(1) = %5.5f \t loghyper(2) = %5.5f \t loghyper(3) = %5.5f\n', loghyper(1), loghyper(2), loghyper(3));
     
    
end