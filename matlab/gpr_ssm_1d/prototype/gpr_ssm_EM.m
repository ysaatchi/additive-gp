function loghyper = gpr_ssm_EM(loghyper, stfunc, x, y)

[x, sort_idx_train] = sort(x);
y = y(sort_idx_train);

for i = 1:10
    
    [E, nlml] = gpr_ssm_estep(loghyper, stfunc, x, y);
    
    %[dd, dy, dh] = jf_checkgrad({'gpr_ssm_mstep', stfunc, x, y, E}, loghyper, 1e-8);
    %dd
    %[dy dh]

    loghyper = minimize(loghyper, 'gpr_ssm_mstep2', -30, stfunc, x, y, E);
    %fprintf('loghyper(1) = %5.5f \t loghyper(2) = %5.5f \t loghyper(3) = %5.5f\n', loghyper(1), loghyper(2), loghyper(3));
end
