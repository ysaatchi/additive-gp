function logtheta2 = gpr_additive_vbem(stfunc, logtheta, X, y, numIter)

for i = 1:numIter
%for i = 1:4
    %%% E-step
    ESS = gpr_additive_vbe(stfunc, logtheta, X, y);
    %%% M-step
    %[d, dy, dh] = jf_checkgrad({'gpr_additive_vbm', stfunc, X, y, ESS}, logtheta, 1e-8);
    
    %tic; logtheta = minimize(logtheta, 'gpr_additive_vbm', -20, stfunc,X,y, ESS); toc;
    logtheta = minimize(logtheta, 'gpr_additive_vbm_wrap_elad', -50, stfunc, X, y, ESS);
end
logtheta2 = logtheta;