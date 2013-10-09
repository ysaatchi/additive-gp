function [necdll, dnecdll] = gpr_additive_vbm_wrap(logtheta, stfunc, X, y, ESS)

[N,D] = size(X);
necdll = gpr_additive_vbm(logtheta, stfunc, X, y, ESS);
[d, dy, dh] = jf_checkgrad({'gpr_additive_vbm', stfunc, X, y, ESS}, logtheta, 1e-8);
dnecdll = dh;
