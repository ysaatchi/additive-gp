function lml = gpc_additive_la_wrap(logtheta, stfunc, likfunc, X, y, Xstar, numNewton)

D = size(X,2);

[p_hat, p_hat_star, F, Ftest, nlml] = gpc_additive_la(stfunc, likfunc, logtheta, X, y, Xstar, numNewton);

alpha.mu_ell = 0;
alpha.mu_sf = 0;
alpha.std_ell = 2;
alpha.std_sf = 2;
for d = 1:D
    nlml = nlml + log(alpha.std_ell) + ((logtheta(d) - alpha.mu_ell)^2)/(2*alpha.std_ell^2);
end
for d = D+1:(2*D)
    nlml = nlml + log(alpha.std_sf) + ((logtheta(d) - alpha.mu_sf)^2)/(2*alpha.std_sf^2);
end

lml = -nlml;