function sample = mrandnorm(mu, Sigma)

assert(size(mu,1) == size(Sigma,1));
assert(size(mu,1) == size(Sigma,2));

%sample = mu + qm_cholesky(Sigma)*randn(length(mu),1);
sample = mu + chol(Sigma,'lower')*randn(length(mu),1);
