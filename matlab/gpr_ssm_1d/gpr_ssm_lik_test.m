clear;

N = 5000;
covfunc = {'covSum', {'covMatern3iso','covNoise'}};
logtheta = [log(0.3); log(0.25); log(1e-10)];
t = 20*(rand(N,1)-0.5);
%y = chol(feval('covMatern3iso', logtheta(1:2), t))'*randn(N,1);  
y = gpr_ffbs_prior(logtheta, 'st_matern3', t);

loghyper = 2*logtheta;
loghyper(1) = -(loghyper(1)/2) + log(sqrt(3));
loghyper(2) = 0;

[t, sort_idx] = sort(t);
y = y(sort_idx);
delta_t = diff(t);
[zz, V0] = feval('st_matern3', exp(loghyper(1)),exp(loghyper(2)), -1);
[Phis, Qs] = arrayfun(@(x)feval('st_matern3',exp(loghyper(1)),exp(loghyper(2)),x), delta_t, 'UniformOutput', false);
[nlmlf, nssef] = gpr_ssm_lik(loghyper, t, y, V0, Phis, Qs);

exp(-0.5*log(gamma_rnd(1 + N/2, 1 + nssef)))

cf = {'covSum', {'covMatern3iso', 'covNoise'}};
logtheta(2) = 0;
K = feval(cf{:}, logtheta, t);
L = chol(K,'lower');
alpha = L'\(L\y);
nsse = 0.5*y'*alpha;

exp(-0.5*log(gamma_rnd(1 + N/2, 1 + nsse)))