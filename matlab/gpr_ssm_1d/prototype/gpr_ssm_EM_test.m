clear;
N = 500;
%rand('state',18);
%randn('state',20);
covfunc = {'covSum', {'covSEiso','covNoise'}};
logtheta = [log(5); log(2.0); log(1e-6)];
t = linspace(-10,10,N)';
t(rand(N,1)<0.3) = [];
y = chol(feval(covfunc{:}, logtheta, t))'*randn(length(t),1);

figure; plot(t, y, '.');

stfunc = 'st_matern7'; nu = 7;

logtheta_init = [log(0.5); log(12.0); log(1e-7)];
loghyper = 2*logtheta_init;
loghyper(1) = -(loghyper(1)/2) + log(sqrt(nu));

%loghyper = gpr_ssm_EM(loghyper, stfunc, t, y);

loghyper = gpr_ssm_EM_fast(loghyper, stfunc, t, y);

logtheta_learned(1) = log(sqrt(nu)) - loghyper(1);
logtheta_learned(2) = loghyper(2)/2;
logtheta_learned(3) = loghyper(3)/2;
