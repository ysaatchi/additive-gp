clear;
stfunc = 'st_matern7'; 
likfunc = 'lik_cumGauss';
switch stfunc
    case 'st_exp'
        nu = 1;
    case 'st_matern3'
        nu = 3;
    case 'st_matern7'
        nu = 7;
    otherwise 
        error('Invalid stfunc, quitting...');
end

N = 10000;
rand('state',18);
randn('state',20);

logtheta = [log(1.0); log(2.0); log(1e-5)];
t = linspace(-5,5,N)';
t(rand(N,1)<0.3) = [];

f = gpr_ffbs_prior(logtheta, stfunc, t);
%f = chol(feval(covfunc{:}, logtheta, t))'*randn(length(t),1);
%pp = 1 ./ (1 + exp(-f));
pp = 0.5*erfc(-f./sqrt(2));
y = zeros(size(pp));
RR = rand(size(pp));
y(RR < pp) = 1;
y(RR >= pp) = -1;

loghyper = 2*logtheta;
loghyper(1) = -(loghyper(1)/2) + log(sqrt(nu));

tstar = 10*rand(10,1);
[train_pred, test_pred, posterior, nlZ] = gpc_ssm(loghyper, stfunc, likfunc, t, y, tstar);
%[p_full, mu_full, s2_full, nlZ_full] = binaryEPGP(logtheta(1:2), 'covMatern3iso', t, y, tstar);

mus = train_pred(:,1);
vars = train_pred(:,2);
h1 = figure; hold on;
ff = [mus + 2*sqrt(vars); flipdim(mus-2*sqrt(vars),1)];
fill([t; flipdim(t, 1)], ff, [7 7 7]/8, 'EdgeColor', [7 7 7]/8);
plot(t, mus, '-k'); 
t_plus = t(y==1);
t_minus = t(y==-1);
plot(t_plus, ones(size(t_plus))*(min(mus - 2*sqrt(vars))-0.1), '.g');
plot(t_minus, ones(size(t_minus))*(min(mus - 2*sqrt(vars))-0.1), '.r');
plot(t, f, 'LineWidth', 1.2);
%axis tight;
getPDF(h1, 'gmp-synth');
!mv gmp-synth.pdf ~/PhD/doc/thesis/Chapter1/

%[mus2, vars2] = gpc_ssm2(loghyper, stfunc, likfunc, t, y);

%DEBUG
%y = f + randn(size(f))*0.05;
%[nlml, train_pred, test_pred] = gpr_ssm(loghyper, stfunc, t, y, max(t) + 0.1);
%[mus, vars] = gpc_ssm2(loghyper, stfunc, 'lik_gauss', t, y);

