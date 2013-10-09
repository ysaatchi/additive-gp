clear;
N = 2000;
rand('state',18);
randn('state',20);
covfunc = {'covSum', {'covSEiso','covNoise'}};
logtheta = [log(1.0); log(3.0); log(0.05)];
t = linspace(-1,1,N)';
t(rand(N,1)<0.3) = [];
y = chol(feval(covfunc{:}, logtheta, t))'*randn(length(t),1);
tstar = linspace(-10,10,301)';
tstar(abs(tstar) < 1) = [];

stfunc = 'st_matern3'; covfunc = 'covMatern3iso';
doPlot = true;
test_learning = true;

[nlml, train_pred, test_pred, logtheta] = gpr_ssm_main(stfunc, zeros(3,1), t, y, tstar, test_learning);

cf = {'covSum', {covfunc, 'covNoise'}};
K = feval(cf{:}, logtheta, t);
L = chol(K, 'lower');
alpha = L'\(L\y);
nlml2 = 0.5*y'*alpha + sum(log(diag(L))) + 0.5*length(t)*log(2*pi);
[Kss, Kstar] = feval(covfunc, logtheta, t, tstar);
mu2 = Kstar' * alpha;
vv = L\Kstar;
var2 = Kss - sum(vv.*vv)';

if doPlot
    figure; plot(t, y, '+');
    hold on;
    plot(tstar, test_pred(:,1), '-k', 'MarkerSize', 1.5);
    if (max(max(abs(test_pred - [mu2 var2])))) > 1e-3
        warning('Problem with gpr_ssm_fast');
    end
end

