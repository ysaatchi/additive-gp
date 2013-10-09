N = 50;
%rand('state',18);
%randn('state',20);
covfunc = {'covSum', {'covSEiso','covNoise'}};
signal_var = 3.0;
noise_var = 0.3;
logtheta = [log(1.0); log(signal_var)/2; log(noise_var)/2];
t = rand(N,1);
y = chol(feval(covfunc{:}, logtheta, t))'*randn(N,1);

[nlml, posterior_sample] = gpr_ffbs_spline(2*logtheta, t, y);

for i = 1:length(t)
    for j = 1:length(t)
        a = min(t(i),t(j));
        b = max(t(i),t(j));
        K(i,j) = signal_var*((b*a^2)/2 - (a^3)/6);
    end
end
K = K + exp(2*logtheta(end))*eye(N);
L = chol(K)';
alpha = solve_chol(L',y);
nlml2 = 0.5*y'*alpha + sum(log(diag(L))) + 0.5*N*log(2*pi);