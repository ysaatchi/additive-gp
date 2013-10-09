function [nlml, posterior_sample] = gpr_ffbs_spline(loghyper, t, y)

%Forward filtering backward sampling for 1D GP
%inplemented using naive kalman filtering equations for readability
%for fast implementation use C++ code

%because we are in 1D loghyper will usually be of length 3
%i.e., loghyper = log([lambda; signal_var; noise_var])
assert(max(t) < 1 && min(t) > 0);
assert(length(loghyper) == 3);

lambda = exp(loghyper(1)); %cov "decay" parameter 
signal_var = exp(loghyper(2));
noise_var = exp(loghyper(3));

stfunc = 'st_spline';

[t, sort_idx] = sort(t);
y = y(sort_idx);

[mu0, V0] = feval(stfunc, lambda, signal_var, t(1), true); %prior mean and cov of latent state
D = length(mu0); %D is unlikely to be more than 5 or 6
H = zeros(1,D);
H(1) = 1;
R = noise_var;

T = length(t);
%absorb first observation
pm = H*mu0;
pv = H*V0*H' + R;
nlml = 0.5*(log(2*pi) + log(pv) + ((y(1) - pm)^2)/pv);
kalman_gain = (V0*H')/pv;
mu = mu0 + kalman_gain*(y(1) - pm);
V = (eye(D) - kalman_gain*H)*V0;
filter(1).mu = mu;
filter(1).V = V;
%FORWARD FILTERING
delta_t = diff(t);
for i = 2:T
    [Phi, Q] = feval(stfunc, lambda, signal_var, delta_t(i-1), false);
    pm = H*(Phi*mu);
    P = Phi*V*Phi' + Q;
    pv = H*P*H' + R;
    nlml_i = 0.5*(log(2*pi) + log(pv) + ((y(i) - pm)^2)/pv);
    nlml = nlml + nlml_i;
    kalman_gain = (P*H')/pv;
    mu = Phi*mu + kalman_gain*(y(i) - pm);
    V = (eye(D) - kalman_gain*H)*P;
    filter(i-1).Phi = Phi;
    filter(i-1).P = P;
    filter(i).mu = mu;
    filter(i).V = V;
end
%BACKWARD SAMPLING
numMC = 1;
posterior_sample = zeros(numMC, T);
for mc = 1:numMC
    X = zeros(D,T);
    X(:,T) = mrandnorm(filter(T).mu, filter(T).V);
    for i = (T-1):-1:1
        mu = filter(i).mu;
        V = filter(i).V;
        Phi = filter(i).Phi;
        P = filter(i).P;
        L = V*Phi'*(P\eye(D));
        mu_s = mu + L*(X(:,i+1) - Phi*mu);
        Sigma_s = V - L*P*L';
        X(:,i) = mrandnorm(mu_s, Sigma_s);
    end
    posterior_sample(mc, sort_idx) = H*X;
end
    


