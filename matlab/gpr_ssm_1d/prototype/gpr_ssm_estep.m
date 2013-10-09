function [E, nlml] = gpr_ssm_estep(loghyper, stfunc, x, y)

assert(length(loghyper) == 3);
assert(size(x,2) == 1);

lambda = exp(loghyper(1)); %cov "decay" parameter 
signal_var = exp(loghyper(2));
noise_var = exp(loghyper(3));

[t, sort_idx] = sort(x);
y = y(sort_idx);

[mu0, V0] = feval(stfunc, lambda, signal_var, -1); %prior mean and cov of latent state
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
    [Phi, Q] = feval(stfunc, lambda, signal_var, delta_t(i-1));
    P = Phi*V*Phi' + Q;
    PhiMu = Phi*mu;
    pm = H*PhiMu;
    pv = H*P*H' + R;
    nlml_i = 0.5*(log(2*pi) + log(pv) + ((y(i) - pm)^2)/pv);
    nlml = nlml + nlml_i;
    kalman_gain = (P*H')/pv;
    mu = PhiMu + kalman_gain*(y(i) - pm);
    V = (eye(D) - kalman_gain*H)*P;
    filter(i-1).Phi = Phi;
    filter(i-1).P = P;
    filter(i).mu = mu;
    filter(i).V = V;
end
W = (eye(D) - kalman_gain*H)*(Phi*filter(T-1).V);
%BACKWARD SMOOTHING
mu_s = filter(T).mu;
V_s = filter(T).V;
E(T).mu = mu_s;
E(T).V = V_s;
for i = (T-1):-1:1
    mu = filter(i).mu;
    V = filter(i).V;
    Phi = filter(i).Phi;
    P = filter(i).P;
    CC = chol(P,'lower');
    L = V*Phi'*(CC'\(CC\eye(D)));
    mu_s = mu + L*(mu_s - Phi*mu);
    V_s = V + L*(V_s - P)*L';
    if i < (T-1)
        W = filter(i+1).V*L' + E(i+1).L*(W - filter(i+1).Phi*filter(i+1).V)*L';
    end
    E(i).mu = mu_s;
    E(i).V = V_s;
    E(i).L = L;
    E(i).W = W;
end


    


