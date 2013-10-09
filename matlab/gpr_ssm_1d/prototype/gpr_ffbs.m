function [nlml, train_sample, test_sample, nsse] = gpr_ffbs(loghyper, stfunc, x, y, xstar)

assert(length(loghyper) == 3);

lambda = exp(loghyper(1)); %cov "decay" parameter 
signal_var = exp(loghyper(2));
noise_var = exp(loghyper(3));

[t, sort_idx] = sort([x; xstar]); assert((size(t,1) > 0) && (size(t,2) == 1));
is_train = (sort_idx < (length(x)+1));  
y = y(sort_idx(is_train));

[mu0, V0] = feval(stfunc, lambda, signal_var, -1); %prior mean and cov of latent state
D = length(mu0); %D is unlikely to be more than 5 or 6
H = zeros(1,D);
H(1) = 1;
R = noise_var;

T = length(t);
nlml = 0; nsse = 0; %nlml = neg log marg lik; nsse = standardized squared error, i.e. -0.5*y'*inv(K)*y
j = 1;
%absorb first observation
if (is_train(1))
    pm = H*mu0;
    pv = H*V0*H' + R;
    nlml = 0.5*(log(2*pi) + log(pv) + ((y(j) - pm)^2)/pv);
    nsse = 0.5*((y(j) - pm)^2)/pv;
    kalman_gain = (V0*H')/pv;
    mu = mu0 + kalman_gain*(y(1) - pm);
    V = (eye(D) - kalman_gain*H)*V0;
    j = j+1;
else
    mu = mu0;
    V = V0;
end
filter(1).mu = mu;
filter(1).V = V;
%FORWARD FILTERING
delta_t = diff(t);
for i = 2:T
    [Phi, Q] = feval(stfunc, lambda, signal_var, delta_t(i-1));
    %min(eig(Q))
    P = Phi*V*Phi' + Q;
    PhiMu = Phi*mu;
    if (is_train(i))
        pm = H*PhiMu;
        pv = H*P*H' + R;
        nlml_i = 0.5*(log(2*pi) + log(pv) + ((y(j) - pm)^2)/pv);
        nlml = nlml + nlml_i;
        nsse = nsse + 0.5*((y(j) - pm)^2)/pv;
        kalman_gain = (P*H')/pv;
        mu = PhiMu + kalman_gain*(y(j) - pm);
        V = (eye(D) - kalman_gain*H)*P;
        j = j+1;
    else
        mu = PhiMu;
        V = P;
    end
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
        %min(eig(Sigma_s))
        Sigma_s = Sigma_s + 1e-6*eye(size(Sigma_s)); %add jitter
        X(:,i) = mrandnorm(mu_s, Sigma_s);
    end
    posterior_sample(mc, sort_idx) = H*X;
end
train_sample = posterior_sample(:, 1:length(x));
test_sample = posterior_sample(:, length(x)+1:end);
    


