function [nlml, train_pred, test_pred, E, nsse, nlogdet] = gpr_ssm_diag(loghyper, stfunc, noise, x, y, xstar)

assert(length(loghyper) == 2);
assert(size(x,2) == 1);
assert(size(xstar,2) == 1); %TODO also add support for empty xstar

lambda = exp(loghyper(1)); %cov "decay" parameter 
signal_var = exp(loghyper(2));

[t, sort_idx] = sort([x; xstar]);
is_train = (sort_idx < (length(x)+1));  
y = y(sort_idx(is_train));

[mu0, V0] = feval(stfunc, lambda, signal_var, -1); %prior mean and cov of latent state
D = length(mu0); %D is unlikely to be more than 5 or 6
H = zeros(1,D);
H(1) = 1;

T = length(t);
nlml = 0; nsse = 0; nlogdet = 0;
j = 1;
%absorb first observation
if (is_train(1))
    pm = H*mu0;
    pv = H*V0*H' + noise(1);
    
    nlml = 0.5*(log(2*pi) + log(pv) + ((y(j) - pm)^2)/pv);
    nsse = nsse +  ((y(j) - pm)^2)/pv;
    nlogdet = nlogdet + log(pv);
    
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
fprintf('Number of coincident inputs = %i\n', sum(delta_t == 0));
for i = 2:T
    if delta_t(i-1) == 0
        Phi = eye(D);
        P = V;
        PhiMu = mu;
    else
        [Phi, Q] = feval(stfunc, lambda, signal_var, delta_t(i-1));
        P = Phi*V*Phi' + Q;
        PhiMu = Phi*mu;
    end
    if (is_train(i))
        pm = H*PhiMu;
        pv = H*P*H' + noise(j);
        nlml_i = 0.5*(log(2*pi) + log(pv) + ((y(j) - pm)^2)/pv);
        
        nlml = nlml + nlml_i;
        nsse = nsse + ((y(j) - pm)^2)/pv;
        nlogdet = nlogdet + log(pv);
    
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

%BACKWARD SMOOTHING
mu_s = filter(T).mu;
V_s = filter(T).V;
posterior_mean(T) = H*mu_s;
posterior_var(T) = H*V_s*H';
E(T).mu = mu_s;
E(T).V = V_s;
for i = (T-1):-1:1
    mu = filter(i).mu;
    V = filter(i).V;
    Phi = filter(i).Phi;
    P = filter(i).P;
    P = P + 1e-8*eye(size(P));
    L = V*Phi'*(P\eye(D));
    mu_s = mu + L*(mu_s - Phi*mu);
    V_s = V + L*(V_s - P)*L';
    posterior_mean(i) = H*mu_s;
    posterior_var(i) = H*V_s*H';
    E(i).mu = mu_s;
    E(i).V = V_s;
    E(i).L = L;
end
all_pred(sort_idx,1) = posterior_mean;
all_pred(sort_idx,2) = posterior_var;
train_pred = all_pred(1:length(x),:);
test_pred = all_pred(length(x)+1:end,:);


    


