function [train_pred, test_pred, posterior, nlZ] = gpc_ssm(loghyper, stfunc, likfunc, x, y, xstar) %TODO: add xstar

sgns = unique(y);
assert(length(sgns) == 2 && sgns(1) == -1 && sgns(2) == 1);
assert(size(x,1) == size(y,1));
assert(size(x,2) == 1); 
assert(size(xstar,2) <= 1); %xstar can be empty

lambda = exp(loghyper(1)); %cov "decay" parameter 
signal_var = exp(loghyper(2));

[t, sort_idx] = sort([x; xstar]);
is_train = (sort_idx < (length(x)+1));  
y = y(sort_idx(is_train));
delta_t = diff(t);
N = length(t);
D = length(feval(stfunc, lambda, signal_var, -1));

H = zeros(1,D);
H(1) = 1;

iter = 1;
dnlZ = Inf;
maxEPIter = 2;
tol = 1e-2;
while ((dnlZ > (tol*N)) && (iter <= maxEPIter)) %tol nats per observation and less than max no of iterations
    
    nlZ = 0;
    j = 1; %training set counter
    %FILTERING
    [mu0, V0] = feval(stfunc, lambda, signal_var, -1);
    if (is_train(1))
        if iter == 1
            [mus_up(j), vars_up(j)] = feval(likfunc, y(j), mu0(1), V0(1,1));
        end
        pm = mu0(1);
        pv = V0(1,1) + vars_up(j);
        kalman_gain = (V0*H')/pv;
        mu = mu0 + kalman_gain*(mus_up(j) - pm);
        V = (eye(D) - kalman_gain*H)*V0;
        nlZ = nlZ + 0.5*(log(2*pi) + log(pv) + ((mus_up(j) - pm)^2)/pv);
        j = j+1;
    else
        mu = mu0;
        V = V0;
    end
    filter(1).mu = mu;
    filter(1).V = V;
    
    for i = 2:N
        [Phi, Q] = feval(stfunc, lambda, signal_var, delta_t(i-1));
        mu_fwd = Phi*mu;
        V_fwd = Phi*V*Phi' + Q;
        if (is_train(i))
            if iter == 1
                [mus_up(j), vars_up(j)] = feval(likfunc, y(j), mu_fwd(1), V_fwd(1,1));
            end
            pm = mu_fwd(1);
            pv = V_fwd(1,1) + vars_up(j);
            kalman_gain = (V_fwd*H')/pv;
            mu = mu_fwd + kalman_gain*(mus_up(j) - pm);
            V = (eye(D) - kalman_gain*H)*V_fwd;
            nlZ = nlZ + 0.5*(log(2*pi) + log(pv) + ((mus_up(j) - pm)^2)/pv);
            j = j+1;
        else
            mu = mu_fwd;
            V = V_fwd;
        end
        filter(i).mu = mu;
        filter(i).V = V;
        filter(i-1).Phi = Phi;
        filter(i-1).P = V_fwd;
    end
    
    %SMOOTHING
    W = (eye(D) - kalman_gain*H)*(Phi*filter(N-1).V);
    mu_s = filter(end).mu;
    V_s = filter(end).V;
    posterior(N).mu = mu_s;
    posterior(N).V = V_s;
    posterior(N).W = W;
    mus(N) = mu_s(1);
    vars(N) = V_s(1,1);
    for i = (N-1):-1:1
        mu = filter(i).mu;
        V = filter(i).V;
        Phi = filter(i).Phi;
        P = filter(i).P;
        %P = P + 1e-8*eye(size(P));
        CC = chol(P,'lower');
        L = V*Phi'*(CC'\(CC\eye(D)));
        mu_s = mu + L*(mu_s - Phi*mu);
        V_s = V + L*(V_s - P)*L';
        if i < (N-1)
            W = filter(i+1).V*L' + Lplus*(W - filter(i+1).Phi*filter(i+1).V)*L';
        end
        Lplus = L;
        posterior(i).mu = mu_s;
        posterior(i).V = V_s;
        posterior(i).W = W;
        mus(i) = mu_s(1);
        vars(i) = V_s(1,1);
    end
    
    %LIKELIHOOD MESSAGE UPDATE
    j = 1;
    for i = 1:N
        if (is_train(i))
            [mus_up(j), vars_up(j), logZ_up(j)] = feval(likfunc, y(j), posterior(i).mu(1), posterior(i).V(1,1));
            j = j+1;
        end
    end
    nlZ = nlZ - sum(logZ_up);
    if iter > 1
        dnlZ = abs(nlZ - nlZ_prev);
    end
    fprintf('Iteration %i : nlZ = %5.5f, dnlZ = %5.5f\n', iter, nlZ, dnlZ);
    
    nlZ_prev = nlZ;
    iter = iter + 1;
    
end

all_pred(sort_idx,1) = mus';
all_pred(sort_idx,2) = vars';
train_pred = all_pred(1:length(x),:);
test_pred = all_pred(length(x)+1:end,:);




