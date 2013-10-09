function [necdll, dnecdll] = gpr_additive_vbm(logtheta, stfunc, X, y, ESS)

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

[N, D] = size(X);
assert(length(y) == N);
assert(length(logtheta) == 2*D+1);

%hyperparameter conversion
loghyper = 2*logtheta;
loghyper(1:D) = -(loghyper(1:D)/2) + log(sqrt(nu));

necdll = 0;
dnecdll = zeros(size(logtheta));
F = zeros(N,D);
Varsum = zeros(N,D);

for d = 1:D

    lambda = exp(loghyper(d)); %cov "decay" parameter
    signal_var = exp(loghyper(d+D));
    
    [t, sort_idx] = sort(X(:,d));
    
    T = length(t);
    
    [mu0, V0, deriv] = feval(stfunc, lambda, signal_var, -1);
    
    p = length(mu0);
    Ex = ESS(d).Ex;
    Vx = ESS(d).Vx;
    Exx = ESS(d).Exx;
    mu = Ex{1};
    V = Vx{1};
    E11 = V + mu*mu';
    V0_inv = (V0\eye(p));
    Ub = log(det(V0)) + trace(V0_inv*E11);
    dUb_lambda = trace(V0_inv*deriv.dVdlambda') - trace((V0_inv*deriv.dVdlambda'*V0_inv)*E11);
    dUb_sigvar = trace(V0_inv*deriv.dVdSigvar') - trace((V0_inv*deriv.dVdSigvar'*V0_inv)*E11);
    
    delta_t = diff(t);
    for i = 2:T
 
        [Phi, Q, deriv] = feval(stfunc, lambda, signal_var, delta_t(i-1)); 
        Q = Q + 1e-4*eye(size(Q));
        mu_prev = Ex{i-1};
        V_prev = Vx{i-1};
        mu = Ex{i};
        V = Vx{i};
        Eii_prev = V_prev + mu_prev*mu_prev';
        Eadj = Exx{i-1} + mu*mu_prev';
        Eii = V + mu*mu';
        CC = chol(Q,'lower');
        Q_inv = CC'\(CC\eye(p));
        Ub = Ub + log(det(Q)) + trace(Q_inv*Eii) - 2*trace(Phi'*Q_inv*Eadj) + trace(Phi'*Q_inv*Phi*Eii_prev);
        A = (deriv.dPhidlambda'*Q_inv - Phi'*(Q_inv*deriv.dQdlambda*Q_inv));
        dUb_lambda = dUb_lambda + trace(Q_inv*deriv.dQdlambda') - trace((Q_inv*deriv.dQdlambda'*Q_inv)*Eii) ...
            - 2*trace(A*Eadj) + trace((Phi'*Q_inv*deriv.dPhidlambda + A*Phi)*Eii_prev);
        A =  - Phi'*(Q_inv*deriv.dQdSigvar*Q_inv);
        dUb_sigvar = dUb_sigvar + trace(Q_inv*deriv.dQdSigvar') - trace((Q_inv*deriv.dQdSigvar'*Q_inv)*Eii)  ...
            - 2*trace(A*Eadj) + trace((A*Phi)*Eii_prev);
        
    end
    %fprintf('UB on dimension %i = %5.3f\n', d, Ub);
    necdll = necdll + Ub;
    dnecdll([d, D+d]) = [lambda*dUb_lambda; signal_var*dUb_sigvar];
    
    %prepare stuff for observation model
    train_means = cell2mat(Ex);
    train_means = train_means(1:p:end);
    F(sort_idx, d) = train_means;
    train_vars = cell2mat(Vx);
    train_vars = train_vars(1:p:end,1);
    Varsum(sort_idx, d) = train_vars;

end

%observation model
noise_var = exp(loghyper(end));
Ub_noise = N*log(noise_var) + sum((y - sum(F,2)).^2 + sum(Varsum,2))/noise_var;
%fprintf('UB_noise = %5.3f\n', Ub_noise);
necdll = necdll + Ub_noise;
dnecdll(end) = N/noise_var - sum((y - sum(F,2)).^2 + sum(Varsum,2))/(noise_var^2);
dnecdll(end) = noise_var*dnecdll(end);
%convert dnecdll to logtheta parameterisation
dnecdll(1:D) = -dnecdll(1:D);
dnecdll(D+1:end) = 2*dnecdll(D+1:end);


