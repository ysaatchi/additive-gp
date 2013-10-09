function [Ub, dUb] = gpr_ssm_mstep(loghyper, stfunc, x, y, E)

lambda = exp(loghyper(1)); %cov "decay" parameter 
signal_var = exp(loghyper(2));
noise_var = exp(loghyper(3));

[t, sort_idx] = sort(x);
y = y(sort_idx);

T = length(t);

[mu0, V0, deriv] = feval(stfunc, lambda, signal_var, -1);

D = length(mu0);

mu = E(1).mu;
V = E(1).V;
E11 = V + mu*mu';
V0_inv = (V0\eye(D));
Ub = log(det(V0)) + trace(V0_inv*E11);
trace(V0_inv*E11);
dUb_lambda = trace(V0_inv*deriv.dVdlambda') - trace((V0_inv*deriv.dVdlambda'*V0_inv)*E11); 
dUb_sigvar = trace(V0_inv*deriv.dVdSigvar') - trace((V0_inv*deriv.dVdSigvar'*V0_inv)*E11); 

delta_t = diff(t);

for i = 2:T
    [Phi, Q, deriv] = feval(stfunc, lambda, signal_var, delta_t(i-1)); %Q = Q + 1e-8*eye(size(Q));
    mu_prev = E(i-1).mu;
    V_prev = E(i-1).V;
    mu = E(i).mu;
    V = E(i).V;
    Eii_prev = V_prev + mu_prev*mu_prev';
    Eadj = E(i-1).W + mu*mu_prev';
    Eii = V + mu*mu';
    CC = chol(Q,'lower');
    Q_inv = CC'\(CC\eye(D));
    Ub = Ub + log(det(Q)) + trace(Q_inv*Eii) - 2*trace(Phi'*Q_inv*Eadj) + trace(Phi'*Q_inv*Phi*Eii_prev);
    A = (deriv.dPhidlambda'*Q_inv - Phi'*(Q_inv*deriv.dQdlambda*Q_inv)); 
    dUb_lambda = dUb_lambda + trace(Q_inv*deriv.dQdlambda') - trace((Q_inv*deriv.dQdlambda'*Q_inv)*Eii) ...
                 - 2*trace(A*Eadj) + trace((Phi'*Q_inv*deriv.dPhidlambda + A*Phi)*Eii_prev);
    A =  - Phi'*(Q_inv*deriv.dQdSigvar*Q_inv);
    dUb_sigvar = dUb_sigvar + trace(Q_inv*deriv.dQdSigvar') - trace((Q_inv*deriv.dQdSigvar'*Q_inv)*Eii)  ...
                 - 2*trace(A*Eadj) + trace((A*Phi)*Eii_prev);
end
dUb_noise = 0;
for i = 1:T
    mu = E(i).mu;
    V = E(i).V;
    Eii = V + mu*mu';
    Ub = Ub + log(noise_var) + ((y(i)^2) - 2*(y(i)*mu(1)) + Eii(1,1))/noise_var;
    %log(noise_var) + ((y(i)^2) - 2*(y(i)*mu(1)) + Eii(1,1))/noise_var;
    dUb_noise = dUb_noise + 1/noise_var - ((y(i)^2) - 2*(y(i)*mu(1)) + Eii(1,1))/(noise_var^2);
end

dUb = [lambda; signal_var; noise_var].*[dUb_lambda; dUb_sigvar; dUb_noise];


