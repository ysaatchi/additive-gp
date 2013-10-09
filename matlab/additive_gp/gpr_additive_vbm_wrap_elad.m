function [necdll, dnecdll] = gpr_additive_vbm_wrap_elad(logtheta, stfunc, X, y, ESS)

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
loghypers = 2*logtheta;
loghypers(1:D) = -(loghypers(1:D)/2) + log(sqrt(nu));

necdll = 0;
dnecdll = zeros(size(logtheta));
F = zeros(N,D);
Varsum = zeros(N,D);
ds = 1:D;

%cache data before run -- essential!
for d = ds
    lh = [loghypers(d); loghypers(D+d); loghypers(end)];
    lambda = exp(lh(1));
    sigvar = exp(lh(2));
    [t, sort_idx] = sort(X(:,d));
   % T = length(t);
    [mu0, V0, deriv0] = feval(stfunc, lambda, sigvar, -1);
%     p = size(V0,1); %ssm latent dim
%     nu = 2*p - 1;
%     delta_t = diff(t);
%     %if any(delta_t < 1e-8)
%     %    warning('Very close / coincident inputs in dimension %i\n', d);
%     %end

    p = length(mu0);
    Ex = ESS(d).Ex;
    Vx = ESS(d).Vx;
    Exx = ESS(d).Exx;
%     mu = Ex{1};
%     V = Vx{1};
%     E11 = V + mu*mu';
%     V0_inv = (V0\eye(p));
%     Ub = log(det(V0)) + trace(V0_inv*E11);
%     dUb_lambda = trace(V0_inv*deriv.dVdlambda') - trace((V0_inv*deriv.dVdlambda'*V0_inv)*E11);
%     dUb_sigvar = trace(V0_inv*deriv.dVdSigvar') - trace((V0_inv*deriv.dVdSigvar'*V0_inv)*E11);
     delta_t = diff(t);
    
%     [Phis, Qs, derivs] = arrayfun(@(x)feval(stfunc,lambda,sigvar,x), delta_t, 'UniformOutput', false);
    Phis = {};
    Qs = {};
    derivs={};
    parfor x = 1:length(delta_t)
        [Phis_i, Qs_i,derivs_i] = feval(stfunc,lambda,sigvar,delta_t(x));
        Phis{x} = Phis_i;
        Qs{x} = Qs_i;
        derivs{x}=derivs_i;
    end
    
    derivsCat = {deriv0,derivs{:}};
%     ssm_data(d).loghyper = lh;
%     ssm_data(d).t = t;
%     ssm_data(d).V0 = V0;
%     ssm_data(d).Phis = Phis;
%     ssm_data(d).Qs = Qs;
%     ssm_data(d).sort_idx = sort_idx;
%     fprintf('ell = %5.5f \t sigvar = %5.5f \t noise = %5.5f \n', sqrt(nu)/exp(lh(1)), exp(lh(2)), exp(lh(3)));
    %Ubpn - Ub + Ub_noise
    [Ubpn, dUb] = gpr_ssm_mstep_mex(lh, t, y, Ex, Vx, Exx, V0, Phis, Qs, derivsCat);
    necdll = necdll+Ubpn;
    ldUb_lambda = dUb(1);
    sdUb_sigvar = dUb(2);
    dnecdll([d, D+d]) = [ldUb_lambda; sdUb_sigvar];
    
    %prepare stuff for observation model
    train_means = cell2mat(Ex);
    train_means = train_means(1:p:end);
    F(sort_idx, d) = train_means;
    train_vars = cell2mat(Vx);
    train_vars = train_vars(1:p:end,1);
    Varsum(sort_idx, d) = train_vars;
end

%observation model
noise_var = exp(loghypers(end));
Ub_noise = N*log(noise_var) + sum((y - sum(F,2)).^2 + sum(Varsum,2))/noise_var;
%fprintf('UB_noise = %5.3f\n', Ub_noise);
necdll = necdll + Ub_noise; %EG don't need
dnecdll(end) = N/noise_var - sum((y - sum(F,2)).^2 + sum(Varsum,2))/(noise_var^2);
dnecdll(end) = noise_var*dnecdll(end);
%convert dnecdll to logtheta parameterisation
dnecdll(1:D) = -dnecdll(1:D);
dnecdll(D+1:end) = 2*dnecdll(D+1:end);