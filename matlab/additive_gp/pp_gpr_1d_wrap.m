%function logp = pp_gpr_1d_wrap(lh, stfunc, t, y, log_noise_var, alpha)

function [nlml, dnlml] = pp_gpr_1d_wrap(phi, stfunc, X, y)

%tic
[N, D] = size(X);

assert(size(phi,1) == D+3);

wgt = phi(1:D); %projection weights
logtheta = phi(D+1:D+3); %scalar GP hyperparameters

%map hypers into SSM parameters
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
loghyper = 2*logtheta;
loghyper(1) = -(loghyper(1)/2) + log(sqrt(nu));

lambda = exp(loghyper(1)); %cov "decay" parameter 
signal_var = exp(loghyper(2));
noise_var = exp(loghyper(3));

x = X*wgt; %project

[t, sort_idx] = sort(x); 
y = y(sort_idx);
X = X(sort_idx, :);

[mu0, V0, deriv0] = feval(stfunc, lambda, signal_var, -1); %prior mean and cov of latent state

% Run covariance function over all projected input
[Phis, Qs, derivs] = arrayfun(@(x)feval(stfunc,lambda,signal_var,wgt, X(x,:)', X(x-1,:)'), 2:length(x), 'UniformOutput', false);
%tic
%[nlml, dnlml_dW, dnlml_dl, dnlml_ds, dnlml_dn] = gpr_pgr_1d_mex([lambda;signal_var;noise_var; log_noise_var], t, y, V0, Phis, Qs, V, deriv);
[nlml, dnlml_dW, dnlml_dl, dnlml_ds, dnlml_dn] = gpr_pgr_1d_mex([lambda;signal_var;noise_var], t, y, V0, Phis, Qs, mu0, deriv0, derivs, wgt);

%toc
dnlml = [dnlml_dW; -lambda*dnlml_dl; 2*signal_var*dnlml_ds; 2*noise_var*dnlml_dn];
