function [nlml, dnlml] = gp_ppr_1d(phi, stfunc, X, y)

%OLD: finite difference version

e = 1e-8;
tphi = phi;
nlml = gp_ppr_core(stfunc, phi, X, y);
dnlml = zeros(size(phi));
for j = 1:length(phi)
   fprintf('.'); if j == length(phi); fprintf('\n'); end  
   tphi(j) = phi(j)+e;                               % perturb a single dimension
   f2 = gp_ppr_core(stfunc, tphi, X, y);
   tphi(j) = phi(j)-e ;
   f1 = gp_ppr_core(stfunc, tphi, X, y);
   tphi(j) = phi(j);                                 % reset it
   dnlml(j) = (f2 - f1)/(2*e);
end

function nlml = gp_ppr_core(stfunc, phi, X, y)

[N,D] = size(X);
assert(size(phi,1) == D+3);

w = phi(1:D);
logtheta = phi(D+1:D+3);

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

lambda = exp(loghyper(1));
sigvar = exp(loghyper(2));
noise_var = exp(loghyper(3));
[zz, V0] = feval(stfunc, lambda, sigvar, -1);
Xw = X*w; %1D 
[tt, sort_idx] = sort(Xw);
delta_t = diff(tt);
[Phis, Qs] = arrayfun(@(x)feval(stfunc,lambda,sigvar,x), delta_t, 'UniformOutput', false);

[nlml, nsse, nlogdet] = gpr_ssm_nlml(loghyper(1:2), noise_var*ones(N,1), Xw, y(sort_idx), V0, Phis, Qs);