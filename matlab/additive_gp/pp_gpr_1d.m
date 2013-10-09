function [nlml, dnlml] = pp_gpr_1d(phi, stfunc, X, y)

tic
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

[mu0, V0, deriv] = feval(stfunc, lambda, signal_var, -1); %prior mean and cov of latent state
p = length(mu0); %p is unlikely to be more than 5 or 6
H = zeros(1,p);
H(1) = 1;

T = length(t);
%absorb first observation
pm = mu0(1); %a constant (usually zero)
pv = V0(1,1) + noise_var;
dpv_dl = deriv.dVdlambda(1,1);
dpv_ds = deriv.dVdSigvar(1,1);
dpv_dn = 1;
nlml = 0.5*(log(2*pi) + log(pv) + ((y(1) - pm)^2)/pv);
dnlml_dpv = 0.5*(1/pv - ((y(1) - pm)^2)/(pv*pv));
dnlml_dl = dnlml_dpv*dpv_dl;
dnlml_ds = dnlml_dpv*dpv_ds;
dnlml_dn = dnlml_dpv*dpv_dn;

kalman_gain = (V0*H')/pv;
dg_dl = (pv*(deriv.dVdlambda*H') - dpv_dl*(V0*H'))/(pv*pv);
dg_ds = (pv*(deriv.dVdSigvar*H') - dpv_ds*(V0*H'))/(pv*pv);
dg_dn = -(V0*H')/(pv*pv);
mu = mu0 + kalman_gain*(y(1) - pm);
dmu_dl = dg_dl*(y(1) - pm);
dmu_ds = dg_ds*(y(1) - pm);
dmu_dn = dg_dn*(y(1) - pm);
dmu_dW = zeros(p,D);
V = (eye(p) - kalman_gain*H)*V0;
dV_dl = deriv.dVdlambda - ((kalman_gain*H)*deriv.dVdlambda + (dg_dl*H)*V0);
dV_ds = deriv.dVdSigvar - ((kalman_gain*H)*deriv.dVdSigvar + (dg_ds*H)*V0);
dV_dn = - (dg_dn*H)*V0;

dV_dW = zeros(p,p,D);
dP_dW = zeros(p,p,D);
dPhiMu_dW = zeros(p,D);
dg_dW = zeros(p,D);
dnlml_dW = zeros(D,1);

%FORWARD PASS : sufficient to compute marginal likelihood and
%derivatives...
for i = 2:T
    
    [Phi, Q, deriv] = feval(stfunc, lambda, signal_var, wgt, X(i,:)', X(i-1,:)');
    %[Phi, Q, deriv] = feval(stfunc, lambda, signal_var, delta_t(i-1));
    
    P = Phi*V*Phi' + Q;
    dP_dl = Phi*V*deriv.dPhidlambda' + (Phi*dV_dl + deriv.dPhidlambda*V)*Phi' + deriv.dQdlambda;
    dP_ds = Phi*dV_ds*Phi' + deriv.dQdSigvar;
    dP_dn = Phi*dV_dn*Phi';
    for d = 1:D
        dP_dW(:,:,d) = Phi*V*deriv.dPhidW(:,:,d)' + ...
            (Phi*dV_dW(:,:,d) + deriv.dPhidW(:,:,d)*V)*Phi' + deriv.dQdW(:,:,d);
    end
    
    PhiMu = Phi*mu;
    dPhiMu_dl = deriv.dPhidlambda*mu + Phi*dmu_dl;
    dPhiMu_ds = Phi*dmu_ds;
    dPhiMu_dn = Phi*dmu_dn;
    for d = 1:D
        dPhiMu_dW(:,d) = deriv.dPhidW(:,:,d)*mu + Phi*dmu_dW(:,d);
    end
    
    pm = PhiMu(1);
    dpm_dl = dPhiMu_dl(1);
    dpm_ds = dPhiMu_ds(1);
    dpm_dn = dPhiMu_dn(1);
    dpm_dW = dPhiMu_dW(1,:)';
    
    pv = P(1,1) + noise_var;
    dpv_dl = dP_dl(1,1);
    dpv_ds = dP_ds(1,1);
    dpv_dn = dP_dn(1,1) + 1;
    dpv_dW = squeeze(dP_dW(1,1,:));
    
    nlml_i = 0.5*(log(2*pi) + log(pv) + ((y(i) - pm)^2)/pv);
    nlml = nlml + nlml_i;
    dnlml_dpv = 0.5*(1/pv - ((y(i) - pm)^2)/(pv*pv));
    dnlml_dpm = -0.5*(2*(y(i) - pm)/pv);
    dnlml_dl = dnlml_dl + dnlml_dpv*dpv_dl + dnlml_dpm*dpm_dl;
    dnlml_ds = dnlml_ds + dnlml_dpv*dpv_ds + dnlml_dpm*dpm_ds;
    dnlml_dn = dnlml_dn + dnlml_dpv*dpv_dn + dnlml_dpm*dpm_dn;
    dnlml_dW = dnlml_dW + dnlml_dpv*dpv_dW + dnlml_dpm*dpm_dW;
    
    kalman_gain = (P*H')/pv;
    dg_dl = (pv*(dP_dl*H') - dpv_dl*(P*H'))/(pv*pv);
    dg_ds = (pv*(dP_ds*H') - dpv_ds*(P*H'))/(pv*pv);
    dg_dn = (pv*(dP_dn*H') - dpv_dn*(P*H'))/(pv*pv);
    for d = 1:D
        dg_dW(:,d) = (pv*(dP_dW(:,:,d)*H') - dpv_dW(d)*(P*H'))/(pv*pv);
    end
    
    mu = PhiMu + kalman_gain*(y(i) - pm);
    dmu_dl = dPhiMu_dl + dg_dl*(y(i) - pm) - kalman_gain*dpm_dl;
    dmu_ds = dPhiMu_ds + dg_ds*(y(i) - pm) - kalman_gain*dpm_ds;
    dmu_dn = dPhiMu_dn + dg_dn*(y(i) - pm) - kalman_gain*dpm_dn;
    for d = 1:D
        dmu_dW(:,d) = dPhiMu_dW(:,d) + dg_dW(:,d)*(y(i) - pm) - kalman_gain*dpm_dW(d);
    end
    
    V = (eye(p) - kalman_gain*H)*P;
    dV_dl = dP_dl - ((kalman_gain*H)*dP_dl + (dg_dl*H)*P);
    dV_ds = dP_ds - ((kalman_gain*H)*dP_ds + (dg_ds*H)*P);
    dV_dn = dP_dn - ((kalman_gain*H)*dP_dn + (dg_dn*H)*P);
    for d = 1:D
        dV_dW(:,:,d) = dP_dW(:,:,d) - ((kalman_gain*H)*dP_dW(:,:,d) + (dg_dW(:,d)*H)*P);
    end
    
end
toc
dnlml = [dnlml_dW; -lambda*dnlml_dl; 2*signal_var*dnlml_ds; 2*noise_var*dnlml_dn];
    


