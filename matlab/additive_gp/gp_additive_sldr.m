function [nll, dnll] = gp_additive_sldr(theta, X, y, d)

% negative log likelihood and its derivatives of the following model
% y = \psi(XC')w + e
% \psi and w are potentially infinite dim 
% X is N*D
% C is d*D
% e is N*1
% y is N*1

% theta consists of d*D elements representing C
% all other hyperparameters are in turn
% log lengthscales (ell)
% log(sqrt(amplitude)) 
% log(noise_std)


[N,D] = size(X);
assert(length(y) == N);

C = reshape(theta(1:d*D), d, D);
loghyper = theta(d*D+1:end); 
assert(length(loghyper) == 2*d+1);
ell = exp(loghyper(1:d));
sigma = exp(loghyper(d+1:2*d));
sigma_noise = exp(loghyper(end));
K_add = zeros(N);
for i = 1:d
    K_add = K_add + sigma(i)^2*exp(-(pwdist(X*C',i).^2)./(2*ell(i)^2));
end
K = K_add + sigma_noise^2*eye(N);

L = chol(K, 'lower');
alpha = L'\(L\y);
nll = sum(log(diag(L))) + 0.5*y'*alpha + (N/2)*log(2*pi);

%derivs w.r.t. C & ell & amp & v
% pre-compute
W = L'\(L\eye(N)) - alpha*alpha';
%dC
dC = zeros(size(C));
for i = 1:d
   for j = 1:D
      Kd = sigma(i)^2*exp(-(pwdist(X*C',i).^2)./(2*ell(i)^2));
      Y = Kd .* -(((1/(ell(i)^2)) * pwdist(X*C', i)) .* pwdist(X, j));
      dC(i,j) = sum(sum(W .* Y))/2;
   end
end

%dell & damp
dell = zeros(size(ell));
damp = dell;
for i = 1:d
    Kd = sigma(i)^2*exp(-(pwdist(X*C',i).^2)./(2*ell(i)^2));
    Y = Kd .* ((1/(ell(i)^2)) * (pwdist(X*C',i).^2));
    dell(i) = sum(sum(W .* Y))/2;
    damp(i) = sum(sum(W .* Kd));
end 

%noise
v = sigma_noise^2;
dv = v*sum(diag(W));

dnll = [dC(:); dell; damp; dv];
