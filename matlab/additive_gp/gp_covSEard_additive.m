function [nlml, dnlml, mus, vars] = gp_covSEard_additive(logtheta, x, y, xstar)

assert((nargin == 3 && nargout == 2) || (nargin == 4 && nargout > 2));

[N, D] = size(x);
assert(length(logtheta) == 2*D+1);

% K = zeros(N);
% for d = 1:D 
%     K = K + feval('covSEard', logtheta([d,D+d]), x(:,d));
% end
% K = K + exp(2*logtheta(end))*eye(N);

K = covSEard_additive(logtheta, x);

L = chol(K, 'lower'); 
clear K;
alpha = L'\(L\y);
nlml = 0.5*y'*alpha + sum(log(diag(L))) + 0.5*N*log(2*pi);

dnlml = zeros(size(logtheta));       
if nargout == 2
    W = L'\(L\eye(N))-alpha*alpha'; clear L;               
    for d = 1:D
        Kd = feval('covSEard', logtheta([d,D+d]), x(:,d));
        dnlml(d) = sum(sum(W.*(Kd.*sq_dist(x(:,d)'/exp(logtheta(d))))))/2;
        dnlml(d+D) = sum(sum(W.*(2*Kd)))/2;
    end
    %dnlml(end) = sum(sum(W.*(2*K_noise)))/2;
    dnlml(end) = sum(diag(W).*exp(2*logtheta(end)));
end

if nargin == 4 && nargout > 2
    assert(size(x,2) == size(xstar,2));
    [M, D] = size(xstar);
    Kstar = zeros(N,M);
    Kss = zeros(M,1);
    for d = 1:D
        %[kssd, kstard] = feval('covSEardOLD', logtheta([d,D+d]), x(:,d), xstar(:,d));
        [kstard] = feval('covSEard', logtheta([d,D+d]), x(:,d), xstar(:,d));
        [kssd] = feval('covSEard', logtheta([d,D+d]), xstar(:,d), 'diag');
        
        Kstar = Kstar + kstard;
        Kss = Kss + kssd;
    end
    mus = Kstar'*alpha;
    v = L\Kstar;
    %vars = (Kss - diag(Kstar'*(L'\(L\Kstar)))) + exp(2*logtheta(end));
    vars = (Kss - sum(v.*v)') + exp(2*logtheta(end));
end