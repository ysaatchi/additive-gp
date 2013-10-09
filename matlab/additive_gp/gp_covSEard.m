function [nlml, dnlml, mus, vars] = gp_covSEard(logtheta, x, y, xstar)

assert((nargin == 3 && nargout == 2) || (nargin == 4 && nargout > 2));

covfunc = {'covSum', {'covSEard', 'covNoise'}};
N = size(x,1);
K = feval('covSEard', logtheta(1:end-1), x);
K = K + exp(2*logtheta(end))*eye(N);    
L = chol(K, 'lower');
clear K;
alpha = L'\(L\y);

nlml = 0.5*y'*alpha + sum(log(diag(L))) + 0.5*N*log(2*pi);

dnlml = zeros(size(logtheta));       % set the size of the derivative vector
if nargout == 2
    W = L'\(L\eye(N))-alpha*alpha';                % precompute for convenience
    clear L;
    for i = 1:length(dnlml)
        dnlml(i) = sum(sum(W.*feval(covfunc{:}, logtheta, x,[], i)))/2;
    end
end

if nargin == 4 && nargout > 2
    assert(size(x,2) == size(xstar,2));
    %[Kss, Kstar] = feval('covSEard', logtheta(1:end-1), x, xstar);
    [Kstar] = feval('covSEard', logtheta(1:end-1), x, xstar);
    [Kss] = feval('covSEard', logtheta(1:end-1), xstar, 'diag');
    mus = Kstar'*alpha;
    v = L\Kstar;
    vars = (Kss - sum(v.*v)') + exp(2*logtheta(end));
end