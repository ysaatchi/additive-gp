function [K, KXZ] = covMatern3_additive(loghyper, X, Z,i)

if nargin == 0, K = '2*D'; return; end
if nargin<3, Z = []; end                                   % make sure, z exists

[N, D] = size(X);
assert(length(loghyper) == 2*D);
ell = exp(loghyper(1:D));
sf2 = exp(2*loghyper(D+1:(2*D)));

if (nargin == 2 && nargout == 1)
    
    K = zeros(N);
    for d = 1:D
        x = X(:,d);
        x = sqrt(3)*x/ell(d);
        A = sqrt(sq_dist(x'));
        Kd = sf2(d)*exp(-A).*(1+A);
        K = K + Kd;
    end
    
elseif (nargin == 3 && nargout == 2)
    
    % cross covariances Kxz
    M = size(Z,1);
    KXZ = zeros(N,M);
    K = sum(sf2);
    for d = 1:D
        z = Z(:,d);
        x = X(:,d);
        x = sqrt(3)*x/ell(d);
        z = sqrt(3)*z/ell(d);
        B = sqrt(sq_dist(x',z'));
        B = sf2(d)*exp(-B).*(1+B);
        KXZ = KXZ + B;
    end
    
elseif (nargin == 3 && nargout == 1)
    dg = strcmp(Z,'diag') && numel(Z)>0;        % determine mode
    if dg                                                               % vector kxx
        K = zeros(size(X,1),1);
        K = sum(sf2)*exp(-K/2);
    else
        % cross covariances Kxz
        M = size(Z,1);
        K = zeros(N,M);
        for d = 1:D
            z = Z(:,d);
            x = X(:,d);
            x = sqrt(3)*x/ell(d);
            z = sqrt(3)*z/ell(d);
            B = sqrt(sq_dist(x',z'));
            B = sf2(d)*exp(-B).*(1+B);
            K = K + B;
        end
    end
elseif (nargin > 3 && nargout == 1)
    
    %derivatives
    assert(i >= 1 && i <= 2*D);
    K = zeros(N);
    if i <= D
        for d = 1:D
            if i == d
                x = X(:,i);
                x = sqrt(3)*x/ell(i);
                K = sf2(i)*sq_dist(x').*exp(-sqrt(sq_dist(x')));
            end
        end
    else
        for d = 1:D
            if d == (i - D)
                x = X(:,d);
                x = sqrt(3)*x/ell(d);
                A = sqrt(sq_dist(x'));
                K = 2*sf2(d)*exp(-A).*(1+A);
            end
        end
    end
    
else
    
    error('Invalid call to covMatern3_additive, quitting...');
    
end

