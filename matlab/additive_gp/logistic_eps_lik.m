%y = [];
%epsi = 1e-3;
%for f = -20:0.01:20
%    y = [y; (1-2*epsi)/(1 + exp(-f)) + epsi];
%end

function [ll, nabla, W] = logistic_eps_lik(F, y)

epsilon = 1e-5;

if nargin == 1
    
    assert(nargout == 1);
    ff = sum(F,2);
    k = (1 - 2*epsilon);
    ll = (k./(1 + exp(-ff))) + epsilon;
    
elseif nargin == 2
    
    uy = unique(y); assert(length(uy) == 2 && uy(1) == 0 && uy(2) == 1);
    
    ff = sum(F,2);
    k = (1 - 2*epsilon);
    pp = (k./(1 + exp(-ff))) + epsilon;
    nn = 1 - pp;
    ll = sum(log(pp(y==1))) + sum(log(nn(y==0)));
    if nargout > 1
        nabla = (k*(y.*exp(-ff) - (1-y))) ./ (1 + exp(-ff));
        W = k*exp(-ff) ./ ((1 + exp(-ff)).^2);
    end
    
else
    
    error('Invalid call to logistic_eps_lik, quitting...');
    
end

