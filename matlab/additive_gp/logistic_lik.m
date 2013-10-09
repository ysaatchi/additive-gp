function [ll, nabla, W] = logistic_lik(F, y)

if nargin == 1
    
    assert(nargout == 1);
    ff = sum(F,2);
    ll = 1 ./ (1 + exp(-ff));
    
elseif nargin == 2

    uy = unique(y); assert(length(uy) == 2 && uy(1) == 0 && uy(2) == 1);
    ff = sum(F,2);
    pp = 1 ./ (1 + exp(-ff));
    nn = 1 - pp;
    ll = sum(log(pp(y==1))) + sum(log(nn(y==0)));
    if nargout > 1
        nabla = y - pp;
        W = pp .* (1 - pp);
    end

else
     
    error('Invalid call to logistic_lik, quitting...');
    
end
