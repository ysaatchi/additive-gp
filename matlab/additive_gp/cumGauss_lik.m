function [pp, nabla, W] = cumGauss_lik(F, y)

f = sum(F,2);
y(y==0) = -1; %dirty hack for now
yf = y.*f;
pp = (1+erf(yf/sqrt(2)))/2;                                    

if nargout>1                             % dlp, derivative of log likelihood
    
    n_p = zeros(size(f));   % safely compute Gaussian over cumulative Gaussian
    ok = yf>-5;                     % normal evaluation for large values of yf
    n_p(ok) = (exp(-yf(ok).^2/2)/sqrt(2*pi))./pp(ok);
    
    bd = yf<-6;                                 % tight upper bound evaluation
    n_p(bd) = sqrt(yf(bd).^2/4+1)-yf(bd)/2;
    
    interp = ~ok & ~bd;            % linearly interpolate between both of them
    tmp = yf(interp);
    lam = -5-yf(interp);
    n_p(interp) = (1-lam).*(exp(-tmp.^2/2)/sqrt(2*pi))./pp(interp) + ...
        lam .*(sqrt(tmp.^2/4+1)-tmp/2);
    
    nabla = y.*n_p;                         % dlp, derivative of log likelihood
    
    W = n_p.^2 + yf.*n_p;
end
