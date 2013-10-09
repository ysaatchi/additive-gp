function [Phi, Q, deriv] = st_matern3(lambda, sigvar, varargin)

%matern3 : case where nu = 3/2; p = 1

if length(varargin) == 1
    delta = varargin{1};
elseif length(varargin) == 3
    w = varargin{1};
    x = varargin{2};
    xprev = varargin{3};
    delta = w'*(x-xprev);
else
    error('Invalid call to st_matern3, quitting...');
end

p = 1;
lp_fact = sum(log(1:p));
l2p_fact = sum(log(1:(2*p)));
logq = log(2*sigvar) + p*log(4) + 2*lp_fact + (2*p + 1)*log(lambda) - l2p_fact;
q = exp(logq);
dq = (q*(2*p + 1))/lambda;

if delta < 0
    %initial state distribution
    Phi = [0;0];
    Q = [sigvar 0; 0 lambda^2*sigvar];
    if nargout > 2
        deriv.dVdSigvar = Q./sigvar;
        deriv.dVdlambda = [0 0; 0 2*lambda*sigvar];
    end
else
    
    Phi = [exp(-lambda*delta)*(1 + lambda*delta),  delta*exp(-lambda*delta);
        -(lambda^2)*delta*exp(-lambda*delta),    exp(-lambda*delta)*(1 - lambda*delta)];
    
    if nargout > 2
        
        dPhidlambda = [delta/exp(lambda*delta) - (delta*(lambda*delta + 1))/exp(lambda*delta), -delta^2/exp(lambda*delta);
            (lambda^2*delta^2)/exp(lambda*delta) - (2*lambda*delta)/exp(lambda*delta), (delta*(lambda*delta - 1))/exp(lambda*delta) - delta/exp(lambda*delta)];
        
    end
    
    Q = zeros(size(Phi));
    Q(1,1) = 1/(4*lambda^3) - (4*delta^2*lambda^2 + 4*delta*lambda + 2)/(8*lambda^3*exp(2*delta*lambda));
    Q(1,2) = delta^2/(2*exp(2*delta*lambda));
    Q(2,1) = Q(1,2);
    Q(2,2) = 1/(4*lambda) - (2*delta^2*lambda^2 - 2*delta*lambda + 1)/(4*lambda*exp(2*delta*lambda));
    
    Q = q*Q;
    
    if nargout > 2
    
        dQdlambda = zeros(size(Q));
        dQdlambda(1,1) = (3*(4*delta^2*lambda^2 + 4*delta*lambda + 2))/(8*lambda^4*exp(2*delta*lambda)) - 3/(4*lambda^4)...
            - (8*lambda*delta^2 + 4*delta)/(8*lambda^3*exp(2*delta*lambda)) + (delta*(4*delta^2*lambda^2 + 4*delta*lambda + 2))/(4*lambda^3*exp(2*delta*lambda));
        dQdlambda(1,2) = -delta^3/exp(2*delta*lambda);
        dQdlambda(2,1) = dQdlambda(1,2);
        dQdlambda(2,2) = (2*delta^2*lambda^2 - 2*delta*lambda + 1)/(4*lambda^2*exp(2*delta*lambda)) - 1/(4*lambda^2) + ...
            (2*delta - 4*delta^2*lambda)/(4*lambda*exp(2*delta*lambda)) + (delta*(2*delta^2*lambda^2 - 2*delta*lambda + 1))/(2*lambda*exp(2*delta*lambda));
        
        dQdlambda = q*dQdlambda + dq*(Q./q);
        dQdSigvar = Q ./ sigvar;
        
        deriv.dPhidlambda = dPhidlambda;
        deriv.dQdlambda = dQdlambda;
        deriv.dQdSigvar = dQdSigvar;
        
        if nargin == 5 %compute derivatives w.r.t. w for PPR
            
            dPhiddelta = [lambda/exp(lambda*delta) - (lambda*(lambda*delta + 1))/exp(lambda*delta), 1/exp(lambda*delta) - (lambda*delta)/exp(lambda*delta);
                (lambda^3*delta)/exp(lambda*delta) - lambda^2/exp(lambda*delta), (lambda*(lambda*delta - 1))/exp(lambda*delta) - lambda/exp(lambda*delta)];
            
            dQddelta = zeros(size(Q));
            dQddelta(1,1) = (4*delta^2*lambda^2 + 4*delta*lambda + 2)/(4*lambda^2*exp(2*delta*lambda)) - (8*delta*lambda^2 + 4*lambda)/(8*lambda^3*exp(2*delta*lambda));
            dQddelta(1,2) = delta/exp(2*delta*lambda) - (delta^2*lambda)/exp(2*delta*lambda);
            dQddelta(2,1) = dQddelta(1,2);
            dQddelta(2,2) = (2*delta^2*lambda^2 - 2*delta*lambda + 1)/(2*exp(2*delta*lambda)) + (2*lambda - 4*delta*lambda^2)/(4*lambda*exp(2*delta*lambda));
        
            dQddelta = q*dQddelta;
            
            dPhidW = zeros(2,2,length(w));
            dQdW = dPhidW;
            for d = 1:length(w)
                dPhidW(:,:,d) = dPhiddelta*(x(d)-xprev(d));
                dQdW(:,:,d) = dQddelta*(x(d)-xprev(d));
            end
            
            deriv.dPhidW = dPhidW;
            deriv.dQdW = dQdW;
            
        end
            
        
    end
    
end

