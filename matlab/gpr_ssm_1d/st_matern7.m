function [Phi, Q, deriv] = st_matern7(lambda, sigvar, varargin)

%matern7 : case where nu = 7/2; p = 3
%use as surrogate for squared exponential kernel

if length(varargin) == 1
    delta = varargin{1};
elseif length(varargin) == 3
    w = varargin{1};
    x = varargin{2};
    xprev = varargin{3};
    delta = w'*(x-xprev);
else
    error('Invalid call to st_matern7, quitting...');
end

p = 3;
lp_fact = sum(log(1:p));
l2p_fact = sum(log(1:(2*p)));
logq = log(2*sigvar) + p*log(4) + 2*lp_fact + (2*p + 1)*log(lambda) - l2p_fact;
q = exp(logq);
dq = (q*(2*p + 1))/lambda;

if delta < 0 %prior state distribution
    
    Phi = zeros(4,1);
    Q = zeros(p+1);
    Q(1,1) = 5/(32*lambda^7);
    Q(1,3) = -9/(288*lambda^5);
    Q(2,2) = 1/(32*lambda^5);
    Q(2,4) = -9/(288*lambda^3);
    Q(3,3) = 1/(32*lambda^3);
    Q(4,4) = 5/(32*lambda);
    
    for i = 1:(p+1)
        for j = 1:(i-1)
            Q(i,j) = Q(j,i);
        end
    end
    
    Q = q*Q;
    
    if nargout > 2
        
        dVdlambda(1,1) = -35/(32*lambda^8);
        dVdlambda(1,3) = 5/(32*lambda^6);
        dVdlambda(2,2) = -5/(32*lambda^6);
        dVdlambda(2,4) = 3/(32*lambda^4);
        dVdlambda(3,3) = -3/(32*lambda^4);
        dVdlambda(4,4) = -5/(32*lambda^2);
        for i = 1:(p+1)
            for j = 1:(i-1)
                dVdlambda(i,j) = dVdlambda(j,i);
            end
        end
        
        deriv.dVdSigvar = Q./sigvar;
        deriv.dVdlambda = q*dVdlambda + dq*(Q./q);
        
    end
    
                                                                                                                                                                                                                               
else

    Phi = [(lambda^3*delta^3 + 3*lambda^2*delta^2 + 6*lambda*delta + 6)/(6*exp(lambda*delta)),              (delta*(lambda^2*delta^2 + 2*lambda*delta + 2))/(2*exp(lambda*delta)),                                (delta^2*(lambda*delta + 1))/(2*exp(lambda*delta)),                                                delta^3/(6*exp(lambda*delta))
                                            -(lambda^4*delta^3)/(6*exp(lambda*delta)), (- lambda^3*delta^3 + lambda^2*delta^2 + 2*lambda*delta + 2)/(2*exp(lambda*delta)),               (delta*(- lambda^2*delta^2 + 2*lambda*delta + 2))/(2*exp(lambda*delta)),                              -(delta^2*(lambda*delta - 3))/(6*exp(lambda*delta))
                    (lambda^4*delta^2*(lambda*delta - 3))/(6*exp(lambda*delta)),                    (lambda^3*delta^2*(lambda*delta - 4))/(2*exp(lambda*delta)), -((lambda*delta - 1)*(- lambda^2*delta^2 + 4*lambda*delta + 2))/(2*exp(lambda*delta)),                (delta*(lambda^2*delta^2 - 6*lambda*delta + 6))/(6*exp(lambda*delta))
    -(lambda^4*delta*(lambda^2*delta^2 - 6*lambda*delta + 6))/(6*exp(lambda*delta)),    -(lambda^3*delta*(lambda^2*delta^2 - 7*lambda*delta + 8))/(2*exp(lambda*delta)),         -(lambda^2*delta*(lambda*delta - 2)*(lambda*delta - 6))/(2*exp(lambda*delta)), -(lambda^3*delta^3 - 9*lambda^2*delta^2 + 18*lambda*delta - 6)/(6*exp(lambda*delta))];
 
     
    Q = zeros(size(Phi));
    Q(1,1) = 5/(32*lambda^7) - (64*delta^6*lambda^6 + 192*delta^5*lambda^5 + 480*delta^4*lambda^4 + 960*delta^3*lambda^3 + 1440*delta^2*lambda^2 + 1440*delta*lambda + 720)/(4608*lambda^7*exp(2*delta*lambda));
    Q(1,2) = delta^6/(72*exp(2*delta*lambda));
    Q(1,3) = (9/exp(2*delta*lambda) - 9)/(288*lambda^5) + delta^5/(24*exp(2*delta*lambda)) + delta/(16*lambda^4*exp(2*delta*lambda)) - (delta^6*lambda)/(72*exp(2*delta*lambda)) + delta^2/(16*lambda^3*exp(2*delta*lambda)) + delta^3/(24*lambda^2*exp(2*delta*lambda)) + delta^4/(48*lambda*exp(2*delta*lambda));
    Q(1,4) = (delta^4*(delta^2*lambda^2 - 6*delta*lambda + 3))/(72*exp(2*delta*lambda));
    Q(2,2) = 1/(32*lambda^5) - (4*delta^6*lambda^6 - 12*delta^5*lambda^5 + 6*delta^4*lambda^4 + 12*delta^3*lambda^3 + 18*delta^2*lambda^2 + 18*delta*lambda + 9)/(288*lambda^5*exp(2*delta*lambda));
    Q(2,3) = (delta^4*(delta*lambda - 3)^2)/(72*exp(2*delta*lambda));
    Q(2,4) = (9/exp(2*delta*lambda) - 9)/(288*lambda^3) + (5*delta^3)/(24*exp(2*delta*lambda)) + delta/(16*lambda^2*exp(2*delta*lambda)) - (5*delta^4*lambda)/(16*exp(2*delta*lambda)) + delta^2/(16*lambda*exp(2*delta*lambda)) + (delta^5*lambda^2)/(8*exp(2*delta*lambda)) - (delta^6*lambda^3)/(72*exp(2*delta*lambda));
    Q(3,3) = 1/(32*lambda^3) - (4*delta^6*lambda^6 - 36*delta^5*lambda^5 + 102*delta^4*lambda^4 - 84*delta^3*lambda^3 + 18*delta^2*lambda^2 + 18*delta*lambda + 9)/(288*lambda^3*exp(2*delta*lambda));
    Q(3,4) = (delta^2*(delta^2*lambda^2 - 6*delta*lambda + 6)^2)/(72*exp(2*delta*lambda));
    Q(4,4) = 5/(32*lambda) - (4*delta^6*lambda^6 - 60*delta^5*lambda^5 + 318*delta^4*lambda^4 - 708*delta^3*lambda^3 + 666*delta^2*lambda^2 - 198*delta*lambda + 45)/(288*lambda*exp(2*delta*lambda));
    
    for i = 1:(p+1)
        for j = 1:(i-1)
            Q(i,j) = Q(j,i);
        end
    end
    
    Q = q*Q; 
    
    if nargout > 2
        
        dPhidlambda(1,1) = (3*lambda^2*delta^3 + 6*lambda*delta^2 + 6*delta)/(6*exp(lambda*delta)) - (delta*(lambda^3*delta^3 + 3*lambda^2*delta^2 + 6*lambda*delta + 6))/(6*exp(lambda*delta));
        dPhidlambda(1,2) = (delta*(2*lambda*delta^2 + 2*delta))/(2*exp(lambda*delta)) - (delta^2*(lambda^2*delta^2 + 2*lambda*delta + 2))/(2*exp(lambda*delta));
        dPhidlambda(1,3) = delta^3/(2*exp(lambda*delta)) - (delta^3*(lambda*delta + 1))/(2*exp(lambda*delta));
        dPhidlambda(1,4) = -delta^4/(6*exp(lambda*delta));
        dPhidlambda(2,1) = (lambda^4*delta^4)/(6*exp(lambda*delta)) - (2*lambda^3*delta^3)/(3*exp(lambda*delta));
        dPhidlambda(2,2) = (- 3*lambda^2*delta^3 + 2*lambda*delta^2 + 2*delta)/(2*exp(lambda*delta)) - (delta*(- lambda^3*delta^3 + lambda^2*delta^2 + 2*lambda*delta + 2))/(2*exp(lambda*delta));
        dPhidlambda(2,3) = (delta*(2*delta - 2*lambda*delta^2))/(2*exp(lambda*delta)) - (delta^2*(- lambda^2*delta^2 + 2*lambda*delta + 2))/(2*exp(lambda*delta));
        dPhidlambda(2,4) = (delta^3*(lambda*delta - 3))/(6*exp(lambda*delta)) - delta^3/(6*exp(lambda*delta));
        dPhidlambda(3,1) = (lambda^4*delta^3)/(6*exp(lambda*delta)) + (2*lambda^3*delta^2*(lambda*delta - 3))/(3*exp(lambda*delta)) - (lambda^4*delta^3*(lambda*delta - 3))/(6*exp(lambda*delta));
        dPhidlambda(3,2) = (lambda^3*delta^3)/(2*exp(lambda*delta)) + (3*lambda^2*delta^2*(lambda*delta - 4))/(2*exp(lambda*delta)) - (lambda^3*delta^3*(lambda*delta - 4))/(2*exp(lambda*delta));
        dPhidlambda(3,3) = (delta*(lambda*delta - 1)*(- lambda^2*delta^2 + 4*lambda*delta + 2))/(2*exp(lambda*delta)) - ((lambda*delta - 1)*(4*delta - 2*lambda*delta^2))/(2*exp(lambda*delta)) - (delta*(- lambda^2*delta^2 + 4*lambda*delta + 2))/(2*exp(lambda*delta));
        dPhidlambda(3,4) = - (delta*(6*delta - 2*lambda*delta^2))/(6*exp(lambda*delta)) - (delta^2*(lambda^2*delta^2 - 6*lambda*delta + 6))/(6*exp(lambda*delta));
        dPhidlambda(4,1) = (lambda^4*delta*(6*delta - 2*lambda*delta^2))/(6*exp(lambda*delta)) + (lambda^4*delta^2*(lambda^2*delta^2 - 6*lambda*delta + 6))/(6*exp(lambda*delta)) - (2*lambda^3*delta*(lambda^2*delta^2 - 6*lambda*delta + 6))/(3*exp(lambda*delta));
        dPhidlambda(4,2) = (lambda^3*delta*(7*delta - 2*lambda*delta^2))/(2*exp(lambda*delta)) + (lambda^3*delta^2*(lambda^2*delta^2 - 7*lambda*delta + 8))/(2*exp(lambda*delta)) - (3*lambda^2*delta*(lambda^2*delta^2 - 7*lambda*delta + 8))/(2*exp(lambda*delta));
        dPhidlambda(4,3) = (lambda^2*delta^2*(lambda*delta - 2)*(lambda*delta - 6))/(2*exp(lambda*delta)) - (lambda^2*delta^2*(lambda*delta - 6))/(2*exp(lambda*delta)) - (lambda*delta*(lambda*delta - 2)*(lambda*delta - 6))/exp(lambda*delta) - (lambda^2*delta^2*(lambda*delta - 2))/(2*exp(lambda*delta));
        dPhidlambda(4,4) = (delta*(lambda^3*delta^3 - 9*lambda^2*delta^2 + 18*lambda*delta - 6))/(6*exp(lambda*delta)) - (3*lambda^2*delta^3 - 18*lambda*delta^2 + 18*delta)/(6*exp(lambda*delta));
        
        dQdlambda = zeros(size(Q));
        dQdlambda(1,1) = (7*(64*delta^6*lambda^6 + 192*delta^5*lambda^5 + 480*delta^4*lambda^4 + 960*delta^3*lambda^3 + 1440*delta^2*lambda^2 + 1440*delta*lambda + 720))/(4608*lambda^8*exp(2*delta*lambda)) - (384*delta^6*lambda^5 + 960*delta^5*lambda^4 + 1920*delta^4*lambda^3 + 2880*delta^3*lambda^2 + 2880*delta^2*lambda + 1440*delta)/(4608*lambda^7*exp(2*delta*lambda)) - 35/(32*lambda^8) + (delta*(64*delta^6*lambda^6 + 192*delta^5*lambda^5 + 480*delta^4*lambda^4 + 960*delta^3*lambda^3 + 1440*delta^2*lambda^2 + 1440*delta*lambda + 720))/(2304*lambda^7*exp(2*delta*lambda));
        dQdlambda(1,2) = -delta^7/(36*exp(2*delta*lambda));
        dQdlambda(1,3) = ((48*delta^5*lambda^4)/exp(2*delta*lambda) - (48*delta^6*lambda^5)/exp(2*delta*lambda) + (8*delta^7*lambda^6)/exp(2*delta*lambda))/(288*lambda^5) - (5*(9/exp(2*delta*lambda) + (18*delta^2*lambda^2)/exp(2*delta*lambda) + (12*delta^3*lambda^3)/exp(2*delta*lambda) + (6*delta^4*lambda^4)/exp(2*delta*lambda) + (12*delta^5*lambda^5)/exp(2*delta*lambda) - (4*delta^6*lambda^6)/exp(2*delta*lambda) + (18*delta*lambda)/exp(2*delta*lambda) - 9))/(288*lambda^6);
        dQdlambda(1,4) = - (delta^5*(delta^2*lambda^2 - 6*delta*lambda + 3))/(36*exp(2*delta*lambda)) - (delta^4*(6*delta - 2*delta^2*lambda))/(72*exp(2*delta*lambda));
        dQdlambda(2,2) = (5*(4*delta^6*lambda^6 - 12*delta^5*lambda^5 + 6*delta^4*lambda^4 + 12*delta^3*lambda^3 + 18*delta^2*lambda^2 + 18*delta*lambda + 9))/(288*lambda^6*exp(2*delta*lambda)) - (24*delta^6*lambda^5 - 60*delta^5*lambda^4 + 24*delta^4*lambda^3 + 36*delta^3*lambda^2 + 36*delta^2*lambda + 18*delta)/(288*lambda^5*exp(2*delta*lambda)) - 5/(32*lambda^6) + (delta*(4*delta^6*lambda^6 - 12*delta^5*lambda^5 + 6*delta^4*lambda^4 + 12*delta^3*lambda^3 + 18*delta^2*lambda^2 + 18*delta*lambda + 9))/(144*lambda^5*exp(2*delta*lambda));
        dQdlambda(2,3) = (delta^5*(delta*lambda - 3))/(36*exp(2*delta*lambda)) - (delta^5*(delta*lambda - 3)^2)/(36*exp(2*delta*lambda));
        dQdlambda(2,4) = (7*delta^5*lambda)/(8*exp(2*delta*lambda)) - (3*(1/(32*exp(2*delta*lambda)) + (delta^2*lambda^2)/(16*exp(2*delta*lambda)) + (delta*lambda)/(16*exp(2*delta*lambda)) - 1/32))/lambda^4 - (35*delta^4)/(48*exp(2*delta*lambda)) - delta^3/(8*lambda*exp(2*delta*lambda)) - (7*delta^6*lambda^2)/(24*exp(2*delta*lambda)) + (delta^7*lambda^3)/(36*exp(2*delta*lambda));
        dQdlambda(3,3) = (4*delta^6*lambda^6 - 36*delta^5*lambda^5 + 102*delta^4*lambda^4 - 84*delta^3*lambda^3 + 18*delta^2*lambda^2 + 18*delta*lambda + 9)/(96*lambda^4*exp(2*delta*lambda)) - (24*delta^6*lambda^5 - 180*delta^5*lambda^4 + 408*delta^4*lambda^3 - 252*delta^3*lambda^2 + 36*delta^2*lambda + 18*delta)/(288*lambda^3*exp(2*delta*lambda)) - 3/(32*lambda^4) + (delta*(4*delta^6*lambda^6 - 36*delta^5*lambda^5 + 102*delta^4*lambda^4 - 84*delta^3*lambda^3 + 18*delta^2*lambda^2 + 18*delta*lambda + 9))/(144*lambda^3*exp(2*delta*lambda));
        dQdlambda(3,4) = - (delta^3*(delta^2*lambda^2 - 6*delta*lambda + 6)^2)/(36*exp(2*delta*lambda)) - (delta^2*(6*delta - 2*delta^2*lambda)*(delta^2*lambda^2 - 6*delta*lambda + 6))/(36*exp(2*delta*lambda));
        dQdlambda(4,4) = (- 24*delta^6*lambda^5 + 300*delta^5*lambda^4 - 1272*delta^4*lambda^3 + 2124*delta^3*lambda^2 - 1332*delta^2*lambda + 198*delta)/(288*lambda*exp(2*delta*lambda)) - 5/(32*lambda^2) + (4*delta^6*lambda^6 - 60*delta^5*lambda^5 + 318*delta^4*lambda^4 - 708*delta^3*lambda^3 + 666*delta^2*lambda^2 - 198*delta*lambda + 45)/(288*lambda^2*exp(2*delta*lambda)) + (delta*(4*delta^6*lambda^6 - 60*delta^5*lambda^5 + 318*delta^4*lambda^4 - 708*delta^3*lambda^3 + 666*delta^2*lambda^2 - 198*delta*lambda + 45))/(144*lambda*exp(2*delta*lambda));
        
        for i = 1:(p+1)
            for j = 1:(i-1)
                dQdlambda(i,j) = dQdlambda(j,i);
            end
        end
        
        dQdlambda = q*dQdlambda + dq*(Q./q);
        dQdSigvar = Q ./ sigvar;
        
        deriv.dPhidlambda = dPhidlambda;
        deriv.dQdlambda = dQdlambda;
        deriv.dQdSigvar = dQdSigvar;
        
        if nargin == 5 %compute derivatives w.r.t. w for PPR
           
            dPhiddelta(1,1) = (3*lambda^3*delta^2 + 6*lambda^2*delta + 6*lambda)/(6*exp(lambda*delta)) - (lambda*(lambda^3*delta^3 + 3*lambda^2*delta^2 + 6*lambda*delta + 6))/(6*exp(lambda*delta));
            dPhiddelta(1,2) = (lambda^2*delta^2 + 2*lambda*delta + 2)/(2*exp(lambda*delta)) + (delta*(2*delta*lambda^2 + 2*lambda))/(2*exp(lambda*delta)) - (lambda*delta*(lambda^2*delta^2 + 2*lambda*delta + 2))/(2*exp(lambda*delta));
            dPhiddelta(1,3) = (delta*(lambda*delta + 1))/exp(lambda*delta) + (lambda*delta^2)/(2*exp(lambda*delta)) - (lambda*delta^2*(lambda*delta + 1))/(2*exp(lambda*delta));
            dPhiddelta(1,4) = delta^2/(2*exp(lambda*delta)) - (lambda*delta^3)/(6*exp(lambda*delta));
            dPhiddelta(2,1) = (lambda^5*delta^3)/(6*exp(lambda*delta)) - (lambda^4*delta^2)/(2*exp(lambda*delta));
            dPhiddelta(2,2) = (- 3*lambda^3*delta^2 + 2*lambda^2*delta + 2*lambda)/(2*exp(lambda*delta)) - (lambda*(- lambda^3*delta^3 + lambda^2*delta^2 + 2*lambda*delta + 2))/(2*exp(lambda*delta));
            dPhiddelta(2,3) = (- lambda^2*delta^2 + 2*lambda*delta + 2)/(2*exp(lambda*delta)) + (delta*(2*lambda - 2*lambda^2*delta))/(2*exp(lambda*delta)) - (lambda*delta*(- lambda^2*delta^2 + 2*lambda*delta + 2))/(2*exp(lambda*delta));
            dPhiddelta(2,4) = (lambda*delta^2*(lambda*delta - 3))/(6*exp(lambda*delta)) - (lambda*delta^2)/(6*exp(lambda*delta)) - (delta*(lambda*delta - 3))/(3*exp(lambda*delta));
            dPhiddelta(3,1) = (lambda^5*delta^2)/(6*exp(lambda*delta)) - (lambda^5*delta^2*(lambda*delta - 3))/(6*exp(lambda*delta)) + (lambda^4*delta*(lambda*delta - 3))/(3*exp(lambda*delta));
            dPhiddelta(3,2) = (lambda^4*delta^2)/(2*exp(lambda*delta)) - (lambda^4*delta^2*(lambda*delta - 4))/(2*exp(lambda*delta)) + (lambda^3*delta*(lambda*delta - 4))/exp(lambda*delta);
            dPhiddelta(3,3) = (lambda*(lambda*delta - 1)*(- lambda^2*delta^2 + 4*lambda*delta + 2))/(2*exp(lambda*delta)) - ((lambda*delta - 1)*(4*lambda - 2*lambda^2*delta))/(2*exp(lambda*delta)) - (lambda*(- lambda^2*delta^2 + 4*lambda*delta + 2))/(2*exp(lambda*delta));
            dPhiddelta(3,4) = (lambda^2*delta^2 - 6*lambda*delta + 6)/(6*exp(lambda*delta)) - (delta*(6*lambda - 2*lambda^2*delta))/(6*exp(lambda*delta)) - (lambda*delta*(lambda^2*delta^2 - 6*lambda*delta + 6))/(6*exp(lambda*delta));
            dPhiddelta(4,1) = (lambda^4*delta*(6*lambda - 2*lambda^2*delta))/(6*exp(lambda*delta)) - (lambda^4*(lambda^2*delta^2 - 6*lambda*delta + 6))/(6*exp(lambda*delta)) + (lambda^5*delta*(lambda^2*delta^2 - 6*lambda*delta + 6))/(6*exp(lambda*delta));
            dPhiddelta(4,2) = (lambda^3*delta*(7*lambda - 2*lambda^2*delta))/(2*exp(lambda*delta)) - (lambda^3*(lambda^2*delta^2 - 7*lambda*delta + 8))/(2*exp(lambda*delta)) + (lambda^4*delta*(lambda^2*delta^2 - 7*lambda*delta + 8))/(2*exp(lambda*delta));
            dPhiddelta(4,3) = (lambda^3*delta*(lambda*delta - 2)*(lambda*delta - 6))/(2*exp(lambda*delta)) - (lambda^3*delta*(lambda*delta - 2))/(2*exp(lambda*delta)) - (lambda^3*delta*(lambda*delta - 6))/(2*exp(lambda*delta)) - (lambda^2*(lambda*delta - 2)*(lambda*delta - 6))/(2*exp(lambda*delta));
            dPhiddelta(4,4) = (lambda*(lambda^3*delta^3 - 9*lambda^2*delta^2 + 18*lambda*delta - 6))/(6*exp(lambda*delta)) - (3*lambda^3*delta^2 - 18*lambda^2*delta + 18*lambda)/(6*exp(lambda*delta));
            
            dQddelta = zeros(size(Q));
            dQddelta(1,1) = (64*delta^6*lambda^6 + 192*delta^5*lambda^5 + 480*delta^4*lambda^4 + 960*delta^3*lambda^3 + 1440*delta^2*lambda^2 + 1440*delta*lambda + 720)/(2304*lambda^6*exp(2*delta*lambda)) - (384*delta^5*lambda^6 + 960*delta^4*lambda^5 + 1920*delta^3*lambda^4 + 2880*delta^2*lambda^3 + 2880*delta*lambda^2 + 1440*lambda)/(4608*lambda^7*exp(2*delta*lambda));
            dQddelta(1,2) = delta^5/(12*exp(2*delta*lambda)) - (delta^6*lambda)/(36*exp(2*delta*lambda));
            dQddelta(1,3) = ((48*delta^4*lambda^5)/exp(2*delta*lambda) - (48*delta^5*lambda^6)/exp(2*delta*lambda) + (8*delta^6*lambda^7)/exp(2*delta*lambda))/(288*lambda^5);
            dQddelta(1,4) = (delta^3*(delta^2*lambda^2 - 6*delta*lambda + 3))/(18*exp(2*delta*lambda)) - (delta^4*(6*lambda - 2*delta*lambda^2))/(72*exp(2*delta*lambda)) - (delta^4*lambda*(delta^2*lambda^2 - 6*delta*lambda + 3))/(36*exp(2*delta*lambda));
            dQddelta(2,2) = (4*delta^6*lambda^6 - 12*delta^5*lambda^5 + 6*delta^4*lambda^4 + 12*delta^3*lambda^3 + 18*delta^2*lambda^2 + 18*delta*lambda + 9)/(144*lambda^4*exp(2*delta*lambda)) - (24*delta^5*lambda^6 - 60*delta^4*lambda^5 + 24*delta^3*lambda^4 + 36*delta^2*lambda^3 + 36*delta*lambda^2 + 18*lambda)/(288*lambda^5*exp(2*delta*lambda));
            dQddelta(2,3) = (delta^3*(delta*lambda - 3)^2)/(18*exp(2*delta*lambda)) - (delta^4*lambda*(delta*lambda - 3)^2)/(36*exp(2*delta*lambda)) + (delta^4*lambda*(delta*lambda - 3))/(36*exp(2*delta*lambda));
            dQddelta(2,4) = delta^2/(2*exp(2*delta*lambda)) - (5*delta^3*lambda)/(3*exp(2*delta*lambda)) + (5*delta^4*lambda^2)/(4*exp(2*delta*lambda)) - (delta^5*lambda^3)/(3*exp(2*delta*lambda)) + (delta^6*lambda^4)/(36*exp(2*delta*lambda));
            dQddelta(3,3) = (4*delta^6*lambda^6 - 36*delta^5*lambda^5 + 102*delta^4*lambda^4 - 84*delta^3*lambda^3 + 18*delta^2*lambda^2 + 18*delta*lambda + 9)/(144*lambda^2*exp(2*delta*lambda)) - (24*delta^5*lambda^6 - 180*delta^4*lambda^5 + 408*delta^3*lambda^4 - 252*delta^2*lambda^3 + 36*delta*lambda^2 + 18*lambda)/(288*lambda^3*exp(2*delta*lambda));
            dQddelta(3,4) = (delta*(delta^2*lambda^2 - 6*delta*lambda + 6)^2)/(36*exp(2*delta*lambda)) - (delta^2*lambda*(delta^2*lambda^2 - 6*delta*lambda + 6)^2)/(36*exp(2*delta*lambda)) - (delta^2*(6*lambda - 2*delta*lambda^2)*(delta^2*lambda^2 - 6*delta*lambda + 6))/(36*exp(2*delta*lambda));
            dQddelta(4,4) = (4*delta^6*lambda^6 - 60*delta^5*lambda^5 + 318*delta^4*lambda^4 - 708*delta^3*lambda^3 + 666*delta^2*lambda^2 - 198*delta*lambda + 45)/(144*exp(2*delta*lambda)) + (- 24*delta^5*lambda^6 + 300*delta^4*lambda^5 - 1272*delta^3*lambda^4 + 2124*delta^2*lambda^3 - 1332*delta*lambda^2 + 198*lambda)/(288*lambda*exp(2*delta*lambda));
            for i = 1:(p+1)
                for j = 1:(i-1)
                    dQddelta(i,j) = dQddelta(j,i);
                end
            end
            dQddelta = q*dQddelta;

            dPhidW = zeros(4,4,length(w));
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
