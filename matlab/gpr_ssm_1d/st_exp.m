function [Phi, Q, deriv] = st_exp(lambda, sigvar, delta)

if delta < 0
    Phi = 0;
    Q = sigvar;
    deriv.dVdSigvar = 1;
    deriv.dVdlambda = 0;
else
    Phi = exp(-lambda*delta);
    Q = sigvar * (1 - exp(-2*lambda*delta));
    deriv.dPhidlambda = -delta*exp(-lambda*delta);
    deriv.dQdlambda = sigvar * (2*delta*exp(-2*lambda*delta));
    deriv.dQdSigvar = (1 - exp(-2*lambda*delta));
end