function x = gamma_rnd(a,b)
% Gamma(a) has density function p(x) = x^(a-1)*b^a*exp(-b*x)/gamma(a)
% requires Stats toolbox
x = gamrnd(a,1./b);