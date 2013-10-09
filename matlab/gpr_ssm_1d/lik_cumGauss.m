function [mu, var, logZ] = lik_cumGauss(y, mu_c, var_c)

%proj of product
z = (y*mu_c)/sqrt(1 + var_c);
mu_proj = mu_c + (y*var_c*std_normal(z))/(Phi(z)*sqrt(1 + var_c));
var_proj = var_c - ((var_c^2*std_normal(z))/((1 + var_c)*Phi(z)))*(z + std_normal(z)/Phi(z));
Z_proj = Phi(z);
%divide
var = 1/(1/var_proj - 1/var_c);
mu = var*(mu_proj/var_proj - mu_c/var_c);
logZ = log(Z_proj) + 0.5*(log(2*pi) + log(var_c + var) + ((mu_c - mu)^2)/(var_c + var));

function n = std_normal(x)
n = (1/(sqrt(2*pi)))*exp(-0.5*x*x);

function phi = Phi(x)
phi = 0.5*erfc(-x/sqrt(2));
