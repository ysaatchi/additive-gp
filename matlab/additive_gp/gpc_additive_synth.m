clear;
stfunc = 'st_matern7'; 
likfunc = 'lik_cumGauss';
switch stfunc
    case 'st_exp'
        nu = 1;
    case 'st_matern3'
        nu = 3;
    case 'st_matern7'
        nu = 7;
    otherwise 
        error('Invalid stfunc, quitting...');
end

N = 1000; D = 5;
X = zeros(N-100,D);

logtheta = [log(1.0); log(2.0); log(1e-5)];

f = zeros(N-100,1);
for d = 1:D
    [rr, ri] = sort(rand(N,1));
    xd = linspace(-5,5,N);
    xd = xd(ri(101:end), :);
    X(:,d) = sort(xd);
    f = f + gpr_ffbs_prior(logtheta, stfunc, sort(xd));
end
%f = chol(feval(covfunc{:}, logtheta, t))'*randn(length(t),1);
pp = 1 ./ (1 + exp(-f));
%pp = 0.5*erfc(-f./sqrt(2));
y = zeros(size(pp));
RR = rand(size(pp));
y(RR < pp) = 1;
y(RR >= pp) = 0;