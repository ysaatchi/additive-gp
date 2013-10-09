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

Ns = [500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000];
rand('state',18);
randn('state',20);
numRepeat = 3;
ssm_runtimes = zeros(length(Ns),numRepeat);
full_runtimes = zeros(length(Ns),numRepeat);
i = 1;

for N = Ns
   
    fprintf('N = %i...\n', N);
    
    logtheta = [log(1.0); log(2.0); log(1e-5)];
    t = linspace(-5,5,N)';
    t(rand(N,1)<0.3) = [];
    
    f = gpr_ffbs_prior(logtheta, stfunc, t);
    pp = 0.5*erfc(-f./sqrt(2));
    y = zeros(size(pp));
    RR = rand(size(pp));
    y(RR < pp) = 1;
    y(RR >= pp) = -1;
    
    loghyper = 2*logtheta;
    loghyper(1) = -(loghyper(1)/2) + log(sqrt(nu));
    
    tstar = 10*rand(10,1);
    
    for j = 1:numRepeat 
        tic;
        [train_pred, test_pred, posterior, nlZ] = gpc_ssm(loghyper, stfunc, likfunc, t, y, tstar);
        ssm_runtimes(i,j) = toc;
    end
    
    if N < 10000
        
        for j = 1:numRepeat
            tic;
            [p_full, mu_full, s2_full, nlZ_full] = binaryEPGP(logtheta(1:2), 'covSEiso', t, y, tstar);
            full_runtimes(i,j) = toc;
        end
        
    end
    
    i = i+1;
    
end

save(sprintf('~/PhD/src/matlab/additive_gp/gpr_ssm/gpc_ssm_runtimes_%s.mat', datestr(now, 'yyyymmdd')), 'Ns', 'ssm_runtimes', 'full_runtimes'); 

