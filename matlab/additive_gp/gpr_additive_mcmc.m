function [sample_logthetas, S] = gpr_additive_mcmc(stfunc, alpha, gamma, X, y, Xstar, ystar, numSamples)

%stfunc: state transition function
%alpha & gamma : priors for GP hypers
%X : N*D matrix of inputs
%y : N*1 matrix of targets (assumed to have zero mean)
%numSamples : number of MCMC samples

N = size(X, 1);
assert(length(y) == N);
assert(size(X,2) == size(Xstar,2));
[M, D] = size(Xstar);

sample_logthetas = zeros(2*D+1, numSamples);

F = zeros(N,D);
Ftest = zeros(M,D);
ds = 1:D;
     
for n = 1:numSamples
    
    fprintf('Sampling iteration %d...\n', n);
    
    %%% HYPERPARAMETER (RE)SAMPLING
    if n == 1
        log_noise_var = log(0.1);
        log_sigvars = zeros(D,1);
        log_lambdas = zeros(D,1);
    else
        %first sample noise
        noise_samples = y - sum(F_sample,2);
        log_noise_var = -log(gamma_rnd(gamma.a_noise + N/2, gamma.b_noise + 0.5*(sum(noise_samples.^2))));
        fprintf('noise_var = %5.5f\n', exp(log_noise_var));
        
        %sample others
        for d = ds
            
            td = ssm_data(d).t;
            yd = y - sum(F_sample(:, ds(ds ~= d)), 2);
            yd = yd(ssm_data(d).sort_idx_train);
            x_init = ssm_data(d).loghyper(1:2);
            x_sample = mh_sampler('gp_lik_wrap', x_init, 0.1, 5, {stfunc, td, yd, log_noise_var, alpha}); 
            log_lambdas(d) = x_sample(1);
            log_sigvars(d) = x_sample(2);
            fprintf('ell = %5.5f \t sigvar = %5.5f \n', sqrt(nu)/exp(log_lambdas(d)), exp(log_sigvars(d)));
            
        end
    end
    
    %cache data before run -- essential!
    for d = ds
        lh = [log_lambdas(d); log_sigvars(d); log_noise_var];
        lambda = exp(lh(1));
        sigvar = exp(lh(2));
        [t, sort_idx_train] = sort(X(:,d));
        [tstar, sort_idx_test] = sort(Xstar(:,d));
        [tt, sort_idx] = sort([t; tstar]);
        is_train = (sort_idx < (length(t)+1));
        [zz, V0] = feval(stfunc, lambda, sigvar, -1);
        p = size(V0,1); %ssm latent dim
        nu = 2*p - 1;
        delta_t = diff(tt);
        
        %[Phis, Qs] = arrayfun(@(x)feval(stfunc,lambda,sigvar,x), delta_t, 'UniformOutput', false);
        Phis = {};
        Qs = {};
        parfor x = 1:length(delta_t)
            [Phis_i, Qs_i] = feval(stfunc,lambda,sigvar,delta_t(x));
            Phis{x} = Phis_i;
            Qs{x} = Qs_i;
        end
        
        ssm_data(d).loghyper = lh;
        ssm_data(d).t = t;
        ssm_data(d).tstar = tstar;
        ssm_data(d).V0 = V0;
        ssm_data(d).Phis = Phis;
        ssm_data(d).Qs = Qs;
        ssm_data(d).sort_idx_train = sort_idx_train;
        ssm_data(d).sort_idx_test = sort_idx_test;
        ssm_data(d).is_train = is_train;
        fprintf('ell = %5.5f \t sigvar = %5.5f \t noise = %5.5f \n', sqrt(nu)/exp(lh(1)), exp(lh(2)), exp(lh(3)));
    end
    
    %First run backfitting so that F converges to posterior mean
    numGaussSeidel = 20;
    for i = 1:numGaussSeidel
        Fprev = F;
        for d = ds
            yd = y - sum(F(:, ds(ds ~= d)), 2);
            [nlml, Ex, Vx, Exx] = ...
                gpr_ssm_fb(ssm_data(d).loghyper, ssm_data(d).t, yd(ssm_data(d).sort_idx_train), ssm_data(d).tstar, ssm_data(d).V0, ssm_data(d).Phis, ssm_data(d).Qs);
            %read out the training and test means
            Ex_train = Ex(ssm_data(d).is_train);
            train_means = cell2mat(Ex_train);
            train_means = train_means(1:p:end);
            train_means(ssm_data(d).sort_idx_train) = train_means;
            Ex_test = Ex(~ssm_data(d).is_train);
            test_means = cell2mat(Ex_test);
            test_means = test_means(1:p:end);
            test_means(ssm_data(d).sort_idx_test) = test_means;
            F(:, d) = train_means;
            Ftest(:, d) = test_means;
        end
        delta = sum((Fprev(:) - F(:)).^2);
        fprintf('delta F = %5.8f\n', delta);
        if delta < 1e-3
            break;
        end
    end
 
    %Now generate samples from the posterior over F in F_sample 
    F_sample = F;
    F_sample_test = Ftest;
    N_f = 50;
    F_samples = zeros(N, N_f);
    F_samples_test = zeros(M, N_f);
    for i = 1:N_f
        for d = ds
            yd = y - sum(F_sample(:, ds(ds ~= d)), 2);
            [nlml, train_sample, test_sample] = ...
                gpr_ssm_ffbs(ssm_data(d).loghyper, ssm_data(d).t, yd(ssm_data(d).sort_idx_train), ssm_data(d).tstar, ssm_data(d).V0, ssm_data(d).Phis, ssm_data(d).Qs, 1, round(rand*1e9));
            F_sample(ssm_data(d).sort_idx_train, d) = train_sample';
            F_sample_test(ssm_data(d).sort_idx_test, d) = test_sample';
            %figure(d); plot(ssm_data(d).t, yd(ssm_data(d).sort_idx_train), '.r'); hold on; plot(ssm_data(d).t, train_sample, '.k'); axis tight; pause(1); hold off;
        end
        F_samples(:, i) = sum(F_sample, 2);
        F_samples_test(:, i) = sum(F_sample_test, 2);
    end
    
    %record performance and hypers
    means = mean(F_samples, 2);
    vars = var(F_samples, 0, 2) + exp(log_noise_var);
    mnlp_train = 0.5*(log(2*pi) + mean(log(vars)) + mean(((y - means).^2)./vars));
    nmse_train = mean(((y - means).^2))/mean(y.^2);
    fprintf('Training NMSE (additive-mcmc) = %3.5f\n', nmse_train);
    fprintf('Training MNLP (additive-mcmc) = %3.5f\n', mnlp_train);
    
    means = mean(F_samples_test, 2);
    vars = var(F_samples_test, 0, 2) + exp(log_noise_var);
    mnlp_test = 0.5*(log(2*pi) + mean(log(vars)) + mean(((ystar - means).^2)./vars));
    nmse_test = mean(((ystar - means).^2))/mean(ystar.^2);
    fprintf('Testing NMSE (additive-mcmc) = %3.5f\n', nmse_test);
    fprintf('Testing MNLP (additive-mcmc) = %3.5f\n', mnlp_test);
    
    log_ells = 0.5*log(nu) - log_lambdas;
    log_sigfs = 0.5*log_sigvars;
    sample_logthetas(1:(2*D), n) = [log_ells; log_sigfs];
    sample_logthetas(end, n) = 0.5*log_noise_var;
    
    S(n).mnlp_train_mcmc = mnlp_train;
    S(n).nmse_train_mcmc = nmse_train;
    S(n).mnlp_test_mcmc = mnlp_test;
    S(n).nmse_test_mcmc = nmse_test;
    %S(n).F = F;
    %S(n).Ftest = Ftest;
    %S(n).F_samples = F_samples;
    %S(n).F_samples_test = F_samples_test;
 
end
