N = 1000;
%rand('state',18);
%randn('state',20);
covfunc = {'covSum', {'covSEiso','covNoise'}};
logtheta = [log(1.0); log(1.0); log(0.1)];
t = 20*(rand(N,1)-0.5);
y = chol(feval(covfunc{:}, logtheta, t))'*randn(N,1);  
%tstar = linspace(-25,25,1001)';
tstar = 1;

cov_test = 'se';
doPlot = true;

switch cov_test
    
    case 'exp'

        loghyper = 2*logtheta;
        loghyper(1) = -loghyper(1);
        [nlml, posterior_sample] = gpr_ffbs(loghyper, 'st_exp', t, y);
        nlmlf = gpr_ffbs_fast(loghyper, 'st_exp', t, y);

        K = exp(loghyper(2))*exp(-exp(loghyper(1))*abs(repmat(t', N, 1) - repmat(t, 1, N))) + exp(loghyper(3))*eye(N);
        L = chol(K)';                        
        alpha = solve_chol(L',y);   
        nlml2 = 0.5*y'*alpha + sum(log(diag(L))) + 0.5*length(t)*log(2*pi);
        
    case 'matern3'
        
        loghyper = 2*logtheta;
        loghyper(1) = -(loghyper(1)/2) + log(sqrt(3));
        %tic;
        [nlml, train_sample, test_sample, nsse] = gpr_ffbs(loghyper, 'st_matern3', t, y, tstar);
        [t, sort_idx] = sort(t);
        y = y(sort_idx);
        delta_t = diff(t);
        [zz, V0] = feval('st_matern3', exp(loghyper(1)),exp(loghyper(2)), -1);
        [Phis, Qs] = arrayfun(@(x)feval('st_matern3',exp(loghyper(1)),exp(loghyper(2)),x), delta_t, 'UniformOutput', false); 
        [nlmlf, nssef] = gpr_ssm_lik(loghyper, t, y, V0, Phis, Qs);
        %toc
        %tic;
        %[nlmlf, fsample, fstar_sample] = gpr_ffbs_fast(loghyper, 'st_matern3', t, y, tstar);
        %toc
        %muf = mean(fstar_sample);
        %varf = var(fstar_sample);
        
        cf = {'covSum', {'covMatern3iso', 'covNoise'}};
        K = feval(cf{:}, logtheta, t);
        L = chol(K,'lower');                        
        alpha = L'\(L\y);   
        nlml2 = 0.5*y'*alpha + sum(log(diag(L))) + 0.5*length(t)*log(2*pi);
        [Kss, Kstar] = feval('covMatern3iso', logtheta, t, tstar);     
  		mu2 = Kstar' * alpha;                                      
  		vv = L\Kstar;
  		var2 = Kss - sum(vv.*vv)';
  		
  		if doPlot
        	figure; plot(t, y, '+');
        	hold on;
        	plot(t, mean(posterior_sample,1), '+k', 'MarkerSize', 2);
        	plot(t, mean(fsample,1), '+r', 'MarkerSize', 2);
        	plot(tstar, mean(fstar_sample,1), '-r', 'MarkerSize', 1.5);
        	plot(tstar, mu2, '-k', 'MarkerSize', 1.5);
        end
        
    case 'matern7' %use as squared exponential!
        
        loghyper = 2*logtheta;
        loghyper(1) = -(loghyper(1)/2) + log(sqrt(7));
        tic;
        nlml = gpr_ffbs(loghyper, 'st_matern7', t, y, tstar) ;
        toc
        %tic;
        %[nlmlf, fsample, fstar_sample] = gpr_ffbs_fast(loghyper, 'st_matern7', t, y, tstar);
        %toc
        %muf = mean(fstar_sample);
        %varf = var(fstar_sample);
        
        %matern-7 covariance
        ell = exp(logtheta(1)); 
        sf2 = exp(2*logtheta(2));
        noise = exp(2*logtheta(3));
        tt = sqrt(7)*t/ell;
        A = sqrt(sq_dist(tt'));
        K = (sf2*exp(-A).*(1 + A + (2/5)*(A.^2) + (1/15)*(A.^3))) + noise*eye(length(t));
        L = qm_cholesky(K);
        alpha = L'\(L\y); 
        nlml2 = 0.5*y'*alpha + sum(log(diag(L))) + 0.5*length(t)*log(2*pi);
        %compare with SE cov
        nlml3 = gpr(logtheta, covfunc, t, y);
        if doPlot
        	figure; plot(t, y, '+');
        	hold on;
        	plot(t, mean(posterior_sample,1), '+k', 'MarkerSize', 2);
        	plot(t, mean(fsample,1), '+r', 'MarkerSize', 2);
        	plot(tstar, mean(fstar_sample,1), '-r', 'MarkerSize', 1.5);
        	%plot(tstar, mu2, '-k', 'MarkerSize', 1.5); TODO add this
        end
        
        
    case 'se'
        
        loghyper = 2*logtheta;
        loghyper(1) = -loghyper(1) + log(0.5);
        tic;
        nlml = gpr_ffbs(loghyper, 'st_se', t, y, tstar);
        toc
        covfunc = {'covSum', {'covSEard', 'covNoise'}};
        K = feval(covfunc{:}, logtheta, t);
        L = chol(K)';                        
        alpha = solve_chol(L',y);   
        nlml2 = 0.5*y'*alpha + sum(log(diag(L))) + 0.5*length(t)*log(2*pi);
        %tic;
        %[nlmlf, fsample, fstar_sample] = gpr_ffbs_fast(loghyper, 'st_se', t, y, tstar);
        %toc
        
    otherwise
        
        error('unrecognized cov_test...');
        
end
