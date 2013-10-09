function ESS = gpr_additive_vbe(stfunc, logtheta, X, y)

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

[N, D] = size(X);
assert(length(y) == N);
assert(length(logtheta) == 2*D+1);

%hyperparameter conversion
loghypers = 2*logtheta;
loghypers(1:D) = -(loghypers(1:D)/2) + log(sqrt(nu));

ds = 1:D;

%cache data before run -- essential!
for d = ds
    lh = [loghypers(d); loghypers(D+d); loghypers(end)];
    lambda = exp(lh(1));
    sigvar = exp(lh(2));
    [t, sort_idx] = sort(X(:,d));
    [zz, V0] = feval(stfunc, lambda, sigvar, -1);
    p = size(V0,1); %ssm latent dim
    nu = 2*p - 1;
    delta_t = diff(t);
    %if any(delta_t < 1e-8)
    %    warning('Very close / coincident inputs in dimension %i\n', d);
    %end
    
%     [Phis, Qs] = arrayfun(@(x)feval(stfunc,lambda,sigvar,x), delta_t, 'UniformOutput', false);
    Phis = {};
    Qs = {};
    parfor x = 1:length(delta_t)
        [Phis_i, Qs_i] = feval(stfunc,lambda,sigvar,delta_t(x));
        Phis{x} = Phis_i;
        Qs{x} = Qs_i;
    end
    ssm_data(d).loghyper = lh;
    ssm_data(d).t = t;
    ssm_data(d).V0 = V0;
    ssm_data(d).Phis = Phis;
    ssm_data(d).Qs = Qs;
    ssm_data(d).sort_idx = sort_idx;
    fprintf('ell = %5.5f \t sigvar = %5.5f \t noise = %5.5f \n', sqrt(nu)/exp(lh(1)), exp(lh(2)), exp(lh(3)));
end

F = zeros(N,D);
numGaussSeidel = 40;
for i = 1:numGaussSeidel
    Fprev = F;
    for d = ds
        yd = y - sum(F(:, ds(ds ~= d)), 2);
        [nlml, Ex, Vx, Exx] = gpr_ssm_estepEG(ssm_data(d).loghyper, ssm_data(d).t, yd(ssm_data(d).sort_idx), ssm_data(d).V0, ssm_data(d).Phis, ssm_data(d).Qs);
        train_means = cell2mat(Ex);
        train_means = train_means(1:p:end);
        F(ssm_data(d).sort_idx, d) = train_means;
        ESS(d).Ex = Ex;
        ESS(d).Vx = Vx;
        ESS(d).Exx = Exx;
    end
    delta = sum((Fprev(:) - F(:)).^2);
    fprintf('delta F = %5.3f\n', delta); 
    if delta < 1e-3
        break;
    end
end