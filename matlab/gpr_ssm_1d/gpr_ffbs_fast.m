function [nlml, fsample, fstar_sample, nsse] = gpr_ffbs_fast(loghyper, stfunc, t, y, tstar)

debug = false;

lambda = exp(loghyper(1));
sigvar = exp(loghyper(2));
[t, sort_idx_train] = sort(t);
y = y(sort_idx_train);
[tstar, sort_idx_test] = sort(tstar);
tt = sort([t; tstar]);
[zz, V0] = feval(stfunc, lambda, sigvar, -1);
delta_t = diff(tt);

[Phis, Qs] = arrayfun(@(x)feval(stfunc,lambda,sigvar,x), delta_t, 'UniformOutput', false); 
[nlml, train_sample, test_sample] = gpr_ssm_ffbs(loghyper, t, y, tstar, V0, Phis, Qs, 1, round(rand*1e9));

if debug
    [nlml, train_sample, test_sample] = gpr_ssm_ffbs(loghyper, t, y, tstar, V0, Phis, Qs, 5000, round(rand*1e9));
    [nlml, train_pred, test_pred] = gpr_ssm_fast(loghyper, stfunc, t, y, tstar);
    figure; hold on;
    pp = 1:length(t);
    pp(rand(length(pp),1) < 0.5) = [];
    for n = 1:length(pp)
        plot(t, train_sample(pp(n),:), '-r');
    end
    plot(t, train_pred(:,1), 'LineWidth', 2);
    plot(t, train_pred(:,1) + 2*sqrt(train_pred(:,2)), 'k--', 'LineWidth', 2);
    plot(t, train_pred(:,1) - 2*sqrt(train_pred(:,2)), 'k--', 'LineWidth', 2);
end

fsample(:, sort_idx_train) = train_sample;
fstar_sample(:, sort_idx_test) = test_sample;

    
