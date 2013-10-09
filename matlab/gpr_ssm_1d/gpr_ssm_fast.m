function [nlml, train_pred, test_pred] = gpr_ssm_fast(loghyper, stfunc, t, y, tstar)
%Wrapper for C++ code : sort input locations and pass to C++

lambda = exp(loghyper(1));
sigvar = exp(loghyper(2));

[t, sort_idx_train] = sort(t);
y = y(sort_idx_train);

if nargin == 4 
    
    [zz, V0] = feval(stfunc, lambda, sigvar, -1);
    delta_t = diff(t);
    [Phis, Qs] = arrayfun(@(x)feval(stfunc,lambda,sigvar,x), delta_t, 'UniformOutput', false);
    
    [nlml, Ex, Vx, Exx] = gpr_ssm_estep(loghyper, t, y, V0, Phis, Qs);
    
    D = length(Ex{1});
    train_means = cell2mat(Ex);
    train_means = train_means(1:D:end);
    train_means(sort_idx_train) = train_means;
    train_vars = cell2mat(Vx);
    train_vars = train_vars(1:D:end,1);
    train_vars(sort_idx_train) = train_vars;
    train_pred = [train_means, train_vars];
    
end   

if nargin == 5
    
    [tstar, sort_idx_test] = sort(tstar);
    [tt, sort_idx] = sort([t; tstar]);
    [zz, V0] = feval(stfunc, lambda, sigvar, -1);
    delta_t = diff(tt);
    [Phis, Qs] = arrayfun(@(x)feval(stfunc,lambda,sigvar,x), delta_t, 'UniformOutput', false);
    
    [nlml, Ex, Vx, Exx] = gpr_ssm_fb(loghyper, t, y, tstar, V0, Phis, Qs);
    
    D = length(Ex{1});
    is_train = (sort_idx < (length(t)+1));
    
    Ex_train = Ex(is_train);
    Vx_train = Vx(is_train);
    train_means = cell2mat(Ex_train);
    train_means = train_means(1:D:end);
    train_means(sort_idx_train) = train_means;
    train_vars = cell2mat(Vx_train);
    train_vars = train_vars(1:D:end,1);
    train_vars(sort_idx_train) = train_vars;
    train_pred = [train_means, train_vars];
    
    Ex_test = Ex(~is_train);
    Vx_test = Vx(~is_train);
    test_means = cell2mat(Ex_test);
    test_means = test_means(1:D:end);
    test_means(sort_idx_test) = test_means;
    test_vars = cell2mat(Vx_test);
    test_vars = test_vars(1:D:end,1);
    test_vars(sort_idx_test) = test_vars;
    test_pred = [test_means, test_vars];
    
end



