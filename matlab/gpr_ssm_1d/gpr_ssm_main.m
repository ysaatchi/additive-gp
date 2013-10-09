function [nlml, train_pred, test_pred, logtheta] = gpr_ssm_main(stfunc, logtheta, x, y, xstar, learn_hypers)

% O(N) GPR for scalar inputs

% INPUTS:
% stfunc := 'st_exp' | 'st_matern3' |'st_matern7' (*)
% logtheta : [log(ell); log(sigma_f); log(sigma_n)]
%   if known, simply set this and turn learning off
%   if not, use this argument to initialize hypers for learning
% x : N*1 vector of input locations
% y : N*1 vector of targets
% xstar : M*1 vector of test input locations
% learn_hypers: flag to turn hyper learning via EM on/off

% OUTPUTS:
% nlml : negative log marginal likelihood of the (learned) model
% train_pred : N*2 matrix of training set predictions
%   (column 1: training means, column2: training variances)
% test_pred : M*2 matrix of test set predictions
%   (column 1: test means, column2: test variances)
% logtheta : if learning is on, then the learned hypers


if (nargin < 5) || (nargin > 6) 
    error('Incorrect number of arguments, quitting');
end

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

if (nargin == 6) && (learn_hypers)
    %Learn hyperparameters using logtheta as the initialization
    loghyper = 2*logtheta;
    loghyper(1) = -(loghyper(1)/2) + log(sqrt(nu));
    
    loghyper = gpr_ssm_EM_fast(loghyper, stfunc, x, y);
    
    logtheta(1) = log(sqrt(nu)) - loghyper(1);
    logtheta(2) = loghyper(2)/2;
    logtheta(3) = loghyper(3)/2;
end

loghyper = 2*logtheta;
loghyper(1) = -(loghyper(1)/2) + log(sqrt(nu));
tic;
[nlml, train_pred, test_pred] = gpr_ssm_fast(loghyper, stfunc, x, y, xstar);
toc


