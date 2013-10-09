function sample = gpr_ffbs_prior(logtheta, stfunc, x)

%map hypers into SSM parameters
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
loghyper = 2*logtheta;
loghyper(1) = -(loghyper(1)/2) + log(sqrt(nu));

[nlml, train_sample, test_sample] = gpr_ffbs(loghyper, stfunc, [], [], x);

sample = test_sample';