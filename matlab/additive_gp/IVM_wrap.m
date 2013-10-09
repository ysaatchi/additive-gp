function pstar = IVM_wrap(X, y, XTest, options)

uy = unique(y);
assert(length(uy) == 2 && uy(1) == -1 && uy(2) == 1);

% Train the IVM.
model = ivmRun(X, y, options);
% Display the final model.
ivmDisplay(model);
[mu, varSigma] = ivmPosteriorMeanVar(model, XTest);
mu = mu + model.noise.bias;
pstar = 0.5*erfc(-mu/sqrt(2));



