% Linear classifier
function [pstar, wj] = linear_logistic(X, y, Xstar, numIter)

%X is NxD
%y is Nx1 (1 for positive class, 0 for negative)
if nargin < 2
    error('Invalid call to linear_logistic, quitting...');
end

if nargin < 4
    numIter = 10;
end
X = [ones(size(X,1), 1) X];
w = zeros(size(X,2),1);
wj = zeros(size(X,2),numIter);
for j = 1:numIter
    sigma = 1./(1 + exp(-(X*w)));
    del = X'*(sigma - y);
    H = bsxfun(@times, sigma.*(1-sigma), X)'*X;
    w = w - H \ del;
    wj(:,j) = w;
end
if nargin < 3
    pstar=wj;
    return
end
Xstar = [ones(size(Xstar,1), 1) Xstar];
pstar = 1./(1 + exp(-(Xstar*wj(:,end))));

