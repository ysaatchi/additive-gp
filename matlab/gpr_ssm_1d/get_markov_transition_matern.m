function [Phi, Q, dPhi_dl, dQ_dl, dPhi_dd, dQ_dd] = get_markov_transition_matern(p)

% p gives the number of times the sample functions are differentiable
% (e.g. Matern3 --> p = 1; once differentiable)

lambda = sym('lambda', 'positive');
t = sym('t', 'positive');
syms s;
%generalize this to arbitrary p
for i = 1:p
    M(i,i) = s;
    M(i,i+1) = -1;
end
M(p+1,p+1) = s;
poly_str = char(expand((lambda + 1)^(p+1)));
idx = findstr('+', poly_str);
idx = [0, idx];
for i = 1:length(idx)-1
    N(p+1,i) = sym(poly_str(idx(i)+1:idx(i+1)-1));
end
B = M + N;
expA = ilaplace(inv(B));
Phi = simplify(expA);
Outer = Phi(:,end)*Phi(:,end)';
delta = sym('delta', 'positive');
Q = int(Outer, t, 0, delta);
simplify(Q);
%also return derivs w.r.t. lambda & delta
if nargout > 2
    dQ_dl = diff(Q, lambda);
    dPhi_dl = diff(Phi, lambda);
    dQ_dd = diff(Q, delta);
    dPhi_dd = diff(Phi, t);
end
