function [Phi, Q] = get_markov_transition_spline(m)

t = sym('t', 'positive');
syms s;
%generalize this to arbitrary p
for i = 1:(m-1)
    A(i,i) = s;
    A(i,i+1) = -1;
end
A(m,m) = s;
expA = ilaplace(inv(A));
Phi = simplify(expA);
Outer = Phi(:,end)*Phi(:,end)';
delta = sym('delta', 'positive');
Q = int(Outer, t, 0, delta);
simplify(Q);
