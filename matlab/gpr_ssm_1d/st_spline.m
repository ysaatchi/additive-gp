function [Phi, Q] = st_spline(lambda, signal_var, delta, first_obs)

%no dependence on lambda!

if first_obs
    Phi = [0; 0];
    Q = [delta^3/3 delta^2/2; delta^2/2 delta];
    Q = signal_var*Q;
else
    Phi = [1 delta; 0 1];
    Q = [delta^3/3 delta^2/2; delta^2/2 delta];
    Q = signal_var*Q;
end