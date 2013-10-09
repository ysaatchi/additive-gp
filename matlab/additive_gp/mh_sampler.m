function [sample, val] = mh_sampler(pfunc, x_init, mh_std, num_accept, pfunc_args)

na = 0; ni = 0;
x = x_init;
D = length(x_init);
xnew = true;
while ((na < num_accept) && (ni < num_accept*10))
    fprintf('.');
    xp = x + mh_std*randn(D,1);
    if xnew
        fx = feval(pfunc, x, pfunc_args{:});
    end
    fxp = feval(pfunc, xp, pfunc_args{:});
    if fxp > (fx + log(rand))
        fprintf('Accepted function value = %5.3f\n', fxp);
        x = xp;
        na = na + 1;
        xnew = true;
    else
        xnew = false;
    end
    ni = ni + 1;
end
sample = x;
val = fxp;
if (na < num_accept)
    warning('Acceptance rate is too low, consider decreasing step size...');
end 
fprintf('Acceptance rate = %3.2f\n', na / ni); 

        

