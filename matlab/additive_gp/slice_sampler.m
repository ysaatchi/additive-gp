function S = slice_sampler(pfunc, x_init, w, burn_in, pfunc_args)

D = length(x_init);
assert(size(x_init,1) >= size(x_init,2));
x = x_init;
S = zeros(D, burn_in);

for n = 1:burn_in
    %fprintf('Slice sampler iteration %d...\n', n);
    for d = 1:D
        pstar = feval(pfunc, x, pfunc_args{:});
        u = log(rand()) + pstar;
        %step-out
        r = rand();
        xl = x;
        xr = x;
        xp = x;
        xld = x(d) - r*w;
        xrd = xld + w;
        xl(d) = xld;
        xr(d) = xrd;
        while (feval(pfunc, xl, pfunc_args{:}) > u)
            fprintf('xl = %5.3f\n', xl);
            xld = xld - w;
            xl(d) = xld;
        end
        while (feval(pfunc, xr, pfunc_args{:}) > u)
            fprintf('xr = %5.3f\n', xr);
            xrd = xrd + w;
            xr(d) = xrd;
        end
        while true
            fprintf('.');
            xpd = xld + rand()*(xrd - xld);
            xp(d) = xpd;
            fprintf('xp = %5.3f\n', xp);
            if (feval(pfunc, xp, pfunc_args{:}) > u)
                x = xp;
                break;
            else
                %modify the interval
                if (xpd > x(d))
                    xrd = xpd;
                    xr(d) = xrd;
                else
                    xld = xpd;
                    xl(d) = xld;
                end
            end
        end
    end
    S(:,n) = x;
end
                
        