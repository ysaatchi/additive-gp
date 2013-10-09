function [Phi, Q] = st_se(lambda, sigvar, delta)

%squared exponential kernel : lambda = 1/(2*ell^2)

N = 4; %degree of approximation to true infinitely differentiable process
N_fact = prod(1:N);
q = sigvar*N_fact*((4*lambda)^N)*sqrt(pi/lambda);

%compute matrix of vector process
c = zeros(2*N+1,1);
for n = 0:N
    c(2*n + 1) = (N_fact*((-1)^n)*((4*lambda)^(N-n)))/(prod(1:n));
end
c = c(end:-1:1);
r = roots(c);
p = poly(r(r < 0));

F = zeros(N);
F(N,:) = -p(end:-1:2);
for n = 1:N-1
    F(n,n+1) = 1;
end

Q_noise = zeros(N);
Q_noise(end,end) = q;

if delta < 0
    
    %solve the algebraic ricatti equation F*V0 + V0*F' + Q_noise = 0
    V0 = care(F, zeros(N,1), Q_noise);
    Phi = zeros(N,1);
    Q = V0;
    
else
    
    Phi = expm(F*delta);
    %use matrix fraction decomposition to evaluate process covariance
    F2 = zeros(2*N);
    F2(1:N, 1:N) = F;
    F2(N+1:end, 1:N) = 0;
    F2(1:N, N+1:end) = Q_noise;
    F2(N+1:end, N+1:end) = -F';
    CD = expm(F2*delta)*[eye(N); eye(N)];
    C = CD(1:N, :);
    D = CD(N+1:end, :);
    Q = C*(D\eye(N)) - Phi*Phi';
              
end

   