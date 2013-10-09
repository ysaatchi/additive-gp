clear;
close all;
%rand('state',222);
%randn('state',222);

data_id = 'pumadyn-8nm';

load(sprintf('~/PhD/datasets/%s/Dataset.data', data_id));
Xall = Dataset(:, 1:end-1); %inputs --> all bar last column as standard
%Xall = pca_normalization(Xall);
yall = Dataset(:, end);
Ntrain = 7000;
X = Xall(1:Ntrain, :);
y = yall(1:Ntrain);
Xstar = Xall((Ntrain+1):end, :);
ystar = yall((Ntrain+1):end);
mu_y = mean(y);
y = y - mu_y;
ystar = ystar - mu_y;
[N,D] = size(X);
M = size(Xstar,1);
stfunc = 'st_matern3';

maxNumProj = 20;

%[d dy dh] = jf_checkgrad({'pp_gpr_1d', stfunc, X(1:100,:), y(1:100)}, rand(D+3, 1), 1e-8);

[fstar, phis, nmses] = pp_gpr(stfunc, X, y, Xstar, ystar, maxNumProj);

nmse_test = mean(((ystar - fstar).^2))/mean(ystar.^2);

