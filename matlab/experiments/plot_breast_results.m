clear;
close all;

load run_breast_20110807.mat;

XTest = zeros(1000,D);
for d = 1:D
    XTest(:,d) = linspace(min(XTrain(:,d)) - 2, max(XTrain(:,d)) + 2, 1000)';
end

numNewton = 30;
logtheta = [ones(D,1)*log(results.additive_grid.optEll); ones(D,1)*log(results.additive_grid.optSigf)];
[p_hat, pstar, F, Ftest, nlml] = gpc_additive_la('st_matern7', 'logistic_eps_lik', logtheta, XTrain, yTrain, XTest, numNewton);

%[p2, mu2, s2, nlZ] = binaryLaplaceGP(logtheta, 'covMatern3_additive', 'logistic', XTrain, yTrain2, XTest);

for d = 1:D
    subplot(3,3,d);
    Xp = XTrain(yTrain == 1, d);
    Xm = XTrain(yTrain == 0, d);
    [n1, xout1] = hist(Xp, unique(Xp));
    bar(xout1,n1,'g'); grid; hold
    [n2, xout2] = hist(Xm, unique(Xm));
    bar(xout2,n2,'r'); 
    [xd, sort_idx] = sort(XTest(:,d));
    mu = Ftest(sort_idx, d);
    plot(xd, mu, '-k', 'LineWidth', 2);
    xlim([-20 20]); 
    hold off;
end

%getPDF(hf, 'breast-add');
%!mv synth-add.pdf ~/PhD/doc/thesis/Chapter3/figures/