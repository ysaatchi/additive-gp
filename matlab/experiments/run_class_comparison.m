% RUN CLASS COMPARISON
%
% for classification comparison between additiveLA and other known
% classification methods
%
% Elad Gilboa, Yunus Saatci 2013
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function results = run_class_comparison(XTrain,yTrain,XTest,yTest,popt)

global folderPath

[N,D] = size(XTrain);
yTrain2 = yTrain;
yTrain2(yTrain==0) = -1;
yTrain2 = reshape(yTrain2,length(yTrain2),1);
yTest2 = yTest;
yTest2(yTest==0) = -1;
yTest2 = reshape(yTest2,length(yTest2),1);

%% LINEAR LOGISTIC
results.linear_logistic = [];
if(~isempty(popt.lin))
    tic
    [pstar_lin, wj] = linear_logistic(XTrain, yTrain,XTest);
    %pstar_lin = 1./(1 + exp(-(XTest*wj(:,end))));
    yPred = zeros(size(yTest));
    yPred(pstar_lin > 0.5) = 1;
    test_error = 1 - sum(yPred==yTest)/size(yTest, 1);
    nll_lin = -mean(yTest.*log(pstar_lin) + (1-yTest).*log(1-pstar_lin));
    exec_time = toc;
    fprintf('TEST ERROR (LINEAR) = %5.5f\n', test_error);
    fprintf('TEST NLL (LINEAR) = %5.5f \n', nll_lin);
    fprintf('RUNTIME (s) = %5.1f \n', exec_time);
    results.linear_logistic.test_error = test_error;
    results.linear_logistic.nll = nll_lin;
    results.linear_logistic.exec_time = exec_time;
end

%% SVM (TENSOR)

% Uncomment if you have libsvn

% results.svm = [];
% if(~isempty(popt.svm))
%     tic
%     %grid search
%     Cs = popt.svm.Cs;
%     gammas = popt.svm.gamms;
%     nlls = zeros(length(Cs), length(gammas));
%     tes = nlls;
%     for i = 1:length(Cs)
%         for j = 1:length(gammas)
%             
%             svm_options = ['-c ', num2str(Cs(i)) ' -g ', num2str(gammas(j)), ' -b 1'];
%             model = svmtrain(yTrain2, XTrain, svm_options); %optionally add SVM options as a third argument
%             [yPred, accuracy, prob_estimates] = svmpredict(yTest2, XTest, model, '-b 1'); %optionally add SVM options as a 4th argument
%             test_error = 1 - accuracy(1)/100;
%             pstar_svm = prob_estimates(:,2);
%             nll_svm = -mean(yTest.*log(pstar_svm) + (1-yTest).*log(1-pstar_svm));
%             nlls(i,j) = nll_svm;
%             tes(i,j) = test_error;
%             
%         end
%     end
%     exec_time = toc;
%     [mini, minj] = find(nlls == min(min(nlls)));
%     fprintf('TEST ERROR (SVM) = %5.5f\n', tes(mini, minj));
%     fprintf('TEST NLL (SVM) = %5.5f \n', nlls(mini, minj));
%     fprintf('RUNTIME (s) = %5.1f \n', exec_time);
%     results.svm.test_error = tes(mini, minj);
%     results.svm.exec_time = exec_time;
%     results.svm.nll = nlls(mini, minj);
%     results.svm.optC = Cs(mini);
%     results.svm.optGamma = gammas(minj);
% end

%% IVM (TENSOR)

% uncomment if you have IVM 

% results.ivm = [];
% if(~isempty(popt.ivm))
%     % IVM Set up
%     options = ivmOptions;
%     results.ivm.numPseudo = popt.ivm.numPseudo;
%     for inumActive = 1:length(popt.ivm.numPseudo)
%         tic;
%         options.numActive = popt.ivm.numPseudo(inumActive);
%         dVal = 500;
%         options.numActive = min([dVal N]);
%         options.kern = {'rbf', 'white'};
%         options.display = 0;
%         pstar_ivm = IVM_wrap(XTrain, yTrain2, XTest, options);
%         yPred = zeros(size(yTest));
%         yPred(pstar_ivm > 0.5) = 1;
%         test_error = 1 - sum(yPred==yTest)/size(yTest, 1);
%         nll_ivm = -mean(yTest.*log(pstar_ivm) + (1-yTest).*log(1-pstar_ivm));
%         exec_time = toc;
%         fprintf('TEST ERROR (IVM) = %5.5f\n', test_error);
%         fprintf('NLL (IVM) = %5.5f \n', nll_ivm);
%         fprintf('RUNTIME (s) = %5.1f \n', exec_time);
%         results.ivm.test_error(inumActive) = test_error;
%         results.ivm.nll(inumActive) = nll_ivm;
%         results.ivm.exec_time(inumActive) = exec_time;
%     end
% end

%% ADDITIVE (approx)
results.addLA = [];
results.additive_mcmc = [];
if(~isempty(popt.addLA))
    if ~popt.addLA.runMCMC
        tic
        %grid search
        ells = popt.addLA.ells;
        sigfs = popt.addLA.sigfs;
        nlls = zeros(length(ells), length(sigfs));
        tes = nlls;
        for i = 1:length(ells)
            for j = 1:length(sigfs)
                
                numNewton = popt.addLA.numNewton;
                numGS = popt.addLA.numGS;
                logtheta = [ones(D,1)*log(ells(i)); ones(D,1)*log(sigfs(j))];
                [p_hat, pstar, F, Ftest, nlml] = gpc_additive_la('st_matern7', 'logistic_eps_lik', logtheta, XTrain, yTrain, XTest, numNewton,numGS);
                yPred = zeros(size(yTest));
                yPred(pstar > 0.5) = 1;
                test_error = 1 - sum(yPred==yTest)/size(yTest, 1);
                nll_fast = -mean(yTest.*log(pstar) + (1-yTest).*log(1-pstar));
                nlls(i,j) = nll_fast;
                tes(i,j) = test_error;
                
            end
        end
        exec_time = toc;
        [mini, minj] = find(nlls == min(min(nlls)));
        fprintf('TEST ERROR (ADDITIVE-FAST) = %5.5f\n', tes(mini, minj));
        fprintf('TEST NLL (ADDITIVE-FAST) = %5.5f \n', nlls(mini, minj));
        fprintf('RUNTIME (s) = %5.1f \n', exec_time);
        results.addLA.test_error = tes(mini, minj);
        results.addLA.exec_time = exec_time;
        results.addLA.nll = nlls(mini, minj);
        results.addLA.optEll = ells(mini);
        results.addLA.optSigf = sigfs(minj);
    else
        %%% ADDITIVE MCMC
        tic
        numMCMC = 20; selects = 10:2:20;
        numNewton = 30;
        mh_std = 0.2;
        [lmls, logthetas, nlmls, p_hat, p_hat_star] = gpc_additive_mcmc('st_matern7', 'logistic_eps_lik', XTrain, yTrain, XTest, numMCMC, numNewton, mh_std);
        pstar = mean(p_hat_star(:, selects), 2);
        yPred = zeros(size(yTest));
        yPred(pstar > 0.5) = 1;
        test_error = 1 - sum(yPred==yTest)/size(yTest, 1);
        nll_fast = -mean(yTest.*log(pstar) + (1-yTest).*log(1-pstar));
        exec_time = toc;
        fprintf('TEST ERROR (ADDITIVE -- MCMC) = %5.5f \n', test_error);
        fprintf('NLL (ADDITIVE -- MCMC) = %5.5f \n', nll_fast);
        fprintf('RUNTIME (s) = %5.1f \n', exec_time);
        results.additive_mcmc.test_error = test_error;
        results.additive_mcmc.nll = nll_fast;
        results.additive_mcmc.exec_time = exec_time;
    end
end

results.addfull = [];
results.fullgp = [];
if N <= 9000
    if(~isempty(popt.addfull))
        %%% FULL GP (ADDITIVE)
        %%[d, dy, dh] = jf_checkgrad({'binaryLaplaceGP', 'covMatern3_additive', 'logistic', XTrain, yTrain2}, ones(2*D,1), 1e-8);
        tic
        logtheta2 = minimize(zeros(2*D,1), 'binaryLaplaceGP', -50, 'covMatern3_additive', 'logistic', XTrain, yTrain2);
        [p2, mu2, s2, nlZ] = binaryLaplaceGP(logtheta2, 'covMatern3_additive', 'logistic', XTrain, yTrain2, XTest);
        yPred = zeros(size(yTest));
        yPred(p2 > 0.5) = 1;
        test_error = 1 - sum(yPred==yTest)/size(yTest, 1);
        nll_full_add = -mean(yTest.*log(p2) + (1-yTest).*log(1-p2));
        exec_time = toc;
        fprintf('TEST ERROR (ADDITIVE -- NAIVE) = %5.5f \n', test_error);
        fprintf('NLL (ADDITIVE -- NAIVE) = %5.5f \n', nll_full_add);
        fprintf('RUNTIME (s) = %5.1f \n', exec_time);
        results.addfull.test_error = test_error;
        results.addfull.nll = nll_full_add;
        results.addfull.exec_time = exec_time;
    end
    if(~isempty(popt.fullgp))
        %%% FULL GP (TENSOR)
        tic
        logtheta2 = minimize(zeros(D+1,1), 'binaryLaplaceGP', -50, 'covSEard', 'logistic', XTrain, yTrain2);
        [p2, mu2, s2, nlZ] = binaryLaplaceGP(logtheta2, 'covSEard', 'logistic', XTrain, yTrain2, XTest);
        yPred = zeros(size(yTest));
        yPred(p2 > 0.5) = 1;
        test_error = 1 - sum(yPred==yTest)/size(yTest, 1);
        nll_full = -mean(yTest.*log(p2) + (1-yTest).*log(1-p2));
        exec_time = toc;
        fprintf('TEST ERROR (FULL GP) = %5.5f \n', test_error);
        fprintf('NLL (FULL GP) = %5.5f \n', nll_full);
        fprintf('RUNTIME (s) = %5.1f \n', exec_time);
        results.fullgp.test_error = test_error;
        results.fullgp.nll = nll_full;
        results.fullgp.exec_time = exec_time;
    end
end

if(~isempty(popt.fitc))
    tic;
    x = XTrain;
    y = yTrain2;
    
    results.fitc.numPseudo = popt.fitc.numPseudo;
    for inumActive = 1:length(popt.fitc.numPseudo)
        % Set the inducing inputs in a random subset
        numPseudo = popt.fitc.numPseudo;
        M = min(N,numPseudo);

        % initialize pseudo-inputs to a random subset of training inputs
        [dum,I] = sort(rand(N,1)); clear dum;
        I = I(1:M);
        Xu = x(I,:);

        [n,nin] = size(x);

        % Create the covariance functions
        pl = prior_t('s2',10);
        pm = prior_sqrtunif();
        gpcf1 = gpcf_matern32('lengthScale', 5, 'magnSigma2', 0.05, 'lengthScale_prior', pl, 'magnSigma2_prior', pm);
        %gpcf2 = gpcf_ppcs3('nin',nin,'lengthScale', 5, 'magnSigma2', 0.05, 'lengthScale_prior', pl, 'magnSigma2_prior', pm);

        % Create the likelihood structure
        lik = lik_logit();

        % Create the FIC GP structure so that inducing inputs are not optimized
        gp = gp_set('type', 'FIC', 'lik', lik, 'cf', gpcf1, 'X_u', Xu, ...
            'jitterSigma2', 1e-4, 'infer_params', 'covariance');
        % alternative models
        % gp = gp_set('type', 'PIC', 'lik', lik, 'cf', gpcf1, 'X_u', Xu, ...
        %            'jitterSigma2', 1e-4, 'infer_params', 'covariance',...
        %            'tr_index', trindex);
        %gp = gp_set('type', 'CS+FIC', 'lik', lik, 'cf', {gpcf1 gpcf2}, 'X_u', Xu, ...
        %            'jitterSigma2', 1e-4, 'infer_params', 'covariance');

        if(strcmp(popt.fitc.aprox,'Laplace'))
            % --- MAP estimate with Laplace approximation ---
            % Set the approximate inference method to Laplace approximation
            gp = gp_set(gp, 'latent_method', 'Laplace');
        else
            % Set the approximate inference method to EP approximation
            gp = gp_set(gp, 'latent_method', 'EP');
        end

        % Set the options for the scaled conjugate optimization
        opts=optimset('TolFun',1e-3,'TolX',1e-3,'Display','iter');
        % Optimize with the scaled conjugate gradient method
        gp=gp_optim(gp,x,y,'opt',opts);

        % make prediction to the data points
        maxrow = min(size(x,1),20000);
        [Ef, Varf] = gp_pred(gp,x(1:maxrow,:),y(1:maxrow), XTest);


        %     err = mean(((Ef>0)-yTest).^2)
        yPred = zeros(size(yTest));
        yPred(Ef > 0) = 1;
        test_error = 1 - sum(yPred==yTest)/size(yTest, 1);

        % PROBIT
        %  p = (erf(Ef/sqrt(2))+1)/2;

        % LOGIT
        p = 1./(1+exp(-Ef));
        inxp = find(p == 1);
        p(inxp) = 0.99999;
            
        nll_fitc = -mean(yTest.*log(p) + (1-yTest).*log(1-p));

        exec_time = toc;
        results.fitc.test_error(inumActive) = test_error;
        results.fitc.nll(inumActive) = nll_fitc;
        results.fitc.exec_time(inumActive) = exec_time;
    end
end

if(~isempty(popt.savefile))
    save(sprintf([folderPath.matlab,'/additive_gp/%s_%s.mat'], popt.savefile, datestr(now, 'yyyymmdd')));
end

end
