function [h] = plot_class_runtime_N_comp(file,marker)
%clear all
% file = 'reg_runtime_N_comp_8W_8D'
% file = 'results_comp_kilian.mat';


fntsz = 14;


load(file)
o.Ns = [2000:2000:10000, 20000:10000:50000]';
linear_logistic = zeros(length(o.Ns),1);
svm = zeros(length(o.Ns),1);
ivm = zeros(length(o.Ns),1);
fitc = zeros(length(o.Ns),1);
additive_grid = zeros(length(o.Ns),1);
additive_full = zeros(length(o.Ns),1);
full_gp = zeros(length(o.Ns),1);


for i = 1:length(o.Ns)
   
    linear_logistic(i)= o.LINEAR(i+1);
%     linear_logistic_test_error(i) = o.results(i).linear_logistic.test_error(1);
%     additive_mcmc_nll(i) = o.results(i).linear_logistic.nll(1);
    
    svm(i) = o.SVM(i+1)*50;
%     svm_test_error(i) = o.results(i).svm.test_error(1);
%     svm_nll(i) = o.results(i).svm.nll(1);
    
    ivm(i) = o.IVM(i+1)*50;
%     ivm_test_error(i) = o.results(i).ivm.test_error(1);
%     ivm_nll(i) = o.results(i).ivm.nll(1);
    
    additive_grid(i) = o.ADDITIVE_FAST(i+1)*50;
%     additive_grid_test_error(i) = o.results(i).additive_grid.test_error(1);
%     additive_grid_nl(i) = o.results(i).additive_grid.nll(1);

    fitc(i) = o.FITC(i+1);
    
    if(~isempty(o.ADDITIVE_FULL(i)))
        additive_full(i) = o.ADDITIVE_FULL(i+1);
%         additive_full_test_error(i) = o.results(i).additive_full.test_error(1);
%         additive_full_nl(i) = o.results(i).additive_full.nll(1);
    else
        additive_full(i) = NaN;
%         additive_full_test_error(i) = NaN;
%         additive_full_nl(i) = NaN;
    end
    
    if(~isempty(o.ADDITIVE_FULL(i)))       % NEED TO CHANGE TO FULLGP
        full_gp(i) = o.ADDITIVE_FULL(i+1)*50;
%         additive_full_test_error(i) = o.results(i).additive_full.test_error(1);
%         additive_full_nl(i) = o.results(i).additive_full.nll(1);
    else
        full_gp(i) = NaN;
%         additive_full_test_error(i) = NaN;
%         additive_full_nl(i) = NaN;
    end

end
% figure;
% hold on
h = loglog(o.Ns,[additive_grid],marker,'LineWidth',2.5,'Color',[0 0 1]); 
hold on
h = loglog(o.Ns,[fitc],marker,'LineWidth',2.5,'Color',[1 0 0]);
h = loglog(o.Ns,[ivm],marker,'LineWidth',2.5,'Color',[0.7 0 0]);
h = loglog(o.Ns,[svm],marker,'LineWidth',2.5,'Color',[0 1 0]);
h = loglog(o.Ns,[full_gp],marker,'LineWidth',2.5,'Color',[0 0 0]);
hold off
legend('Additive-LA','FIC','IVM','SVM','Full-GP')
ylabel('Runtime (s)','fontsize',fntsz)
xlabel('Input Size (N)','fontsize',fntsz)
set(gca,'fontsize',fntsz);
axis([1e3 1e5 50 1e5])

deltaNs = log(o.Ns(end))-log(o.Ns(2));
slop_fullgp = (log(full_gp(4))-log(full_gp(2)))/(log(o.Ns(4))-log(o.Ns(2)))
slop_svm = (log(svm(end))-log(svm(2)))/deltaNs
slop_ivm = (log(ivm(end))-log(ivm(2)))/deltaNs
slop_additive_grid=(log(additive_grid(end))-log(additive_grid(2)))/deltaNs
slop_fitc = (log(fitc(end))-log(fitc(2)))/deltaNs