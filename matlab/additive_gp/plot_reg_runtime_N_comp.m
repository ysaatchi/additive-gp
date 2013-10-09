function [h] = plot_reg_runtime_N_comp(file,marker,clacSlop,lnzs,fig,algnames,algcolors)
%clear all
% file = 'reg_runtime_N_comp_8W_8D'
% file = 'results_comp_kilian.mat';
fntsz = 14;

PPGPR = 1;
SPGP = 5;
ADDVB = 3;
FULLGP =6;

load(file)

additive_mcmc = zeros(length(o.Ns),1);
ppr_mcmc = zeros(length(o.Ns),1);
pp_gpr = zeros(length(o.Ns),1);
additive_vb = zeros(length(o.Ns),1);
full_gp_add = zeros(length(o.Ns),1);
full_gp = zeros(length(o.Ns),1);
spgp = zeros(length(o.Ns),1);

for i = 1:length(o.Ns)
   
    additive_mcmc(i) = o.results(i).additive_mcmc.exec_time;
    additive_mcmc_nmse(i) = o.results(i).additive_mcmc.nmse(1);
    additive_mcmc_mnlp(i) = o.results(i).additive_mcmc.mnlp(1);
    
    ppr_mcmc(i) = o.results(i).ppr_mcmc.exec_time;
    ppr_mcmc_nmse(i) = o.results(i).ppr_mcmc.nmse(1);
    ppr_mcmc_mnlp(i) = o.results(i).ppr_mcmc.mnlp(1);
    
    pp_gpr(i) = o.results(i).pp_gpr.exec_time;
    pp_gpr_nmse(i) = o.results(i).pp_gpr.nmses(1);
    pp_gpr_mnlp(i) = o.results(i).pp_gpr.mnlp(1);
    
    additive_vb(i) = o.results(i).additive_vb.exec_time;
    additive_vb_nmse(i) = o.results(i).additive_vb.nmse(1);
    additive_vb_mnlp(i) = o.results(i).additive_vb.mnlp(1);
    
    if(~isempty(o.results(i).full_gp_add))
        full_gp_add(i) = o.results(i).full_gp_add.exec_time;
        full_gp_add_nmse(i) = o.results(i).full_gp_add.nmse(1);
        full_gp_add_mnlp(i) = o.results(i).full_gp_add.mnlp(1);
    else
        full_gp_add(i) = NaN;
        full_gp_add_nmse(i) = NaN;
        full_gp_add_mnlp(i) = NaN;
    end
    if(~isempty(o.results(i).full_gp))
        full_gp(i) = o.results(i).full_gp.exec_time;
        full_gp_nmse(i) = o.results(i).full_gp.nmse(1);
        full_gp_mnlp(i) = o.results(i).full_gp.mnlp(1);
    else
        full_gp(i) = NaN;
        full_gp_nmse(i) = NaN;
        full_gp_mnlp(i) = NaN;
    end
    spgp(i) = o.results(i).spgp.exec_time;
    spgp_nmse(i) = o.results(i).spgp.nmse(1);
    spgp_mnlp(i) = o.results(i).spgp.mnlp(1);
end
% figure;
% hold on
algdata = {pp_gpr,ppr_mcmc,additive_vb,additive_mcmc,spgp,full_gp};
h = fig;
for algi = 1:length(algcolors)
%     h = loglog(o.Ns,algdata{algi},marker);
%     set(h(algi),'LineWidth',lnzs,'Color',algcolors{algi});
    loglog(o.Ns,algdata{algi},marker{:},'MarkerFaceColor',algcolors{algi},'LineWidth',lnzs,'Color',algcolors{algi});
end
legend(algnames{1},algnames{2},algnames{3},algnames{4},algnames{5},algnames{6});
ylabel('Runtime (s)','fontsize',fntsz)
xlabel('N','fontsize',fntsz)
set(gca,'fontsize',fntsz);

if(clacSlop == true)
    deltaNs = log(o.Ns(end))-log(o.Ns(1));
    slopfullgp = (log(full_gp(5))-log(full_gp(1)))/(log(o.Ns(5))-log(o.Ns(1)))
    slop_pp_gpr = (log(pp_gpr(end))-log(pp_gpr(1)))/deltaNs
    slop_ppr_mcmc = (log(ppr_mcmc(end))-log(ppr_mcmc(1)))/deltaNs
    slop_additive_vb=(log(additive_vb(end))-log(additive_vb(1)))/deltaNs
    slop_additive_mcmc = (log(additive_mcmc(end))-log(additive_mcmc(1)))/deltaNs
    slop_spgp = (log(spgp(end))-log(spgp(1)))/deltaNs
end