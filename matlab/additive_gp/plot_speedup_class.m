close all
fileNames = {'reg_runtime_N_resutls_1_1D','reg_runtime_N_resutls_1_8D','reg_runtime_N_resutls_1_32D',...
    'reg_runtime_N_resutls_8_1D','reg_runtime_N_resutls_8_8D','reg_runtime_N_results_8_32D'};
yBar = zeros(3,length(fileNames));
yName = {};
fullgp = {};
Ncell = 5;
for i = 1:length(fileNames)
   load(fileNames{i})
   fullgp = o.FULL_GP(Ncell);
   aspeed = fullgp/o.ADDITIVE_VB(Ncell);
   pspeed = fullgp/o.PPGPR(Ncell);
   sspeed = fullgp/o.SPGP(Ncell);
   
   yBar(:,i) = [sspeed,aspeed,pspeed]';
   yName{i} = fileNames{i};
end

figure;
bar(yBar');
XTick=get(gca,'XTick');
xticklabel_rotate(XTick,45,yName,'interpreter','none')
legend('SPGP','VBEM','PPGPR');
ylabel('Speedup')

fileNames = {'reg_runtime_N_comp_8W_1D','reg_runtime_N_comp_8W_8D',...
    'run_pumadyn8_fm1000_8W',...
    'run_pumadyn8_fm7168_8W',...
    'run_pumadyn8_nm_8W',...
    'run_elevators_8W','run_kin40k_8W','run_pumadyn32_nm_8W'};
yBarTime = zeros(length(fileNames),5);
yBarMNLP = zeros(length(fileNames),6);
yName = {};
for i = 1:length(fileNames)
   load(fileNames{i})
   if(length(o.results)> 5)
       Ncell = 5;
   else
       Ncell = 1;
   end
   fullgp.time = o.results(Ncell).full_gp.exec_time;
   fullgp.mnlp = o.results(Ncell).full_gp.mnlp;
   
   additive_mcmc.time = o.results(Ncell).additive_mcmc.exec_time;
   additive_mcmc.mnlp = mean(o.results(Ncell).additive_mcmc.mnlp(max(3,end-Ncell):end));
   
   additive_vb.time = o.results(Ncell).additive_vb.exec_time;
   additive_vb.mnlp = o.results(Ncell).additive_vb.mnlp;
   
   ppr_mcmc.time = o.results(Ncell).ppr_mcmc.exec_time;
   ppr_mcmc.mnlp = mean(o.results(Ncell).ppr_mcmc.mnlp(max(3,end-Ncell):end));
   
   pp_gpr.time = o.results(Ncell).pp_gpr.exec_time;
   pp_gpr.mnlp = o.results(Ncell).pp_gpr.mnlp;
   
   spgp.time = o.results(Ncell).spgp.exec_time;
   spgp.mnlp = o.results(Ncell).spgp.mnlp;
   
   
   yBarTime(i,:) = ([spgp.time,additive_mcmc.time,additive_vb.time,...
       ppr_mcmc.time,pp_gpr.time]/fullgp.time).^(-1);
   
   yBarMNLP(i,:) = ([spgp.mnlp,additive_mcmc.mnlp,additive_vb.mnlp,...
       ppr_mcmc.mnlp,pp_gpr.mnlp,fullgp.mnlp]); 
   
       
   yName{i} = fileNames{i};
end

figure;
bar(yBarTime);
XTick=get(gca,'XTick');
xticklabel_rotate(XTick,45,yName,'interpreter','none');
legend('SPGP','Additive-MCMC','Additive-VB','PPGPR-MCMC','PPGPR-Greedy');
ylabel('Speedup')
figure;
bar(yBarMNLP);
XTick=get(gca,'XTick');
xticklabel_rotate(XTick,45,yName,'interpreter','none');
legend('SPGP','Additive-MCMC','Additive-VB','PPGPR-MCMC','PPGPR-Greedy');
ylabel('MNLP');