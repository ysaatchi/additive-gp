close all

fig1 = figure(1);
hold on
algnames = {'PPGPR-Greedy','PPGPR-MCMC','Additive-VB','Additive-MCMC','SPGP','Full-GP'};
algcolors = {[0 0 1],[0.85 0 0],[0.77 0 0],[0.62 0 0],[0.5 0 0],[0 0 0]};

plot_reg_runtime_N_comp('reg_runtime_N_comp_8W_8D_fix',{'-'},true,2.5,fig1,algnames,algcolors);

% hold on
% llpumadyn8 = plot_reg_runtime_N_comp('run_pumadyn8_fm7168_8W.mat','O');
% markerColor = get(llpumadyn8,'Color');
% for i = 1:length(markerColor)
%     set(llpumadyn8(i),'MarkerFaceColor',markerColor{i},'MarkerSize',10);
% end
% 
llpumadyn8 = plot_reg_runtime_N_comp('run_pumadyn8_nm_8W.mat',{'O','MarkerSize',7},false,1,fig1,algnames,algcolors);
% markerColor = get(llpumadyn8,'Color');
% for i = 1:length(markerColor)
%     set(llpumadyn8(i),'MarkerFaceColor',markerColor{i},'MarkerSize',7);
% end


algcolors = {[0 0 1]};
plot_reg_runtime_N_comp('reg_runtime_N_comp_1W_8D_fix',{'--'},true,1.5,fig1,algnames(1),algcolors);
legend(algnames{1},algnames{2},algnames{3},algnames{4},algnames{5},algnames{6});
fntsz=14;
ylabel('Runtime (s)','fontsize',fntsz)
xlabel('N','fontsize',fntsz)
set(gca,'fontsize',fntsz);


% llpumadyn8 = plot_reg_runtime_N_comp('run_pumadyn8_nm_1W.mat','x',false,1);
% markerColor = get(llpumadyn8,'Color');
% for i = 1:length(markerColor)
%     set(llpumadyn8(i),'MarkerFaceColor',markerColor{i},'MarkerSize',10);
% end

%
% llpumadyn8 = plot_reg_runtime_N_comp('run_pumadyn8_fm1000_1W.mat','*');
% markerColor = get(llpumadyn8,'Color');
% for i = 1:length(markerColor)
%     set(llpumadyn8(i),'MarkerFaceColor',markerColor{i},'MarkerSize',7);
% end
% %lina([1.1e3 1.5e3],[600 3e3],'Marker','.','LineStyle','-')
% text(6800,100,...
%      '\uparrow pumadyn8-nm7168-1W',...
%      'FontSize',16)
% text(950,20,...
%      '\uparrow pumadyn8-fm1000-1W',...
%      'FontSize',16)
% title('reg-runtime-N-comp-1W-8D')
 axis([1e3,5e4,1e1,1e4])
%line([7168 7168],[50 150],'Marker','.','LineStyle','-')


% fig2 = figure(2);
% plot_reg_runtime_N_comp('reg_runtime_N_comp_8W_8D_fix','-',true,2.5);
% hold on
% llpumadyn8 = plot_reg_runtime_N_comp('run_pumadyn8_nm_8W.mat','O',false,1);
% markerColor = get(llpumadyn8,'Color');
% for i = 1:length(markerColor)
%     set(llpumadyn8(i),'MarkerFaceColor',markerColor{i},'MarkerSize',7);
% end
% 
% plot_reg_runtime_N_comp('reg_runtime_N_comp_1W_8D_fix','--',true,1.5);
% 
% llpumadyn8 = plot_reg_runtime_N_comp('run_pumadyn8_nm_1W.mat','x',false,1);
% markerColor = get(llpumadyn8,'Color');
% for i = 1:length(markerColor)
%     set(llpumadyn8(i),'MarkerFaceColor',markerColor{i},'MarkerSize',10);
% end
% axis([10^(3.85),10^(3.86),40,1e3]);
% [h_m h_i]=inset(fig1,fig2,0.6);


text(6800,30,...
     '\uparrow pumadyn8-nm7168',...
     'FontSize',16)
% text(950,20,...
%      '\uparrow pumadyn8-fm1000-8W',...
%      'FontSize',16)
% title('reg-runtime-N-comp-8W-8D')
% %line([7168 7168],[50 150],'Marker','.','LineStyle','-')
axis([1e3,5e4,1e1,8000])

return
%%
figure
plot_class_runtime_N_comp('class_runtimes_N_results1','-');
axis([2e3,5e4,70,10e3])

