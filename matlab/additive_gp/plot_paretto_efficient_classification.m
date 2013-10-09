synth=[
0.59  161
0.74 2244
1.06 525
01.36 850
0.69 65.1
0.58  345
]

telescope = [
0.34  2345
NaN  inf
0.36 3340
0.36 7331
0.69 118
0.30  8070
]

IJCNN = [
0.16  14505
NaN  inf
1.18 4390
0.82 16728
0.69  369
0.05  22170
]

%%


semilogx(telescope(1,2),telescope(1,1),['bd'],telescope(2,2),telescope(2,1),['bx'],telescope(3,2),telescope(3,1),['b^'],...
    telescope(4,2),telescope(4,1),['b*'],telescope(5,2),telescope(5,1),['bo'],telescope(6,2),telescope(6,1),['b<'],'MarkerSize',10,'MarkerFaceColor','b')
legend('Additive-LA','Full-GP','FIC-50','FIC 10%','IVM 50','SVM');
hold on

semilogx(synth(1,2),synth(1,1),['rd'],synth(2,2),synth(2,1),['rx'],synth(3,2),synth(3,1),['r^'],...
    synth(4,2),synth(4,1),['r*'],synth(5,2),synth(5,1),['ro'],synth(6,2),synth(6,1),['r<'],'MarkerSize',10,'MarkerFaceColor','r')


semilogx(IJCNN(1,2),IJCNN(1,1),['gd'],IJCNN(2,2),IJCNN(2,1),['gx'],IJCNN(3,2),IJCNN(3,1),['g^'],...
    IJCNN(4,2),IJCNN(4,1),['g*'],IJCNN(5,2),IJCNN(5,1),['go'],IJCNN(6,2),IJCNN(6,1),['g<'],'MarkerSize',10,'MarkerFaceColor','g')

% pareto lines
semilogx([telescope(5,2),telescope(5,2),telescope(1,2),telescope(6,2),telescope(6,2)+1e10], [telescope(5,1)+1e4,telescope(5,1),telescope(1,1),telescope(6,1),telescope(6,1)],['b'])
semilogx([synth(5,2),synth(5,2),synth(1,2),synth(6,2),synth(6,2)+1e10], [synth(5,1)+1e4,synth(5,1),synth(1,1),synth(6,1),synth(6,1)],['r'])
semilogx([IJCNN(5,2),IJCNN(5,2),IJCNN(6,2),IJCNN(6,2)+1e10], [IJCNN(5,1)+1e4,IJCNN(5,1),IJCNN(6,1),IJCNN(6,1)],['g'])
text(1e4,0.3+0.1, 'telescope','FontSize',14,'Color','b')
text(0.7e4,0.58+0.1, 'synthetic','FontSize',14,'Color','r')
text(2e4,0.05+0.1, 'IJCNN','FontSize',14,'Color','g')


hold off
axis([50,10e4,0,1.5]);
ylabel('MNLL','FontSize',fntsz);
xlabel('Runtime (s)','FontSize',fntsz);
set(gca,'fontsize',14);
