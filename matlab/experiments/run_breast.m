clear variables;
%clear functions;

data_id = 'breast';
runMCMC = false;

global folderPath

load([folderPath.datasets,'/breast.mat']);
y(y==-1) = 0;

fprintf('Number of negative classes = %i\n', sum(y == 0));
fprintf('Number of positive classes = %i\n', sum(y == 1));

[Nall,D] = size(X);
[dum,I] = sort(rand(Nall,1)); clear dum;
X(I,:) = X;
y(I)=y;

 for d = 1:D
     xd = (X(:,d) - mean(X(:,d)))./std(X(:,d));
     X(:,d) = xd;
 end
y2 = y;
y2(y2 == 0) = -1;

N = floor(0.8*Nall);
XTrain = X(1:N, :);
XTest = X(N+1:end, :);
yTrain = y(1:N);
yTrain2 = y2(1:N);
yTest = y(N+1:end);
yTest2 = y2(N+1:end);
M = length(yTest);



popt.lin = [];%true;]
popt.svm=[];%
popt.svm.Cs = [];%2.^(linspace(-2,10,7));
popt.svm.gamms = [];%2.^(linspace(-10,3,7));
popt.ivm=[];%
popt.ivm.numPseudo = [50,0.1*N];
popt.addLA=[];%
popt.addLA.runMCMC = false;
popt.addLA.ells = sqrt(2.^(linspace(-5,10,7)));
popt.addLA.sigfs = sqrt(2.^(linspace(-3,11,7)));
popt.addLA.numNewton = 5;
popt.addLA.numGS = 5;
popt.addfull = [];
popt.fullgp = [];%true;
popt.fitc.numPseudo = [50,0.1*N];
popt.fitc.aprox = 'EP';
popt.savefile = [];
results = run_class_comparison(XTrain,yTrain,XTest,yTest,popt);
