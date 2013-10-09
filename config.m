global folderPath

currFolder = 'C:/Users/Elad/Dropbox/gp/fastGP/released code/AdditiveGP'; %<current code directory>

folderPath.datasets = [currFolder,'/datasets'];
folderPath.source = [currFolder,'/source'];
folderPath.lib = [currFolder,'/lib'];
folderPath.GPML =  'C:/Users/Elad/Dropbox/gp/GPPack/gpml-matlab-v3.2-2013-01-15';
folderPath.GPstuff =  'C:/Users/Elad/Dropbox/gp/GPstuff-3.3';
 
% uncomment to complie mex files
% folderPath.boost =  'C:/boost_1_45_0';          

addpath(genpath(folderPath.GPML));
addpath(genpath(folderPath.GPstuff));
addpath(genpath(folderPath.source));
addpath(genpath(folderPath.lib));