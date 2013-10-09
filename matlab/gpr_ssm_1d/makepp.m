global folderPath

lapacklib = fullfile(matlabroot, 'extern', 'lib', 'win64', 'microsoft', 'libmwlapack.lib');
blaslib = fullfile(matlabroot, 'extern', 'lib', 'win64', 'microsoft', 'libmwblas.lib');

mex(['-I',folderPath.boost],'-largeArrayDims', 'gpr_pgr_1d_mex.cpp', lapacklib, blaslib)
%mex('-IC:/boost_1_45_0','-largeArrayDims', 'gpr_ssm_mstep_mex.cpp', lapacklib, blaslib)