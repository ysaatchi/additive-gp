#!/bin/sh

/export/matlab/R2009b/bin/mex -I/home/cunningham/MATLAB/FASTGP/boost_1_45_0 -largeArrayDims gpr_ssm_estep.cpp  -lmwlapack -lmwblas
/export/matlab/R2009b/bin/mex -I/home/cunningham/MATLAB/FASTGP/boost_1_45_0 -largeArrayDims gpr_ssm_estepEG.cpp  -lmwlapack -lmwblas
/export/matlab/R2009b/bin/mex -I/home/cunningham/MATLAB/FASTGP/boost_1_45_0 -largeArrayDims gpr_ssm_mstep_mex.cpp  -lmwlapack -lmwblas
/export/matlab/R2009b/bin/mex -I/home/cunningham/MATLAB/FASTGP/boost_1_45_0 -largeArrayDims gpr_ssm_fb.cpp  -lmwlapack -lmwblas
/export/matlab/R2009b/bin/mex -I/home/cunningham/MATLAB/FASTGP/boost_1_45_0 -largeArrayDims gpr_ssm_lik.cpp  -lmwlapack -lmwblas
/export/matlab/R2009b/bin/mex -I/home/cunningham/MATLAB/FASTGP/boost_1_45_0 -largeArrayDims gpr_ssm_ffbs.cpp  -lmwlapack -lmwblas
/export/matlab/R2009b/bin/mex -I/home/cunningham/MATLAB/FASTGP/boost_1_45_0 -largeArrayDims gpr_ssm_fb_diag.cpp  -lmwlapack -lmwblas
/export/matlab/R2009b/bin/mex -I/home/cunningham/MATLAB/FASTGP/boost_1_45_0 -largeArrayDims gpr_ssm_nlml.cpp  -lmwlapack -lmwblas
/export/matlab/R2009b/bin/mex -I/home/cunningham/MATLAB/FASTGP/boost_1_45_0 -largeArrayDims gpr_pgr_1d_mex.cpp  -lmwlapack -lmwblas
