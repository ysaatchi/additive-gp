global folderPath

if( isequal(computer,'PCWIN64') )
    %% WIN 64bit
    lapacklib = fullfile(matlabroot, ...
      'extern', 'lib', 'win64', 'microsoft', 'libmwlapack.lib');
    blaslib = fullfile(matlabroot, ...
      'extern', 'lib', 'win64', 'microsoft', 'libmwblas.lib');
elseif ( isequal(computer,'PCWIN32') )
    %% WIN 32bit
    lapacklib = fullfile(matlabroot, ...
      'extern', 'lib', 'win32', 'microsoft', 'libmwlapack.lib');
    blaslib = fullfile(matlabroot, ...
      'extern', 'lib', 'win32', 'microsoft', 'libmwblas.lib');
else
    %% UNIX
        lapacklib = '-lmwlapack';
        blaslib = '-lmwblas';
end

mexfiles = {'gpr_ssm_estep','gpr_ssm_estepEG','gpr_ssm_mstep_mex','gpr_ssm_fb','gpr_ssm_lik','gpr_ssm_ffbs','gpr_ssm_fb_diag',...
    'gpr_ssm_nlml','gpr_pgr_1d_mex'};

for i = 1:length(mexfiles)
    mex(['-I',folderPath.boost],'-largeArrayDims', [mexfiles{i},'.cpp'], lapacklib, blaslib)
end