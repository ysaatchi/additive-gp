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

mexfiles = {'covSEard_additive'};

for i = 1:length(mexfiles)
    mex(['-I',folderPath.boost],'-largeArrayDims', [mexfiles{i},'.cpp'], lapacklib, blaslib)
end
