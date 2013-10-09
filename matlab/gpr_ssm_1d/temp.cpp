//You can include any C libraries that you normally use
#include "mex.h"
#include "lapack.h"
#include <math.h>
#include <vector>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <time.h>
#include <assert.h>
#include <stdlib.h>
#include "boost/multi_array.hpp"
#define PI 3.14159265358979323846

using namespace std;

//[nlml, Ex, Vx, Exxprev] = gpr_ssm_estep(logtheta, x, y, V0, Phi_cell, Q_cell)

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    
    //Inputs
    if (nrhs != 6) {
        mexErrMsgTxt("Incorrect number of arguments (should be 6). Quitting...");
    }
    
    bool debug = false;
    
    
    
}
