//You can include any C libraries that you normally use
#include "mex.h"
#include "lapack.h"
#include <math.h>
#include <vector>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <time.h>
#include <stdlib.h>
#define PI 3.14159265358979323846

using namespace std;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

//K = covSEard_additive(logtheta, X) 
    
    //Inputs
    if (nrhs != 2) {
        mexErrMsgTxt("Incorrect number of arguments (should be 2). Quitting...");
    }
    
    bool debug = false;
    
    double* logtheta = mxGetPr(prhs[0]);
    double* X = mxGetPr(prhs[1]);
    int N = mxGetM(prhs[1]);
    int D = mxGetN(prhs[1]);
    
    //size checks
    if (mxGetM(prhs[0]) != 2*D+1) {
        mexErrMsgTxt("Check input argument sizes! Quitting...");
    }
    //Outputs
    if (nlhs != 1) {
        mexErrMsgTxt("Invalid number of output arguments (should be 1). Quitting...");
    }
    
    plhs[0] = mxCreateDoubleMatrix(N, N, mxREAL);
    
    double* K = mxGetPr(plhs[0]);
    
    //MAIN CODE
    
    for (int i = 0; i < N; i++) {
        
        for (int j = 0; j < N; j++) { 
            
            double k = 0;
            
            for (int d = 0; d < D; d++) { 
                
                double ell2 = exp(2*logtheta[d]);
                double sv = exp(2*logtheta[D+d]);
                k += sv * exp(-0.5*((X[N*d + i] - X[N*d + j])*(X[N*d + i] - X[N*d + j]))/ell2);
                
            }
            
            if (i == j) {
                
                k += exp(2*logtheta[2*D]);
                
            }
            
            K[j*N + i] = k;
            
        }
        
    }
   
       
}