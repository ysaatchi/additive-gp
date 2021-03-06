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

//[nlml, nsse, nlogdet] = gpr_ssm_nlml(logtheta, noise, x, y, V0, Phi_cell, Q_cell)

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    
    //Inputs
    if (nrhs != 7) {
        mexErrMsgTxt("Incorrect number of arguments (should be 7). Quitting...");
    }
    
    bool debug = false;
    
    int D = mxGetM(prhs[4]);
    
    double* logtheta = mxGetPr(prhs[0]);
    double* noise = mxGetPr(prhs[1]);
    double* x = mxGetPr(prhs[2]);
    double* y = mxGetPr(prhs[3]);
    double* V0 = mxGetPr(prhs[4]);
    const mxArray* Phi_cell = prhs[5];
    const mxArray* Q_cell = prhs[6];
    
    //size checks
    if (mxGetM(prhs[0]) != 2 || (mxGetM(prhs[2]) != mxGetM(prhs[3]))) {
        mexErrMsgTxt("Check input argument sizes! Quitting...");
    }
    if (!(mxGetN(prhs[0]) == 1 && mxGetN(prhs[2]) == 1 && mxGetN(prhs[3]) == 1)) {
        mexErrMsgTxt("All vectors passed in must be column vectors! Quitting...");
    }
    //Outputs
    if (nlhs != 3) {
        mexErrMsgTxt("Invalid number of output arguments (should be 3). Quitting...");
    }
    int T = mxGetM(prhs[1]);
    plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
    plhs[2] = mxCreateDoubleMatrix(1, 1, mxREAL);
    double* nlml = mxGetPr(plhs[0]);
    double* nsse = mxGetPr(plhs[1]);
    double* nlogdet = mxGetPr(plhs[2]);
    typedef boost::multi_array<double, 3> array_3D;
    typedef boost::multi_array<double, 2> array_2D;
    typedef std::vector<double> array_1D;
    //MAIN CODE
   
    double lambda = exp(logtheta[0]); //cov "decay" parameter
    double signal_var = exp(logtheta[1]);
    
    array_1D kalman_gain(D);
    array_1D mu(D);
    array_2D V(boost::extents[D][D]);
    array_1D PhiMu(D);
    array_2D Phi(boost::extents[D][D]);
    array_2D Q(boost::extents[D][D]);
    array_2D Z(boost::extents[D][D]);
    array_2D P(boost::extents[D][D]);
    //output variables
    nlml[0] = 0;
    nsse[0] = 0;
    nlogdet[0] = 0;
    
    /* BEGIN KALMAN FILTERING */
    ////cout << "Forward filtering..." << endl;
    //absorb first observation
    int t = 0;
    
    double pm = 0;
    double pv = V0[0] + noise[t];
    double lp = log(pv);
    double ne = (pow((y[t] - pm), 2) / pv); 
    nsse[0] += lp;
    nlogdet[0] += ne;
    nlml[0] += 0.5 * (log(2 * PI) + lp + ne);
    for (int d = 0; d < D; d++) {
        double kgd = V0[d] / pv;
        kalman_gain[d] = kgd;
        mu[d] = kgd * (y[t] - pm);
    }
    for (int i = 0; i < D; i++) {
        for (int j = 0; j < D; j++) {
            V[i][j] = V0[j*D + i] - kalman_gain[i]*V0[j*D];
        }
    }
       
    //filter forward in "time"
    for (t = 1; t < T; t++) {

        array_1D Phi_v(D*D);
        array_1D Q_v(D*D);
      
        memcpy(&Phi_v[0], mxGetPr(mxGetCell(Phi_cell, t-1)), sizeof(double)*D*D);
        memcpy(&Q_v[0], mxGetPr(mxGetCell(Q_cell, t-1)), sizeof(double)*D*D);
        
        for (int i = 0; i < D; i++) {
            for (int j = 0; j < D; j++) {
                Phi[i][j] = Phi_v[D*j + i];
            }
        }
        for (int i = 0; i < D; i++) {
            for (int j = 0; j < D; j++) {
                Q[i][j] = Q_v[D*j + i];
            }
        }
        
        memset(&PhiMu[0], 0, sizeof (double) * D);
        for (int i = 0; i < D; i++) {
            for (int j = 0; j < D; j++) {
                PhiMu[i] += Phi[i][j] * mu[j];
            }
        }
        
        //TODO: replace below with standard matrix mul
        for (int i = 0; i < D; i++) {
            for (int j = 0; j < D; j++) {
                Z[i][j] = 0;
                for (int k = 0; k < D; k++) {
                    Z[i][j] += Phi[i][k] * V[k][j];
                }
            }
        }
        for (int i = 0; i < D; i++) {
            for (int j = 0; j < D; j++) {
                P[i][j] = Q[i][j];
                for (int k = 0; k < D; k++) {
                    P[i][j] += Z[i][k] * Phi[j][k];
                }
            }
        }
        
        pm = PhiMu[0];
        pv = P[0][0] + noise[t];
        lp = log(pv);
        ne = (pow((y[t] - pm), 2) / pv); 
        nsse[0] += lp;
        nlogdet[0] += ne;
        nlml[0] += 0.5 * (log(2 * PI) + lp + ne);
        for (int d = 0; d < D; d++) {
            double kgd = (P[d][0]) / pv;
            kalman_gain[d] = kgd;
            mu[d] = PhiMu[d] + kgd * (y[t] - pm);
        }
        for (int i = 0; i < D; i++) {
            for (int j = 0; j < D; j++) {
                V[i][j] = P[i][j] - kalman_gain[i]*(P[0][j]);
            }
        }   
        
        
    }
    /* END KALMAN FILTERING */
    
}
