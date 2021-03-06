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
#include "boost/array.hpp"
#define PI 3.14159265358979323846

using namespace std;

//[nlml, Ex, Vx, Exxprev] = gpr_ssm_estep(logtheta, x, y, V0, Phi_cell, Q_cell)

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    
    //Inputs
    if (nrhs != 6) {
        mexErrMsgTxt("Incorrect number of arguments (should be 6). Quitting...");
    }
    
    bool debug = false;
    
    int D = mxGetM(prhs[3]);
    
    double* logtheta = mxGetPr(prhs[0]);
    double* x = mxGetPr(prhs[1]);
    double* y = mxGetPr(prhs[2]);
    double* V0 = mxGetPr(prhs[3]);
    const mxArray* Phi_cell = prhs[4];
    const mxArray* Q_cell = prhs[5];
    
    //size checks
    if (mxGetM(prhs[0]) != 3 || (mxGetM(prhs[1]) != mxGetM(prhs[2]))) {
        mexErrMsgTxt("Check input argument sizes! Quitting...");
    }
    if (!(mxGetN(prhs[0]) == 1 && mxGetN(prhs[1]) == 1 && mxGetN(prhs[2]) == 1)) {
        mexErrMsgTxt("All vectors passed in must be column vectors! Quitting...");
    }
    //Outputs
    if (nlhs != 4) {
        mexErrMsgTxt("Invalid number of output arguments (should be 4). Quitting...");
    }
    int T = mxGetM(prhs[1]);
    plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
    plhs[1] = mxCreateCellMatrix(T, 1);
    plhs[2] = mxCreateCellMatrix(T, 1);
    plhs[3] = mxCreateCellMatrix(T-1, 1);
    double* nlml = mxGetPr(plhs[0]);
    
    //MAIN CODE
   
    double lambda = exp(logtheta[0]); //cov "decay" parameter
    double signal_var = exp(logtheta[1]);
    double noise_var = exp(logtheta[2]);
    //store filtering results TODO: MUST FIX SO THAT THESE ARE STORED ON HEAP
    typedef boost::multi_array<double, 3> array_3D;
    typedef boost::multi_array<double, 2> array_2D;
    typedef std::vector<double> array_1D;
    array_2D mus(boost::extents[T][D]);
    array_3D Phis(boost::extents[T][D][D]);
    array_3D Ps(boost::extents[T][D][D]);
    array_3D Vs(boost::extents[T][D][D]);
    
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
    
    /* BEGIN KALMAN FILTERING */
    //////cout << "Forward filtering..." << endl;
    //absorb first observation
    int t = 0;
    
    double pm = 0;
    double pv = V0[0] + noise_var;
    nlml[0] += 0.5 * (log(2 * PI) + log(pv) + (pow((y[t] - pm), 2) / pv));
    for (int d = 0; d < D; d++) {
        double kgd = V0[d] / pv;
        kalman_gain[d] = kgd;
        mu[d] = kgd * (y[t] - pm);
        mus[t][d] = mu[d];
    }
    for (int i = 0; i < D; i++) {
        for (int j = 0; j < D; j++) {
            V[i][j] = V0[j*D + i] - kalman_gain[i]*V0[j*D];
            Vs[t][i][j] = V[i][j];
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
        
        double pm = PhiMu[0];
        double pv = P[0][0] + noise_var;
        nlml[0] += 0.5 * (log(2 * PI) + log(pv) + (pow((y[t] - pm), 2) / pv));
        for (int d = 0; d < D; d++) {
            double kgd = (P[d][0]) / pv;
            kalman_gain[d] = kgd;
            mu[d] = PhiMu[d] + kgd * (y[t] - pm);
            mus[t][d] = mu[d];
        }
        for (int i = 0; i < D; i++) {
            for (int j = 0; j < D; j++) {
                V[i][j] = P[i][j] - kalman_gain[i]*(P[0][j]);
                Vs[t][i][j] = V[i][j];
            }
        }   
        
        //Save Phi and P for smoothing
        for (int i = 0; i < D; i++) {
            for (int j = 0; j < D; j++) {
                Phis[t - 1][i][j] = Phi[i][j];
                Ps[t - 1][i][j] = P[i][j];
            }
        }
        
    }
    /* END KALMAN FILTERING */
    
    /* BEGIN BACKWARD SMOOTHING */
    
    //////cout << "Backward smoothing..." << endl;
    
    mxArray* mut = mxCreateDoubleMatrix(D, 1, mxREAL);
    mxArray* Vt = mxCreateDoubleMatrix(D, D, mxREAL);
    mxArray* Wt = mxCreateDoubleMatrix(D, D, mxREAL);
    double* mu_s = mxGetPr(mut);
    double* V_s = mxGetPr(Vt);
    double* W_s = mxGetPr(Wt);
    //mu_s = filter(T).mu;
    //V_s = filter(T).V;
    for (int i = 0; i < D; i++) {
        mu_s[i] = mus[T-1][i];
        for (int j = 0; j < D; j++) {
            V_s[D*j + i] = Vs[T-1][i][j];
        }
    }
    mxSetCell(plhs[1], T-1, mxDuplicateArray(mut));
    mxSetCell(plhs[2], T-1, mxDuplicateArray(Vt));
    //W = (eye(D) - kalman_gain*H)*(Phi*filter(T-1).V);
    array_2D W1(boost::extents[D][D]);
    for (int i = 0; i < D; i++) {
        for (int j = 0; j < D; j++) {
            W1[i][j] = 0;
            if (i == j) {
                W1[i][j] = 1;
            }
            if (j == 0) {
                W1[i][j] -= kalman_gain[i];
            }
        }
    }
    for (int i = 0; i < D; i++) {
        for (int j = 0; j < D; j++) {
            Z[i][j] = 0;
            for (int k = 0; k < D; k++) {
                Z[i][j] += Phis[T-2][i][k] * Vs[T-2][k][j];
            }
        }
    }
    for (int i = 0; i < D; i++) {
        for (int j = 0; j < D; j++) {
            W_s[D*j + i] = 0;
            for (int k = 0; k < D; k++) {
                W_s[D*j + i] += W1[i][k]*Z[k][j];
            }
        }
    }
    
    array_2D L(boost::extents[D][D]);
    array_2D L_next(boost::extents[D][D]);
    
    for (t = T-2; t >= 0; t--) {
        
        //L = V*Phi'*(P\eye(D));
        for (int i = 0; i < D; i++) {
            mu[i] = mus[t][i];
            for (int j = 0; j < D; j++) {
                V[i][j] = Vs[t][i][j];
                Phi[i][j] = Phis[t][i][j];
                P[i][j] = Ps[t][i][j];
            }
        }

        array_1D A(D*D);
        for (int i = 0; i < D; i++) {
            for (int j = 0; j < D; j++) {
                A[j * D + i] = P[i][j];
            }
        }
        
        mwSignedIndex info;
        mwSignedIndex D_lp = (mwSignedIndex) D;
        char UorL = 'L';
        dpotrf(&UorL, &D_lp, &A[0], &D_lp, &info);
        if (info != 0){
            ////cout << "WARNING from gpr_ssm_matern: Matrix is probably not positive definite!" << endl;
        }
        dpotri(&UorL, &D_lp, &A[0], &D_lp, &info);
        
        array_2D Pinv(boost::extents[D][D]);
        for (int i = 0; i < D; i++) {
            for (int j = 0; j <= i; j++) {
                Pinv[i][j] = A[j * D + i];
                if (j != i) {
                    Pinv[j][i] = Pinv[i][j];
                }
            }
        }
        
        for (int i = 0; i < D; i++) {
            for (int j = 0; j < D; j++) {
                Z[i][j] = 0;
                for (int k = 0; k < D; k++) {
                    Z[i][j] += V[i][k] * Phi[j][k];
                }
            }
        }
        for (int i = 0; i < D; i++) {
            for (int j = 0; j < D; j++) {
                L[i][j] = 0;
                for (int k = 0; k < D; k++) {
                    L[i][j] += Z[i][k] * Pinv[k][j];
                }
                //printf("L[%i][%i] = %3.3f\n", i, j, L[i][j]);
            }
        }
        //mu_s = mu + L*(mu_s - Phi*mu);
        for (int i = 0; i < D; i++) {
            PhiMu[i] = 0;
            for (int j = 0; j < D; j++) {
                PhiMu[i] += Phi[i][j] * mu[j];
            }
        }
        array_1D mu_s_next(D);
        for (int i = 0; i < D; i++) {
            mu_s_next[i] = mu_s[i];
        }
        for (int i = 0; i < D; i++) {
            mu_s[i] = mu[i];
            for (int j = 0; j < D; j++) {
                mu_s[i] += L[i][j]*(mu_s_next[j] - PhiMu[j]);
            }
        }
        //V_s = V + L*(V_s - P)*L';
        array_2D dV(boost::extents[D][D]);
        for (int i = 0; i < D; i++) {
            for (int j = 0; j < D; j++) {
                dV[i][j] = V_s[D*j + i] - P[i][j];
            }
        }
        for (int i = 0; i < D; i++) {
            for (int j = 0; j < D; j++) {
                Z[i][j] = 0;
                for (int k = 0; k < D; k++) {
                    Z[i][j] += L[i][k] * dV[k][j];
                }
            }
        }        
        for (int i = 0; i < D; i++) {
            for (int j = 0; j < D; j++) {
                V_s[D*j + i] = V[i][j];
                for (int k = 0; k < D; k++) {
                    V_s[D*j + i] += Z[i][k] * L[j][k];
                }
            }
        }
        /*if i < (T-1)
            W = filter(i+1).V*L' + E(i+1).L*(W - filter(i+1).Phi*filter(i+1).V)*L';
          end*/
        if (t < T-2) {
            for (int i = 0; i < D; i++) {
                for (int j = 0; j < D; j++) {
                    W1[i][j] = 0;
                    for (int k = 0; k < D; k++) {
                        W1[i][j] += Vs[t+1][i][k] * L[j][k];
                    }
                }
            }
            double PhiV;
            array_2D dW(boost::extents[D][D]);
            for (int i = 0; i < D; i++) {
                for (int j = 0; j < D; j++) {
                    PhiV = 0;
                    for (int k = 0; k < D; k++) {
                        PhiV += Phis[t+1][i][k] * Vs[t+1][k][j];
                    }
                    dW[i][j] = W_s[D*j + i] - PhiV;
                }
            }
            for (int i = 0; i < D; i++) {
                for (int j = 0; j < D; j++) {
                    Z[i][j] = 0;
                    for (int k = 0; k < D; k++) {
                        Z[i][j] += L_next[i][k] * dW[k][j];
                    }
                }
            }
            for (int i = 0; i < D; i++) {
                for (int j = 0; j < D; j++) {
                    W_s[D*j + i] = W1[i][j];
                    for (int k = 0; k < D; k++) {
                        W_s[D*j + i] += Z[i][k] * L[j][k];
                    }
                }
            }
        }
        for (int i = 0; i < D; i++) {
            for (int j = 0; j < D; j++) {
                L_next[i][j] = L[i][j];
            }
        }
                
        mxSetCell(plhs[1], t, mxDuplicateArray(mut));
        mxSetCell(plhs[2], t, mxDuplicateArray(Vt));
        mxSetCell(plhs[3], t, mxDuplicateArray(Wt));   
    
    }
    
    mus.resize(boost::extents[1][1]);
    Vs.resize(boost::extents[1][1][1]);
    Phis.resize(boost::extents[1][1][1]);
    Ps.resize(boost::extents[1][1][1]);
    
}
