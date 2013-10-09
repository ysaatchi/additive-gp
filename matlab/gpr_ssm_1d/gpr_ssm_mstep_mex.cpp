//You can include any C libraries that you normally use
#include "mex.h"
#include "lapack.h"
#include "blas.h"
#include <math.h>
#include <vector>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <time.h>
#include <stdlib.h>
#include "boost/multi_array.hpp"
#define PI 3.14159265358979323846

using namespace std;

//helper method declarations
double spd_matrix_inverse(int D, const double* A, double* A_inv);
void sq_matrix_mult(char* op1, char* op2, int D, double* A, double* B, double* C);

//[ecdll, decdll] = gpr_ssm_mstep(loghyper, x, y, Ex, Vx, Exx, V0, Phis, Qs, derivs)

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    
    //Inputs
    if (nrhs != 10) {
        mexErrMsgTxt("Incorrect number of arguments (should be 10). Quitting...");
    }
    
    bool debug = false;
    
    int D = mxGetM(prhs[6]);
    
    double* logtheta = mxGetPr(prhs[0]);
    
    double* x = mxGetPr(prhs[1]);
    double* y = mxGetPr(prhs[2]);
    const mxArray* Ex = prhs[3]; //cell array of Expected Suff Stats
    const mxArray* Vx = prhs[4];
    const mxArray* Exx = prhs[5];
    double* V0 = mxGetPr(prhs[6]);
    const mxArray* Phis = prhs[7];
    const mxArray* Qs = prhs[8];
    const mxArray* derivs = prhs[9];
    
    //size checks
    if (mxGetM(prhs[0]) != 3 || (mxGetM(prhs[1]) != mxGetM(prhs[2]))) {
        mexErrMsgTxt("Check input argument sizes! Quitting...");
    }
    if (!(mxGetN(prhs[0]) == 1 && mxGetN(prhs[1]) == 1 && mxGetN(prhs[2]) == 1 && mxGetN(prhs[3]) == 1)) {
        mexErrMsgTxt("All vectors passed in must be column vectors! Quitting...");
    }
    //Outputs
    if (nlhs != 2) {
        mexErrMsgTxt("Invalid number of output arguments (should be 2). Quitting...");
    }
    
    plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(3, 1, mxREAL);
    double* ecdll = mxGetPr(plhs[0]);
    double* decdll = mxGetPr(plhs[1]);
    memset(decdll, 0, sizeof(double)*3);
    //MAIN CODE
    
    int T = mxGetM(prhs[1]);
    typedef boost::multi_array<double, 3> array_3D;
    typedef boost::multi_array<double, 2> array_2D;
    typedef std::vector<double> array_1D;
    
    double lambda = exp(logtheta[0]); //cov "decay" parameter
    double signal_var = exp(logtheta[1]);
    double noise_var = exp(logtheta[2]);
    //store filtering results
    array_1D mu(D);
    array_1D V(D*D);
    array_1D Phi(D*D);
    array_1D Q(D*D);
    array_1D Eii(D*D);
    array_1D Z(D*D);
    array_1D ZZ(D*D);
    double trace1;
    double trace2;
    double trace3;
    
    ecdll[0] = 0;
    
    array_1D V0_inv(D*D);
    double logdet = 0.0;
    logdet = spd_matrix_inverse(D, &V0[0], &V0_inv[0]);
    memcpy(&mu[0], mxGetPr(mxGetCell(Ex, 0)), sizeof(double)*D);
    memcpy(&V[0], mxGetPr(mxGetCell(Vx, 0)), sizeof(double)*D*D);
    for (int i = 0; i < D; i++) {
        for (int j = 0; j < D; j++) {
            Eii[j*D + i] = V[j*D + i] + mu[i]*mu[j];
        }
    }
    trace1 = 0.0;
    for (int i = 0; i < D; i++) {
        for (int k = 0; k < D; k++) {
            trace1 += V0_inv[k*D + i]*Eii[i*D + k];
        }
    }
    ecdll[0] += logdet + trace1;
    double term = logdet + trace1;
    ////cout << "trace1 = " << trace1 << endl;
    
    double* dV0;
    
    
    for (int hh = 0; hh < 2; hh++) {
        if (hh == 0) {
            dV0 = mxGetPr(mxGetField(mxGetCell(derivs, 0), 0, "dVdlambda"));    
        }
        if (hh == 1) {
            dV0 = mxGetPr(mxGetField(mxGetCell(derivs, 0), 0, "dVdSigvar"));
        }
        sq_matrix_mult("T", "N", D, dV0, &V0_inv[0], &Z[0]);
        sq_matrix_mult("N", "N", D, &V0_inv[0], &Z[0], &ZZ[0]);
        trace1 = 0.0;
        for (int i = 0; i < D; i++) {
            for (int k = 0; k < D; k++) {
                trace1 += V0_inv[k*D + i]*dV0[k*D + i];
            }
        }
        trace2 = 0.0;
        for (int i = 0; i < D; i++) {
            for (int k = 0; k < D; k++) {
                trace2 += ZZ[k*D + i]*Eii[i*D + k];
            }
        }
        decdll[hh] = trace1 - trace2;
    }
    ecdll[0] += log(noise_var) + ((y[0]*y[0]) - 2*(y[0]*mu[0]) + Eii[0])/noise_var;
    decdll[2] = 1/noise_var - ((y[0]*y[0])- 2*(y[0]*mu[0]) + Eii[0])/(noise_var*noise_var);
    
    for (int t = 1; t < T; t++) {
        
        ////cout << "t = " << t << endl;
        
        array_1D mu_prev(D);
        array_1D V_prev(D*D);
        array_1D W_prev(D*D);
        array_1D Eii_prev(D*D);
        array_1D Eadj(D*D);
        memcpy(&mu_prev[0], mxGetPr(mxGetCell(Ex, t-1)), sizeof(double)*D);
        memcpy(&V_prev[0], mxGetPr(mxGetCell(Vx, t-1)), sizeof(double)*D*D);
        memcpy(&W_prev[0], mxGetPr(mxGetCell(Exx, t-1)), sizeof(double)*D*D);
        memcpy(&mu[0], mxGetPr(mxGetCell(Ex, t)), sizeof(double)*D);
        memcpy(&V[0], mxGetPr(mxGetCell(Vx, t)), sizeof(double)*D*D);
        memcpy(&Phi[0], mxGetPr(mxGetCell(Phis, t-1)), sizeof(double)*D*D);
        memcpy(&Q[0], mxGetPr(mxGetCell(Qs, t-1)), sizeof(double)*D*D);
        for (int i = 0; i < D; i++) {
            for (int j = 0; j < D; j++) {
                Eii_prev[j*D + i] = V_prev[j*D + i] + mu_prev[i]*mu_prev[j];
            }
        }
        for (int i = 0; i < D; i++) {
            for (int j = 0; j < D; j++) {
                Eadj[j*D + i] = W_prev[j*D + i] + mu[i]*mu_prev[j];
            }
        }
        for (int i = 0; i < D; i++) {
            for (int j = 0; j < D; j++) {
                Eii[j*D + i] = V[j*D + i] + mu[i]*mu[j];
            }
        }
        array_1D Q_inv(D*D);
        //add jitter to Q first
        for (int i = 0; i < D; i++) {
            for (int j = 0; j < D; j++) {
                if (i == j) {
                    //Q[j*D + i] += 1e-10;
					Q[j*D + i] += 1e-4;
                }
            }
        }
        logdet = spd_matrix_inverse(D, &Q[0], &Q_inv[0]);
        
        //Ub = Ub + log(det(Q)) + trace(Q_inv*Eii) - 2*trace(Phi'*Q_inv*Eadj) + trace(Phi'*Q_inv*Phi*Eii_prev);
        trace1 = 0.0;
        for (int i = 0; i < D; i++) {
            for (int k = 0; k < D; k++) {
                trace1 += Q_inv[k*D + i]*Eii[i*D + k];
            }
        }
        trace2 = 0.0;
        sq_matrix_mult("T", "N", D, &Phi[0], &Q_inv[0], &Z[0]);
        for (int i = 0; i < D; i++) {
            for (int k = 0; k < D; k++) {
                trace2 += Z[k*D + i]*Eadj[i*D + k];
            }
        }
        sq_matrix_mult("N", "N", D, &Z[0], &Phi[0], &ZZ[0]);
        trace3 = 0.0;
        for (int i = 0; i < D; i++) {
            for (int k = 0; k < D; k++) {
                trace3 += ZZ[k*D + i]*Eii_prev[i*D + k];
            }
        }
        ecdll[0] += logdet + trace1 - 2*trace2 + trace3;        //Ub
        ////cout << "trace1 = " << trace1 << endl;
        ////cout << "trace2 = " << trace2 << endl;
        ////cout << "trace3 = " << trace3 << endl;
        ////cout << "ecdll term = " << term << endl;
        
        //A = (deriv.dPhidlambda'*Q_inv - Phi'*(Q_inv*deriv.dQdlambda*Q_inv));
        //dUb_lambda = dUb_lambda + trace(Q_inv*deriv.dQdlambda') - trace((Q_inv*deriv.dQdlambda'*Q_inv)*Eii) ...
        //- 2*trace(A*Eadj) + trace((Phi'*Q_inv*deriv.dPhidlambda + A*Phi)*Eii_prev);
        // A =  - Phi'*(Q_inv*deriv.dQdSigvar*Q_inv);
        // dUb_sigvar = dUb_sigvar + trace(Q_inv*deriv.dQdSigvar') - trace((Q_inv*deriv.dQdSigvar'*Q_inv)*Eii) ...
        // - 2*trace(A*Eadj) + trace((A*Phi)*Eii_prev);
        
        //derivs w.r.t. lambda
        double* dPhidlambda = mxGetPr(mxGetField(mxGetCell(derivs, t), 0, "dPhidlambda"));
        double* dQdlambda = mxGetPr(mxGetField(mxGetCell(derivs, t), 0, "dQdlambda"));
        array_1D A1(D*D);
        array_1D A2(D*D);
        array_1D A(D*D);
        sq_matrix_mult("T", "N", D, dPhidlambda, &Q_inv[0], &A1[0]);
        sq_matrix_mult("N", "N", D, dQdlambda, &Q_inv[0], &Z[0]);
        sq_matrix_mult("N", "N", D, &Q_inv[0], &Z[0], &ZZ[0]);
        sq_matrix_mult("T", "N", D, &Phi[0], &ZZ[0], &A2[0]);
        for (int i = 0; i < D*D; i++) {
            A[i] = A1[i] - A2[i];
            ////cout << "A[" << i << "] = " << A[i] << endl; 
        }
        trace1 = 0.0;
        for (int i = 0; i < D; i++) {
            for (int k = 0; k < D; k++) {
                trace1 += Q_inv[k*D + i]*dQdlambda[k*D + i];
            }
        }
        trace2 = 0.0;
        for (int i = 0; i < D; i++) {
            for (int k = 0; k < D; k++) {
                trace2 += ZZ[k*D + i]*Eii[i*D + k];
            }
        }
        trace3 = 0.0;
        for (int i = 0; i < D; i++) {
            for (int k = 0; k < D; k++) {
                trace3 += A[k*D + i]*Eadj[i*D + k];
            }
        }
        sq_matrix_mult("N", "N", D, &Q_inv[0], dPhidlambda, &Z[0]);
        sq_matrix_mult("T", "N", D, &Phi[0], &Z[0], &ZZ[0]);
        array_1D APhi(D*D);
        sq_matrix_mult("N", "N", D, &A[0], &Phi[0], &APhi[0]);
        double trace4 = 0.0;
        for (int i = 0; i < D; i++) {
            for (int k = 0; k < D; k++) {
                trace4 += (ZZ[k*D + i] + APhi[k*D + i])*Eii_prev[i*D + k];
            }
        }
        decdll[0] += trace1 - trace2 - 2*trace3 + trace4;       //dUb_lambda
        
        //derivs w.r.t. sigvar
        double* dQdSigvar = mxGetPr(mxGetField(mxGetCell(derivs, t), 0, "dQdSigvar"));
        sq_matrix_mult("N", "N", D, dQdSigvar, &Q_inv[0], &Z[0]);
        sq_matrix_mult("N", "N", D, &Q_inv[0], &Z[0], &ZZ[0]);
        sq_matrix_mult("T", "N", D, &Phi[0], &ZZ[0], &A2[0]);
        for (int i = 0; i < D*D; i++) {
            A[i] = - A2[i];
        }
        trace1 = 0.0;
        for (int i = 0; i < D; i++) {
            for (int k = 0; k < D; k++) {
                trace1 += Q_inv[k*D + i]*dQdSigvar[k*D + i];
            }
        }
        trace2 = 0.0;
        for (int i = 0; i < D; i++) {
            for (int k = 0; k < D; k++) {
                trace2 += ZZ[k*D + i]*Eii[i*D + k];
            }
        }
        trace3 = 0.0;
        for (int i = 0; i < D; i++) {
            for (int k = 0; k < D; k++) {
                trace3 += A[k*D + i]*Eadj[i*D + k];
            }
        }
        sq_matrix_mult("N", "N", D, &A[0], &Phi[0], &APhi[0]);
        trace4 = 0.0;
        for (int i = 0; i < D; i++) {
            for (int k = 0; k < D; k++) {
                trace4 += APhi[k*D + i]*Eii_prev[i*D + k];
            }
        }
        decdll[1] += trace1 - trace2 - 2*trace3 + trace4;               //dUb_sigvar
        
        //noise
        //EG  //ecdll[0] += log(noise_var) + ((y[t]*y[t]) - 2*(y[t]*mu[0]) + Eii[0])/noise_var;     //necdll
        // term = log(noise_var) + ((y[t]*y[t]) - 2*(y[t]*mu[0]) + Eii[0])/noise_var;       //EG , term is not used after
        ////cout << "ecdll term = " << term << endl;
        decdll[2] += 1/noise_var - ((y[t]*y[t])- 2*(y[t]*mu[0]) + Eii[0])/(noise_var*noise_var);    //???
        
    }
    
    for (int i = 0; i < 3; i++) {
        decdll[i] *= exp(logtheta[i]);      //dnecdll
    }
    
}

double spd_matrix_inverse(int D, const double* A, double* A_inv) {
    typedef std::vector<double> array_1D;
    ////cout << "Start" << endl;
    array_1D B(D*D);
    memcpy(&B[0], A, sizeof(double)*D*D);
    mwSignedIndex info;
    mwSignedIndex D_lp = (mwSignedIndex) D;
    char UorL = 'L';
    dpotrf(&UorL, &D_lp, &B[0], &D_lp, &info);
    if (info != 0){
        //cout << "WARNING from spd_matrix_inverse: Matrix is probably not positive definite!" << endl;
    }
    double ld = 0.0;
    
    ////cout << "D = " << D << endl;
    for (int i = 0; i < D; i++) {
        
        ld += log(B[i*D + i]);
        
    }
    ld *= 2.0;
    
    ////cout << "Mid" << endl;
    dpotri(&UorL, &D_lp, &B[0], &D_lp, &info);
    for (int i = 0; i < D; i++) {
        for (int j = 0; j <= i; j++) {
            A_inv[j*D + i] = B[j * D + i];
            if (j != i) {
                A_inv[i*D + j] = A_inv[j*D + i];
            }
        }
    }
    
    ////cout << "End" << endl;
    return ld;
    
}

void sq_matrix_mult(char* op1, char* op2, int D, double* A, double* B, double* C) {
    
    double one = 1.0;
    double zero = 0.0;
    mwSignedIndex D_lp = (mwSignedIndex) D;
    /* Pass arguments to Fortran by reference */
    dgemm(op1, op2, &D_lp, &D_lp, &D_lp, &one, A, &D_lp, B, &D_lp, &zero, C, &D_lp);
    
}