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
#include "boost/multi_array.hpp"
#include "boost/numeric/ublas/matrix.hpp"
#include "boost/numeric/ublas/io.hpp"
#define PI 3.14159265358979323846

using namespace std;
typedef boost::multi_array<double, 3> array_3D;
typedef boost::multi_array<double, 2> array_2D;
typedef std::vector<double> array_1D;
    
void matrixMatrixProd(array_2D &A,array_2D &B,array_2D &Z,int isize, int jsize, int ksize)
{
     for (int i=0; i<isize; i++)
     	for (int j=0; j<jsize; j++)
        	for(int k=0; i<ksize; k++)
                	Z[i][k] += (A[i][j] * B[j][k]);
}

void matrixVectorProd(array_2D &A,array_2D &B,array_2D &Z,int isize, int jsize, int ksize)
{
     for (int i=0; i<isize; i++)
     	for (int j=0; j<jsize; j++)
        	for(int k=0; i<ksize; k++)
                	Z[i][k] += (A[i][j] * B[j][k]);
}

// void xtimesy(double x, double *y, double *z, size_t m, size_t n)
// {
//   mwSize i,j,count=0;
//   
//   for (i=0; i<n; i++) {
//     for (j=0; j<m; j++) {
//       *(z+count) = x * *(y+count);
//       count++;
//     }
//   }
// }

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

//[nlml, dnlml_dW, dnlml_dl, dnlml_ds, dnlml_dn] = gpr_pgr_1d_mex([lambda;signal_var;noise_var], t, y, V0, Phis, Qs, mu0, deriv0, derivs, wgt);

    //Inputs
    if (nrhs != 10) {
        mexErrMsgTxt("Incorrect number of arguments (should be 10). Quitting...");
    }
    
    bool debug = false;
    
    // get the number of rows in V0
    int D = mxGetM(prhs[3]);
    int W = mxGetM(prhs[9]);
    
    using namespace boost::numeric::ublas;
    matrix<double> m (D, D);
    
    double* theta = mxGetPr(prhs[0]);
    double* x = mxGetPr(prhs[1]);
    double* y = mxGetPr(prhs[2]);
    double* V0 = mxGetPr(prhs[3]);
    const mxArray* Phi_cell = prhs[4];
    const mxArray* Q_cell = prhs[5];
    double* mu0 = mxGetPr(prhs[6]);
    const mxArray* deriv0 = prhs[7];
    const mxArray* deriv_cell = prhs[8];
    
    //deriv0. structure
    double *dVdSigvar = mxGetPr(mxGetFieldByNumber(deriv0, 0, 0));
    double *dVdlambda = mxGetPr(mxGetFieldByNumber(deriv0, 0, 1));
    
    //deriv. structure
    double *dPhidlambda = mxGetPr(mxGetFieldByNumber(deriv_cell, 0, 0));
    double *dQdlambda = mxGetPr(mxGetFieldByNumber(deriv_cell, 0, 1));
    double *dQdSigvar = mxGetPr(mxGetFieldByNumber(deriv_cell, 0, 2));
    double *dPhidW = mxGetPr(mxGetFieldByNumber(deriv_cell, 0, 3));  
    double *dQdW = mxGetPr(mxGetFieldByNumber(deriv_cell, 0, 4));  
    
    
    
    //size checks
    if (mxGetM(prhs[0]) != 3 || (mxGetM(prhs[1]) != mxGetM(prhs[2]))) {
        mexErrMsgTxt("Check input argument sizes! Quitting...");
    }
    if (!(mxGetN(prhs[0]) == 1 && mxGetN(prhs[1]) == 1 && mxGetN(prhs[2]) == 1)) {
        mexErrMsgTxt("All vectors passed in must be column vectors! Quitting...");
    }
    //Outputs
    if (nlhs != 5) {
        mexErrMsgTxt("Invalid number of output arguments (should be 5). Quitting...");
    }
    int T = mxGetM(prhs[1]);
    
    plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(W, 1, mxREAL);
    plhs[2] = mxCreateDoubleMatrix(1, 1, mxREAL);  //need to check size!!
    plhs[3] = mxCreateDoubleMatrix(1, 1, mxREAL);
    plhs[4] = mxCreateDoubleMatrix(1, 1, mxREAL);
   
    double* nlml = mxGetPr(plhs[0]);
    double* dnlml_dW = mxGetPr(plhs[1]);
    double* dnlml_dl = mxGetPr(plhs[2]);
    double* dnlml_ds = mxGetPr(plhs[3]);
    double* dnlml_dn = mxGetPr(plhs[4]);
            
    //MAIN CODE
   
    double lambda = (theta[0]); //cov "decay" parameter
    double signal_var = (theta[1]);
    double noise_var = (theta[2]);
    
    
    array_1D kalman_gain(D);
    array_1D mu(D);
    array_2D V(boost::extents[D][D]);
    array_1D PhiMu(D);
    array_2D Phi(boost::extents[D][D]);
    array_2D Q(boost::extents[D][D]);
    array_2D Z(boost::extents[D][D]);
    array_2D P(boost::extents[D][D]);
    
    array_1D dmu_dl(D);
    array_1D dmu_ds(D);
    array_1D dmu_dn(D);
    array_1D dg_dl(D);
    array_1D dg_ds(D);
    array_1D dg_dn(D);
      
    
    //output variables
    nlml[0] = 0;
    dnlml_dW[0] = 0;
    dnlml_dl[0]  = 0;
    dnlml_ds[0]  = 0;
    dnlml_dn[0]  = 0;
    
    /* BEGIN KALMAN FILTERING */
    //cout << "Forward filtering..." << endl;
    //absorb first observation
    int t = 0;
    double pm = mu0[0];
    double pv = V0[0] + noise_var;
    // dpv_dl = deriv.dVdlambda(1,1);
    double dpv_dl = dVdlambda[0];
    //dpv_ds = deriv.dVdSigvar(1,1);
    double dpv_ds = dVdSigvar[0];
    //dpv_dn = 1;
    double dpv_dn = 1;
    //nlml = 0.5*(log(2*pi) + log(pv) + ((y(1) - pm)^2)/pv);
    nlml[0] += 0.5 * (log(2 * PI) + log(pv) + (pow((y[t] - pm), 2) / pv));
    // dnlml_dpv = 0.5*(1/pv - ((y(1) - pm)^2)/(pv*pv));
    double dnlml_dpv = 0.5*(1/pv - (pow((y[t] - pm), 2)/(pv*pv)));
    dnlml_dl[0] += dnlml_dpv*dpv_dl;
    dnlml_ds[0] += dnlml_dpv*dpv_ds;
    dnlml_dn[0] += dnlml_dpv*dpv_dn;
    
    // kalman_gain = (V0*H')/pv;
    // dg_dl = (pv*(deriv.dVdlambda*H') - dpv_dl*(V0*H'))/(pv*pv);
    // dg_ds = (pv*(deriv.dVdSigvar*H') - dpv_ds*(V0*H'))/(pv*pv);
    // dg_dn = -(V0*H')/(pv*pv);
    // mu = mu0 + kalman_gain*(y(1) - pm);
    // dmu_dl = dg_dl*(y(1) - pm);
    // dmu_ds = dg_ds*(y(1) - pm);
    // dmu_dn = dg_dn*(y(1) - pm);
    double kgd,dVld,dVsd;
    for (int d = 0; d < D; d++) {
         kgd = (V0[d]) / pv;
         dVld = (dVdlambda[d]);
         dVsd = (dVdSigvar[d]);
        
        kalman_gain[d] = kgd;
        dg_dl[d] = (pv*(dVld) - dpv_dl*(kgd)/pv);
        dg_ds[d] = (pv*(dVsd) - dpv_ds*(kgd)/pv);
        dg_dn[d] = -(kgd)/(pv);
        mu[d] = mu0[d]+kgd * (y[t] - pm);
        dmu_dl[d] = dg_dl[d]*(y[t] - pm);
    	dmu_ds[d] = dg_ds[d]*(y[t] - pm);
    	dmu_dn[d] = dg_dn[d]*(y[t] - pm);
    }
    
    //dmu_dW = zeros(p,D);
    array_2D dmu_dW(boost::extents[D][W]);
    for (int i = 0; i < D; i++) {
        for (int j = 0; j < W; j++) {
            dmu_dW[i][j] = 0;
        }
    }
    
    //V = (eye(p) - kalman_gain*H)*V0;
    for (int i = 0; i < D; i++) {
        for (int j = 0; j < D; j++) {
            V[i][j] = V0[j*D + i] - kalman_gain[i]*(V0[j*D]);
        }
    }
    
    
    //dV_dl = deriv.dVdlambda - ((kalman_gain*H)*deriv.dVdlambda + (dg_dl*H)*V0);
    //dV_ds = deriv.dVdSigvar - ((kalman_gain*H)*deriv.dVdSigvar + (dg_ds*H)*V0);
    //dV_dn = - (dg_dn*H)*V0;
    for (int d = 0; d < D; d++) {
         kgd = (V0[d]) / pv;
         dVld = (dVdlambda[d]);
         dVsd = (dVdSigvar[d]);
        
         //dV_dl[d] = dVdlambda - ((kalman_gain[d])*dVdlambda + (dg_dl[d])*V0);
        
    }

    //dV_dW = zeros(p,p,D); D=p, W=D
    array_3D dV_dW(boost::extents[D][D][W]);
    //dP_dW = zeros(p,p,D);
    array_3D dP_dW(boost::extents[D][D][W]);
    //dPhiMu_dW = zeros(p,D);
    array_2D dPhiMu_dW(boost::extents[D][W]);
    //dg_dW = zeros(p,D);
    array_2D dg_dW(boost::extents[D][W]);
    
    //dnlml_dW = zeros(D,1); already done by mxCreateDoubleMatrix
    
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
        pv = P[0][0] + noise_var;
        nlml[0] += 0.5 * (log(2 * PI) + log(pv) + (pow((y[t] - pm), 2) / pv));
        //nsse[0] += 0.5 * (pow((y[t] - pm), 2) / pv);
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
       
}
