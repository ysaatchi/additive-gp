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
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/operation.hpp>
#define PI 3.14159265358979323846

// using namespace std;
// typedef boost::multi_array<double, 3> matrix<double>;
// typedef boost::multi_array<double, 2> matrix<double>;
// typedef std::vector<double> vector<double>;

using namespace boost::numeric::ublas;
typedef boost::numeric::ublas::matrix<double> matrix_type;

typedef boost::numeric::ublas::vector<matrix_type> mat_vec_type;


// void matrixMatrixProd(matrix_type &A,matrix_type &B,matrix_type &Z,int isize, int jsize, int ksize)
// {
//      for (int i=0; i<isize; i++)
//      	for (int j=0; j<jsize; j++)
//         	for(int k=0; i<ksize; k++)
//                 	Z[i][k] += (A[i][j] * B[j][k]);
// }
//
// void matrixVectorProd(matrix_type &A,matrix_type &B,matrix_type &Z,int isize, int jsize, int ksize)
// {
//      for (int i=0; i<isize; i++)
//      	for (int j=0; j<jsize; j++)
//         	for(int k=0; i<ksize; k++)
//                 	Z[i][k] += (A[i][j] * B[j][k]);
// }

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
void array2vector(matrix_type &vec, double* arry, int size)
{
    for(int i=0;i<size;i++)
        vec(i,0) = arry[i];
    
}

matrix_type array2vector(double* arry, int size)
{
    matrix_type vec(size,1);
    for(int i=0;i<size;i++)
        vec(i,0) = arry[i];
    return vec;
}

void array2matrix(matrix_type &mat, double* arry, int sizeI, int sizeJ)
{
    for(int i=0;i<sizeI;i++)
        for(int j=0;j<sizeJ;j++)
            mat(i,j) = arry[i+j*sizeI];
}


// void array2matrix3(matrix_type &mat, double* arry, int sizeI, int sizeJ, int sizeK)
// {
//     //matrix_type mat1(1,1,1);
//     //mat1(0,0,0) = 0;
//     for(int i=0;i<sizeI;i++)
//         for(int j=0;j<sizeJ;j++)
//             for(int k=0;k<sizeK;k++)
//                 mat(i,j,k) = arry[i+j*sizeI+k*sizeI*sizeJ];
//
// }

void zeroVector(matrix_type &vec)
{
    for(int i=0;i<vec.size1();i++)
        vec(i,0) = 0;
}

void zeroMatrix(matrix_type &mat, int sizeI, int sizeJ)
{
    for(int i=0;i<sizeI;i++)
        for(int j=0;j<sizeJ;j++)
            mat(i,j) =0;
//     else
//     {
//         for(int i=0;i<sizeI;i++)
//             for(int j=0;j<sizeJ;j++)
//                 for(int k=0;k<sizeK;k++)
//                     mat(i,j,k) = 0;
//     }
}

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
    
    
    
    double* theta = mxGetPr(prhs[0]);
    double* x = mxGetPr(prhs[1]);
    double* y = mxGetPr(prhs[2]);
    matrix_type V0(D,D);
    array2matrix(V0, mxGetPr(prhs[3]), D, D);
    const mxArray* Phi_cell = prhs[4];
    const mxArray* Q_cell = prhs[5];
    double* mu0 = mxGetPr(prhs[6]);
    const mxArray* deriv0 = prhs[7];
    const mxArray* deriv_cell = prhs[8];
    
    matrix_type mu0vec(D,1);
    array2vector(mu0vec, mu0, D);
    
    // allocate temp matrices for within calcuations
    matrix_type tMDD(D,D);
    matrix_type tMDW(D,W);
    matrix_type tMWD(W,D);
    matrix_type tMWW(W,W);
    matrix_type tVD(D,1);
     matrix_type tVD2(D,1);
    matrix_type tVW(W,1);
    matrix_type tVW2(W,1);
    
    
    //deriv0. structure
    double *dVdSigvar = mxGetPr(mxGetFieldByNumber(deriv0, 0, 0));
    matrix_type dVdSigvarMat(D,D);
    array2matrix(dVdSigvarMat, dVdSigvar, D, D);
    //printMyMat(dVdSigvarMat,D,D,"dVdSigvarMat");
    
    double *dVdlambda = mxGetPr(mxGetFieldByNumber(deriv0, 0, 1));
    matrix_type dVdlambdaMat(D,D);
    array2matrix(dVdlambdaMat, dVdlambda, D, D);
    //printMyMat(dVdlambdaMat,D,D,"dVdlambdaMat");
    
    //deriv. structure
    double *dPhidlambda,*dQdlambda,*dQdSigvar,*dPhidW,*dQdW;
    
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
    
    
    matrix_type kalman_gain(D,1);
    matrix_type mu(D,1);
    matrix_type V(D,D);
    matrix_type PhiMu(D,1);
    matrix_type Phi(D,D);
    matrix_type Q(D,D);
    matrix_type Z(D,D);
    matrix_type P(D,D);
    
    matrix_type dmu_dl(D,1);
    matrix_type dmu_ds(D,1);
    matrix_type dmu_dn(D,1);
    matrix_type dg_dl(D,1);
    matrix_type dg_ds(D,1);
    matrix_type dg_dn(D,1);
    
    
    //output variables
    nlml[0] = 0;
    for(int k=0;k<W;k++){
        dnlml_dW[k] = 0;
    }
    dnlml_dl[0]  = 0;
    dnlml_ds[0]  = 0;
    dnlml_dn[0]  = 0;
    
    /* BEGIN KALMAN FILTERING */
    //cout << "Forward filtering..." << endl;
    //absorb first observation
    int t = 0;
    double pm = mu0[0];
    double pv = V0(0,0) + noise_var;
    // dpv_dl = deriv.dVdlambda(1,1);
    double dpv_dl = dVdlambdaMat(0,0);
    //dpv_ds = deriv.dVdSigvar(1,1);
    double dpv_ds = dVdSigvarMat(0,0);
    //dpv_dn = 1;
    double dpv_dn = 1;
    //nlml = 0.5*(log(2*pi) + log(pv) + ((y(1) - pm)^2)/pv);
    nlml[0] += 0.5 * (log(2 * PI) + log(pv) + (pow((y[t] - pm), 2) / pv));
    // dnlml_dpv = 0.5*(1/pv - ((y(1) - pm)^2)/(pv*pv));
    double dnlml_dpv = 0.5*(1/pv - (pow((y[t] - pm), 2)/(pv*pv)));
    dnlml_dl[0] += dnlml_dpv*dpv_dl;
    dnlml_ds[0] += dnlml_dpv*dpv_ds;
    dnlml_dn[0] += dnlml_dpv*dpv_dn;
    //printMyNum(dnlml_dl[0],"dnlml_dl");
    //printMyNum(dnlml_ds[0],"dnlml_ds");
    //printMyNum(dnlml_dn[0],"dnlml_dn");
    
    matrix_type H(D,1);
    zeroVector(H);
    H(0,0) = 1;
    
    // kalman_gain = (V0*H")/pv;
    // dg_dl = (pv*(deriv.dVdlambda*H') - dpv_dl*(V0*H'))/(pv*pv);
    // dg_ds = (pv*(deriv.dVdSigvar*H') - dpv_ds*(V0*H'))/(pv*pv);
    // dg_dn = -(V0*H')/(pv*pv);
    // mu = mu0 + kalman_gain*(y(1) - pm);
    // dmu_dl = dg_dl*(y(1) - pm);
    // dmu_ds = dg_ds*(y(1) - pm);
    // dmu_dn = dg_dn*(y(1) - pm);
    double kgd,dVld,dVsd;
//     for (int d = 0; d < D; d++) {
//         kgd = (V0(d,0)) / pv;
//         dVld = (dVdlambda[d]);
//         dVsd = (dVdSigvar[d]);
//
//         kalman_gain[d] = kgd;
//         dg_dl[d] = (pv*(dVld) - dpv_dl*(kgd)/pv);
//         dg_ds[d] = (pv*(dVsd) - dpv_ds*(kgd)/pv);
//         dg_dn[d] = -(kgd)/(pv);
//         mu[d] = mu0[d]+kgd * (y[t] - pm);
//         dmu_dl[d] = dg_dl[d]*(y[t] - pm);
//         dmu_ds[d] = dg_ds[d]*(y[t] - pm);
//         dmu_dn[d] = dg_dn[d]*(y[t] - pm);
//     }
    matrix_type V0H(D,1);
    axpy_prod(V0,H,V0H,true);
    kalman_gain = V0H/pv;
    axpy_prod(dVdlambdaMat,H,tVD,true);
    dg_dl =(pv*tVD-dpv_dl*V0H)/(pv*pv);
    //dg_dl = (pv*prod(dVdlambdaMat,H) - dpv_dl*prod(V0,H))/(pv*pv);
    axpy_prod(dVdSigvarMat,H,tVD,true);
    dg_ds =(pv*tVD-dpv_ds*V0H)/(pv*pv);
    // dg_ds = (pv*prod(dVdSigvarMat,H) - dpv_ds*prod(V0,H))/(pv*pv);
    dg_dn=-V0H/(pv*pv);
    //dg_dn = -prod(V0,H)/(pv*pv);
    
    mu = mu0vec + kalman_gain*(y[0] - pm);
    dmu_dl = dg_dl*(y[0] - pm);
    dmu_ds = dg_ds*(y[0] - pm);
    dmu_dn = dg_dn*(y[0] - pm);
    
    //printMyVec(kalman_gain,D,"kalman_gain");
    //printMyVec(dg_dl,D,"dg_dl");
    //printMyVec(dg_dn,D,"dg_dn");
    //printMyVec(mu,D,"mu");
    //printMyVec(dmu_dl,D,"dmu_dl");
    //printMyVec(dmu_ds,D,"dmu_ds");
    //printMyVec(dmu_dn,D,"dmu_dn");
    
    //dmu_dW = zeros(p,D);
    matrix_type dmu_dW(D,W);
    zeroMatrix(dmu_dW,D,W);
    //printMyMat(dmu_dW,D,W,"dmu_dW");
    
    matrix_type kH(D,D);
    axpy_prod(kalman_gain, trans(H),kH,true);
    
    //V = (eye(p) - kalman_gain*H)*V0;
    matrix_type kHV0(D,D);
    axpy_prod(kH,V0,kHV0,true);
    V = V0 - kHV0;
    //printMyMat(V,D,D,"V");
    
    
    //dV_dl = deriv.dVdlambda - ((kalman_gain*H)*deriv.dVdlambda + (dg_dl*H)*V0);
    axpy_prod(kH,dVdlambdaMat,tMDD,true);    //tMDD=(kalman_gain*H)*deriv.dVdlambda
//matrix_type kHl = prod(kH,dVdlambdaMat);
    matrix_type dgdlH(D,D);
    axpy_prod(dg_dl, trans(H),dgdlH,true);   //dgdlH=dg_dl*H
    axpy_prod(dgdlH,V0,tMDD,false);     //tMDD += dgdlH*V0
    //matrix_type dgdlHV0 = prod(dgdlH,V0);
    matrix_type dV_dl = dVdlambdaMat -tMDD; //dV_dl = deriv.dVdlambda - tMDD
//matrix_type dV_dl = dVdlambdaMat - (kHl + dgdlHV0);
    //printMyMat(dV_dl,D,D,"dV_dl");
    
    //dV_ds = deriv.dVdSigvar - ((kalman_gain*H)*deriv.dVdSigvar + (dg_ds*H)*V0);
    axpy_prod(kH,dVdSigvarMat,tMDD,true);     //tMDD=(kalman_gain*H)*deriv.dVdSigvar
//matrix_type kHs = prod(kH,dVdSigvarMat);
    matrix_type dgdsH(D,D);
    axpy_prod(dg_ds, trans(H),dgdsH,true);   //dgdsH=dg_ds*H
    axpy_prod(dgdsH,V0,tMDD,false);     //tMDD += dgdsH*V0
    //matrix_type dgdsHV0 = prod(dgdsH,V0);
    matrix_type dV_ds = dVdSigvarMat - tMDD; //dV_ds = deriv.dVdSigvarMat - tMDD
    //printMyMat(dV_ds,D,D,"dV_ds");
    
    //dV_dn = - (dg_dn*H)*V0;
    matrix_type dgdnH(D,D);
    axpy_prod(dg_dn, trans(H),dgdnH,true);   //dgdnH=dg_dn*H
    axpy_prod(dgdnH,V0,tMDD,true);  //tMDD = dgdnH*V0
    matrix_type dV_dn = - tMDD;
    //printMyMat(dV_dn,D,D,"dV_dn");
    
    //dV_dW = zeros(p,p,D); D=p, W=D
    //matrix_type dV_dW(D,D,W);
    //zeroMatrix(dV_dW,D,D,W);
    
    mat_vec_type dV_dW(W);
    
    //dP_dW = zeros(p,p,D);
    mat_vec_type dP_dW(W);
    
    for(int i=0;i<W;i++){
        matrix_type dV_dW_i(D, D);
        zeroMatrix(dV_dW_i,D,D);
        dV_dW(i) = dV_dW_i;
        //char name[50];
        //sprintf (name, "dV_dW_%d", i);
        //printMyMat(dV_dW_i,D,D,name);
        matrix_type dP_dW_i(D, D);
        zeroMatrix(dP_dW_i,D,D);
        dP_dW(i) = dP_dW_i;
    }
    
    
    
    //zeroMatrix(dP_dW,D,D,W);
    //dPhiMu_dW = zeros(p,D);
    matrix_type dPhiMu_dW(D,W);
    zeroMatrix(dPhiMu_dW,D,W);
    
    //dg_dW = zeros(p,D);
    matrix_type dg_dW(D,W);
    zeroMatrix(dg_dW,D,W);
    
    
    //filter forward in "time"
    
//     matrix_type Phi_v(D*D);
//     matrix_type Q_v(D*D);
    matrix_type deriv_dPhidlambda(D,D);
    matrix_type deriv_dQdlambda(D,D);
    matrix_type deriv_dQdSigvar(D,D);
    mat_vec_type deriv_dPhidW(W);
    mat_vec_type deriv_dQdW(W);
    matrix_type dP_dl(D,D);
    matrix_type dP_ds(D,D);
    matrix_type dP_dn(D,D);
    matrix_type dPhiMu_dl(D,1);
    matrix_type dPhiMu_ds(D,1);
    matrix_type dPhiMu_dn(D,1);
    matrix_type PH(D,1);
    
//     for (t = 1; t < T; t++) {
    for (t = 1; t < T; t++) {
        
        //memcpy(&Phi_v[0], mxGetPr(mxGetCell(Phi_cell, t-1)), sizeof(double)*D*D);
        //array2matrix(Phi, Phi_v, D, D);
        array2matrix(Phi, mxGetPr(mxGetCell(Phi_cell, t-1)), D, D);
        //printMyMat(Phi,D,D,"Phi");
        
        
        //memcpy(&Q_v[0], mxGetPr(mxGetCell(Q_cell, t-1)), sizeof(double)*D*D);
        //array2matrix(Q, Q_v, D, D);
        array2matrix(Q, mxGetPr(mxGetCell(Q_cell, t-1)), D, D);
        //printMyMat(Q,D,D,"Q");
        
//         mxArray *pa = mxGetCell(Q_cell, t-1);
//         mexPrintf("\t\t%s\t\t\n", mxGetClassName(pa));
        ////////////// DERIV STRUCTURE
        
        //deriv_dPhidlambda(D,D) = deriv.dPhidlambda
//         dPhidlambda = mxGetPr(mxGetFieldByNumber(deriv_cell, t-1, 0));
//         dPhidlambda = mxGetPr(mxGetFieldByNumber(deriv_cell, t-1, 0));
//         array2matrix(deriv_dPhidlambda, dPhidlambda, D, D);
//         //printMyMat(deriv_dPhidlambda,D,D,"deriv_dPhidlambda");
        
        mxArray *derive_cell_i = mxGetCell(deriv_cell, t-1);
        dPhidlambda = mxGetPr(mxGetFieldByNumber(derive_cell_i, 0, 0));
        array2matrix(deriv_dPhidlambda, dPhidlambda, D, D);
        //printMyMat(deriv_dPhidlambda,D,D,"deriv_dPhidlambda");
        
//         pa = mxGetFieldByNumber(derive_cell_i, 0, 0);
//         mexPrintf("\t\t%s\t\t\n", mxGetClassName(pa));
        
        //deriv_dQdlambda(D,D) = deriv.dQdlambda
        dQdlambda = mxGetPr(mxGetFieldByNumber(derive_cell_i, 0, 1));
        array2matrix(deriv_dQdlambda, dQdlambda, D, D);
        //printMyMat(deriv_dQdlambda,D,D,"deriv_dQdlambda");
        
        //deriv_dQdSigvar(D,D) = deriv.dQdSigvar
        dQdSigvar = mxGetPr(mxGetFieldByNumber(derive_cell_i, 0, 2));
        array2matrix(deriv_dQdSigvar, dQdSigvar, D, D);
        //printMyMat(deriv_dQdSigvar,D,D,"deriv_dQdSigvar");
        
        for(int k=0;k<W;k++) {
            //deriv_dPhidW(D,D,W) = deriv.dPhidW
            dPhidW = mxGetPr(mxGetFieldByNumber(derive_cell_i, 0, 3))+k*D*D;
            matrix_type dPhidWMat(D,D);
            array2matrix(dPhidWMat, dPhidW, D, D);
            deriv_dPhidW[k] = dPhidWMat;
            
            //char name[50];
            //sprintf (name, "deriv_dPhidW_%d", k);
            //printMyMat(deriv_dPhidW[k],D,D,name);
            
            //deriv_dQdW(D,D,W) = deriv.dQdW
            dQdW = mxGetPr(mxGetFieldByNumber(derive_cell_i, 0, 4))+k*D*D;
            matrix_type dQdWMat(D,D);
            array2matrix(dQdWMat, dQdW, D, D);
            deriv_dQdW[k] = dQdWMat;
            //sprintf (name, "deriv_dQdW_%d", k);
            //printMyMat(deriv_dQdW[k],D,D,name);
        }
        
        
        // P = Phi*V*Phi' + Q;
        matrix_type tPhi = trans(Phi);
        axpy_prod(V,tPhi,tMDD,true);  //tMDD = V*Phi'
        //matrix_type VPhi = prod(V,tPhi);
        P=Q;
        axpy_prod(Phi,tMDD,P,false);  //P += Phi*V*Phi'
        //P = prod(Phi,VPhi)+Q;
        //printMyMat(P,D,D,"P");
        
        // dP_dl = Phi*V*deriv.dPhidlambda' + (Phi*dV_dl + deriv.dPhidlambda*V)*Phi' + deriv.dQdlambda;
        axpy_prod(Phi,V,tMDD,true);    //tMDD=Phi*V
        axpy_prod(tMDD,trans(deriv_dPhidlambda),dP_dl,true);   //dP_dl= Phi*V*deriv.dPhidlambda'
        axpy_prod(Phi,dV_dl,tMDD,true);    //tMDD=Phi*dV_dl
        axpy_prod(deriv_dPhidlambda,V,tMDD,false);    //tMDD+=deriv.dPhidlambda*V
        axpy_prod(tMDD,tPhi,dP_dl,false);    //dP_dl+=tMDD*Phi'
        dP_dl+=deriv_dQdlambda;
        //printMyMat(dP_dl,D,D,"dP_dl");
        
        // dP_ds = Phi*dV_ds*Phi' + deriv.dQdSigvar;
        axpy_prod(Phi,dV_ds,tMDD,true);    //tMDD=Phi*V
        axpy_prod(tMDD,tPhi,dP_ds,true);   //dP_ds = Phi*dV_ds*Phi'
        dP_ds+=deriv_dQdSigvar;
        //printMyMat(dP_ds,D,D,"dP_ds");
        
        //dP_dn = Phi*dV_dn*Phi';
        axpy_prod(Phi,dV_dn,tMDD,true);    //tMDD=Phi*V
        axpy_prod(tMDD,tPhi,dP_dn,true);   //dP_dn = Phi*dV_dn*Phi'
        //printMyMat(dP_dn,D,D,"dP_dn");
        
//     for d = 1:D
//         dP_dW(:,:,d) = Phi*V*deriv.dPhidW(:,:,d)' + ...
//             (Phi*dV_dW(:,:,d) + deriv.dPhidW(:,:,d)*V)*Phi' + deriv.dQdW(:,:,d);
//     end
        matrix_type PhiV(D,D);
        axpy_prod(Phi,V,PhiV,true); //PhiV=Phi*V
        for(int k=0;k<W;k++){
            tMDD = trans(deriv_dPhidW[k]);
            axpy_prod(PhiV,tMDD,dP_dW[k],true);  //dP_dW[k]=Phi*V*deriv.dPhidW(:,:,d)'
            axpy_prod(Phi,dV_dW[k],tMDD,true);    //tMDD=Phi*dV_dW(:,:,d)
            axpy_prod(deriv_dPhidW[k],V,tMDD,false);   //tMDD +=deriv.dPhidW(:,:,d)*V
            axpy_prod(tMDD,tPhi,dP_dW[k],false);        //dP_dW[k] += tMDD*Phi'
            dP_dW[k] += deriv_dQdW[k];
            
            //char name[50];
            //sprintf (name, "dP_dW_%d", k);
            //printMyMat(dP_dW[k],D,D,name);
        }
        
//     PhiMu = Phi*mu;
        axpy_prod(Phi,mu,PhiMu,true);
        //printMyVec(PhiMu,D,"PhiMu");
//     dPhiMu_dl = deriv.dPhidlambda*mu + Phi*dmu_dl;
        axpy_prod(deriv_dPhidlambda,mu,dPhiMu_dl,true);  //dPhiMu_dl = deriv.dPhidlambda*mu
        axpy_prod(Phi,dmu_dl,dPhiMu_dl,false);       //dPhiMu_dl +=  Phi*dmu_dl;
        //printMyVec(dPhiMu_dl,D,"dPhiMu_dl");
//     dPhiMu_ds = Phi*dmu_ds;
        axpy_prod(Phi,dmu_ds,dPhiMu_ds,true);
        
        //printMyVec(dPhiMu_ds,D,"dPhiMu_ds");
//     dPhiMu_dn = Phi*dmu_dn;
        axpy_prod(Phi,dmu_dn,dPhiMu_dn,true);
        //printMyVec(dPhiMu_dn,D,"dPhiMu_dn");
//     for d = 1:D
//         dPhiMu_dW(:,d) = deriv.dPhidW(:,:,d)*mu + Phi*dmu_dW(:,d);
//     end
        for(int j=0;j<W;j++){
            //axpy_prod(deriv_dPhidW[j],mu,column(dPhiMu_dW, j),true); //dPhiMu_dW(:,d) = deriv.dPhidW(:,:,d)*mu
            axpy_prod(deriv_dPhidW[j],mu,tVD,true); //dPhiMu_dW(:,d) = deriv.dPhidW(:,:,d)*mu
            //axpy_prod(Phi,column(dmu_dW, j),column(dPhiMu_dW, j),false); //dPhiMu_dW(:,d) += Phi*dmu_dW(:,d);
            for(int i=0;i<D;i++)
                tVD2(i,0) = dmu_dW(i,j);
            axpy_prod(Phi,tVD2,tVD,false); //dPhiMu_dW(:,d) += Phi*dmu_dW(:,d);
            for(int i=0;i<D;i++)
                dPhiMu_dW(i, j) = tVD(i,0);
            
        }
        //printMyMat(dPhiMu_dW,D,W,"dPhiMu_dW");
        
        pm = PhiMu(0,0);
        double dpm_dl = dPhiMu_dl(0,0);
        double dpm_ds = dPhiMu_ds(0,0);
        double dpm_dn = dPhiMu_dn(0,0);
//     dpm_dW = dPhiMu_dW(1,:)';
        matrix_type dpm_dW(W,1);
        for(int i=0;i<W;i++)
            dpm_dW(W,0)= dPhiMu_dW(0, i);
        //printMyVec(dpm_dW,W,"dpm_dW");
        
        pv = P(0,0) + noise_var;
        double dpv_dl = dP_dl(0,0);
        double dpv_ds = dP_ds(0,0);
        double dpv_dn = dP_dn(0,0) + 1;
//     dpv_dW = squeeze(dP_dW(1,1,:));
        matrix_type dpv_dW(W,1);
        for(int k=0;k<W;k++)
            dpv_dW(k,0) = dP_dW[k](0,0);
        //printMyVec(dpv_dW,W,"dpv_dW");
//
//     nlml_i = 0.5*(log(2*pi) + log(pv) + ((y(i) - pm)^2)/pv);
        nlml[0] += 0.5 * (log(2 * PI) + log(pv) + (pow((y[t] - pm), 2) / pv));
        
//     dnlml_dpv = 0.5*(1/pv - ((y(i) - pm)^2)/(pv*pv));
        dnlml_dpv = 0.5*(1/pv - (pow((y[t] - pm), 2)/(pv*pv)));
        
//     dnlml_dpm = -0.5*(2*(y(i) - pm)/pv);
        double dnlml_dpm = -0.5*(2*(y[t] - pm)/pv);
        
//     dnlml_dl = dnlml_dl + dnlml_dpv*dpv_dl + dnlml_dpm*dpm_dl;
        dnlml_dl[0] += dnlml_dpv*dpv_dl+ dnlml_dpm*dpm_dl;
//     dnlml_ds = dnlml_ds + dnlml_dpv*dpv_ds + dnlml_dpm*dpm_ds;
        dnlml_ds[0] += dnlml_dpv*dpv_ds + dnlml_dpm*dpm_ds;
//     dnlml_dn = dnlml_dn + dnlml_dpv*dpv_dn + dnlml_dpm*dpm_dn;
        dnlml_dn[0] += dnlml_dpv*dpv_dn + dnlml_dpm*dpm_dn;
//     dnlml_dW = dnlml_dW + dnlml_dpv*dpv_dW + dnlml_dpm*dpm_dW;
        for(int k=0;k<W;k++) {
            dnlml_dW[k] += dnlml_dpv*dpv_dW(k,0)+ dnlml_dpm*dpm_dW(k,0);
        }
        
        matrix_type dnlml_dW_vec(D,1);
        array2vector(dnlml_dW_vec,dnlml_dW,D);
        //printMyVec(dnlml_dW_vec,W,"dnlml_dW");
//     kalman_gain = (P*H')/pv;
        axpy_prod(P,H,PH,true);
        kalman_gain = PH/pv;
//     dg_dl = (pv*(dP_dl*H') - dpv_dl*(P*H'))/(pv*pv);
        axpy_prod(dP_dl,H,tVD,true); //tVD = dP_dl*H'
        dg_dl = (pv*tVD - dpv_dl*PH)/(pv*pv);
        
//     dg_ds = (pv*(dP_ds*H') - dpv_ds*(P*H'))/(pv*pv);
        axpy_prod(dP_ds,H,tVD,true); //tVD = dP_ds*H'
        dg_ds = (pv*tVD - dpv_ds*PH)/(pv*pv);
        
//     dg_dn = (pv*(dP_dn*H') - dpv_dn*(P*H'))/(pv*pv);
        axpy_prod(dP_dn,H,tVD,true); //tVD = dP_dn*H'
        dg_dn = (pv*tVD - dpv_dn*PH)/(pv*pv);
//     for d = 1:D
//         dg_dW(:,d) = (pv*(dP_dW(:,:,d)*H') - dpv_dW(d)*(P*H'))/(pv*pv);
//     end
        for(int k=0;k<W;k++)
        {
            axpy_prod(dP_dW[k],H,tVD,true); //tVD = dP_dW(:,:,d)*H'
            tVD = (pv*tVD - dpv_dW(k,0)*PH)/(pv*pv);
            for(int i=0;i<D;i++)
                dg_dW(i,k) = tVD(i,0);
        }
        
//
        mu = PhiMu + kalman_gain*(y[t] - pm);
        dmu_dl = dPhiMu_dl + dg_dl*(y[t] - pm) - kalman_gain*dpm_dl;
        dmu_ds = dPhiMu_ds + dg_ds*(y[t] - pm) - kalman_gain*dpm_ds;
        dmu_dn = dPhiMu_dn + dg_dn*(y[t] - pm) - kalman_gain*dpm_dn;
//     for d = 1:D
//         dmu_dW(:,d) = dPhiMu_dW(:,d) + dg_dW(:,d)*(y(i) - pm) - kalman_gain*dpm_dW(d);
//     end
        for(int k=0;k<W;k++) {
            for(int i=0; i<D;i++)
                dmu_dW(i,k) = dPhiMu_dW(i,k) + dg_dW(i,k)*(y[t] - pm) - kalman_gain(i,0)*dpm_dW(k,0);
        }
        //printMyMat(dmu_dW,D,W,"dmu_dW");
//
//     V = (eye(p) - kalman_gain*H)*P;
        matrix_type kalH(D,D);
        axpy_prod(kalman_gain,trans(H),kalH,true);
        axpy_prod(kalH,P,tMDD);
        V = P - tMDD;
        //printMyMat(V,D,D,"V");
//     dV_dl = dP_dl - ((kalman_gain*H)*dP_dl + (dg_dl*H)*P);
        matrix_type dgdlH(D,D);
        axpy_prod(dg_dl,trans(H),dgdlH,true); //dgdlH = (dg_dl*H)
        axpy_prod(dgdlH,P,tMDD,true);       //tMDD = (dg_dl*H)*P
        axpy_prod(kalH,dP_dl,tMDD,false);    //tMDD += (kalman_gain*H)*dP_dl
        dV_dl = dP_dl - tMDD;
        //printMyMat(dV_dl,D,D,"dV_dl");
        
//     dV_ds = dP_ds - ((kalman_gain*H)*dP_ds + (dg_ds*H)*P);
        matrix_type dgdsH(D,D);
        axpy_prod(dg_ds,trans(H),dgdsH,true); //dgdsH = (dg_ds*H)
        axpy_prod(dgdsH,P,tMDD,true);       //tMDD = (dg_ds*H)*P
        axpy_prod(kalH,dP_ds,tMDD,false);    //tMDD += (kalman_gain*H)*dP_dl
        dV_ds = dP_ds - tMDD;
        //printMyMat(dV_ds,D,D,"dV_ds");
        
//     dV_dn = dP_dn - ((kalman_gain*H)*dP_dn + (dg_dn*H)*P);
        matrix_type dgdnH(D,D);
        axpy_prod(dg_dn,trans(H),dgdnH,true);
        axpy_prod(dgdnH,P,tMDD,true);       //tMDD = (dg_dn*H)*P
        axpy_prod(kalH,dP_dn,tMDD,false);    //tMDD += (kalman_gain*H)*dP_dn
        dV_dn = dP_dn - tMDD;
        //printMyMat(dV_dn,D,D,"dV_dn");
//     for d = 1:D
//         dV_dW(:,:,d) = dP_dW(:,:,d) - ((kalman_gain*H)*dP_dW(:,:,d) + (dg_dW(:,d)*H)*P);
//     end
        for(int k=0;k<W;k++) {
            for(int i=0;i<D;i++)
                tVD(i,0) = dg_dW(i,k);
            matrix_type dgdWH(D,D);
            axpy_prod(tVD,trans(H),dgdWH,true);  //dgdWH=dg_dW(:,d)*H
            axpy_prod(dgdWH,P,tMDD,true);   //tMDD=(dg_dW(:,d)*H)*P
            axpy_prod(kalH,dP_dW[k],tMDD,false); //tMDD+=(kalman_gain*H)*dP_dW(:,:,d)
            
            dV_dW[k] = dP_dW[k] - tMDD;
            
            //char name[50];
            //sprintf (name, "dV_dW_%d", k);
            //printMyMat(dV_dW[k],D,D,name);
        }
        
//
    }
    
}
