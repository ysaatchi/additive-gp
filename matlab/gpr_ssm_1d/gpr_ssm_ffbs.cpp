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
#include <boost/random.hpp>
#include "boost/multi_array.hpp"
#define PI 3.14159265358979323846

using namespace std;

void nrandnorm(int D, double* mean, double* covar, double* sample);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

//gpr_ssm_ffbs(logtheta, x, y, xstar, V0, Phi_cell, Q_cell, int numSamples, int seed) {
    
    //Inputs
    if (nrhs != 9) {
        mexErrMsgTxt("Incorrect number of arguments (should be 9). Quitting...");
    }
    
    bool debug = false;
    
    int D = mxGetM(prhs[4]);
    
    double* logtheta = mxGetPr(prhs[0]);
    double* x = mxGetPr(prhs[1]);
    double* y = mxGetPr(prhs[2]);
    double* xstar = mxGetPr(prhs[3]);
    double* V0 = mxGetPr(prhs[4]);
    const mxArray* Phi_cell = prhs[5];
    const mxArray* Q_cell = prhs[6];
    int numSamples = (int)(mxGetScalar(prhs[7]));
    unsigned int seed_init = (unsigned int)(mxGetScalar(prhs[8]));
    
    //size checks
    if (mxGetM(prhs[0]) != 3 || (mxGetM(prhs[1]) != mxGetM(prhs[2]))) {
        mexErrMsgTxt("Check input argument sizes! Quitting...");
    }
    if (!(mxGetN(prhs[0]) == 1 && mxGetN(prhs[1]) == 1 && mxGetN(prhs[2]) == 1 && mxGetN(prhs[3]) == 1)) {
        mexErrMsgTxt("All vectors passed in must be column vectors! Quitting...");
    }
    //Outputs
    if (nlhs != 3) {
        mexErrMsgTxt("Invalid number of output arguments (should be 3). Quitting...");
    }
    int N = mxGetM(prhs[1]);
    int M = mxGetM(prhs[3]);
    assert(M > 0);
    plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(numSamples, N, mxREAL);
    plhs[2] = mxCreateDoubleMatrix(numSamples, M, mxREAL);
    double* nlml = mxGetPr(plhs[0]);
    double* train_sample = mxGetPr(plhs[1]);
    double* test_sample = mxGetPr(plhs[2]);
    
    //MAIN CODE

    //initialize RNG seeds
    boost::mt19937 generator;
    generator.seed(seed_init); //if you want different seed at each run must pass in a random seed_init
    boost::normal_distribution<> normdist;
    boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > randn(generator, normdist);
    
    //compute merge vectors (true if in training, false if in xstar)
    int T = N + M;
    int train_idx = 0;
    int test_idx = 0;
    int num_train = N;
    int num_test = M;
    bool* is_train = new bool[T];
    memset(is_train, true, sizeof (bool) * T);
    double* xall = new double[T];
    int k = 0;
    if (M > 0) { //can pass an empty xstar
        while ((train_idx < num_train) && (test_idx < num_test))  {
            if (x[train_idx] < xstar[test_idx]) {
                xall[k] = x[train_idx];
                train_idx++;
                k++;
            } else {
                xall[k] = xstar[test_idx];
                is_train[k] = false;
                test_idx++;
                k++;
            }
        }
        if (test_idx < num_test) {
            for (int i = k; i < T; i++) {
                xall[i] = xstar[test_idx];
                is_train[i] = false;
                test_idx++;
            }
        }
        if (train_idx < num_train) {
            for (int i = k; i < T; i++) {
                xall[i] = x[train_idx];
                train_idx++;
            }
        }
        
    }
    ////cout << "train_idx = " << train_idx << " test_idx = " << test_idx << endl;
   
    double lambda = exp(logtheta[0]); //cov "decay" parameter
    double signal_var = exp(logtheta[1]);
    double noise_var = exp(logtheta[2]);
    typedef boost::multi_array<double, 3> array_3D;
    typedef boost::multi_array<double, 2> array_2D;
     typedef std::vector<double> array_1D;
    array_2D mus(boost::extents[T][D]);
    array_3D Phis(boost::extents[T][D][D]);
    array_3D Ps(boost::extents[T][D][D]);
    array_3D Vs(boost::extents[T][D][D]);
    //double mus[T][D];
    //double Vs[T][D][D];
    //double Phis[T][D][D];
    //double Ps[T][D][D];
    array_1D kalman_gain(D);
    array_1D mu(D);
    array_2D V(boost::extents[D][D]);
    array_1D PhiMu(D);
    array_2D Phi(boost::extents[D][D]);
    array_2D Q(boost::extents[D][D]);
    array_2D Z(boost::extents[D][D]);
    array_2D P(boost::extents[D][D]);
    array_2D S(boost::extents[T][D]);
     array_1D Phi_v(D*D);
        array_1D Q_v(D*D);
        array_1D V_vec(D * D);
            array_1D A(D * D);
        array_2D Pinv(boost::extents[D][D]);
            array_2D L(boost::extents[D][D]);
            array_1D mu_s(D);
            array_2D V_s(boost::extents[D][D]);
    //output variables
    nlml[0] = 0;
    
    /* BEGIN KALMAN FILTERING */
    ////cout << "Forward filtering..." << endl;
    //absorb first observation
    int t = 0;
    train_idx = 0;
    if (is_train[0]) {
        double pm = 0;
        double pv = V0[0] + noise_var;
        nlml[0] += 0.5 * (log(2 * PI) + log(pv) + (pow((y[train_idx] - pm), 2) / pv));
        for (int d = 0; d < D; d++) {
            double kgd = (V0[d]) / pv;
            kalman_gain[d] = kgd;
            mu[d] = kgd * (y[train_idx] - pm);
            mus[t][d] = mu[d];
        }
        for (int i = 0; i < D; i++) {
            for (int j = 0; j < D; j++) {
                V[i][j] = V0[j*D + i] - kalman_gain[i]*(V0[j*D]);
                Vs[t][i][j] = V[i][j];
            }
        }
        train_idx++;
    } else {
        for (int d = 0; d < D; d++) {
            mu[d] = 0;
            mus[t][d] = mu[d];
        }
        for (int i = 0; i < D; i++) {
            for (int j = 0; j < D; j++) {
                V[i][j] = V0[j*D + i];
                Vs[t][i][j] = V[i][j];
            }
        }
    }
   
        
    //filter forward in "time"
    for (t = 1; t < T; t++) {
        
        
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
        if (is_train[t]) {
            
            ////cout << "is_train" << endl;
            double pm = PhiMu[0];
            double pv = P[0][0] + noise_var;
            nlml[0] += 0.5 * (log(2 * PI) + log(pv) + (pow((y[train_idx] - pm), 2) / pv));
            for (int d = 0; d < D; d++) {
                double kgd = (P[d][0]) / pv;
                kalman_gain[d] = kgd;
                mu[d] = PhiMu[d] + kgd * (y[train_idx] - pm);
                mus[t][d] = mu[d];
            }
            for (int i = 0; i < D; i++) {
                for (int j = 0; j < D; j++) {
                    V[i][j] = P[i][j] - kalman_gain[i]*(P[0][j]);
                    Vs[t][i][j] = V[i][j];
                }
            }
            train_idx++;
            
        } else {
            
            ////cout << "is_test" << endl;
            for (int d = 0; d < D; d++) {
                mu[d] = PhiMu[d];
                mus[t][d] = mu[d];
            }
            for (int i = 0; i < D; i++) {
                for (int j = 0; j < D; j++) {
                    V[i][j] = P[i][j];
                    Vs[t][i][j] = V[i][j];
                }
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
    
    /* BEGIN BACKWARD SAMPLING numSamples TIMES */
    
    ////cout << "Backward sampling ( " << numSamples << " samples )..." << endl;
    
    for (int nn = 0; nn < numSamples; nn++) {
        
        
        
        for (int i = 0; i < D; i++) {
            mu[i] = mus[T-1][i];
            for (int j = 0; j < D; j++) {
                V[i][j] = Vs[T-1][i][j];
            }
        }
        
        
        for (int i = 0; i < D; i++) {
            for (int j = 0; j < D; j++) {
                V_vec[j * D + i] = V[i][j];
            }
        }
        
        double *R = new double[D];
        for (int i = 0; i < D; i++) {
            R[i] = randn();
        }
        nrandnorm(D, &mu[0], &V_vec[0], R); //R = randnorm(mu,V_vec);
        
        for (int i = 0; i < D; i++) {
            S[T - 1][i] = R[i];
            //printf("R[i] = %3.3f\n", R[i]);
        }
        
        for (t = T - 2; t >= 0; t--) {
            
            for (int i = 0; i < D; i++) {
                mu[i] = mus[t][i];
                for (int j = 0; j < D; j++) {
                    V[i][j] = Vs[t][i][j];
                    Phi[i][j] = Phis[t][i][j];
                    P[i][j] = Ps[t][i][j];
                }
            }
            
            //L = V*Phi'*(P\eye(D));
            
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
                //cout << "WARNING from gpr_ffbs_matern: Matrix is probably not positive definite!" << endl;
            }
            dpotri(&UorL, &D_lp, &A[0], &D_lp, &info);
            
            /*
            int status = clapack_dpotrf(CblasColMajor, CblasLower, D, A, D);
            clapack_dpotri(CblasColMajor, CblasLower, D, A, D); //returns lower triangle of symmetric inverse
            */
            
            
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
            
            //mu_s = mu + L*(X(:,i+1) - Phi*mu);
            //memset(PhiMu, 0, sizeof (double) * D);
            for (int i = 0; i < D; i++) {
                PhiMu[i] = 0;
                for (int j = 0; j < D; j++) {
                    PhiMu[i] += Phi[i][j] * mu[j];
                }
            }
            
            for (int i = 0; i < D; i++) {
                mu_s[i] = mu[i];
                for (int j = 0; j < D; j++) {
                    mu_s[i] += L[i][j]*(S[t + 1][j] - PhiMu[j]);
                }
            }
            //Sigma_s = V - L*P*L';
            for (int i = 0; i < D; i++) {
                for (int j = 0; j < D; j++) {
                    Z[i][j] = 0;
                    for (int k = 0; k < D; k++) {
                        Z[i][j] += L[i][k] * P[k][j];
                    }
                }
            }
            
            for (int i = 0; i < D; i++) {
                for (int j = 0; j < D; j++) {
                    V_s[i][j] = V[i][j];
                    for (int k = 0; k < D; k++) {
                        V_s[i][j] -= Z[i][k] * L[j][k];
                    }
                }
            }
            
            //printf("mu[%i] = %3.3f\n", 0, mu[0]);
            
            /*for (int i = 0; i < D; i++) {
             * for (int j = 0; j < D; j++) {
             * printf("V_s[%i][%i] = %3.3f\n", i, j, V_s[i][j]);
             * }
             * }*/
            //X(t,:) = mrandnorm(mu_s, Sigma_s);
            for (int i = 0; i < D; i++) {
                for (int j = 0; j < D; j++) {
                    V_vec[j * D + i] = V_s[i][j];
                    if (i == j) {
                        V_vec[j * D + i] += 1e-8; //add jitter
                    }
                }
            }
            for (int i = 0; i < D; i++) {
                R[i] = randn();
            }
            nrandnorm(D, &mu_s[0], &V_vec[0], R); //mu and V contain the filtered values of the last state
            //printf("mu_s = %3.3f \t var_s = %3.3f\n", mu_s[0], V_vec[0]);
            for (int i = 0; i < D; i++) {
                S[t][i] = R[i];
            }
            
        }
        //separate S matrix into those associated with training points and those with test points
        
        train_idx = 0;
        test_idx = 0;
        k = 0;
        for (int tt = 0; tt < T; tt++) {
            if (is_train[tt]) {
                //X(i,j) = X[(j*numRows)+i]
                train_sample[train_idx*numSamples + nn] = S[tt][0];
                train_idx++;
                k++;
            } else {
                test_sample[test_idx*numSamples + nn] = S[tt][0];//first element of state vector is function value
                test_idx++;
                k++;
            }
        }
        
    }
    
    mus.resize(boost::extents[1][1]);
    Vs.resize(boost::extents[1][1][1]);
    Phis.resize(boost::extents[1][1][1]);
    Ps.resize(boost::extents[1][1][1]);
    delete[] is_train;
    delete[] xall;
    
}

void nrandnorm(int D, double* mean, double* covar, double* sample) {
    
    //sample initially contains standard normal sample --> transformed into
    //normal random sample with mean and covar as given
    typedef boost::multi_array<double, 2> array_2D;
     typedef std::vector<double> array_1D;
    array_1D randvec(D);
    for (int d = 0; d < D; d++) {
        randvec[d] = sample[d];
    }
    array_1D A(D*D);
    for (int i = 0; i < D*D; i++) {
        A[i] = covar[i];
    }
    
    mwSignedIndex info;
    mwSignedIndex D_lp = (mwSignedIndex) D;
    char UorL = 'L';
    dpotrf(&UorL, &D_lp, &A[0], &D_lp, &info);
    if (info != 0){
        //cout << "WARNING from nrandnorm: Matrix is probably not positive definite!" << endl;
    }
    
    /*
    int status = clapack_dpotrf(CblasColMajor, CblasLower, D, A, D);
    if (status != 0) {
        //cout << "WARNING from nrandnorm: Matrix is probably not positive definite!" << endl;
    }
    */
    array_2D chol(boost::extents[D][D]);
    for (int i = 0; i < D; i++) {
        for (int j = 0; j < D; j++) {
            if (j <= i) {
                chol[i][j] = A[j*D + i];
            } else {
                chol[i][j] = 0;
            }
            //printf("chol[%i][%i] = %3.5f\n", i, j, chol[i][j]);
        }
    }
    
    for (int i = 0; i < D; i++) {
        double ss = 0;
        for (int j = 0; j < D; j++) {
            ss += chol[i][j] * randvec[j];
        }
        sample[i] = mean[i] + ss;
        //printf("sample = %3.3f\n", sample[i]);
    }
    //printf("sample_f = %3.3f\n", sample[0]);
    
}
