AdditiveGP for fast Multidimensional GP analysis
===================

A framework for fast multidimensional GP analysis through additive modeling and SSM analysis.

To read about some preliminary experiments using this code, see:

*Structure Discovery in Nonparametric Regression through Compositional Kernel Search*
by Gilboa, Elad, Yunus Saatçi, and John P. Cunningham. "Scaling Multidimensional Inference for 
Structured Gaussian Processes." to appear TPAMI (2013).


Feel free to email with any questions:
[Elad Gilboa] (gilboae@ese.wustl.edu)



### Instructions:

You'll need Matlab and GPML. For comparisons you might need to
install GPstuff-3.3, IVM, and libsvm. For compilation of the mex
files you will also need the boost_1_45_0.

Before running additiveGP you must call config.m to setup the
paths. You need to change `config.m` for your lib locations.

To check whether the framework runs, go to the source directory and
run 'reg_runtime_N_comparison' for regression and 'run_breast' for
classification.

There are some example experiment scripts `examples/`.


Parameters
==========

for regression:

numSubset = 1000; %subset of data to use for MCMC inference dproj =
D; %number of projection dimensions

numPseudo=500; %number of pseudo inputs for SPGP

numMCMC = 10; %number of full MCMC iterations

rand_init = false; %initialize proj pursuit weight randomly or with
linear model


for classification:

addLA.runMCMC;% whether to run the approximation with MCMC or
Laplace approximation

addLA.ells; %length scale hyperparameter for gpml;

addLA.sigfs; %variance hyperparameter for gpml;

addLA.numNewton; %number of Newton iterations;

addLA.numGS;% number of Gibbs sampling for posterior calculation


If you have any questions about getting this running on your
machine or cluster, please let us know.
