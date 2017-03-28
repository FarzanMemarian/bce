# Author: Farzan Memarian
# Affiliation: PhD student at University of Texas at Austin
# 
# This python code consists of 5 modules, runBCE, bceEstep, bceMstep, calculateAccuracy and learnBCE. 
# It is developed based on Python 3 formatting.  
# It is based on the MATLAB code provided by authors of the following paper:
# H. Wang, H. Shan, A. Banerjee. Bayesian Cluster Ensembles. Statistical Analysis and Data Mining, 2011

from __future__ import print_function
import numpy as np
import scipy as sp
import math
import scipy.io as io
import scipy.special as special

<<<<<<< HEAD
def runBCE(base_labels, true_labels, number_baseclusterers, Palpha, Pbeta):
=======


#def main():
#f __name__ == "__main__":
#    main()

def runBCE(base_labels, true_labels, number_baseclusterers, Palpha, Pbeta):
    # load the data and initial model .runBCE(base_labels, true_labels, number_baseclusterers, Palpha, Pbeta)
    #
>>>>>>> 93938a10d473ce73b4feb3204d2da3f78532d01f
    #   k = number of ensemble clusters
    #   N = number of base clusterings
    #   M = number of data points
    # 
    # base_labels:              M*N, base clustering results to be processed
    # Palpha:                   k*1, initial value for the model parameter of Dirichlet distribution
    # Pbeta:                    cell with N elements for N base clusterings, each element is a
    #                           k*q matrix, i.e., initial value for k parameters of a q-dimensional discrete distribution
    # number_baseclusterers:    1*N, number of clusters in each of N base clustering results
    intVec = np.vectorize(int)  # FM: vectorized version of int function to be applied on an array

<<<<<<< HEAD
# The following lines are commented since they need to be performed outside this package. The following parameters are actually passed to the program rather than being performec inside the runBCE function. 
=======
>>>>>>> 93938a10d473ce73b4feb3204d2da3f78532d01f

#    mat = io.loadmat('Iris.mat')  # FM: mat is a dictionary the way Iris.mat is loaded in python. 
#    # FM: Note that in order for the code to run properly the following information 
#    # must be provided with the specific format mentioned:
#    # base_labels:              an array of size M*N
#    # true_labels:              an array of shape 1*M
#    # number_baseclusterers:    an array of shape 1*N
#    # Palpha:                   an array of shape k*1
#    # Pbeta:                    an array of arrays, which consists of an outer array
#    #                           of size 1*N, each of these arrays is of size k*q
#    base_labels = mat["base_labels"]
#    true_labels = mat["true_labels"]
#    number_baseclusterers = mat["number_baseclusterers"]
#    Palpha = mat["Palpha"]
#    Pbeta = mat["Pbeta"]
    
    
    # PramaLap:                 parameter for laplace smoothing
    PramaLap = 0.000001

    # If use random initialization
    # Palpha = rand(size(Palpha));
    # for i = 1:length(Pbeta)
    #     temp = (size(Pbeta{i}));
    #     [k, q] = size(temp);
    #     temp = temp ./ (sum(temp, 2) * ones(1, q));
    #     Pbeta{i} = temp;
    # end


    
    # learn BCE 
    phiAll, gammaAll, resultAlpha, resultBeta = learnBCE(base_labels, Palpha, Pbeta, PramaLap, \
                                                         number_baseclusterers)

    # calculate accuracy
    k = np.size(np.unique(true_labels))
    M = np.size(true_labels)

    # Obtain the cluster assignments from BCE
<<<<<<< HEAD
    Ensemble_labels = np.zeros((M, 1))
=======
    Ensemble_labels = np.zeros((1, M))
>>>>>>> 93938a10d473ce73b4feb3204d2da3f78532d01f
    wtheta = np.zeros((k, M))  # FM: initializing wtheta for python version.
    for index in range(M):
        wtheta[:, index] = gammaAll[:, index]
        bb = np.nonzero(intVec(wtheta[:, index] == max(wtheta[:, index])))
<<<<<<< HEAD
        Ensemble_labels[index, 0] = bb[0] + 1
=======
        Ensemble_labels[0, index] = bb[0] + 1
>>>>>>> 93938a10d473ce73b4feb3204d2da3f78532d01f

    # Calculate the accuracy based on true labels and BCE results
    accu = calculateAccuracy(true_labels, Ensemble_labels)
    print ('The micro-precision of BCE is ', accu)
<<<<<<< HEAD
    return Ensemble_labels
=======

>>>>>>> 93938a10d473ce73b4feb3204d2da3f78532d01f

def learnBCE(X, oldAlpha, oldBeta, lap, Q):
    #
    # BCE learning
    # 
    #   k = number of ensemble clusters
    #   N = number of base clusterings
    #   M = number of data points
    #
    # Input:
    #   X:          M*N, base clustering results
    #   oldAlpha:   k*1, model parameter for Dirichlet distribution
    #   oldBeta:    cell with N elements for N base clusterings, each element is a
    #               k*q matrix, i.e., k parameters for a q-dimensional discrete distribution
    #   lap:        smoothing parameter
    #   Q:          cell with N elements, each is the number of clusters in base
    #               clustering results         
    # Output:
    #   phiAll:     k*N*M, variational parameters for discrete distributions
    #   gamaAll:    k*M, variational parameters for Dirichlet distributions
    #--------------------------------------------------------------------

    [M, N] = X.shape
    k = np.size(oldAlpha)

    # initial value and variables for iteration
    alpha_t = oldAlpha
    beta_t = oldBeta
    epsilon = 0.01
    time = 500
    e = 100
    t = 1

    # start learning iterations
    print ('learning BCE')
    sample = np.zeros((M, N))
    phiAll = np.zeros((M, k, N))  # FM: added to make it compatible with python
                                  # note that it is assumed that phiAll is a 3D array
    gamaAll = np.zeros((k, M)) 
    while e > epsilon and t < time:
        # E-step
        for s in range(M):
            sample = np.array([X[s, :]])
            estimatedPhi, estimatedGama = bceEstep(alpha_t, beta_t, sample)
            phiAll[s, :, :] = estimatedPhi 
            gamaAll[:, s] = np.reshape(estimatedGama, (3,))

        # M-step
        alpha_tt, beta_tt = bceMstep(alpha_t, phiAll, gamaAll, X, Q, lap)

        # error
        upvalue = 0
        downvalue = 0
        for index in range(np.size(Q)):
            upvalue = upvalue + np.sum(np.sum(abs(beta_t[0, index] - beta_tt[0, index]), 0)) 
            downvalue = downvalue + np.sum(np.sum(beta_t[0, index], 0))
        e = upvalue / downvalue
        print ('t=', t, ', error=', e)

        # update
        alpha_t = alpha_tt
        beta_t = beta_tt

        t = t + 1

    resultAlpha = alpha_t
    resultBeta = beta_t


    return phiAll, gamaAll, resultAlpha, resultBeta


def calculateAccuracy(true_labels, Ensemble_labels):
    #
    # Calculate  micro-precision given clustering results and true labels.
    #
    #   k = number of ensemble clusters
    #   M = number of data points
    #
    # Input:
    #   true_labels:        1*M, true class labels for the data points
    #   Ensemble_labels:    1*M, labels obtained from BCE
    #   
    # Output:
    #   micro_precision:    micro-precision
    #--------------------------------------------------------------------
    
    k = np.size(np.unique(true_labels))
    M = np.size(true_labels)
    intVec = np.vectorize(int)  # FM: vectorized version of int function to be applied on an array
    accurence = np.zeros((k, k))
    for j in range(k):
         for jj in range(k):
            accurence[j, jj] = np.shape(np.nonzero(intVec((intVec(Ensemble_labels == (jj + 1)) * (j + 1)) == true_labels)))[1]
    [rowm, coln] = np.shape(accurence)
    amatrix = accurence
    sumMax = 0
    while rowm >= 1:
        xx = np.amax(np.amax(amatrix, 0), 0)
        [x, y] = np.nonzero(intVec(amatrix == xx)) 
        sumMax = sumMax + xx                      
        iyy = 0
        temp = np.zeros((rowm, rowm - 1))
        for iy in range(rowm):
            if iy == y[0]:
                continue  
            else:                        
                temp[:, iyy] = amatrix[:, iy]
                iyy = iyy + 1
        temp2 = np.zeros((rowm - 1, rowm - 1))
        ixx = 0
        for ix in range(rowm):
            if ix == x[0]:
                continue
            else:                       
                temp2[ixx, :] = temp[ix, :]
                ixx = ixx + 1
        rowm = rowm - 1
        amatrix = np.zeros((rowm, rowm))
        amatrix = temp2
    
    micro_precision = sumMax / M
    return micro_precision

def bceMstep(alpha, phi, gama, X, Q, lap):
    #
    # M-step of BCE
    #
    #   k = number of ensemble clusters
    #   N = number of base clusterings
    #   M = number of data points
    #
    # Input:
    #   phi:    k*N*M, variational parameters for discrete distributions
    #   gama:   k*M, variational parameters for Dirichlet distributions
    #   alpha:  k*1, model parameters for the Dirichlet distribution
    #   X:      M*N, base clustering results
    #   Q:      cell with N elements, each is the number of clusters in base
    #           clustering results
    #   lap:    laplacian smoothing parameter
    #
    # Output:
    #   alpha:  k*1, model paramter for Dirichlet distribution
    #   beta:   cell with N elements for N base clusterings, each element is a
    #           k*q matrix, i.e., k parameters for a q-dimensional discrete distribution   
    # -------------------------------------------------
    
    [M, k, N] = np.shape(phi) 
    
    beta = np.empty((1, N), np.ndarray)  # FM: added to initialize beta as an array of arrays
    #-------update beta----------
    for ind in range(N):
        beta[0, ind] = np.zeros((k, Q[0, ind]))
    
    
    intVec = np.vectorize(int)  # FM: vectorized version of int
    
    for ind in range(N):
        for q in range(Q[0, ind]):
            temp = np.zeros((k, N))
            for s in range(M):
                x = np.array([X[s, :]])
                fil = np.matmul(np.ones((k, 1)), intVec(x == (q + 1)))   
                temp = temp + phi[s, :, :] * fil 
            beta[0, ind][:, q] = temp[:, ind]
    
    # smoothing
    for ind in range(N):
        beta[0, ind] = beta[0, ind] + lap
        beta[0, ind] = beta[0, ind] / np.matmul(np.reshape(np.sum(beta[0, ind], 1), \
                                                (k, 1)), (np.ones((1, Q[0, ind])))) 
    
    
    # -------update alpha-----------
    
    alpha_t = alpha
    epsilon = 0.001
    time = 500
    
    t = 0
    e = 100
    psiGama = special.psi(gama)
    psiSumGama = special.psi(sum(gama, 0))
    while e > epsilon and t < time:
        g = np.reshape(np.sum((psiGama - np.matmul(np.ones((k, 1)), np.reshape(psiSumGama, (1, M)))), 1), (k, 1)) \
                                + M * (special.psi(np.sum(alpha_t, 0)) - special.psi(alpha_t))
        h = -M * special.polygamma(1, alpha_t)
        z = M * special.polygamma(1, np.sum(alpha_t, 0))
        c = np.sum((g / h), 0) / (1 / z + np.sum((1 / h), 0)) 
        delta = (g - c) / h
    
        # line search
        eta = 1
        alpha_tt = alpha_t - delta
        while (np.size(np.nonzero(intVec(alpha_tt <= 0))[0]) > 0):
            eta = eta / 2
            alpha_tt = alpha_t - eta * delta
        e = np.sum(abs(alpha_tt - alpha_t), 0) / np.sum(alpha_t, 0)
        
        alpha_t = alpha_tt
    
        t = t + 1
    
    alpha = alpha_t
    
    return alpha, beta

def bceEstep(alpha, beta, x):
    # 
    # E step of BCE
    # 
    #   k = number of ensemble clusters
    #   N = number of base clusterings
    #   M = number of data points
    # 
    # Input:
    #   alpha:  k*1, parameter for Dirichlet distribution
    #   beta:   cell with N elements for N base clusterings, each element is a
    #           k*q matrix, i.e., k parameters for a q-dimensional discrete distribution
    #   x:      1*N, base clustering results for one data point. 0 indicates
    #           missing base clustering results
    # 
    # output:
    #   phi_t:  k*N, variational parameter for discrete distribution
    #   gama_t: k*1, variational parameter for Dirichlet distribution
    #-------------------------------------------------
    
    k = np.size(alpha)
    N = np.size(x)
    V = np.size(np.nonzero(x))
    intVec = np.vectorize(int)  # FM: vectorized version of int function to be applied on an array
    fil = np.matmul(np.ones((k, 1)), intVec((x != 0)))  # FM: matrix product of two arrays
    
    # initial value for variational parameters
    phi_t = (np.ones((k, N)) * fil) / k
    gama_t = alpha + V / k
    
    # variables for iteration
    epsilon = 0.01
    time = 500
    e = 100
    t = 1
    tempBeta = np.zeros((k, N))  # FM: added to initialize tempBeta
    
    for i in range(k):
        for n in range(N):
            if (x[0, n] != 0):
               tempBeta[i, n] = beta[0, n][i, x[0, n] - 1]  
            else:
               tempBeta[i, n] = -1
    
    
    # Continue iteration, if the error is larger than the threshold, or
    # iteration time is smaller than the predefined steps.
    realmin = np.finfo(np.double).tiny 
    while e > epsilon and t < time:
        # new phi
        phi_tt = np.exp(np.matmul((special.psi(gama_t) - special.psi(np.sum(gama_t, 0))), np.ones((1, N)))) * tempBeta
        phi_tt = phi_tt / np.matmul(np.ones((k, 1)), np.reshape(np.sum(phi_tt + realmin, 0), (1, N)))
        phi_tt = phi_tt * fil
        
        # new gamma
        gama_tt = alpha + np.reshape(np.sum(phi_tt, 1), (k, 1))  # FM: reshape has been used for np.sum to make it vertical
        
        # error of the iteration
        e1 = np.sum(np.sum(np.absolute(phi_tt - phi_t), 0)) / np.sum(np.sum(phi_t, 0))
        e2 = sum(np.absolute(gama_tt - gama_t), 0) / sum(gama_t, 0)
        e = max(e1, e2)
        
        # update the variational parameters
        phi_t = phi_tt
        gama_t = gama_tt
        # disp(['t=',intVecstr(t),', e1,e2,e:',num2str(e1),',',num2str(e2),',',num2str(e)])
        t = t + 1
    
    
    return phi_t, gama_t
