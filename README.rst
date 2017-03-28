This program is written for Python 3. 

The goal of the program is to use the  Basyesian Ensemble Clustering approach introduced in: 
"Hongjun Want, Hanhuai Shan, Arindam Banerjee, Bayesian Cluster Ensembles, 2010"

the program requires as input the result of various base clustering algorithms on a certain data set, then the Bayesian Clustering algorithm is applied and the concensus clustering algorithm will be outputed. 

Note that in order for the code to run, the following information must be provided with the specific format mentioned:

base_labels:              an array of size M*N, it includes the base clusteirng results to be processed

true_labels:              an array of shape 1*M, this is for cases where we know the true labels and just want to evaluate the performance. If it is not known, a vector of ones can be passed. 

number_baseclusterers:    an array of shape 1*N, number of clusters in each of the N base clustering results

Palpha:                   an array of shape k*1, initial value for the model parameter of Drichlet distribution

Pbeta:                    an array of arrays, which consists of an outer array
                          of size 1*N, each of these arrays is of size k*q. each of the N elements corresponds to one of the N base clustering results, and each of these elements is itself a k*q matrix, i.e. initial value for k parameters of a q-dimensional discrete distribution.

All the above information could be read from a dictionary.

To use the package, you need to download it and then go inside it (where the setup.py file is) and run:

pip install .

When the package is installed, then we need to run:

import scipy.io as io

mat = io.loadmat('YOUR_FILE_NAME.mat')  #YOUR_FILE_NAME refers to the name of a dictionary that includes the followint information

# note that inside the package one sample data set, named iris.mat has been provide. But the user can use any dataset that follows the same format and includes the follwing information.

# base_labels:              an array of size M*N

# true_labels:              an array of shape 1*M

# number_baseclusterers:    an array of shape 1*N

# Palpha:                   an array of shape k*1

# Pbeta:                    an array of arrays, which consists of an outer array

#                           of size 1*N, each of these arrays is of size k*q

base_labels = mat["base_labels"]

true_labels = mat["true_labels"]

number_baseclusterers = mat["number_baseclusterers"]

Palpha = mat["Palpha"]

Pbeta = mat["Pbeta"]

import bce  # here the bayesian clustering ensemble package is loaded

ensemble_labels = bce.runBCE(base_labels, true_labels, number_baseclusterers, Palpha, Pbeta)


All the above information could be read from a dictionary. If the package is installed, then we need to run:

import bce

ensemble_labels = bce.runBCE(base_labels, true_labels, number_baseclusterers, Palpha, Pbeta)

