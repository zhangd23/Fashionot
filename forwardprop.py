# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 13:46:10 2018

@author: dzhang
"""

## Neural Network Forward Propagation

import scipy.io
#from scipy import linalg
import numpy as np

#sigmoid function
def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g

#predict function using X, Theta1, Theta2
def forwardprop(X, Theta1, Theta2):
    a1 = np.append(np.ones([m,1]),X,axis=1)
    a2 = sigmoid(a1.dot(Theta1.T))
    a2 = np.append(np.ones([m,1]),a2,axis=1)
    a3 = sigmoid(a2.dot(Theta2.T))
    a3[a3 >= .5] = 1
    a3[a3 < .5] = 0
    return a3


mat = scipy.io.loadmat('ex3data1.mat')

X = mat['X']
y = np.array(mat['y'], dtype=np.float64)
n = mat['X'].shape[1]
m = mat['X'].shape[0]
n2 = 100                #number of neurons in hidden layer 
n3 = 1                  #number of neurons in output layer

Theta1 = np.random.rand(n2, n+1)*2 -1    #random initialization of Theta1
Theta2 = np.random.rand(n3, n2+1)*2 -1   #random initialization of Theta2



#X.T.dot(y).shape Note for syntax

#np.ones([m,1])

#all indexes where true [i for i, x in enumerate(a3==1) if x]