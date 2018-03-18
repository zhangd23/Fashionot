# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 16:26:19 2018

@author: dzhang
"""

import scipy.io
import numpy as np

#sigmoid function
def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g

def sigmoidgradient(z):
    g = sigmoid(z)*(1-sigmoid(z))
    return g

# Cost Function
def NNCostFxn(X, Theta1, Theta2, lam):
    a1 = np.append(np.ones([m,1]),X,axis=1)
    z2 = a1.dot(Theta1.T)
    a2 = sigmoid(z2)
    a2 = np.append(np.ones([np.size(a2,0),1]),a2,axis=1)
    z3 = a2.dot(Theta2.T)
    a3 = sigmoid(z3)

    #Cost Calculation
    J = 1/m*(-y.T.dot(np.log(a3))-(1-y).T.dot(np.log(1-a3)))
    
    #Regularization for cost
    R = lam/(2*m) * sum(sum(np.square(Theta1[0:np.size(Theta1,0), 1:np.size(Theta1,1)]))) \
    + sum(sum(np.square(Theta2[0:np.size(Theta2,0), 1:np.size(Theta2,1)])))
    
    #Update J with Regularization
    J = J + R

    #Back propagation calculations
    #Error calculations
    d3 = a3-y
    d2 = d3.dot(Theta2[0:np.size(Theta2,0), 1:np.size(Theta2,1)])*sigmoidgradient(z2)
    Delta1 = d2.T.dot(a1)
    Delta2 = d3.T.dot(a2)
    Theta1_grad = 1/m*Delta1
    Theta2_grad = 1/m*Delta2
    
    #Regularization for Back propagation
    Theta1_R = Theta1
    Theta1_R[:,0]=0
    Theta2_R = Theta2
    Theta2_R[:,0]=0
    R1 = lam/m*Theta1_R
    R2 = lam/m*Theta2_R
    
    Theta1_grad += R1
    Theta2_grad += R2
    
    grad = np.append(np.ravel(Theta1_grad),np.ravel(Theta2_grad))
    
    return J, grad
    
'''
R1 = lambda/m*Theta1(:, 2:end);
R2 = lambda/m*Theta2(:, 2:end);

R1 = [zeros(size(R1, 1), 1) R1];
R2 = [zeros(size(R2, 1), 1) R2];

Theta1_grad = Theta1_grad + R1;
Theta2_grad = Theta2_grad + R2;
'''

# Load training data
# Replace with script as it is completed
mat = scipy.io.loadmat('ex3data1.mat')

# Setup parameters
X = mat['X']
y = np.array(mat['y'], dtype=np.float64)
n = mat['X'].shape[1]
m = mat['X'].shape[0]
n2 = 100                #number of neurons in hidden layer 
n3 = 1                  #number of neurons in output layer
lam = 1

# Initialize random Thetas
Theta1 = np.random.rand(n2, n+1)*2 -1    #random initialization of Theta1
Theta2 = np.random.rand(n3, n2+1)*2 -1   #random initialization of Theta2

