# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 16:26:19 2018

@author: dzhang
"""

import scipy.io
import scipy.optimize as optimize
import numpy as np
import matrixgenerator as mg



def fixlog(a):
    #helps correct for scenarios where log(0) results in infinity
    #we are looking at log(0
    #get all value of a < 1E-100, set them to 1E-100
    b = a < 1E-30 
    a[b] = 1E-30
    a = np.log(a)
    return a
#sigmoid function
def sigmoid(z):
    #this will blow up for large negative numbers
    #solution is to limit z
    z=np.clip(z,-100,1000) 
    g = 1 / (1 + np.exp(-z))
    return g

def sigmoidgradient(z):
    z=np.clip(z,-100,1000) 
    g = sigmoid(z)*(1-sigmoid(z))
    return g

#Gradient Function
def NNGradFxn(theta,X,y,lam,n2,n3):
    #Theta1 and Theta2 come in rolled up to be compatible with optimization functions
    #have to unfold them
    
    m = X.shape[0] #our X data is m rows (examples) by n columns (pixels)
    n = X.shape[1]    
    
    #unpack the thetas
    Theta1 = theta[0:n2*(n+1)]       
    Theta2 = theta[n2*(n+1):len(theta)]
   # print('theta1sh',Theta1.shape,'theta2sh',Theta2.shape)
    #reshape Theta1
    Theta1 = Theta1.reshape((n2,n+1)) #Theta1 is n2 (hidden layer neurons) by n(input pixels)
    #reshape Theta2
    Theta2 = Theta2.reshape((n3,n2+1)) #Theta2 is n3 (output columns) by n2 (hidden layer neurons)
    
   # print('theta1sh',Theta1.shape,'theta2sh',Theta2.shape)
    a1 = np.append(np.ones([m,1]),X,axis=1)
    z2 = a1.dot(Theta1.T)
    a2 = sigmoid(z2)
    a2 = np.append(np.ones([np.size(a2,0),1]),a2,axis=1)
    z3 = a2.dot(Theta2.T)
    a3 = sigmoid(z3)
    #Back propagation calculations
    #Error calculations
    d3 = a3-y #m x number of output neurons
    tTheta2 = Theta2[0:np.size(Theta2,0), 1:np.size(Theta2,1)] #remove extra column of Theta 
    d2 = (d3 @ tTheta2)*sigmoidgradient(z2)  #operator @ can be used for matrix multiplication
    Delta1 = d2.T.dot(a1)
    Delta2 = d3.T.dot(a2)
    Theta1_grad = (1/m)*Delta1 #grad shoud be n2 x n+1 same as Theta1
    Theta2_grad = (1/m)*Delta2 #grad should be n3 x n2+1 same as Theta2
    
    #Regularization for Back propagation
    Theta1_R = np.array(Theta1[:])
    #important note: Theta1_R= Theta1 creates a reference to Theta1
    Theta1_R[:,0]=0
    Theta2_R = np.array(Theta2[:])
    Theta2_R[:,0]=0
    R1 = (lam/m)*Theta1_R
    R2 = (lam/m)*Theta2_R
    
    Theta1_grad += R1
    Theta2_grad += R2
    #note: need to specify ravel order 
    grad = np.append(np.ravel(Theta1_grad,order='F'),np.ravel(Theta2_grad,order='F'))
    
    return grad
    

# Cost Function
def NNCostFxn(theta,X,y,lam,n2,n3):
    #Theta1 and Theta2 come in rolled up to be compatible with optimization functions
    #have to unfold them
  
    
    m = X.shape[0] #our X data is m rows (examples) by n columns (pixels)
    n = X.shape[1]    
    
    #unpack the thetas
    Theta1 = theta[0:n2*(n+1)]
       
    Theta2 = theta[n2*(n+1):len(theta)]
   # print('theta1sh',Theta1.shape,'theta2sh',Theta2.shape)
    #reshape Theta1
    Theta1 = Theta1.reshape((n2,n+1)) #Theta1 is n2 (hidden layer neurons) by n(input pixels)
    #reshape Theta2
    Theta2 = Theta2.reshape((n3,n2+1)) #Theta2 is n3 (output columns) by n2 (hidden layer neurons)
    
   # print('theta1sh',Theta1.shape,'theta2sh',Theta2.shape)
    a1 = np.append(np.ones([m,1]),X,axis=1)
    z2 = a1.dot(Theta1.T)
    a2 = sigmoid(z2)
    a2 = np.append(np.ones([np.size(a2,0),1]),a2,axis=1)
    z3 = a2.dot(Theta2.T)
    a3 = sigmoid(z3)


    #Cost Calculation
   # print('shapes: a3',a3.shape,'y',y.shape)
    tmpa3 = np.reshape(a3,-1)
    tmpy = np.reshape(y,-1)
    Jpart1 = fixlog(tmpa3).dot(-tmpy)
    Jpart2 = fixlog(1-tmpa3).dot(-(1-tmpy))
    #Jpart1 = -(y.T).dot(fixlog(a3))
    #Jpart2 = -((1-y).T).dot(fixlog(1-a3))
    J = (1/m)*(Jpart1 + Jpart2)
    J = sum(J.reshape(1,-1))
  
   # J = (1/m)*((-y.T.dot(np.log(a3))-(1-y).T.dot(np.log(1-a3))))
    
    #some notes on logistic cost function:
    # y can be 0 or 1. a3 is a guess from 0 to 1
    #case A, y = 1, then cost scales with distance of prediction a3 from 1. 
    #cost = - log(a3)  --> if a3==1 cost = -log(1) which is -
    #                  --> if a3==0 cos = -log(0) which is extremely high
    #case B, y = 0, the cost scaled with distance of a3 from 0
    #cost = -log(1-a3) --> if a3==0, cost = -log(1) = 0
    #                  --> if a3==1, cost = -log(0) which is a very high cost
    #for case where y = 1, ignore the second part so prefix of par 2 is (1-y)
    #for case where y = 0, ignore first part, so prefix is y 
    
  
    
    #Regularization for cost
    R = (lam/(2*m)) * (sum(sum(np.square(Theta1[0:np.size(Theta1,0), 1:np.size(Theta1,1)]))) \
    + sum(sum(np.square(Theta2[0:np.size(Theta2,0), 1:np.size(Theta2,1)]))))
   
    #Update J with Regularization
    J = J + R
    return J
    
'''
R1 = lambda/m*Theta1(:, 2:end);
R2 = lambda/m*Theta2(:, 2:end);

R1 = [zeros(size(R1, 1), 1) R1];
R2 = [zeros(size(R2, 1), 1) R2];

Theta1_grad = Theta1_grad + R1;
Theta2_grad = Theta2_grad + R2;
'''

# Load training data
'''# Replace with script as it is completed
mat = scipy.io.loadmat('ex3data1.mat')

# Setup parameters
X = mat['X']
y = np.array(mat['y'], dtype=np.float64)

n = mat['X'].shape[1]
m = mat['X'].shape[0]'''

X,y = mg.generateMatrices()
X = X.astype(np.float64)
y = y.astype(np.float64)
m = X.shape[0]
n = X.shape[1]

n2 = 100                #number of neurons in hidden layer 
n3 = 1                  #number of neurons in output layer
lam = 1

# Initialize random Thetas
Theta1 = np.random.rand(n2, n+1)*2 -1    #random initialization of Theta1
Theta2 = np.random.rand(n3, n2+1)*2 -1   #random initialization of Theta2


#roll up Theta1,Theta2 to be compatible with optimization functions
#Theta1 = np.ravel(Theta1)
#Theta2 = np.ravel(Theta2)
initial_theta = np.append(np.ravel(Theta1),np.ravel(Theta2))

#Get our cost and Gradient out
tmpJ = NNCostFxn(initial_theta,X,y,lam,n2,n3)
tmpG = NNGradFxn(initial_theta,X,y,lam,n2,n3)
print('cost:',tmpJ,'gradients',tmpG)
#Now that we know how to get gradients and costs, run gradient descent and

#optimize our neural net
#https://docs.scipy.org/doc/scipy-0.13.0/reference/generated/scipy.optimize.fmin_cg.html
#https://stackoverflow.com/questions/18801002/fminunc-alternate-in-numpy

opts = {'maxiter' : None,    # default value.
         'disp' : True,    # non-default value.
         'gtol' : 1e-5,    # default value.
         'norm' : np.inf,  # default value.
         'eps' : 1E-8}
         #'eps' : 1.4901161193847656e-08}  # default value.

'''res2 = optimize.minimize(f, x0, jac=gradf, args=args,
                          method='CG', options=opts)'''

#combine Theta1 and Theta2 into one set of initial parameters

res2 = optimize.minimize(NNCostFxn, x0=initial_theta, jac=NNGradFxn, args=(X,y,lam,n2,n3),
                          method='CG', options=opts)


opt_theta = res2.x
 #unpack the thetas
tTheta1 = opt_theta[0:n2*(n+1)]
tTheta2 = opt_theta[n2*(n+1):len(opt_theta)]
#reshape Theta1
tTheta1 = tTheta1.reshape((n2,n+1)) #Theta1 is n2 (hidden layer neurons) by n(input pixels)
#reshape Theta2
tTheta2 = tTheta2.reshape((n3,n2+1)) #Theta2 is n3 (output columns) by n2 (hidden layer neurons)
    