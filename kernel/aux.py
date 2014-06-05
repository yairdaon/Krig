'''
Created on Apr 29, 2014

@author: daon
'''
import numpy as np
import math

def cov(x,y,r):
    """
    calculate an autocovariance
    """
    temp = np.linalg.norm(x-y)
    return  math.exp(  -temp*temp/(2*r*r)  ) 



def augCovMat(X,r):
    """
    return the augmented covariance matrix for the observations
    """
    
    #decide on the size 
    n = len(X)
    
    # allocate memory
    C = np.zeros( (n+1,n+1) )
    
    # set the values of the covariance, exploiting symmetry
    for i in range(0,n):
        for j in range (0,i+1):
            C[i,j] = cov(X[i],X[j],r)
            C[j,i] = C[i,j]
    
    # set the values of the augmentation (see matrix above)        
    for i in range(0,n):
        C[i,n] = -1.0
        C[n,i] = 1.0
    C[n,n] = 0.0
    return C



def covMat(X,r): 
    """
    return the covariance matrix for the observations
    """
    
    #decide on the size of
    n = len(X)
    
    # allocate memory
    C = np.zeros( (n,n) )
    
    # set the values of the covariance, exploiting symmetry
    for i in range(0,n):
        for j in range (0,i):
            C[i,j] = cov(X[i],X[j],r)
            C[j,i] = C[i,j]
        C[i,i] = cov(X[i], X[i], r)
    return C