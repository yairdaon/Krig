'''
Created on Apr 29, 2014

@author: daon
'''

import numpy as np
import math
import aux
import config as cfg
import type

def kriging(s, CFG):
    '''
    this is where we decide actually which kriging subroutine we use.
    calling them always has the same syntax, though.
    return the interpolated f
    here this is interpreted as a log-likelihood \ log-probability
    '''
    
    # make sure the matrices used in the kriging computation are ready    
    CFG.setMatrices()
    
    # choose among the different algorithms
    if CFG.algType == type.AUGMENTED_COVARIANCE:
        f, sigSquare =  acmSvdKriging(s, CFG)    
        return f, math.sqrt( abs(sigSquare) ) 
    
    if CFG.algType == type.COVARIANCE:
        f, sigSquare =  cmSvdKriging(s, CFG)    
        return f, math.sqrt( abs(sigSquare) ) 
    
    if CFG.algType == type.RASMUSSEN_WILLIAMS:
        f, sigSquare =  rwKriging(s, CFG)    
        return f, math.sqrt( abs(sigSquare) ) 
 
def acmSvdKriging(s, CFG):
    '''
    do krigin using SVD and tychonoff regularization.

    we are looking to solve the following:
    [     |-1]   [   ]     [ ]
    [  C  |-1] * [lam]  =  [c]
    [     |-1]   [   ]     [ ]
    [1 1 1| 0]   [ m ]     [1]
    
    where:
    C is a covariance matrix between observations (have n of those)
    lambda are weights
    m is a lagrange multiplier
    c is the covariance between the given s and the n observations
    
    function parameters: 
    s - where we want to estimate our function \ process
    CFG - an object that contains all the data we need for the computation
    
    returns - mean and standard deviation for point s
    '''
    # unpack the variables
    X = CFG.X
    F = CFG.F
    r = CFG.r
    reg = CFG.reg
        
    # number of samples we have.
    n = len(F)
    
    # create the target c:
    c = np.zeros( n+1 )
    for i in range(0,n):
        c[i] = aux.cov(s,X[i],r)
    c[n] =  1.0
    
    lam = aux.tychonoffSvdSolver(CFG.U , CFG.S , CFG.V ,  c, reg)     # solve!!!
    
    m = lam[n]
    lam = lam[0:n]
    lam = lam/np.sum(lam) #make sure weights sum to 1
    
    # calculate the kriged estiamte
    f = np.zeros( len(F[0]) )
    for i in range(n):
        f = f + lam[i] * F[i]
        
    # calculate the variance    
    sigmaSquare = m + aux.cov(0,0,r) - np.sum(lam*c[0:n])  
    
    # we want to know if the variance turns out to be negative!
    if sigmaSquare  < 0 and -sigmaSquare > reg:
        print("Negative kriged variance. Probably because data points are too close. ")
        
    return f, sigmaSquare   


def cmSvdKriging(s, CFG):
    '''
    do krigin using SVD and tychonoff regularization.

    we are looking to solve the following:
    [     ]   [   ]     [ ]
    [  C  ] * [lam]  =  [c]
    [     ]   [   ]     [ ]
    
    where:
    C is a covariance matrix between observations (have n of those)
    lambda are weights
    c is the covariance between the given s and the n observations
    
    function parameters: 
    s - where we want to estimate our function \ process
    CFG - an object that contains all the data we need for the computation
    
    returns - mean and standard deviation for point s
    '''
    # unpack the variables
    X = CFG.X
    F = CFG.F
    U = CFG.U
    S = CFG.S
    V = CFG.V
    r = CFG.r
    reg = CFG.reg
    
    # number of samples we have.
    n = len(F)
    
    # create the target c:
    c = np.zeros( n )
    for i in range(0,n):
        c[i] = aux.cov(s,X[i],r)
    
    b = np.dot(np.transpose(U), c)
    
    
    # solve for lambda 
    x = b*S/(S*S + reg )
    lam = np.dot( np.transpose(V) ,  np.transpose(x) )
    
    f = np.zeros( len(F[0]) )
    for i in range(n):
        f = f + lam[i] * F[i]
        
    sigmaSquare =  aux.cov(0,0,r) - np.sum(lam*c[0:n])
    
     # we want to know if the variance turns out to be negative!
    if sigmaSquare  < 0 and -sigmaSquare > reg:
        print("Negative kriged variance. Probably because data points are too close. ")
        
    return f, sigmaSquare   


def rwKriging(s, CFG):
    '''
    use algortihm 2.1 from page 19 of the book 
    "Gaussian Processes for Machine Learning" by
    Rasmussen and Wiliams. They solve using Cholesky. Here
    we use the SVD with tychonoff regularization instead.
    '''
    # unpack the variables
    X = CFG.X
    y = np.array( CFG.F )
    y = np.ravel(y)

    # number of samples we have.
    n = len(X)
    
    r = CFG.r
    reg = CFG.reg
    
    

    # create the target c:
    k = np.zeros( n )
    for i in range(0,n):
        k[i] = aux.cov(s,X[i],r)
    
    
    
    # solve for lambda 
    alpha = aux.tychonoffSvdSolver(CFG.U, CFG.S, CFG.V, y, reg)
    
    f = np.zeros( len(CFG.F[0]) )
    for i in range(n):
        f = f + alpha[i] * k[i]
        
    tmp = aux.tychonoffSvdSolver(CFG.U, CFG.S, CFG.V, k, reg)
    sigmaSquare =  aux.cov(0,0,r) - np.sum(k*tmp)
    if sigmaSquare  < 0 and -sigmaSquare > 10*reg:
        print(" negative variance ")
    return f, sigmaSquare   
