'''
Created on Apr 29, 2014

@author: daon
'''

import numpy as np
import math
import kernel.aux as aux
import kernel.config as cfg



def kriging(s, CFG):
    """
    this is where we decide actually which kriging subroutine we use.
    calling them always has the same syntax, though. 
    """
    f, sigSquare =  svdKriging(s, CFG)    
    return f, math.sqrt( abs(sigSquare) ) 
 
def svdKriging(s, CFG):
    """
    do krigin using SVD and tychonoff regularization.

    we are looking to solve the following:
    [     |-1]   [   ]     [ ]
    [  C  |-1] * [lam]  =  [c]
    [     |-1]   [   ]     [ ]
    [--------]   [ m ]     [1]
    [1 1 1| 0]
    
    where:
    C is a covariance matrix between observations (have n of those)
    lambda are weights
    m is a lagrange multiplier
    c is the covariance between the given s and the n observations
    
    function parameters: 
    s - where we want to estimate our function \ process
    CFG - an object that contains all the data we need for the computation
    
    returns - mean and standard deviation for point s
    """
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
    c = np.zeros( n+1 )
    for i in range(0,n):
        c[i] = aux.cov(s,X[i],r)
    c[n] =  1.0
    
    b = np.dot(np.transpose(U), c)
    
    
    # solve for lambda 
    x = b*S/(S*S + reg )
    lam = np.dot( np.transpose(V) ,  np.transpose(x) )
    m = lam[n]
    lam = lam[0:n]
    lam = lam/np.sum(lam) #make sure they ssum to 1
    
    f = np.zeros( len(F[0]) )
   
    for i in range(n):
        f = f + lam[i] * F[i]
    sigmaSquare = m + aux.cov(0,0,r) - np.sum(lam*c[0:n])  
    
    return f, sigmaSquare    