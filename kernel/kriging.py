'''
Created on Apr 29, 2014

@author: daon
'''

import numpy as np
import math
import kernel.aux as aux
import kernel.config as cfg


# this is where we decide actually which kriging subroutine we use.
# calling them always has the same syntax, though. It is C that takes
# different forms.
def kriging(s, CFG):
    
    f, sigSquare =  svdKriging(s, CFG)
    #f, sigSquare = augCovKriging(s, CFG)
    
    return f, math.sqrt( abs(sigSquare) ) 
 

def svdKriging(s, CFG):
    
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
    
    # print "n = " + str(n)
    # now create the target c:
    c = np.zeros( n+1 )
    for i in range(0,n):
        c[i] = aux.cov(s,X[i],r)
    c[n] =  1.0
    
    b = np.dot(np.transpose(U), c)
    
    #s =  np.diag(S)
    # solve for  lambda 
    x = b*S/(S*S + reg )
    # print np.shape(x)
    lam = np.dot( np.transpose(V) ,  np.transpose(x) )
    m = lam[n]
    
    lam = lam[0:n]
    lam = lam/np.sum(lam)
    
    f = np.zeros( len(F[0]) )
    # print " this is f" 
    # print  f
    # sigmaSquare = 0.0
    # find the answer
    for i in range(n):
        f = f + lam[i] * F[i]
        #print "sig2 = " + str(sigmaSquare)
        #print "lam[i] * c[i] = " + str( lam[i]*c[i] )
    #answer = lam
    sigmaSquare = m + aux.cov(0,0,r) - np.sum(lam*c[0:n])  
    
    return f, sigmaSquare    
    
# do kriging. Hewr we calculate the kriging weights
# we are looking to solve the following:
#    
# C*lambda = c + m
# \sum lambda = 1
#
# the solution may be found in page 256 of
# "Regression Models for Time Series Analysis"
# By Kedem and Fokianos, 2002 John Wiley & Sons.
#
# We use Cinv as C^{-1} where C is a covariance matrix between observations
# lambda are weights
# m is a lagrange multiplier
# c is the covariance between the given s and the n observations
#
# parameters: 
# s - where we want to estimate our function \ process
# X - a numpy list of x values. each x is a numpy (row) array 
# F - similar to X.  F[i] is a numpy (row) array where F[i] = f(X[i]).
# Cinv - the inverse of the covariance matrix. 
# r - a hyper parameter, responsible for the spread of the covariance
#
#returns - mean and standard deviation for point s

def weights(s, X, Cinv, r):
    
    # number of samples we have.
    n = len(X)
    
    # now create the target c:
    c = np.zeros( (n,1) )
    
    for i in range(0,n):
        c[i] = aux.cov(s,X[i],r)
          
    # m = 1 - ones'Cinv*c / ones'Cinv*ones , where 'ones' is a vector of ones
    m =  (  1.0 -  np.sum( np.dot( Cinv, c) )    )/(   np.sum(Cinv)   )
    
    # define a vector of ones
    ones = np.ones( (n,1) )
    
    # find the weights lambda
    lam = np.dot(Cinv , c + m*ones)
    
    # if (sum(lam) - 1.0) != 0:
    #    print "sum of lambdas is  " + str( sum(lam) ) 
    # we normalize the lambdas
    #lam = lam/math.fsum(lam)
    
    return lam, m, c



def invCovKriging(s, CFG): #  X, F, Cinv, r):
    
    # unpack the variables
    X = CFG.X
    F = CFG.F
    Cinv = CFG.icm
    r = CFG.r
    
    # calculate weights
    lam, m, c = weights(s, X, Cinv, r)

    f = 0
    #use weights to find new kriged value
    for i in range(len(lam)):
        f = f + F[i]*lam[i]
        
    sigmaSquare =  aux.cov(0,0,r) + m - math.fsum( lam*c ) 
    
    return f, sigmaSquare





# do kriging, the old way.
# we are looking to solve the following:
#    
#[     |-1]   [   ]     [ ]
#[  C  |-1] * [lam]  =  [c]
#[     |-1]   [   ]     [ ]
#[--------]   [ m ]     [1]
#[1 1 1| 0]
#
# the above matrix will be henceforth 
# called "the (augmented) covariance matrix"
#
# where:
# C is a covariance matrix between observations (have n of those)
# lambda are weights
# m is a lagrange multiplier
# c is the covariance between the given s and the n observations
#
# function parameters: 
# s - where we want to estimate our function \ process
# X - a numpy list of x values. each x is a numpy (row) array 
# F - similar to X.  F[i] is a numpy (row) array where F[i] = f(X[i]).
# C - the (augmented) covariance matrix. 
# r - a hyper parameter, responsible for the spread of the covariance
#
#returns - mean and standard deviation for point s



def augCovKriging(s, CFG): # X, F, C, r):
    

    # unpack the variables
    X = CFG.X
    F = CFG.F
    C = CFG.acm
    r = CFG.r
    # number of samples we have.
    n = len(F)
    
    #print "n = " + str(n)
    # now create the target c:
    c = np.zeros( (n+1,1) )
    for i in range(0,n):
        c[i] = aux.cov(s,X[i],r)
    c[n] =  1.0
    
    # solve for v = ( lambda, m )
    v = np.linalg.solve(C,c)
    lam = v[0:n:1]
    m = v[n]
    
    f = np.zeros( (1,len(F[0])) )
    # print " this is f" 
    # print  f
    # sigmaSquare = 0.0
    # find the answer
    for i in range(n):
        f = f + lam[i] * F[i]
        #print "sig2 = " + str(sigmaSquare)
        #print "lam[i] * c[i] = " + str( lam[i]*c[i] )
    #sigma = math.sqrt(sigmaSquare)-
    #answer = lam
    sigmaSquare = 2.0*m - np.sum( v*c ) + aux.cov(0,0,r)  
    
    return f, sigmaSquare
    
    
    
def invAugKriging(s, CFG): # X, F, Cinv, r):
    
    # unpack the variables
    X = CFG.X
    F = CFG.F
    Cinv = CFG.iacm
    r = CFG.r
    
    # number of samples we have.
    n = len(F)
    
    #print "n = " + str(n)
    # now create the target c:
    c = np.zeros( (n+1,1) )
    for i in range(0,n):
        c[i] = aux.cov(s,X[i],r)
    c[n] =  1.0
    
    # solve for v = ( lambda, m )
    v = np.dot(Cinv,c)
    lam = v[0:n:1]
    m = v[n]
    
    f = np.zeros( (1,len(F[0])) )
    # print " this is f" 
    # print  f
    # sigmaSquare = 0.0
    # find the answer
    for i in range(n):
        f = f + lam[i] * F[i]
        #print "sig2 = " + str(sigmaSquare)
        #print "lam[i] * c[i] = " + str( lam[i]*c[i] )
    #sigma = math.sqrt(sigmaSquare)-
    #answer = lam
    sigmaSquare = 2.0*m + aux.cov(0,0,r)  - np.sum( v*c ) 

    #answer =  [f, math.sqrt(max( sigmaSquare,0 )  )]
    return f, sigmaSquare    
     
   
   
