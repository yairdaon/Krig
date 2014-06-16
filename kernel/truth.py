'''
Created on May 12, 2014

@author: daon
'''
import numpy as np

def trueLL(s):
    '''
    return what we believe is the true log-likelihood.
    If you have some external procedure to calculate log-likelihood, 
    this is where it goes
    
    '''
    
    return np.sin(3*s)
    
def gaussian1D(s):
    '''
    the log likelihood of a 1D standard Gaussian
    '''
    return -s*s/2.0

def doubleWell1D(s):
    '''
    this log likelihood is a double well potential (upside down)
    '''
    return -2*s**4 + 15*s**2

def bigPoly1D(s):
    '''
    the following polynomial
    '''
    return -(s**6  + 3.5*s**4  - 2.5*s**3 - 12.5*s**2 + 1.5*s )
    
def norm2D(s):
    '''
    some 2D log likelihood function
    '''
    t = np.linalg.norm(s)
    return -np.array( [t*t] )
    