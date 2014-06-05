'''
Created on May 12, 2014

@author: daon
'''
import numpy as np

def trueLL(s):
    """
    return what we believe is the true log-likelihood.
    If you have some external procedure to calculate log-likelihood, 
    this is where it goes
    """
    #return -2*s**4 + 15*s**2
    #return -s*s
    return -(s**6  + 3.5*s**4  - 2.5*s**3 - 12.5*s**2 + 1.5*s + 9 )
    #return np.ones( np.shape(s))