'''
Created on May 13, 2014

@author: daon
'''
import kernel.aux as aux
import kernel.kriging as kg
import kernel.sampler as smp
import matplotlib.pyplot as plt
import numpy as np


X = []
X.append( np.array ( [ 1.0, 1.0, 1.0 ]))
X.append
F = []
F.append(np.array( [2.45]))
M = 2.0
r = 1.0
smp.sampler(X, F, M, r)
print "X = " + str(X)
print "F = " + str(F)
smp.sampler(X, F, M, r, "Old")
print "X = " + str(X)
print "F = " + str(F)

print "The lengths of X and F should equal:"
print "len(X) = " + str(len(X)) + ", len(F) = " + str(len(F))