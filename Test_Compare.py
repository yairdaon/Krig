'''
Created on May 13, 2014

@author: daon
'''
import kernel.aux as aux
import kernel.kriging as kg
import numpy as np


# create locations where values of log 
# likelihood are known
X = []
x1 =  np.array( [ 1.5001] )
x2 =  np.array( [ 1.4999] )
x3 =  np.array( [ 1.5] )
X.append(x1)
X.append(x2)
X.append(x3)

F = []
for i in range(0,3):
    F.append(   np.array([1.5])   )
    
s = np.array( [ 1.0 ])
r = 1.0


C = aux.augCovMat(X, r)   
print "Different kriging sub-routines:"
print

print "Solve for the augmented covariance matrix:" 
tmp, sig2= kg.augCovKriging(s, X, F, C, r)
print "result: " + str( tmp) + str( sig2 )
print " with error = " + str( abs(tmp - 1.5))

print "* * *"

print "Solve using the inverse covariance matrix:" 
tmp, sig2= kg.invCovKriging(s, X, F, aux.invCovMat(X, r), r)
print "result: " + str( tmp) + str( sig2 )
print " with error = " + str( abs(tmp - 1.5))

print "* * *"

print "Solve using the inverse of the AUGMENTD covariance matrix:"
tmp, sig2= kg.invAugKriging(s, X, F, aux.invAugCovMat(X, r), r)
print "result: " + str( tmp) + str( sig2 )
print " with error = " + str( abs(tmp - 1.5))

