'''
Created on May 13, 2014

@author: daon
'''
import kernel.aux as aux
import kernel.kriging as kg
import numpy as np
import kernel.config as cfg

a = cfg.Config()

x1 =  np.array( [ 1.5001] )
f1 =  np.array([1.5])
cfg.Config.addPair(a, x1, f1)

x2 =  np.array( [ 1.4999] )
f2 =  np.array([1.5])
cfg.Config.addPair(a, x2, f2)

x3 =  np.array( [ 1.5] )
f3 =  np.array([1.5])
cfg.Config.addPair(a, x3, f3)

r = 1.0
cfg.Config.setR(a, r)
cfg.Config.setMatrices(a)
    
s = np.array( [ 1.0 ])


print "Different kriging sub-routines:"
print

print "Solve for the augmented covariance matrix:" 
tmp, sig2= kg.augCovKriging(s, a)
print "result: " + str( tmp) + str( sig2 )
print " with error = " + str( abs(tmp - 1.5))

print "* * *"

print "Solve using the inverse covariance matrix:" 
tmp, sig2= kg.invCovKriging(s, a)
print "result: " + str( tmp) + str( sig2 )
print " with error = " + str( abs(tmp - 1.5))

print "* * *"

print "Solve using the inverse of the AUGMENTD covariance matrix:"
tmp, sig2= kg.invAugKriging(s, a)
print "result: " + str( tmp) + str( sig2 )
print " with error = " + str( abs(tmp - 1.5))

print "* * *"

print "Solve using the SVD of the AUGMENTD covariance matrix with Tychonoff regularization:"
tmp, sig2= kg.svdKriging(s, a)
print "result: " + str( tmp) + str( sig2 )
print " with error = " + str( abs(tmp - 1.5))

