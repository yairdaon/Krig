'''
Created on May 13, 2014

@author: daon
'''
import kernel.kriging as kg
import kernel.aux as aux
import numpy as np


# print np.random.rand( 5,1 )
r = 1.0
X = []
for i in range(13):
    X.append(  np.random.rand( 30, 1)  )
Cinv = aux.invCovMat(X, r)

s = np.array( [ 0,0 ])
n = len(X)
# the weights function is supposed to solve C*lam = c+m
# with sum lam = 1
lam, m, c = kg.weights(s, X, Cinv, r)

#print lam
#print m
#print c

ones = np.ones(c.shape )
v = np.dot(aux.covMat(X,r), lam) - c - m*ones
print("If all of theses are zero, then you're OK!")
print(v)
print(sum(lam) - 1.0)

