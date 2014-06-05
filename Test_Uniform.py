'''
Created on May 9, 2014

@author: daon
'''

import kernel.sampler as smp
import numpy as np
import pylab as P
import kernel.config as cfg


# create locations where values of log 
# likelihood are known
X = []
x1 =  np.array( [ 0.5] )
x2 =  np.array( [ 1.0] )
x3 =  np.array( [ 1.5] )
x4 =  np.array( [ 1.25] )
X.append(x1)
X.append(x2)
X.append(x3)
X.append(x4)

# set all these known values to be the same
# should give a uniform distribution
F = []
for i in range(0,4):
    F.append(   np.array([1.5])   )
# the parameter controlling the covariance
r = 1.0

# we insist that beyond the [-M,M] box,
# all log-likelihood is zero
M = 1.0

# number of iterations before plot histogram
n = 100

# maximal sampled value. should converge to M 
minimum = 5.0

# maximal sampled value. should converge to -M 
maximum = -5.0

a = cfg.Config()
a.setR(r)
a.setM(M)
for i in range(4):
    a.addPair(X[i], F[i])
a.setMatrices()
samples =[]
# loop and demonstrate SLLN
for i in range(0,n):
    
    # use our sampler to sample
    temp = smp.sampler(a)
    samples.append(temp)
    # change the min and max if appropriate
    if (temp > maximum):
        maximum = temp
    if (temp < minimum):
        minimum = temp
        
    # print current step
    #print "----------------------"
    #print "sample number " + str(i+1)
    #print "sample  = " + str(temp)
    #print "maximum = " + str(maximum)
    #print "minimum = " + str(minimum)  

samples = np.array( [samples] )
P.hist(samples)
P.show()