'''
Created on May 6, 2014

@author: daon
'''
#from __future__ import print_function
import kernel.sampler as smp
import matplotlib.pyplot as plt
import numpy as np
import kernel.config as cfg

a = cfg.Config()
k = 60
n = 1500
M = 10.0
r = 1.3

for j in range(0,k):
    
    # locations where log-likelihood of gaussian is known
    x = np.array( [ -10.0 + j*20.0/k ] )    
    cfg.Config.addPair(a, x, -x*x/2.0)
    
cfg.Config.setM(a, M)
cfg.Config.setR(a, r)
cfg.Config.setMatrices(a)


# take n samples
for i in range(n):
    print( "Gaussian test, sample " + str(i) + " of " + str(n))
    smp.sampler(a)

plt.hist(a.X, 60)
plt.show()