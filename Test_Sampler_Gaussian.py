'''
Created on May 6, 2014

@author: daon
'''
#from __future__ import print_function
import kernel.aux as aux
import kernel.kriging as kg
import kernel.sampler as smp
import matplotlib.pyplot as plt
import numpy as np
import emcee as mc
import kernel.config as cfg

a = cfg.Config()


# create locations where values of log 
# likelihood are known
#X = []
#F = []
k = 3
for j in range(0,k):
    
    # locations where log-likelihood of gaussian is known
    x = np.array( [ -10.0 + j*20.0/k ] )
    #X.append( x )
    
    # the log-likelihood of a gaussian
    f = -x*x/2.0
    #F.append(f)
    cfg.Config.addPair(a, x, f)
    
cfg.Config.setM(a, 10.0)
cfg.Config.setMatrices(a)



# We'll sample with 20 walkers.
# sampler = smp.sampler(X, F, M, r)

for i in range (20):
    print("* * * " + str(i) + " * * *")
    smp.sampler(a)


print a.X
print a.F

#X = np.array(X)
#F = np.array(F)
plt.hist(a.X, 100)
plt.show()
# Finally, you can plot the projected histograms of the samples

#plt.hist(sampler.flatchain[:,0], 100)
#plt.show()