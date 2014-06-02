'''
Created on Apr 29, 2014

@author: daon
'''
import kernel.aux as aux
import kernel.kriging as kg
import matplotlib.pyplot as plt
import numpy as np
import kernel.config as cfg


# allocating memory
x = np.arange(0.0, 2.0, 0.01)
n = len(x)
f = np.zeros( n )
upper = np.zeros( n )
lower = np.zeros( n )

#X = []
x1 =  np.array( [ 0.5 ] )
x2 =  np.array( [ 1.0 ] )
x3 =  np.array( [ 1.5 ] )
x4 =  np.array( [ 1.25 ] )
#X.append(x1)
#X.append(x2)
#X.append(x3)
#X.append(x4)

#F = []
f1 = np.array( [  1.5  ])
f2 = np.array( [  1.5  ])
f3 = np.array( [  1.5  ])
f4 = np.array( [  1.5  ])
#F.append(f1) 
#F.append(f2)
#F.append(f3) 
#F.append(f4)

a = cfg.Config()
cfg.Config.addPair(a, x1, f1)
cfg.Config.addPair(a, x2, f2)
cfg.Config.addPair(a, x3, f3)
cfg.Config.addPair(a, x4, f4)
cfg.Config.setR(a, 0.5)
cfg.Config.setMatrices(a)

# calculate the functions for the given input
for j in range(0,n):
    v = kg.kriging(x[j] ,a)
    #print a
    f[j] = v[0]
    upper[j] = v[0] + 1.96*v[1] 
    lower[j] = v[0] - 1.96*v[1] 

curve1  = plt.plot(x, f, label = "kriged value")
curve2  = plt.plot(x, upper, label = "1.96 standard deviations")
curve3  = plt.plot(x, lower)

plt.setp( curve1, 'linewidth', 3.0, 'color', 'k', 'alpha', .5 )
plt.setp( curve2, 'linewidth', 1.5, 'color', 'r', 'alpha', .5 )
plt.setp( curve3, 'linewidth', 1.5, 'color', 'r', 'alpha', .5 )
plt.legend()
plt.title("Kriging with bounds")
plt.show()



