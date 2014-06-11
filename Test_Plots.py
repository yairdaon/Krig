'''
Created on Apr 29, 2014

@author: daon
'''
import kernel.aux as aux
import kernel.kriging as kg
import matplotlib.pyplot as plt
import numpy as np
import kernel.config as cfg
import kernel.type as type
import kernel.truth as truth

# allocating memory
x = np.arange(-10, 10, 0.05)
n = len(x)
f = np.zeros( n )
upper = np.zeros( n )
lower = np.zeros( n )
limit = np.ones( n )

X = []
x1 =  np.array( [ 1.1 ] )
x2 =  np.array( [ 1.0 ] )
x3 =  np.array( [ -1.1] )
x4 =  np.array( [ -3.0] )
X.append(x1)
X.append(x2)
X.append(x3)
X.append(x4)

a = cfg.Config()
for v in X:
    a.addPair(v, truth.trueLL(v))
    
a.setType(type.RASMUSSEN_WILLIAMS)
a.setR(1.3)
a.setMatrices()
limAtInfty = a.getLimitSVD()

# calculate the functions for the given input
for j in range(0,n):    
    v = kg.kriging(x[j] ,a)
    f[j] = v[0]
    upper[j] = v[0] + 1.96*v[1] 
    lower[j] = v[0] - 1.96*v[1]
    limit[j] = limAtInfty

curve1  = plt.plot(x, f, label = "kriged value")
curve2  = plt.plot(x, upper, label = "1.96 standard deviations")
curve3  = plt.plot(x, lower)
curve4  = plt.plot(x, limit, label = "kriged value at infinity")
curve5 =  plt.plot( a.X, a.F, 'bo', label = "sampled points ")

plt.setp( curve1, 'linewidth', 3.0, 'color', 'k', 'alpha', .5 )
plt.setp( curve2, 'linewidth', 1.5, 'color', 'r', 'alpha', .5 )
plt.setp( curve3, 'linewidth', 1.5, 'color', 'r', 'alpha', .5 )
plt.setp( curve4, 'linewidth', 1.5, 'color', 'b', 'alpha', .5 )

#plt.axis([-1, 1, -10, 0])

plt.legend(loc=1,prop={'size':6})    
plt.title("Kriging with bounds using " + a.algType.getDescription() )
plt.show()


