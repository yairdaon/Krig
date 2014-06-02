'''
Created on May 20, 2014

@author: daon
'''

#  Make frames for a movie from  data from a Python formatted plot file

import kernel.aux as aux
import kernel.kriging as kg
import kernel.sampler as smp
import kernel.truth as truth
import numpy as np
import emcee as mc
import matplotlib.pyplot as plt
import string as st
import kernel.config as cfg


#     Parameters for the algorithm
nf    = 60        # The number of evaluations of the true likelihood
r     = 1.3       # The length scale parameter in the Gaussian process covariance function.
M = 10.0
f = truth.trueLL                    # the true log-likelihood function


#    Start the algorithm using these points
StartPoints = []
StartPoints.append( np.array( [ -0.5 ] ) )
StartPoints.append( np.array( [  1.5 ] ) )

#     Initializations of the algorithm
a = cfg.Config()
for point in StartPoints:
    cfg.Config.addPair(a, point, f(point))
cfg.Config.setR(a, r)
cfg.Config.setM(a, M)
cfg.Config.setMatrices(a)


# plot parameters
delay = 5
xMin = -20.0
xMax =  20.0
yMin = -15.0
yMax = 8.0

x = np.arange(xMin, xMax, 0.02)
n = len(x)
y = np.zeros( x.shape )
true = f(x) # the real log likelihood

for frame in range (nf):
    smp.sampler(a)
    for j in range(0,n):
        tmp = kg.kriging(x[j] ,a)
        y[j] = tmp[0]
        
    for k in range(delay):
        plt.figure( frame*delay + k )
        
        curve1  = plt.plot(x, y , label = "kriged value")
        curve2 =  plt.plot(x, true, label = " real value ")
        curve3 =  plt.plot( a.X, a.F, 'bo', label = " sampled points ")
        plt.setp( curve1, 'linewidth', 3.0, 'color', 'k', 'alpha', .5 )
        plt.setp( curve2, 'linewidth', 1.5, 'color', 'r', 'alpha', .5 )
        plt.axis([xMin, xMax, yMin, yMax])
        PlotTitle = 'Kriged Log-Likelihood Changes in Time. r = ' + str(r)
        plt.title( PlotTitle )
        textString = 'using  ' + str(frame + 1) + ' sampled points' 
        plt.text(1.0, 1.0, textString)
        plt.legend( loc = 3)
    
        FrameFileName = "MovieFrames/Frame" + str(frame*delay + k) + ".png"
        plt.savefig(FrameFileName)
        plt.close(frame*delay + k)
        # print "saved file " + FrameFileName