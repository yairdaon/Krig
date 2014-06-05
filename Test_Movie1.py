'''
Created on May 20, 2014

@author: daon
'''

#  Make frames for a movie from  data from a Python formatted plot file

#import kernel.aux as aux
import kernel.kriging as kg
import kernel.sampler as smp
import kernel.truth as truth
import numpy as np
#import emcee as mc
import matplotlib.pyplot as plt
import string as st
import kernel.config as cfg

np.random.seed(1792)

#     Parameters for the algorithm
nf    = 20        # The number of evaluations of the true likelihood
r     = 1.3       # The length scale parameter in the Gaussian process covariance function.
M = 20.0
f = truth.trueLL                    # the true log-likelihood function


#    Start the algorithm using these points
StartPoints = []
StartPoints.append( np.array( [ -0.5 ] ) )
StartPoints.append( np.array( [  0.5 ] ) )

#     Initializations of the algorithm
a = cfg.Config()
a.state = np.random.get_state()
for point in StartPoints:
    cfg.Config.addPair(a, point, f(point))
cfg.Config.setR(a, r)
cfg.Config.setM(a, M)
cfg.Config.setMatrices(a)


# plot parameters
delay = 8
# xMin = -M
# xMax =  M
yMin = -60000000.0
yMax = -yMin

x = np.arange(-M, M, 0.05)
n = len(x)
y = np.zeros( x.shape )
limit = np.zeros( x.shape )
true = f(x) # the real log likelihood

for frame in range (nf+1):
    limAtInfty = cfg.Config.getLimitSVD(a)
    for j in range(0,n):
        tmp = kg.kriging(x[j] ,a)
        y[j] = tmp[0]
        limit[j] = limAtInfty
        
    for k in range(delay):
        plt.figure( frame*delay + k )
        
        curve1  = plt.plot(x, y , label = "kriged value")
        curve2 =  plt.plot(x, true, label = " real value ")
        curve3 =  plt.plot( a.X, a.F, 'bo', label = " sampled points ")
        curve4  = plt.plot(x, limit, 'g', label = "kriged value at infinity")

        plt.setp( curve1, 'linewidth', 3.0, 'color', 'k', 'alpha', .5 )
        plt.setp( curve2, 'linewidth', 1.5, 'color', 'r', 'alpha', .5 )

        plt.axis([-M, M, yMin, yMax])
        PlotTitle = 'Kriged Log-Likelihood Changes in Time. r = ' + str(r)
        plt.title( PlotTitle )
        textString = 'using  ' + str(frame ) + ' sampled points' 
        plt.text(1.0, 1.0, textString)
        plt.legend(loc=1,prop={'size':6})    
        FrameFileName = "MovieFrames/Frame" + str(frame*delay + k) + ".png"
        plt.savefig(FrameFileName)
        plt.close(frame*delay + k)
        if (frame*delay + k) % 15 == 0:
            print "saved file " + FrameFileName + ".  " + str(frame*delay) +  " / " + str(nf*delay) 
    smp.sampler(a)

        
        
        