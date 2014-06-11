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
import matplotlib.pyplot as plt
import kernel.config as cfg
import math
import os
import kernel.type as type

np.random.seed(1792) # for reproducibility
os.system("mkdir MovieFrames")
os.system("rm -f MovieFrames/*.png")     
os.system("rm -f Movie1.mpg")     
              

# RUN PARAMETERS!!!
#     Parameters for the algorithm
r     = 1.3       # The length scale parameter in the Gaussian process covariance function.
f = truth.trueLL                    # the true log-likelihood function
M = 4.0
xMin = -M
xMax = M
yMax = np.max( np.abs(f( np.arange(xMin,xMax, 1.0)))) 
#yMax = math.sqrt( yMax )
yMin = -yMax
x = np.arange(xMin, xMax, 0.05)
delay = 8
nf    =10       # The number of evaluations of the true likelihood

#    Start the algorithm using these points
StartPoints = []
StartPoints.append( np.array( [ 0 ] )  )#-0.5)*(M/2.0) )
StartPoints.append( np.array( [0.5] )  )#-0.5)*(M/2.0) )
##StartPoints.append( np.array( [ -M/2.0 ]))
##StartPoints.append( np.array( [  M/2.0 ]))

#     Initializations of the algorithm
a = cfg.Config()
a.state = np.random.get_state()
for point in StartPoints:
    cfg.Config.addPair(a, point, f(point))
cfg.Config.setR(a, r)
cfg.Config.setM(a, M)
cfg.Config.setType(a, type.RASMUSSEN_WILLIAMS)



kriged = np.zeros( x.shape )
limit = np.zeros( x.shape )
true = f(x) # the real log likelihood
for frame in range (nf+1):
    limAtInfty = cfg.Config.getLimitSVD(a)
    for j in range(0,len(x)):
        kriged[j] = kg.kriging(x[j] ,a)[0]
        limit[j] = limAtInfty
        
    for k in range(delay):
        plt.figure( frame*delay + k )
        
        curve1  = plt.plot(x, kriged , label = "kriged log-likelihood")
        curve2 =  plt.plot(x, true, label = "true log-likelihood")
        curve3 =  plt.plot( a.X, a.F, 'bo', label = "sampled points ")
        curve4  = plt.plot(x, limit, 'g', label = "kriged value at infinity")

        plt.setp( curve1, 'linewidth', 3.0, 'color', 'k', 'alpha', .5 )
        plt.setp( curve2, 'linewidth', 1.5, 'color', 'r', 'alpha', .5 )

        plt.axis([xMin, xMax, yMin, yMax])
        PlotTitle = 'Kriged Log-Likelihood Changes in Time. r = ' + str(r) + " Algorithm: " + a.algType.getDescription()
        plt.title( PlotTitle )
        textString = 'using  ' + str(frame ) + ' sampled points' 
        plt.text(1.0, 1.0, textString)
        plt.legend(loc=1,prop={'size':6})    
        FrameFileName = "MovieFrames/Frame" + str(frame*delay + k) + ".png"
        plt.savefig(FrameFileName)
        plt.close(frame*delay + k)
        if (frame*delay + k) % 10 == 0:
            print( "saved file " + FrameFileName + ".  " + str(frame*delay) +  " / " + str(nf*delay) )
    smp.sampler(a)

os.system("ffmpeg -i MovieFrames/Frame%d.png Movie1.mpg") 
os.system("vlc Movie1.mpg")     
    
            
