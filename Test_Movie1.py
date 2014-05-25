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

nf = 10
delay = 5
xMin = -5.0
xMax = 5.0
yMin = -15.0
yMax = 2.0


a = cfg.Config()

# the log-likelihood of a gaussian
f = truth.trueLL
  
# locations where log-likelihood of gaussian is known
s1 = np.array( [ -0.5  ] )
s2 = np.array( [  1.5  ] )
cfg.Config.addPair(a, s1, f(s1))
cfg.Config.addPair(a, s2, f(s2))

cfg.Config.setR(a, 0.5)
cfg.Config.setM(a, 10.0)
cfg.Config.setMatrices(a)



x = np.arange(-5.0, 5.0, 0.05)
n = len(x)
y = np.zeros( x.shape )
true = np.zeros( x.shape )

true = f(x)

for frame in range (nf):
    smp.sampler(a)
    for j in range(0,n):
        b = kg.kriging(x[j] ,a)
        y[j] = b[0]
        
    for k in range(delay):
        plt.figure( frame*delay + k )
        #print("* * * " + str(i) + " * * *")
        
        curve1  = plt.plot(x, y , label = "kriged value")
        curve2 =  plt.plot(x, true, label = " real value ")
        plt.setp( curve1, 'linewidth', 3.0, 'color', 'k', 'alpha', .5 )
        plt.setp( curve2, 'linewidth', 1.5, 'color', 'r', 'alpha', .5 )
        plt.axis([xMin, xMax, yMin, yMax])
        plt.title("Kriged Log-Likelihood Changes in Time. ")
        textString = 'using  ' + str(frame) + ' sampled points' 
        plt.text(1.0, 1.0, textString)
        plt.legend( loc = 2)
    
        FrameFileName = "MovieFrames/Frame" + str(frame*delay + k) + ".png"
        plt.savefig(FrameFileName)
        plt.close(frame*delay + k)
        print "saved file " + FrameFileName
    

