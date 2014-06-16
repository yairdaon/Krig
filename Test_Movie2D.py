'''
Created on Jun 16, 2014

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!
'''

import unittest
import kernel.kriging as kg
import kernel.sampler as smp
import kernel.truth as truth
import numpy as np
import matplotlib.pyplot as plt
import kernel.config as cfg
import math
import os
import kernel.type as type
from mpl_toolkits.mplot3d import axes3d


class Test(unittest.TestCase):
    '''
    if this does not work, it is most likely because you don't have 
    ffmpeg and\or vlc. go to the tearDown method to change these to
    whatever you have in your system.
    
    this unit test creates a movie. run it and see for yourself!!
    '''

    def setUp(self):
        '''
        this is where most of the setup is done for the movie
        '''
        
        # for reproducibility
        np.random.seed(1792) 
        
        # tell the OS to prepare for the movie and the frames
        os.system("mkdir MovieFrames2")
        os.system("rm -f MovieFrames2/*.png")     
        os.system("rm -f Movie2.mpg")    
        
        #     Initializations of the container object
        
        CFG = cfg.Config()
        
        # make it remember the state of the PRNG
        CFG.state = np.random.get_state()
        
        # The length scale parameter in the Gaussian process covariance function.
        r = 1.3
        CFG.setR(r) 
            
        # the size of the box outside of which the probability is zero
        M = 5.0 
        CFG.setM(M)

        # we know the true log-likelihood in these points
        StartPoints = []
        StartPoints.append( np.array( [ 0, 0 ] )  )#-0.5)*(M/2.0) )
        StartPoints.append( np.array( [0.5, 1.0] )  )#-0.5)*(M/2.0) )
        
        # choose the true log likelihood
        CFG.setLL( truth.norm2D )
        f = CFG.LL # the true log-likelihood function 
        for point in StartPoints:
            CFG.addPair( point, f(point) )
        
        # we use algorithm 2.1 from Rasmussen & Williams book
        CFG.setType( type.RASMUSSEN_WILLIAMS )
        
        # keep the container in scope so we can use it later
        self.CFG = CFG
        
    def tearDown(self):
        '''
        after the test was run, create and show the movie.
        you need two programs installed on your machine to make this work:
        you need ffmpeg to create the movie from the frames python saves 
        and you need vlc to watch the movie
        feel free to change these two lines here according to whatever
        programs you have installed in your system
        '''
         
        os.system("ffmpeg -i MovieFrames2/Frame%d.png Movie2.mpg") 
        os.system("vlc Movie2.mpg")     


    def testMovie2D(self):
        
        
        # the true log-likelihood function
        f = self.CFG.LL 
        
        # the size of the box outside of which the probability is zero
        M = self.CFG.M 
        
        # the bounds on the plot axes
        xMin = -M
        xMax = M
        yMax = 2.0
        yMin = -40.0
        
        # create the two meshgrids the plotter needs
        a  = np.arange(xMin, xMax, 0.1)
        b  = np.arange(xMin, xMax, 0.1)
        X, Y = np.meshgrid(a, b)
        
        # we create each frame many times, so the movie is slower and easier to watch
        delay = 8
        
        # The number of evaluations of the true likelihood
        nf    = 60      
        
        # allocate memory for the arrays to be plotted
        kriged = np.zeros( X.shape )
        
        # the curve of the true log-likelihood
        #true = f(x) # the real log likelihood
        
        p = np.zeros(2)
        
        # create frames for the ffmpeg programs
        for frame in range (nf+1):
            
            # the current a value of the kriged interpolant "at infinity"
            #limAtInfty = self.config.getLimitSVD()
            
            # create the kriged curve and the limit curve
            for j in range(len(a)):
                for i in range(len(b)):
                    p[0] = X[j,i]
                    p[1] = Y[j,i]    
                    kriged[j,i] = kg.kriging( p ,self.CFG )[0]
                    #limit[j] = limAtInfty
            
            # each frame is saved delay times, so we can watch the movie at reasonable speed    
            for k in range(delay):
                fig = plt.figure( frame*delay + k )
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_wireframe(X, Y, kriged, rstride=10, cstride=10)
                ax.set_xlim(xMin, xMax)
                ax.set_ylim(xMin, xMax)
                ax.set_zlim(yMin, yMax)

                PlotTitle = 'Kriged Log-Likelihood Changes in Time. r = ' + str(self.CFG.r) + " Algorithm: " + self.CFG.algType.getDescription()
                plt.title( PlotTitle )
                textString = 'using  ' + str(frame ) + ' sampled points' 
                #plt.text(1.0, 1.0, textString)
                plt.legend(loc=1,prop={'size':7})    
                FrameFileName = "MovieFrames2/Frame" + str(frame*delay + k) + ".png"
                plt.savefig(FrameFileName)
                plt.close(frame*delay + k)
                if (frame*delay + k) % 10 == 0:
                    print( "saved file " + FrameFileName + ".  " + str(frame*delay + k) +  " / " + str(nf*delay) )
                      
            # IMPORTANT - we sample from the kriged log-likelihood. this is crucial!!!!
            smp.sampler(self.CFG)
            

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()