'''
Created on Jun 15, 2014

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
        os.system("mkdir MovieFrames1")
        os.system("rm -f MovieFrames1/*.png")     
        os.system("rm -f Movie1.mpg")    
        
        #     Initializations of the container object
        a = cfg.Config()
        
        # make it remember the state of the PRNG
        a.state = np.random.get_state()
        
        # The length scale parameter in the Gaussian process covariance function.
        r = 1.3
        cfg.Config.setR(a, r) 
            
        # the size of the box outside of which the probability is zero
        M = 4.0 
        cfg.Config.setM(a, M)

        # we know the true log-likelihood in these points
        StartPoints = []
        StartPoints.append( np.array( [ 0 ] )  )#-0.5)*(M/2.0) )
        StartPoints.append( np.array( [0.5] )  )#-0.5)*(M/2.0) )
        ##StartPoints.append( np.array( [ -M/2.0 ]))
        ##StartPoints.append( np.array( [  M/2.0 ]))
        
        f = truth.trueLL # the true log-likelihood function 
        for point in StartPoints:
            cfg.Config.addPair(a, point, f(point))
            
        # we use algorithm 2.1 from Rasmussen & Williams book
        cfg.Config.setType(a, type.RASMUSSEN_WILLIAMS)
        
        # keep the container in scope so we can use it later
        self.config = a
        
    def tearDown(self):
        '''
        after the test was run, create and show the movie.
        you need two programs installed on your machine so this works:
        you need ffmpeg to create the movie from the frames python saves 
        and you need vlc to watch the movie
        feel free to change these two lines here according to whatever
        programs you have installed in your system
        '''
         
        os.system("ffmpeg -i MovieFrames1/Frame%d.png Movie1.mpg") 
        os.system("vlc Movie1.mpg")     


    def testMovie1D(self):
        
        # the true log-likelihood function
        f = truth.trueLL 
        
        # the size of the box outside of which the probability is zero
        M = self.config.M 
        
        # the bounds on the plot axes
        xMin = -M
        xMax = M
        yMax = np.max( np.abs(f( np.arange(xMin,xMax, 1.0)))) 
        yMax = math.sqrt( yMax )
        #yMax = 2.0
        yMin = -yMax
        
        # all the x values for which we plot
        x = np.arange(xMin, xMax, 0.05)
        
        # we create each frame many times, so the movie is slower and easier to watch
        delay = 8
        
        # The number of evaluations of the true likelihood
        nf    = 9       
        
        # allocate memory for the arrays to be plotted
        kriged = np.zeros( x.shape )
        limit = np.zeros( x.shape )
        true = np.zeros( x.shape )
        
        
        # create frames for the ffmpeg programs
        for frame in range (nf+1):
            
            # the current a value of the kriged interpolant "at infinity"
            limAtInfty = self.config.getLimitSVD()
            
            # create the kriged curve and the limit curve
            for j in range(0,len(x)):
                kriged[j] = kg.kriging(x[j] ,self.config)[0]
                limit[j] = limAtInfty
                true[j] = f(x[j]) # the real log likelihood
            
            # each frame is saved delay times, so we can watch the movie at reasonable speed    
            for k in range(delay):
                plt.figure( frame*delay + k )
                
                # here we create the plot. nothing too fascinating here.
                curve1  = plt.plot(x, kriged , label = "kriged log-likelihood")
#                 print x.shape
#                 print true.shape
                curve2 =  plt.plot(x, true, label = "true log-likelihood")
                curve3 =  plt.plot( self.config.X, self.config.F, 'bo', label = "sampled points ")
                curve4  = plt.plot(x, limit, 'g', label = "kriged value at infinity")
        
                plt.setp( curve1, 'linewidth', 3.0, 'color', 'k', 'alpha', .5 )
                plt.setp( curve2, 'linewidth', 1.5, 'color', 'r', 'alpha', .5 )
        
                
                plt.axis([xMin, xMax, yMin, yMax])
                PlotTitle = 'Kriged Log-Likelihood Changes in Time. r = ' + str(self.config.r) + " Algorithm: " + self.config.algType.getDescription()
                plt.title( PlotTitle )
                textString = 'using  ' + str(frame ) + ' sampled points' 
                plt.text(1.0, 1.0, textString)
                plt.legend(loc=1,prop={'size':7})    
                FrameFileName = "MovieFrames1/Frame" + str(frame*delay + k) + ".png"
                plt.savefig(FrameFileName)
                plt.close(frame*delay + k)
                if (frame*delay + k) % 10 == 0:
                    print( "saved file " + FrameFileName + ".  " + str(frame*delay + k) +  " / " + str(nf*delay) )
                    
            # IMPORTANT - we sample from the kriged log-likelihood. this is crucial!!!!
            smp.sampler(self.config)
        


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()