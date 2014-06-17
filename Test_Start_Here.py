'''
Created on Jun 17, 2014

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!
'''
import unittest
import kernel.config as cfg
import kernel.truth as truth
import kernel.sampler as smp
import pylab as P
import matplotlib.pyplot as plt
import numpy as np


class Test(unittest.TestCase):
    '''
    an easy unit test, to show you how the sampler works.
    the goal is to sample a gaussian.
    if you want to actually mess with the hard wired 
    parameters, go to the other tests (preferably Test_Gaussian, which is similar).
    '''


    def setUp(self):
        '''
        how to create and populate the container object
        '''
        # Welcome!!!
        print("Welcome to my package. I hope you'll enjoy the ride. The script that runs this test is Test_Start_Here.py. ")
        
        # create a container object. Everything is initialized to a default value
        self.CFG = cfg.Config()
        
        # set log-likelihood to that of 1D standard normal
        self.LL = truth.gaussian1D
        
        # the container knows the true log-likelihood so it can set itself up and go.
        self.CFG.quickSetup(1)
        
        # reproducibility 
        np.random.seed(235321)
        
        # take initial samples, calculate log likelihood and improve kriged log-likelihood
        initial = 35
        for i in range(initial):
            print( "Initial samples " + str(i+1) + " of " + str(initial))
            smp.sampler(self.CFG)
          
          
    def testBeginner(self):    
        '''
        an easy unit test, to show you how the sampler works
        '''  
        
        # now we've had enough - we just want to sample the posterior:
        self.CFG.setAddSamplesToDataSet( False )
        
        # take some samples. We DO NOT incorporate these into the data set
        n = 2000
        
        # allocate memory for the data
        self.samples = np.zeros(n)    
        
        for i in range(n):
            print( "Gaussian test, posterior sample " + str(i+1) + " of " + str(n))
            self.samples[i] = smp.sampler(self.CFG)

        
            
    def tearDown(self):
        '''
        after the sampling is done, we plot
        '''
        
        P.figure()
        tmp, bins, patches = P.hist(self.samples, 15, normed=1, histtype='stepfilled')
        P.setp(patches, 'facecolor', 'g', 'alpha', 0.75)
        P.title("Samples from the kriged (posterior) log-likelihood interpolating a Gaussian")
        P.show()

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()