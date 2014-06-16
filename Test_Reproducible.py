'''
Created on Jun 14, 2014

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!
'''
import unittest
import numpy as np
import kernel.config as cfg
import kernel.truth as truth
import kernel.sampler as smp

class Test(unittest.TestCase):
    '''
    make sure we can reproduce results.
    try commenting out one of the seeds and see how this fails
    '''


    def setUp(self):
        
        np.random.seed(127)
        
        # parameters:
        r = 1.0 # length scale hyper parameter
        M = 10.0 # size of box with non-vanishing probability
        f = truth.trueLL # the real log-likelihood
        
        # start the algorithm using these points
        StartPoints = []
        StartPoints.append( np.array( [ -1.5 ] ) )
        StartPoints.append( np.array( [  1.5 ] ) )

        # creating the container object...
        a = cfg.Config()
        for point in StartPoints:
            cfg.Config.addPair(a, point, f(point)) # ...populating it with (point,value) pairs...
        cfg.Config.setR(a, r) # ...setting the hyper parametr r...
        cfg.Config.setM(a, M) # ...setting the box size M...
        cfg.Config.setMatrices(a) #... set the matrices we use for kriging...
        smp.sampler(a) # ... and sample two points using the given seed
        smp.sampler(a) # note: the sampler adds these points to the container a on its own
        
        # after all of this - this is what the sampler samples
        self.x = smp.sampler(a)
        
    def tearDown(self):
        pass


    def testReproducibility(self):
        '''
        tests that using the same seed, we can reproduce results.
        note: this is what we did in the setUp method above EXCEPT
        the last two lines.
        '''
        
        np.random.seed(127)
        r = 1.0
        M = 10.0
        f = truth.trueLL
        #    Start the algorithm using these points
        StartPoints = []
        StartPoints.append( np.array( [ -1.5 ] ) )
        StartPoints.append( np.array( [  1.5 ] ) )

        #     Initializations of the algorithm
        a = cfg.Config()
        for point in StartPoints:
            cfg.Config.addPair(a, point, f(point))
        cfg.Config.setR(a, r)
        cfg.Config.setM(a, M)
        cfg.Config.setMatrices(a)
        smp.sampler(a)
        smp.sampler(a)
        
        # now we put the sample in y
        self.y = smp.sampler(a)   
        
        # and compare
        self.assertEqual(self.x, self.y)  

    def testReproducibilityFails(self):
            '''
            tests that using different seeds, we cannot expect to
            reproduce results. 
            note: this is what we did in the testReproducibility method 
            above EXCEPT for the first line and the last.
            '''
            
            # Seed is commented, so we cannot expect reproducibility
            #np.random.seed(127)
            r = 1.0
            M = 10.0
            f = truth.trueLL
            #    Start the algorithm using these points
            StartPoints = []
            StartPoints.append( np.array( [ -1.5 ] ) )
            StartPoints.append( np.array( [  1.5 ] ) )
    
            #     Initializations of the algorithm
            a = cfg.Config()
            for point in StartPoints:
                cfg.Config.addPair(a, point, f(point))
            cfg.Config.setR(a, r)
            cfg.Config.setM(a, M)
            cfg.Config.setMatrices(a)
            smp.sampler(a)
            smp.sampler(a)
            
            # now we put the sample in y
            self.y = smp.sampler(a)   
            
            # and compare. they should be different!
            self.assertTrue( not np.array_equal(self.x, self.y) )        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()