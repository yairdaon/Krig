'''
Created on Jun 5, 2014

@author: daon
'''
import unittest
import numpy as np
import kernel.config as cfg
import kernel.truth as truth
import kernel.sampler as smp
class Test(unittest.TestCase):
    """
    make sure we can reproduce results.
    try commenting out one of the seeds and see how this fails
    """


    def setUp(self):
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
        self.x = smp.sampler(a)
        
    def tearDown(self):
        pass


    def testReproducibility(self):
        
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
        self.y = smp.sampler(a)   
        self.assertEqual(self.x, self.y)  


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()