'''
Created on Jun 5, 2014

@author: daon
'''
import unittest
import kernel.config as cfg
import numpy as np
import kernel.sampler as smp

class Test(unittest.TestCase):
    """
    test the sampler. make sure that the min and max values we get
    are close to the boundary we set
    """


    def setUp(self):
        r = 1.0
        M = 0.1
        X = []
        x1 =  np.array( [ 0.5] )
        x2 =  np.array( [ 1.0] )
        x3 =  np.array( [ 1.5] )
        x4 =  np.array( [ 1.25] )
        X.append(x1)
        X.append(x2)
        X.append(x3)
        X.append(x4)

        a = cfg.Config()
        a.setR(r)
        a.setM(M)
        # set all these known values to be the same
        for i in range (len(X)):
            a.addPair(X[i], np.array( [ 1.5  ] ) )
    
        a.setMatrices()
        self.a = a

            
    def tearDown(self):
        pass


    def testMinMax(self):
        # number of iterations used to demonstrate MinMax
        n = 500

        # maximal sampled value. should converge to M 
        minimum = 0

        # maximal sampled value. should converge to -M 
        maximum = 0

        # loop and demonstrate SLLN
        for i in range(n):
            print( "MinMax test, sample " + str(i) + " of " + str(n))
            temp = smp.sampler(self.a)
            if (temp > maximum):
                maximum = temp
            if (temp < minimum):
                minimum = temp
        M = self.a.M
        print "MinMax test: "
        print "maximum = " + str(maximum) + " should be close to " + str(M)
        print "minimum = " + str(minimum) + " should be close to " + str(-M) 
        maximum = max( maximum, -minimum )
        self.assertTrue( np.allclose(M, maximum)    )

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()