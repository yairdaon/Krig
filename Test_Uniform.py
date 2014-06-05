'''
Created on Jun 5, 2014

@author: daon
'''
import unittest 
import kernel.config as cfg
import numpy as np
import kernel.kriging as kg
class TestKriging(unittest.TestCase):
    """ 
    test kriging by mkin sure the procedure outputs
    a constant 1.5 when it is given constant 1.5 
    input
    """
    def setUp(self):
        
        r = 1.0
        M = 1.0

        # create locations where values of log 
        # likelihood are known
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
        
    def testUniform(self):
        #self.assertTrue(    np.allclose( np.array([1.5]) , kg.kriging(np.array([5.0]),self.a )[0] )      )
        self.assertTrue(    np.allclose( np.array([1.5]) , kg.kriging(np.array([20.0]),self.a )[0] )      )



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()