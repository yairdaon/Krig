'''
Created on Apr 29, 2014

@author: daon
'''
import unittest
import kernel.kriging as kg
import kernel.aux as aux
import numpy as np
import kernel.config as cfg

class Test(unittest.TestCase):
    
    def setUp(self):
        
        x1 =  np.matrix( [ 0 , 1 ] )
        x2 =  np.matrix( [ 1 , 0 ] )
        x3 =  np.matrix( [ 0 ,-1 ] )
        x4 =  np.matrix( [-1 , 0 ] )
        f1 = np.array( [0] )
        f2 = np.array( [2] )
        f3 = np.array( [0] )
        f4 = np.array( [-2] )

        a = cfg.Config()
        cfg.Config.addPair(a, x1, f1)
        cfg.Config.addPair(a, x2, f2)
        cfg.Config.addPair(a, x3, f3)
        cfg.Config.addPair(a, x4, f4)
        cfg.Config.setR(a, 1.0)
        cfg.Config.setMatrices(a)
        
        self.a = a
        
    def testSimple(self):
        s = np.array( [0, 0] )

        b , c = kg.kriging(s, self.a)
        self.assertTrue(np.allclose( np.array( [0] ) , b ))