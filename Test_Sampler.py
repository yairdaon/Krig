'''
Created on Jun 15, 2014

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!
'''
import unittest
import kernel.sampler as smp
import numpy as np
import kernel.config as cfg
import kernel.truth as truth


class Test(unittest.TestCase):
    '''
    we test the sampler.py module. 
    '''


    def setUp(self):
        '''
        set up the test. this means creating the container object 
        that holds all the data, settings, variables and flags 
        required for a successful run
        '''
        
        # for reproducibility purposes 
        np.random.seed(5012)
        
        # the size of the box outside of which we have zero probability
        M = 2.0
        
        # the length scale of the covariance function. this is a hyper parameter.
        r = 1.0
        
        # create and populate the container object:
        a = cfg.Config() # call the constructor...
        a.addPair(np.array ( [ 1.0, 1.0, 1.0 ]), np.array( [2.45])) #...add data...  
        a.setR(r) # ...set the length scale hyper parameter...
        a.setM(M) # ...set the box size...
        a.setMatrices() # ...calcualte the matrices neede for kriging...
        a.LL = truth.norm2D
        self.config = a # ... and keep the container in the scope of the test.


    def tearDown(self):
        pass


    def testSampler1(self):
        '''
        here we sample from the sampler. We choose to incorporate
        the sampled data into our data set. we set the appropirate variable to True
        (see below) although we do not need to - the container object chooses
        this option by default
        '''
                             
        # choose to add samples to data set
        self.config.setAddSamplesToDataSet( True )
                                                                                                                                      
       # take three samples
        smp.sampler(self.config)
        smp.sampler(self.config)
        smp.sampler(self.config)
                             
        self.assertEqual(len( self.config.X ) , 4 )
        self.assertEqual( len(self.config.X[0]) , len(self.config.X[1]) )             
        self.assertEqual( len(self.config.X[0]) , len(self.config.X[2]) )   
        self.assertEqual( len(self.config.X[0]) , len(self.config.X[3]) )             
          
                    
    def testSampler2(self):
        '''
        here we sample from the sampler. We choose NOT to incorporate
        the sampled data into our data set. we set the appropirate variable to False
        '''
        
        # choose to add samples to data set
        self.config.setAddSamplesToDataSet( False )
        
        # take four samples
        smp.sampler(self.config)
        smp.sampler(self.config)
        smp.sampler(self.config)
        smp.sampler(self.config)
        
        # take another one and incorporate it
        self.config.setAddSamplesToDataSet( True )
        smp.sampler(self.config)
        
        self.assertEqual(len( self.config.X ) , 2)
        
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()