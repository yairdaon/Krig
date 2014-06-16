'''
Created on Jun 5, 2014

@author: daon
'''
import unittest
import numpy as np
import kernel.sampler as smp 
import kernel.config as cfg 

class Test(unittest.TestCase):
    '''
    test the sampler. make sure that the min and max values we get
    are close to the boundary we set
    '''


    def setUp(self):
        '''
        set the test up, define everything we need
        '''
        
        # the charactersistc distance hyper parameter
        r = 1.0
        
        # size of box. if |x|_sup > M then P(x) = 0
        M = 0.5
        
        # the points in which we know the log likelihood
        X = []
        x1 =  np.array( [ 0.5] )
        x2 =  np.array( [ 1.0] )
        x3 =  np.array( [ 1.5] )
        x4 =  np.array( [ 1.25] )
        X.append(x1)
        X.append(x2)
        X.append(x3)
        X.append(x4)

        # setting up the container object:
        a = cfg.Config() # create the object
        for x in X: # set log-likelihoods
            a.addPair(x, np.array( [ 1.5 ] ) )
        a.setMatrices() #calculate the matrices required for kriging
        a.setR(r)  # set the hyper parameter r
        a.setM(M)  # set the size of the box
        a.setAddSamplesToDataSet( False ) # don't add samples to data or test will run for a long time
        
        # keep the container object in the scope of the whole test
        self.a = a

            
    def tearDown(self):
        pass


    def testMinMax1(self):
        '''
        we let the sampler run for many times. It samples from a 
        uniform distribution in [-M,M]. We expect the max and min
        values sampled to be close to -M and M. Being somewhat lazy, 
        we only check that max( maximum , -minimum) is close to M
        '''
        
        # set the seed, for reproducibility purposes
        np.random.seed(18394)
        
        # number of iterations used to demonstrate MinMax
        n = 10000

        # (initial) minimal sampled value. should converge to -M 
        minimum = 0

        # (initial) maximal sampled value. should converge to M 
        maximum = 0
        
        # do not print too much stuff to the screen. Change this if you wish to
        output = True
        
        # do many times:
        for i in range(n):
            
            # take a sample
            sample = smp.sampler(self.a) 
            
            # update the max
            if (sample > maximum):
                maximum = sample
            
            #update the min
            if (sample < minimum):
                minimum = sample
                
            # print to console?
            if output:
                print( "Test_MinMax, sample " + str(i) + " of " + str(n))
                print("Min = " + str(minimum) + ", Max = " + str(maximum) )
            
            if np.allclose(self.a.M, maximum):
                print("time is " + str(i))
                break
        M = self.a.M
        
        # print to console?
        if output:
            print "MinMax test: "
            print "maximum = " + str(maximum) + " should be close to " + str(M)
            print "minimum = " + str(minimum) + " should be close to " + str(-M) 
        maximum = max( maximum, -minimum )
        
    def testMinMax2(self):
            '''
            same as testMinMax1, just using different seed
            '''
            
            # set the seed, for reproducibility purposes
            np.random.seed(17)
            
            # number of iterations used to demonstrate MinMax
            n = 10000
    
            # (initial) minimal sampled value. should converge to -M 
            minimum = 0
    
            # (initial) maximal sampled value. should converge to M 
            maximum = 0
    
            # do many times:
            for i in range(n):
                
                # take a sample
                sample = smp.sampler(self.a) 
                
                # update the max
                if (sample > maximum):
                    maximum = sample
                
                #update the min
                if (sample < minimum):
                    minimum = sample
                
                #print( "Test_MinMax, sample " + str(i) + " of " + str(n))
                #print("Min = " + str(minimum) + ", Max = " + str(maximum) )
                if np.allclose(self.a.M, maximum):
                    print("time is " + str(i))
                    break
            M = self.a.M
            print "MinMax test: "
            print "maximum = " + str(maximum) + " should be close to " + str(M)
            print "minimum = " + str(minimum) + " should be close to " + str(-M) 
            maximum = max( maximum, -minimum )
            self.assertTrue( np.allclose(M, maximum)    )

if __name__ == "__main__":
    unittest.main()
