'''
Created on Aug 5, 2014

@author: Yair Daon. email: fisrtname.lsatname@gmail.com
Feel free to write to me about my code!
'''
import unittest


class Test(unittest.TestCase):


    def testDensity(self):
        def setUp(self):
        
        # for reproducibility purposes
        np.random.seed( 1243 )
        
        # create the container object
        CFG = cfg.Config()
        
        # set plot bounds
        #self.M = 1
        
        # set prior loglikelihood to exponential
#         prior = lambda x: -np.linalg.norm(x)
#         CFG.setPrior(prior)
        
        # set true LL
        likelihood = truth.bigPoly1D
        CFG.setLL(likelihood)
        
        # use RW's algorithm
        CFG.setType(type.RASMUSSEN_WILLIAMS)
       
        # quick setup
        CFG.quickSetup(1)
        
        # create sampler...
        self.sampler = smp.Sampler ( CFG )
        k =  7 # ...decide how many initial points we take to resolve the log-likelihood
        for j in range(0,k): 
            print( "Initial samples " + str(j+1) + " of " + str(k))
            self.sampler.sample() # ... sample, incorporate into data set, repeat k times.
            
        self.CFG = CFG
     
    def testPrior(self):
        
        # allocating memory
        x = np.arange(-self.CFG.M, self.CFG.M, 0.05)
        n = len(x)
        f = np.zeros( n )
        true = np.zeros( n )
        prior = np.ones( n )

        # calculate the curves for the given input
        for j in range(0,n):    
            
            # do kriging, get avg value and std dev
            v = kg.kriging(x[j] , self.CFG) 
            f[j] =  (v[0]) # set the interpolant
            prior[j] = self.CFG.prior(x[j])   # set the limiting curve
            true[j]  = self.CFG.LL(   x[j])  
        
        #move to normal, non-exponential, scale
        fe = np.exp(f)
        priore = np.exp(prior)
        truee = np.exp(true)
        Fe = np.exp(np.asarray( self.CFG.F ))
        X =  np.asarray( self.CFG.X )
        
        # create a figure with two subplts
        fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True)

        # first plot
        curve1  = ax0.plot(x, f, label = "kriged LL")
        curve2  = ax0.plot(x, true, label = "true LL")
        #curve3  = ax0.plot(x, prior, label = "prior log-likelihood")
        curve4 =  ax0.plot(  self.CFG.X ,    self.CFG.F  , 'bo', label = "sampled points ")
        
        ax0.setp( curve1, 'linewidth', 3.0, 'color', 'k', 'alpha', .5 )
        ax0.setp( curve2, 'linewidth', 1.5, 'color', 'r', 'alpha', .5 )
        ax0.setp( curve4, 'linewidth', 1.5, 'color', 'b', 'alpha', .5 )
        
        ax0.legend(loc=1,prop={'size':7})    
        ax0.title("Kriged log likelihood")
        
        # scond plot
        curve5  = ax1.plot(x, fe, label = "exp(kriged LL)")
        curve6  = ax1.plot(x, true, label = "unnormalized density")
        #curve7  = ax1.plot(x, prior, label = "prior log-likelihood")
        curve8 =  ax1.plot(  X ,   Fe  , 'bo', label = "sampled points ")
        
        ax1.setp( curve5, 'linewidth', 3.0, 'color', 'k', 'alpha', .5 )
        ax1.setp( curve6, 'linewidth', 1.5, 'color', 'r', 'alpha', .5 )
        ax1.setp( curve8, 'linewidth', 1.5, 'color', 'b', 'alpha', .5 )
        # save
        fig.savefig("graphics/Test_Density: Kriged LL and resulting density")
        
        
        
        plt.close()


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testDensity']
    unittest.main()