'''
Created on May 19, 2014

@author: daon
'''
        
import numpy as np
import kernel.aux as aux
import math
   

class Config:   
    """
    This class holds all the data needed to do kriging, resample etc.
    """
    
    
    
    # state of PRNG, for reproducibility purposes
    state = np.random.get_state()
        
    def __init__(self):
        
        # list observations... 
        self.X = []
        # ...and corresponding log-likelihoods
        self.F = []
    
        # a hyper parameter
        self.r = 1.0
    
        # the regularization used in the kriging procedure
        self.reg = 100*np.finfo(np.float).eps

    
    def addPair(self,x,f):
        """ 
        add a location and its log likelihood to the lists
        """
        
        self.X.append(x)
        self.F.append(f)
        
    def setR( self,t):
        """
        set the hyper parameter r and the regularization
        we used in the kriging procedure
        """
        self.r = t
        self.reg = 100*math.sqrt(self.r)*np.finfo(np.float).eps
                          
    def setMatrices(self):
        """ 
        calculates the SVD of the augmented covariance matrix. 
        """
        
        self.acm = aux.augCovMat(self.X,self.r)    # acm  = augmented covariance matrix
        U, S, V = np.linalg.svd(self.acm, full_matrices = True, compute_uv = True)
        self.U = U
        self.S = S
        self.V = V
          
    def setM(self, M):
        """ 
        M is the size of box we're interested in. We set the probability outside the box
        to be zero
        """
        self.M = M


    def condition(self):
        """
        use SVD to find condition number of matrix
        """
        
        s = np.linalg.svd( self.acm )[1]
        self.acmCond = max(s)/min(s)
        
        s = np.linalg.svd( self.cm )[1]
        self.cmCond = max(s)/min(s)
        
     
    def getLimitSVD(self):
        """
        returns the kriged value "at infinity"
        very similar to the SVDkriging procedure
        """
       
        F = self.F
        S = self.S
        reg = self.reg
        n = len(F)
        
        c = np.zeros( n+1 )
        c[n] =  1.0
        b = np.dot(np.transpose(self.U), c)
        
        x = b*S/(S*S + reg )
        lam = np.dot( np.transpose(self.V) ,  np.transpose(x) )
        lam = lam[0:n]   
    
        lim = np.zeros( len(F[0]) )
        for i in range(n):
            lim = lim + lam[i] * F[i]    
        
        return lim