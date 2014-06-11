'''
Created on May 19, 2014

@author: daon
'''
        
import numpy as np
import kernel.aux as aux
import math
import kernel.type as type
   

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
    
        # some parameters, default values
        self.r = 1.0 # hyper paramenter
        self.M = 100.0
        self.algType = type.AUGMENTED_COVARIANCE
        
        # the regularization used in the kriging procedure
        self.reg = 100*np.finfo(np.float).eps
        
        self.matricesReady = False

    
        
    
    def addPair(self,x,f):
        """ 
        add a location and its log likelihood to the lists
        """
        self.X.append(x)
        self.F.append(f)
        self.matricesReady = False
        
    def setR( self,t):
        """
        set the hyper parameter r and the regularization
        we used in the kriging procedure
        """
        self.r = t
        self.reg = 100*math.sqrt(self.r)*np.finfo(np.float).eps
                          
    def setMatrices(self):
        """ 
        calculates the matrices needed to carry out the kriging calculation.
        we use this procedure when we add a sampled "ground truth" point 
        once the matrices are ready we set matricesReady = True
        """
        if self.matricesReady == False:
            if self.algType == type.AUGMENTED_COVARIANCE:
                self.acm = aux.augCovMat(self.X,self.r)    # acm  = augmented covariance matrix
                self.U, self.S, self.V = np.linalg.svd(self.acm, full_matrices = True, compute_uv = True)
        
            if self.algType == type.COVARIANCE:
                self.cm = aux.covMat(self.X,self.r)    # cm  = covariance matrix
                self.U, self.S, self.V = np.linalg.svd(self.cm, full_matrices = True, compute_uv = True)
            
            if self.algType == type.RASMUSSEN_WILLIAMS:
                self.cm = aux.covMat(self.X,self.r)    # cm  = covariance matrix
                self.U, self.S, self.V = np.linalg.svd(self.cm, full_matrices = True, compute_uv = True)
                
        self.matricesReady = True
        
    def setType(self, algType):
        """
        here we set the variable that decides whether we use the augmented coavriance matrix for kriging or not.
        if we use the augmented covariance matrix, we're using an UNBIASED predictor - we're forcing the weights 
        to sum to 1.
        """        
        self.algType = algType
        self.matricesReady = False
        
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
        
        
        
        
        
        if self.algType == type.AUGMENTED_COVARIANCE:
            
            # make sure the matrices are ready
            self.setMatrices()
            
            # unpack
            F = self.F
            S = self.S
            n = len(F)
            
            # set target
            c = np.zeros( n+1 )
            c[n] =  1.0
            lam = aux.tychonoffSvdSolver(self.U, self.S, self.V, c, self.reg)
            lim = np.zeros( len(F[0]) )
            for i in range(n):
                lim = lim + lam[i] * F[i]    
            return lim
            
            
        if self.algType == type.RASMUSSEN_WILLIAMS:
            return 0
            
        
    
        