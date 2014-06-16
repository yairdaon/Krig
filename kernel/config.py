'''
Created on May 19, 2014

@author: daon
'''
        
import numpy as np
import aux
import math
import type
import truth
import kriging as kg
   

class Config:   
    '''
    This will be referred to as a container class. It holds all
    the data needed to do kriging, resample etc. An instance of 
    this class is passed to the sampler, the kriging procedure
    etc.
    '''
        
    def __init__(self):
        '''
        we set many default parameters here. you may change them if you 
        are sure you understand what they do. change them using their 
        setters method, though. not here
        '''
        
        # list observations... 
        self.X = []
        # ...and corresponding log-likelihoods
        self.F = []
        
        # the regularization used in the kriging procedure
        self.reg = 100*np.finfo(np.float).eps
        
        # are the matrices we use ready or do we need to calculate them
        self.matricesReady = False
        
        
        # some parameters, default values
        self.r = 1.0 # hyper paramenter
        self.M = 100.0
        self.algType = type.AUGMENTED_COVARIANCE # type of kriging algorithm
        
        # by default we incorporate new samples to our data set
        # we can set this to false. Then we can just sample from 
        # the posterior
        self.addSamplesToDataSet = True
        
        
        # state of PRNG, for reproducibility purposes
        self.state = np.random.get_state()
        
        # point to the true log-likelihood that we'll use
        self.LL = truth.trueLL
    
        
    
    def addPair(self,x,f):
        ''' 
        add a location and its log likelihood to the lists
        '''
        self.X.append(x)
        self.F.append(f)
        
        # we need to recalculate the matrices, so the matrices aren't tready
        self.matricesReady = False
        
    def setR( self,t):
        '''
        set the hyper parameter r and the regularization
        we used in the kriging procedure
        '''
        
        # the length scale of the covariance function
        self.r = t
        
        # the regularization we use in the tychonoff solver
        self.reg = 100*math.sqrt(self.r)*np.finfo(np.float).eps
        
        # parameters changed, so we need to recalculate the matrices
        self.matricesReady = False
    
    def setLL(self, likelihood):
        '''
        we use this to tell the sampler what reality is
        sometimes the sampler wants to know what is the 
        true log-likelihood. here we set the appropriate 
        variable to tell the sampler what the true log 
        likelihood is.
        '''
        self.LL = likelihood   
                           
    def setMatrices(self):
        ''' 
        calculates the matrices needed to carry out the kriging calculation.
        we use this procedure when we add a sampled "ground truth" point 
        once the matrices are ready we set matricesReady = True
        '''
        if self.matricesReady == False:
            if self.algType == type.AUGMENTED_COVARIANCE:
                self.acm = aux.augCovMat(self.X,self.r)    # acm  = augmented covariance matrix
                self.U, self.S, self.V = np.linalg.svd(self.acm, full_matrices = True, compute_uv = True)
        
            elif self.algType == type.COVARIANCE:
                self.cm = aux.covMat(self.X,self.r)    # cm  = covariance matrix
                self.U, self.S, self.V = np.linalg.svd(self.cm, full_matrices = True, compute_uv = True)
            
            elif self.algType == type.RASMUSSEN_WILLIAMS:
                self.cm = aux.covMat(self.X,self.r)    # cm  = covariance matrix
                self.U, self.S, self.V = np.linalg.svd(self.cm, full_matrices = True, compute_uv = True)
            else: 
                print("Your algorithm type is not valid. Algorithm type set to default.")
                self.algType = type.AUGMENTED_COVARIANCE
                self.acm = aux.augCovMat(self.X,self.r)    # acm  = augmented covariance matrix
                self.U, self.S, self.V = np.linalg.svd(self.acm, full_matrices = True, compute_uv = True)
            self.matricesReady = True    
                
        
    def setAddSamplesToDataSet(self, addOrNot):
        '''
        addOrNot has to be boolean. The variable
        addSamplesToDataSet decides whether we simply sample the
        posterior (if set to False) or we sample the posterior, calcualte
        its corresponding log-likelihood and use that to get a better
        kriged interpolant
        '''
        
        self.addSamplesToDataSet = addOrNot
       
    def setType(self, algType):
        '''
        here we set the variable that decides which algorithm we use for kriging or not.
        if we use the augmented covariance matrix, we're using an UNBIASED predictor. This amounts to
        forcing the weights to sum to 1.
        '''        
        if self.algType != algType:
            self.algType = algType
            self.matricesReady = False
        
    def setM(self, M):
        ''' 
        M is the size, in the sup norm, of box weconsider. We set the probability outside the box
        to be zero. 
        Precisely: P(x) = 0 for all x such that |x|_inf > M. Equivalently: log P(x) = -inf for all
        x such that |x|_inf > M
        '''
        self.M = M


    def condition(self):
        '''
        use SVD to find condition number of matrix
        '''
        
        s = np.linalg.svd( self.acm )[1]
        self.acmCond = max(s)/min(s)
        
        s = np.linalg.svd( self.cm )[1]
        self.cmCond = max(s)/min(s)
        
     
    def getLimitSVD(self):
        '''
        returns the kriged value "at infinity"
        very similar to the  corresponding kriging procedure
        '''
        
        # if we solve for the augmented covariance matrix
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
            
        # if we use algorithm 2.1 (page 19) from Rasmussen & Williams' book
        # title "Gaussian Processes for Machine Learning"    
        if self.algType == type.RASMUSSEN_WILLIAMS:
            return 0
        
            
        
    
        