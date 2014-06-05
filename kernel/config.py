'''
Created on May 19, 2014

@author: daon
'''
        
import numpy as np
import kernel.aux as aux
import math
   

class Config:   
         
    X = []
    F = []
    r = 1.0
    state = np.random.get_state()
    icm  = np.array( [] )
    acm  = np.array( [] )
    cm   = np.array( [] )
    iacm = np.array( [] )
    
    def __init__(self):
        pass
    
    def addPair(self,x,f):
        self.X.append(x)
        self.F.append(f)
        
    def setR( self,t):
        self.r = t
        self.reg = 100*math.sqrt(self.r)*np.finfo(np.float).eps
                          
    def setMatrices(self):
        
        #print("settin  matrices")
        #self.cm  = aux.covMat(self.X,self.r)       # cm   = covariance matrix
        #self.icm = aux.invCovMat(self.X,self.r)    # icm  = inverse covariance matrix
        #print np.shape(self.cm)
        #print
        
        self.acm = aux.augCovMat(self.X,self.r)    # acm  = augmented covariance matrix
        # self.iacm= aux.invAugCovMat(self.X,self.r) # iacm = inverse augmented covariance matrix
        U, S, V = np.linalg.svd(self.acm, full_matrices = True, compute_uv = True)
        self.U = U
        self.S = S
        self.V = V
        #self.condition()
        #print(n)
        #print(np.shape(self.cm))
        #print( self.cm 
        # print "icmCond = " + str( self.icmCond ) 
        # print "acmCond = " + str( self.acmCond ) 
        # print "cmCond = "  + str( self.cmCond  ) 
        # print "iacmCond = "+ str( self.iacmCond ) 
  
    def setM(self, N):
        self.M = N


    # use SVD to find condition number of matrix
    def condition(self):
        
        s = np.linalg.svd( self.icm )[1]
        self.icmCond = max(s)/min(s)
        
        s = np.linalg.svd( self.acm )[1]
        self.acmCond = max(s)/min(s)
        
        s = np.linalg.svd( self.cm )[1]
        self.cmCond = max(s)/min(s)
        
        #s = np.linalg.svd( self.iacm )[1]
        #self.iacmCond = max(s)/min(s)
        
        
    # calculates the kriged value at infinity    
    def getLimit(self):
        
        n = len(self.X)
        m = 1.0/np.sum(self.icm)
        lam = np.dot( self.icm, np.ones( (n,1) ) )
        
        lam = m*lam
        self.limAtInfty = np.zeros( np.shape( self.F[0] ))
        for i in range(n):
            self.limAtInfty = self.limAtInfty + lam[i]*self.F[i] 
        return self.limAtInfty
 

    def getLimitSVD(self):
           
        
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