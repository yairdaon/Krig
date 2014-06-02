'''
Created on May 19, 2014

@author: daon
'''
        
import numpy as np
import aux as aux
import math
   
   
# we use the fo        
class Config:   
         
    X = []
    F = []
    r = 1.0
             
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
        self.icm = aux.invCovMat(self.X,self.r)    # icm  = inverse covariance matrix
        self.acm = aux.augCovMat(self.X,self.r)    # acm  = augmented covariance matrix
        self.cm  = aux.covMat(self.X,self.r)       # cm   = covariance matrix
        self.iacm= aux.invAugCovMat(self.X,self.r) # iacm = inverse augmented covariance matrix
        U, S, V = np.linalg.svd(self.acm, full_matrices = True, compute_uv = True)
        self.U = U
        self.S = S
        self.V = V
        self.condition()
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
        
        s = np.linalg.svd( self.iacm )[1]
        self.iacmCond = max(s)/min(s)
        
        