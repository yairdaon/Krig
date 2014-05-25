'''
Created on May 19, 2014

@author: daon
'''
        
import numpy as np
import aux as aux
        
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
                  
    def setMatrices(self):
        self.icm = aux.invCovMat(self.X,self.r) # inverse covariance matrix
        self.acm = aux.augCovMat(self.X,self.r) # augmented covariance matrix
        self.cm  = aux.covMat(self.X,self.r)    # covariance matrix
        self.iacm= aux.invAugCovMat(self.X,self.r) # inverse augmented covariance matrix
    
    def setM(self, N):
        self.M = N

