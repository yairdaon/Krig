'''
Created on Jun 9, 2014

@author: daon
'''

class Type(object):
    '''
    a static class with the instances seen below.
    Do NOT create instances of this class on your own, this is NOT what this
    class is built for. The instances on the bottom are a substitute
    for a flag.
    '''


    def __init__(self, typeString):
        '''
        Constructor
        '''
        self._matrixType = typeString

    def getDescription(self):
        """
        return the type of algorithm an instance defines
        """
        return self._matrixType
    
AUGMENTED_COVARIANCE = Type( "Augmented Covariance" ) # Use augmented covariance matrix. Unbiased predictor.
COVARIANCE           = Type( "Covarince Matrix" ) # Covariance matrix. Not an unbiased predictor.
RASMUSSEN_WILLIAMS   = Type( " Rasmussen Williams") # algorithm 2.1 in R&W "gaussian Process for Machine Learning"      
  
    