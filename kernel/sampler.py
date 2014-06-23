'''
Created on May 2, 2014

@author: daon
'''
import numpy as np
import kriging as kg 
import truth 
import aux 
import config as cfg
import emcee as mc






def lnprob(s, CFG):
    '''
    return the interpolated f
    here this is interpreted as a log-likelihood \ log-probability
    input:
    s - the point in space fr which we estimate the log-likelihood
    X - a list of locations in space
    F - a list of corresponding precalculated log-likelihoods
    C -  an augmented covariance matrix
    M - an artificial bound on the distribution. we insist that
    if |x| > M (the sup norm) then the likelihood
    is zero (so the log-likelihood is -infinity)
    a hyper parameter, see the documentation for aux.cov(...) procedure
    '''
    
    M = CFG.M
    # we ensure M > |s| in sup norm
    # we should also make sure that all observations 
    # satisfy |X[j]| < M
    if (np.linalg.norm(s, np.inf)  >  M):
        return -np.inf
    
    # do kriging to estimate the the log likelihood in a new 
    # location, given previous observations
    mu, sig = kg.kriging(s, CFG)

    # return the interpolated value only - no use for the std dev
    return mu
    

def sampler(CFG):
    '''
    this procedure samples a distribution. this distribution is defined by 
    kriging some previously collected data and interpolating it to give a 
    distribution over states space.
    input:    
    CFG is a container object that holds all required data. see the config.py module
    '''
    #print("sampler...")
    # unpack the data in CFG
    X = CFG.X
    M = CFG.M
        
    # the number of space dimensions, corresponds to the length 
    ndim = len(X[0])
    
    # set number of walkers
    nwalkers =  10*ndim
    
    # set burn in time
    burn = 150*ndim**(1.5)

    
    # the initial set of positions are uniform  in the box [-M,M]^ndim
    pos = np.random.rand(ndim * nwalkers) #choose U[0,1]
    pos = ( 2*pos  - 1.0 )*M # shift and stretch
    pos = pos.reshape((nwalkers, ndim)) # reshape
    
    # Initialize the sampler with the chosen specs.
    sam = mc.EnsembleSampler(nwalkers, ndim, lnprob, args=[ CFG ])
    
    counter = 0;
    keepLooping = True
    while ( keepLooping ):
        
        #allow at most ten loops
        counter = counter + 1
        if counter > 5: break
        
        # Run some steps as a burn-in. use last as a sample
        pos, prob, state = sam.run_mcmc(pos, burn, rstate0 = CFG.state)
        CFG.state = np.random.get_state()
        
        for n in range(nwalkers):
            
            # record the position of first walker
            s = pos[n,:]
            value, variance = kg.kriging(s, CFG)
            valueAtInf , varianceAtInf = kg.setGetLimit(CFG)
            
            # somewhat arbitrary, but good enough for now
            if variance/varianceAtInf > 0.5:
                #print "used it!!! " + str(n)
                keepLooping = False
                break
        
    # append to the list of exact log likelihoods if instructed to do so
    if CFG.addSamplesToDataSet == True:
        
        # calculate the corresponding log likelihood
        f = np.array( [ CFG.LL(s) ] )
    
        # incorporate the new sample to our data set
        cfg.Config.addPair(CFG, s, f)
        
        # update the required matrices required fot the kriging calculation
        cfg.Config.setMatrices(CFG)
    
    # return the sample
    return s