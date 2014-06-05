'''
Created on May 2, 2014

@author: daon
'''
import numpy as np
import kernel.kriging as kg 
import kernel.truth as truth
import kernel.aux as aux
import kernel.config as cfg
import emcee as mc




# return the interpolated f
# here this is interpreted as a log-likelihood \ log-probability
# input:
# s - the point in space fr which we estimate the log-likelihood
# X - a list of locations in space
# F - a list of corresponding precalculated log-likelihoods
# C -  an augmented covariance matrix
# M - an artificial bound on the distribution. we insist that
# if |x| > M (the sup norm) then the likelihood
# is zero (so the log-likelihood is -infinity)
# a hyper parameter, see the documentation for aux.cov(...) procedure
def lnprob(s, CFG):
    
    M = CFG.M
    # we insist that M > |s| in sup norm
    # we should also make sure that all observations 
    # satisfy |X[j]| < M
    if (np.linalg.norm(s, np.inf)  >  M):
        return -np.inf
    
    # use  kriging to estimate the the log likelihood in a new 
    # location, given previous observations
    mu, sig = kg.kriging(s, CFG)

    # return the interpolated value only - no use for the std dev
    return mu
    

# this procedure samples a distribution. this distribution is defined by 
# kriging some previously collected data and interpolating it to give a 
# distribution over states space.
# input:    
# X are locations
# F are calulated log-likelihood values (expensive to calculate)
# C, M, r - see documentation above for lnprob

def sampler(CFG): #X, F, M, r, *args):
    # print np.random.get_state()
    # unpack the data in CFG
    X = CFG.X
    F = CFG.F
    M = CFG.M
    r = CFG.r
    
    # the number of space dimensions, corresponds to the length 
    ndim = len(X[0])
    nwalkers =  2*ndim + 4
    burn = 150*ndim**(1.5)

    
    # the initial set of positions are uniform  in the box [-M,M]^ndim
    p0 = np.random.rand(ndim * nwalkers) #choose U[0,1]
    p0 = ( p0  - 0.5 )*M # shift and stretch
    p0 = p0.reshape((nwalkers, ndim)) # reshape
    
    # Initialize the sampler with the chosen specs.
    sam = mc.EnsembleSampler(nwalkers, ndim, lnprob, args=[ CFG ])
    
    # Run some steps as a burn-in.
    

    pos, prob, state = sam.run_mcmc(p0, burn, rstate0 = CFG.state)
    CFG.state = np.random.get_state()
    #print CFG.state
    
    # record the position of first walker
    s = pos[0,:]
    
    # calculate the corresponding log likelihood
    f = np.array( [ truth.trueLL(s) ] )
    
    
    # append to the list of "known" log likelihoods
    cfg.Config.addPair(CFG, s, f)
    
    cfg.Config.setMatrices(CFG)
    
    # return the sample
    #print s
    return s