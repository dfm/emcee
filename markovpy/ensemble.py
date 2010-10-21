#!/usr/bin/env python
# encoding: utf-8
# 
#  mcsampler.py
#  markovpy
#  
#  Created by Dan F-M on 2010-10-18.
# 
#  This is a Markov chain Monte Carlo (MCMC) sampler based on:
#
#  Goodman & Weare, Ensemble Samplers With Affine Invariance 
#    Comm. App. Math. Comp. Sci., Vol. 5 (2010), No. 1, 65–80
#

import numpy as np
from mcsampler import MCSampler

class EnsembleSampler(MCSampler):
    """Ensemble sampling following Goodman & Weare (2009)"""
    def __init__(self,nwalkers,a=2.):
        self.nwalkers = nwalkers
        self.a = a
    
    def sample_pdf(self,loglike,bounds,proposal,N,args):
        np.random.seed()
        
        W = self.nwalkers # number of walkers
        
        npars = np.shape(bounds)[0]
        old_p,old_prob = [],[]
        chain = []
        
        # initialize walkers
        for i in range(W):
            old_p.append((bounds[:,1]-bounds[:,0])*np.random.rand(npars)+bounds[:,0])
            chain.append(old_p[-1])
            old_prob.append(loglike(old_p[-1],*args))
        
        nacc = 0
        
        for it in range(N):
            for i in range(W):
                z = ((self.a-1.)*np.random.rand()+1)**2./self.a
                rint = np.random.randint(W-1)
                if rint >= i:
                    rint += 1
                new_p = old_p[rint]+z*(old_p[i]-old_p[rint])
                accept = False
                if np.all(new_p > bounds[:,0]) and np.all(new_p < bounds[:,1]):
                    new_prob = loglike(new_p,*args)
                    diff = (npars-1.)*np.log(z)+new_prob-old_prob[i]
                
                    if diff > 0:
                        accept = True
                    else:
                        rn = np.random.rand()
                        if rn < np.exp(diff):
                            accept = True
                
                if accept:
                    old_prob[i] = new_prob
                    old_p[i] = new_p
                    nacc += 1
                
                chain.append(old_p[i])
        
        acceptfrac = float(nacc)/N/W
        
        if acceptfrac > 0.1:
            print '''Ensemble sampling completed successfully:
    Acceptance fraction: %.3f
'''%(acceptfrac)
        else:
            print 'Warning: acceptance fraction < 10\%'
        
        return np.array(chain),acceptfrac
            
    