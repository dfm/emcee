#!/usr/bin/env python
# encoding: utf-8
"""
A test the breaks MarkovPy

History
-------
2011-08-08 - Created by Dan Foreman-Mackey

"""

import numpy as np
import pylab as pl
np.random.seed()
import markovpy

class Gaussian:
    def __init__(self):
        # dimensions of space
        self.nwalkers = 100
        self.ndim     = 20
        
        self.initial_position = [np.random.rand(self.ndim)+5.0 for i in xrange(self.nwalkers)]
        self.means            = 10.0*np.random.rand(self.ndim)
        self.variances        = np.random.rand(self.ndim)
        self.inv_var          = np.diagflat(1.0/self.variances)

    def lnprob(self, x, mu, ivar):
        """
        Value at x of a multi-dimensional Gaussian with mean mu
        and inverse variance sig2
        """
        diff = x-mu
        return -np.dot(diff,np.dot(ivar,diff))/2.0
    
    def sample(self):
        self.sampler = \
                markovpy.EnsembleSampler(self.nwalkers,self.ndim,self.lnprob,
                        postargs=(self.means,self.inv_var),threads=10)
        # burn-in
        pos,prob,state = self.sampler.run_mcmc(self.initial_position, None, 1000)
        

# try it
gaussian = Gaussian()
gaussian.sample()

chain = gaussian.sampler.get_chain()
for par in xrange(gaussian.ndim):
    pl.figure()
    samples = chain[:,par,:].flatten()
    pl.hist(samples,100,normed=True,histtype='step',color='k')
    x = np.linspace(min(samples),max(samples),1000)
    y = np.exp(-(x-gaussian.means[par])**2/2.0/gaussian.variances[par])/np.sqrt(2*np.pi*gaussian.variances[par])
    pl.plot(x,y)

pl.show()


