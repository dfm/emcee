#!/usr/bin/env python
# encoding: utf-8
"""
quickstart.py

Created by Dan Foreman-Mackey on Jun 03, 2011
"""

import time

import pylab as pl
import numpy as np
np.random.seed()

import pyest

# dimensions of space
nwalkers = 100
ndim     = 20

initial_position = [np.random.rand(ndim)+5.0 for i in xrange(nwalkers)]
means            = 10.0*np.random.rand(ndim)
variances        = np.random.rand(ndim)
inv_var          = np.diagflat(1.0/variances)

# PDF that we're going to sample
def lnprob(x):
    """
    Value at x of a multi-dimensional Gaussian with mean mu
    and inverse variance sig2
    """
    diff = x-means
    return -np.dot(diff,np.dot(inv_var,diff))/2.0

sampler = pyest.EnsembleSampler(nwalkers,ndim,lnprob,threads=1)
# burn-in
pos,prob,state = sampler.run_mcmc(initial_position, None, 500)
# final chain
#sampler.clear_chain()
start = time.time()
print '** Starting sampling **'
sampler.run_mcmc(np.array(pos), state, 1000)
print '** Finished sampling **'
print '                Time (s): ',time.time()-start
print 'Mean acceptance fraction: ',np.mean(sampler.acceptance_fraction)

# plotting
chain = sampler.get_chain()
for par in xrange(ndim):
    pl.figure()
    samples = chain[:,par,:].flatten()
    pl.hist(samples,100,normed=True,histtype='step',color='k')
    x = np.linspace(min(samples),max(samples),1000)
    y = np.exp(-(x-means[par])**2/2.0/variances[par])/np.sqrt(2*np.pi*variances[par])
    pl.plot(x,y)

pl.show()



