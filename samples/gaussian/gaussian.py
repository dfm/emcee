#!/usr/bin/env python

import time

import numpy as np
np.random.seed()

import markovpy

# PDF that we're going to sample
def lnprob(x,*args):
    """
    Value at x of a multi-dimensional Gaussian with mean mu
    and inverse variance sig2
    """
    mu,sig2 = tuple(*args)
    diff = x-mu
    return -np.dot(diff,np.dot(sig2,diff))/2.0

# dimensions of space
nwalkers = 100
ndim     = 20

initial_position = [np.random.rand(ndim)+5.0 for i in xrange(nwalkers)]
means            = 10.0*np.random.rand(ndim)
variances        = np.random.rand(ndim)
inv_var          = np.diagflat(1.0/variances)

sampler = markovpy.EnsembleSampler(nwalkers,ndim,lnprob,postargs=(means,inv_var))
# burn-in
pos,prob,state = sampler.run_mcmc(initial_position, None, 500)
# final chain
sampler.clear_chain()
start = time.time()
print '** Starting sampling **'
sampler.run_mcmc(np.array(pos), state, 1000)
print '** Finished sampling **'
print '                Time (s): ',time.time()-start
print 'Mean acceptance fraction: ',np.mean(sampler.acceptance_fraction())
# plotting

import pylab as pl

chain = sampler.get_chain()
for par in xrange(ndim):
    pl.figure()
    samples = chain[:,par,:].flatten()
    pl.hist(samples,100,normed=True,histtype='step',color='k')
    x = np.linspace(min(samples),max(samples),1000)
    y = np.exp(-(x-means[par])**2/2.0/variances[par])/np.sqrt(2*np.pi*variances[par])
    pl.plot(x,y)

pl.show()
