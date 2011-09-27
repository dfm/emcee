#!/usr/bin/env python
# encoding: utf-8
"""
Defines various nose unit tests

"""

import numpy as np
np.random.seed(1)
from ensemble import EnsembleSampler

logprecision = -4

def lnprob_gaussian(x, icov):
    """
    Value at x of a multi-dimensional Gaussian with mean mu
    and inverse cov icov
    """
    return -np.dot(x,np.dot(icov,x))/2.0

class Tests:
    def setUp(self):
        self.nwalkers = 100
        self.ndim     = 2

        self.mean = np.zeros(self.ndim)
        self.cov  = 0.5-np.random.rand(self.ndim*self.ndim).reshape((self.ndim,self.ndim))
        self.cov  = np.triu(self.cov)
        self.cov += self.cov.T - np.diag(self.cov.diagonal())
        self.cov  = np.dot(self.cov,self.cov)
        self.icov = np.linalg.inv(self.cov)
        self.p0   = [0.1*np.random.randn(self.ndim) for i in xrange(self.nwalkers)]

    def tearDown(self):
        pass

    def test_gaussian(self,threads=1):
        self.sampler = EnsembleSampler(self.nwalkers,self.ndim,lnprob_gaussian,
                        postargs=[self.icov],threads=threads)
        pos,prob,state = self.sampler.run_mcmc(self.p0, None, 1000)

        chain = self.sampler.chain
        flatchain = np.zeros([self.ndim,chain.shape[-1]*self.nwalkers])
        for i in range(self.ndim):
            flatchain[i,:] = chain[:,i,:].flatten()

        maxdiff = 10.**(logprecision)
        assert np.all((np.mean(flatchain,axis=-1)-self.mean)**2 < maxdiff)
        assert np.all((np.cov(flatchain)-self.cov)**2 < maxdiff)

    def test_multi_gaussian(self):
        self.test_gaussian(threads=self.ndim/2)

if __name__ == '__main__':
    import pylab as pl
    tests = Tests()
    tests.setUp()

    try:
        tests.test_multi_gaussian()
    except Exception as e:
        print e

    acorr = tests.sampler.acorr
    pl.plot(acorr,'k')

    chain = tests.sampler.chain
    truth = np.random.multivariate_normal(tests.mean,tests.cov,chain.shape[-1])
    for i in range(tests.ndim):
        pl.figure()
        samps = chain[:,i,:].flatten()
        pl.hist(samps,100,normed=True)
        pl.hist(truth[:,i],100,normed=True,histtype='step')

    pl.show()

