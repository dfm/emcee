#!/usr/bin/env python
# encoding: utf-8
"""
Defines various nose unit tests

"""

import time

import numpy as np

from mh import MHSampler
from ensemble import EnsembleSampler

logprecision = -4

def lnprob_gaussian(x, icov):
    return -np.dot(x,np.dot(icov,x))/2.0

class Tests:
    def setUp(self):
        self.nwalkers = 100
        self.ndim     = 5

        self.N = 1000

        self.mean = np.zeros(self.ndim)
        self.cov  = 0.5-np.random.rand(self.ndim*self.ndim).reshape((self.ndim,self.ndim))
        self.cov  = np.triu(self.cov)
        self.cov += self.cov.T - np.diag(self.cov.diagonal())
        self.cov  = np.dot(self.cov,self.cov)
        self.icov = np.linalg.inv(self.cov)
        self.p0   = [0.1*np.random.randn(self.ndim) for i in xrange(self.nwalkers)]

        self.truth = np.random.multivariate_normal(self.mean,self.cov,100000)

    def check_sampler(self, N=None, p0=None):
        if N is None:
            N = self.N
        if p0 is None:
            p0 = self.p0

        for i in self.sampler.sample(p0, iterations=N):
            pass

        assert np.mean(self.sampler.acceptance_fraction) > 0.25
        chain = self.sampler.flatchain
        maxdiff = 10.**(logprecision)
        assert np.all((np.mean(chain,axis=0)-self.mean)**2/self.N**2 < maxdiff)
        assert np.all((np.cov(chain, rowvar=0)-self.cov)**2/self.N**2 < maxdiff)

    def test_mh(self):
        self.sampler = MHSampler(self.cov, self.ndim, lnprob_gaussian, args=[self.icov])
        self.check_sampler(N=self.N*self.nwalkers, p0=self.p0[0])

    def test_ensemble(self):
        self.sampler = EnsembleSampler(self.nwalkers, self.ndim, lnprob_gaussian, args=[self.icov])
        self.check_sampler()

    def test_parallel(self):
        self.sampler = EnsembleSampler(self.nwalkers, self.ndim, lnprob_gaussian, args=[self.icov], threads=2)
        self.check_sampler()

