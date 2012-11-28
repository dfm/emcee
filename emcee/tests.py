#!/usr/bin/env python
# encoding: utf-8
"""
Defines various nose unit tests

"""

import numpy as np

from .mh import MHSampler
from .ensemble import EnsembleSampler
from .ptsampler import PTSampler

logprecision = -4


def lnprob_gaussian(x, icov):
    return -np.dot(x, np.dot(icov, x)) / 2.0

class LnprobGaussian(object):
    def __init__(self, icov):
        self.icov = icov

    def __call__(self, x):
        return lnprob_gaussian(x, self.icov)

def ln_flat(x):
    return 0.0

class Tests:
    def setUp(self):
        self.nwalkers = 100
        self.ndim = 5

        self.ntemp = 20

        self.N = 1000

        self.mean = np.zeros(self.ndim)
        self.cov = 0.5 - np.random.rand(self.ndim ** 2) \
                .reshape((self.ndim, self.ndim))
        self.cov = np.triu(self.cov)
        self.cov += self.cov.T - np.diag(self.cov.diagonal())
        self.cov = np.dot(self.cov, self.cov)
        self.icov = np.linalg.inv(self.cov)
        self.p0 = [0.1 * np.random.randn(self.ndim)
                for i in range(self.nwalkers)]
        self.truth = np.random.multivariate_normal(self.mean, self.cov, 100000)

    def check_sampler(self, N=None, p0=None):
        if N is None:
            N = self.N
        if p0 is None:
            p0 = self.p0

        for i in self.sampler.sample(p0, iterations=N):
            pass

        assert np.mean(self.sampler.acceptance_fraction) > 0.25
        chain = self.sampler.flatchain
        maxdiff = 10. ** (logprecision)
        assert np.all((np.mean(chain, axis=0) - self.mean) ** 2 / self.N ** 2
                < maxdiff)
        assert np.all((np.cov(chain, rowvar=0) - self.cov) ** 2 / self.N ** 2
                < maxdiff)

    def check_pt_sampler(self, N=None, p0=None):
        if N is None:
            N = self.N
        if p0 is None:
            p0 = self.p0

        for i in self.sampler.sample(p0, iterations=N):
            pass

        # Weaker assertions on acceptance fraction
        assert np.mean(self.sampler.acceptance_fraction) > 0.1
        assert np.mean(self.sampler.tswap_acceptance_fraction) > 0.1
        
        maxdiff = 10.0**logprecision
        
        chain=np.reshape(self.sampler.chain[0,...], (-1, self.sampler.chain.shape[-1]))

        assert np.all((np.mean(chain, axis=0) - self.mean)**2.0 / N**2.0 < maxdiff)
        assert np.all((np.cov(chain, rowvar=0) - self.cov)**2.0 / N**2.0 < maxdiff)


    def test_mh(self):
        self.sampler = MHSampler(self.cov, self.ndim, lnprob_gaussian,
                args=[self.icov])
        self.check_sampler(N=self.N * self.nwalkers, p0=self.p0[0])

    def test_ensemble(self):
        self.sampler = EnsembleSampler(self.nwalkers, self.ndim,
                            lnprob_gaussian, args=[self.icov])
        self.check_sampler()

    def test_parallel(self):
        self.sampler = EnsembleSampler(self.nwalkers, self.ndim,
                lnprob_gaussian, args=[self.icov], threads=2)
        self.check_sampler()

    def test_pt_sampler(self):
        self.sampler = PTSampler(self.ntemp, self.nwalkers, self.ndim, LnprobGaussian(self.icov), ln_flat)
        p0=np.random.multivariate_normal(mean=self.mean, cov=self.cov, size=(self.ntemp, self.nwalkers))
        self.check_pt_sampler(p0=p0)

    def test_blobs(self):
        lnprobfn = lambda p: (-0.5 * np.sum(p ** 2), np.random.rand())
        self.sampler = EnsembleSampler(self.nwalkers, self.ndim, lnprobfn)
        self.check_sampler()

        # Make sure that the shapes of everything are as expected.
        assert (self.sampler.chain.shape == (self.nwalkers, self.N, self.ndim)
                and len(self.sampler.blobs) == self.N
                and len(self.sampler.blobs[0]) == self.nwalkers), \
                        "The blob dimensions are wrong."

        # Make sure that the blobs aren't all the same.
        blobs = self.sampler.blobs
        assert np.any([blobs[-1] != blobs[i] for i in range(len(blobs) - 1)])
