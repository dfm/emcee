#!/usr/bin/env python
# encoding: utf-8
"""
Defines various nose unit tests

"""

import numpy as np

from .autocorr import integrated_time
from .ensemble import EnsembleSampler

logprecision = -4


def lnprob_gaussian(x, icov):
    return -np.dot(x, np.dot(icov, x)) / 2.0


def lnprob_gaussian_nan(x, icov):
    # if walker's parameters are zeros => return NaN
    if not (np.array(x)).any():
        result = np.nan
    else:
        result = -np.dot(x, np.dot(icov, x)) / 2.0

    return result


def ln_flat(x):
    return 0.0


class Tests:

    def setUp(self):
        np.random.seed(42)

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
        assert np.all(self.sampler.acceptance_fraction > 0)

        chain = self.sampler.flatchain
        maxdiff = 10. ** (logprecision)
        assert np.all((np.mean(chain, axis=0) - self.mean) ** 2 / self.N ** 2
                      < maxdiff)
        assert np.all((np.cov(chain, rowvar=0) - self.cov) ** 2 / self.N ** 2
                      < maxdiff)

    def test_ensemble(self):
        self.sampler = EnsembleSampler(self.nwalkers, self.ndim,
                                       lnprob_gaussian, args=[self.icov])
        self.check_sampler()

    def test_nan_lnprob(self):
        self.sampler = EnsembleSampler(self.nwalkers, self.ndim,
                                       lnprob_gaussian_nan,
                                       args=[self.icov])

        # If a walker is right at zero, ``lnprobfn`` returns ``np.nan``.
        p0 = self.p0
        p0[0] = 0.0

        try:
            self.check_sampler(p0=p0)
        except ValueError:
            # This should fail *immediately* with a ``ValueError``.
            return

        assert False, "We should never get here."

    def test_inf_nan_params(self):
        self.sampler = EnsembleSampler(self.nwalkers, self.ndim,
                                       lnprob_gaussian, args=[self.icov])

        # Set one of the walkers to have a ``np.nan`` value.
        p0 = self.p0
        p0[0][0] = np.nan

        try:
            self.check_sampler(p0=p0)

        except ValueError:
            # This should fail *immediately* with a ``ValueError``.
            pass

        else:
            assert False, "The sampler should have failed by now."

        # Set one of the walkers to have a ``np.inf`` value.
        p0[0][0] = np.inf

        try:
            self.check_sampler(p0=p0)

        except ValueError:
            # This should fail *immediately* with a ``ValueError``.
            pass

        else:
            assert False, "The sampler should have failed by now."

        # Set one of the walkers to have a ``np.inf`` value.
        p0[0][0] = -np.inf

        try:
            self.check_sampler(p0=p0)

        except ValueError:
            # This should fail *immediately* with a ``ValueError``.
            pass

        else:
            assert False, "The sampler should have failed by now."

    def test_parallel(self):
        self.sampler = EnsembleSampler(self.nwalkers, self.ndim,
                                       lnprob_gaussian, args=[self.icov],
                                       threads=2)
        self.check_sampler()

    def test_blobs(self):
        lnprobfn = lambda p: (-0.5 * np.sum(p ** 2), np.random.rand())
        self.sampler = EnsembleSampler(self.nwalkers, self.ndim, lnprobfn)
        self.check_sampler()

        # Make sure that the shapes of everything are as expected.
        assert (self.sampler.chain.shape == (self.N, self.nwalkers, self.ndim)
                and len(self.sampler.blobs) == self.N
                and len(self.sampler.blobs[0]) == self.nwalkers), \
            "The blob dimensions are wrong."

        # Make sure that the blobs aren't all the same.
        blobs = self.sampler.blobs
        assert np.any([blobs[-1] != blobs[i] for i in range(len(blobs) - 1)])

    def test_run_mcmc_resume(self):

        self.sampler = s = EnsembleSampler(self.nwalkers, self.ndim,
                                           lnprob_gaussian, args=[self.icov])

        # first time around need to specify p0
        try:
            s.run_mcmc(None, self.N)
        except ValueError:
            pass

        s.run_mcmc(self.p0, N=self.N)
        assert s.chain.shape[0] == self.N

        # this doesn't actually check that it resumes with the right values, as
        # that's non-trivial... so we just make sure it does *something* when
        # None is given and that it records whatever it does
        s.run_mcmc(None, N=self.N)
        assert s.chain.shape[0] == 2 * self.N

    def test_autocorr_multi_works(self):
        xs = np.random.randn(16384, 2)

        acls_multi = integrated_time(xs) # This throws exception unconditionally in buggy impl's
        acls_single = np.array([integrated_time(xs[:,i]) for i in range(xs.shape[1])])

        assert np.all(np.abs(acls_multi - acls_single) < 2)

    def test_gh226(self):
        # EnsembleSampler.sample errors when iterations is not a multiple of thin
        m_true = -0.9594
        b_true = 4.294
        f_true = 0.534

        # Generate some synthetic data from the model.
        N = 50
        x = np.sort(10 * np.random.rand(N))
        yerr = 0.1 + 0.5 * np.random.rand(N)
        y = m_true * x + b_true
        y += np.abs(f_true * y) * np.random.randn(N)
        y += yerr * np.random.randn(N)

        A = np.vstack((np.ones_like(x), x)).T
        C = np.diag(yerr * yerr)
        cov = np.linalg.inv(np.dot(A.T, np.linalg.solve(C, A)))
        b_ls, m_ls = np.dot(cov, np.dot(A.T, np.linalg.solve(C, y)))

        def lnlike(theta, x, y, yerr):
            m, b, lnf = theta
            model = m * x + b
            inv_sigma2 = 1.0 / (yerr ** 2 + model ** 2 * np.exp(2 * lnf))
            return -0.5 * (np.sum((y - model) ** 2 * inv_sigma2 -
                                  np.log(inv_sigma2)))

        def lnprior(theta):
            m, b, lnf = theta
            if -5.0 < m < 0.5 and 0.0 < b < 10.0 and -10.0 < lnf < 1.0:
                return 0.0
            return -np.inf

        def lnprob(theta, x, y, yerr):
            lp = lnprior(theta)
            if not np.isfinite(lp):
                return -np.inf
            return lp + lnlike(theta, x, y, yerr)

        nll = lambda *args: -lnlike(*args)

        # minimize(nll, [m_true, b_true, np.log(f_true)], args=(x, y, yerr))
        init_guess = np.array([-0.95612643,  4.23596208, -0.66826006])

        ndim, nwalkers = 3, 100
        pos = [init_guess + 1e-4 * np.random.randn(ndim) for
               i in range(nwalkers)]

        sampler = EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr))
        s = sampler.sample(pos, iterations=65, thin=2)
        for i in range(65):
            next(s)
        np.testing.assert_equal(sampler.chain.shape, (32, nwalkers, ndim))

        sampler = EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr))
        s = sampler.sample(pos, iterations=65, thin=3)
        for i in range(65):
            next(s)
