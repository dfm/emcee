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


def lnprob_gaussian_nan(x, icov):
    # if walker's parameters are zeros => return NaN
    if not (np.array(x)).any():
        result = np.nan
    else:
        result = -np.dot(x, np.dot(icov, x)) / 2.0

    return result


def log_unit_sphere_volume(ndim):
    if ndim % 2 == 0:
        logfactorial = 0.0
        for i in range(1, ndim / 2 + 1):
            logfactorial += np.log(i)
        return ndim / 2.0 * np.log(np.pi) - logfactorial
    else:
        logfactorial = 0.0
        for i in range(1, ndim + 1, 2):
            logfactorial += np.log(i)
        return (ndim + 1) / 2.0 * np.log(2.0) \
            + (ndim - 1) / 2.0 * np.log(np.pi) - logfactorial


class LogLikeGaussian(object):
    def __init__(self, icov):
        """Initialize a gaussian PDF with the given inverse covariance
        matrix.  If not ``None``, ``cutoff`` truncates the PDF at the
        given number of sigma from the origin (i.e. the PDF is
        non-zero only on an ellipse aligned with the principal axes of
        the distribution).  Without this cutoff, thermodynamic
        integration with a flat prior is logarithmically divergent."""

        self.icov = icov

    def __call__(self, x):
        dist2 = lnprob_gaussian(x, self.icov)

        return dist2


class LogPriorGaussian(object):
    def __init__(self, icov, cutoff=None):
        self.icov = icov
        self.cutoff = cutoff

    def __call__(self, x):
        dist2 = lnprob_gaussian(x, self.icov)

        if self.cutoff is not None:
            if -dist2 > self.cutoff * self.cutoff / 2.0:
                return float('-inf')
            else:
                return 0.0
        else:
            return 0.0


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
        assert np.all(self.sampler.acceptance_fraction > 0)

        chain = self.sampler.flatchain
        maxdiff = 10. ** (logprecision)
        assert np.all((np.mean(chain, axis=0) - self.mean) ** 2 / self.N ** 2
                      < maxdiff)
        assert np.all((np.cov(chain, rowvar=0) - self.cov) ** 2 / self.N ** 2
                      < maxdiff)

    def check_pt_sampler(self, cutoff, N=None, p0=None):
        if N is None:
            N = self.N
        if p0 is None:
            p0 = self.p0

        for i in self.sampler.sample(p0, iterations=N):
            pass

        # Weaker assertions on acceptance fraction
        assert np.mean(self.sampler.acceptance_fraction) > 0.1, \
            "acceptance fraction < 0.1"
        assert np.mean(self.sampler.tswap_acceptance_fraction) > 0.1, \
            "tswap acceptance frac < 0.1"

        maxdiff = 10.0 ** logprecision

        chain = np.reshape(self.sampler.chain[0, ...],
                           (-1, self.sampler.chain.shape[-1]))

        # np.savetxt('/tmp/chain.dat', chain)

        log_volume = self.ndim * np.log(cutoff) \
            + log_unit_sphere_volume(self.ndim) \
            + 0.5 * np.log(np.linalg.det(self.cov))
        gaussian_integral = self.ndim / 2.0 * np.log(2.0 * np.pi) \
            + 0.5 * np.log(np.linalg.det(self.cov))

        lnZ, dlnZ = self.sampler.thermodynamic_integration_log_evidence()
        print(self.sampler.get_autocorr_time())

        assert np.abs(lnZ - (gaussian_integral - log_volume)) < 3 * dlnZ, \
            ("evidence incorrect: {0:g} versus correct {1:g} (uncertainty "
             "{2:g})").format(lnZ, gaussian_integral - log_volume, dlnZ)
        assert np.all((np.mean(chain, axis=0) - self.mean) ** 2.0 / N ** 2.0
                      < maxdiff), 'mean incorrect'
        assert np.all((np.cov(chain, rowvar=0) - self.cov) ** 2.0 / N ** 2.0
                      < maxdiff), 'covariance incorrect'

    def test_mh(self):
        self.sampler = MHSampler(self.cov, self.ndim, lnprob_gaussian,
                                 args=[self.icov])
        self.check_sampler(N=self.N * self.nwalkers, p0=self.p0[0])

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

    def test_pt_sampler(self):
        cutoff = 10.0
        self.sampler = PTSampler(self.ntemp, self.nwalkers, self.ndim,
                                 LogLikeGaussian(self.icov),
                                 LogPriorGaussian(self.icov, cutoff=cutoff))
        p0 = np.random.multivariate_normal(mean=self.mean, cov=self.cov,
                                           size=(self.ntemp, self.nwalkers))
        self.check_pt_sampler(cutoff, p0=p0, N=1000)

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

    def test_run_mcmc_resume(self):

        self.sampler = s = EnsembleSampler(self.nwalkers, self.ndim,
                                           lnprob_gaussian, args=[self.icov])

        # first time around need to specify p0
        try:
            s.run_mcmc(None, self.N)
        except ValueError:
            pass

        s.run_mcmc(self.p0, N=self.N)
        assert s.chain.shape[1] == self.N

        # this doesn't actually check that it resumes with the right values, as
        # that's non-trivial... so we just make sure it does *something* when
        # None is given and that it records whatever it does
        s.run_mcmc(None, N=self.N)
        assert s.chain.shape[1] == 2 * self.N
