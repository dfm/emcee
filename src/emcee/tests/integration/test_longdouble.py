import numpy as np
import pytest

import emcee


def test_longdouble_doesnt_crash_bug_312():

    def log_prob(x, ivar):
        return -0.5 * np.sum(ivar * x ** 2)

    ndim, nwalkers = 5, 20
    ivar = 1. / np.random.rand(ndim).astype(np.longdouble)
    p0 = np.random.randn(nwalkers, ndim).astype(np.longdouble)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=[ivar])
    sampler.run_mcmc(p0, 100)


@pytest.mark.parametrize("cls", [emcee.backends.Backend,
                                 emcee.backends.TempHDFBackend])
def test_longdouble_actually_needed(cls):

    mjd = np.longdouble(58000.)
    sigma = 100*np.finfo(np.longdouble).eps*mjd

    def log_prob(x):
        assert x.dtype == np.longdouble
        return -0.5 * np.sum(((x-mjd)/sigma) ** 2)

    ndim, nwalkers = 1, 20
    steps = 1000
    p0 = sigma*np.random.randn(nwalkers, ndim).astype(np.longdouble) + mjd
    assert not all(p0 == mjd)

    with cls(dtype=np.longdouble) as backend:
        sampler = emcee.EnsembleSampler(nwalkers,
                                        ndim,
                                        log_prob,
                                        backend=backend)
        sampler.run_mcmc(p0, steps)

        samples = sampler.get_chain().reshape((-1,))
        assert samples.dtype == np.longdouble

        assert not np.all(samples == mjd)
        assert np.abs(np.mean(samples)-mjd) < 10*sigma/np.sqrt(len(samples))
        assert 0.1*sigma < np.std(samples) < 10*sigma
