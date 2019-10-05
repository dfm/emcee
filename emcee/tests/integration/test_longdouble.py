import numpy as np
import emcee


def test_longdouble_doesnt_crash_bug_312():

    def log_prob(x, ivar):
        return -0.5 * np.sum(ivar * x ** 2)

    ndim, nwalkers = 5, 20
    ivar = 1. / np.random.rand(ndim).astype(np.longdouble)
    p0 = np.random.randn(nwalkers, ndim).astype(np.longdouble)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=[ivar])
    sampler.run_mcmc(p0, 100)
