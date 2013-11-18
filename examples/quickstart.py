#!/usr/bin/env python
"""
Sample code for sampling a multivariate Gaussian using emcee.

"""

from __future__ import print_function
import numpy as np
import emcee
import time


# First, define the probability distribution that you would like to sample.
def lnprob(x, mu, icov):
    diff = x - mu
    return -np.dot(diff, np.dot(icov, diff)) / 2.0


#vectorized probability distribution for bcast option
def vec_lnprob(ensemble, means, icov):
    ensemble_t = np.array(ensemble).T
    """Multidimensional gaussian.
    Inputs:
        ensemble [numpy.ndarray] (ndim, nwalkers,) - ensemble of #=`nwalker`
        walkers
        means [numpy.ndarray] (ndim,) - means of distribution,
        icov [numpy.ndarray]  (ndim, ndim,) - inverse covariance matrix.
    Output:
        [numpy.ndarray] (nwalkers,) - ln of probability for walkers in
        `ensemble`.
    """

    diff = ensemble_t - means[:, np.newaxis]

    return -0.5 * np.einsum('i..., i...', diff, np.dot(icov, diff))


# We'll sample a 50-dimensional Gaussian...
ndim = 50
# ...with randomly chosen mean position...
means = np.random.rand(ndim)
# ...and a positive definite, non-trivial covariance matrix.
cov = 0.5 - np.random.rand(ndim ** 2).reshape((ndim, ndim))
cov = np.triu(cov)
cov += cov.T - np.diag(cov.diagonal())
cov = np.dot(cov, cov)

# Invert the covariance matrix first.
icov = np.linalg.inv(cov)

# We'll sample with 250 walkers.
nwalkers = 250

# Choose an initial set of positions for the walkers.
p0 = [np.random.rand(ndim) for i in xrange(nwalkers)]

# Initialize the sampler with the chosen specs.
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[means, icov])
poolsampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[means, icov])
vecsampler = emcee.EnsembleSampler(nwalkers, ndim, vec_lnprob, args=[means, icov], bcast=True)

# Run 100 steps as a burn-in.
pos, prob, state = sampler.run_mcmc(p0, 100)
ppos, pprob, pstate = poolsampler.run_mcmc(p0, 100)
vpos, vprob, vstate = vecsampler.run_mcmc(p0, 100)

# Reset the chain to remove the burn-in samples.
sampler.reset()
poolsampler.reset()
vecsampler.reset()

##################################################################
#############Timing for Eggbox density sampling###################
##################################################################

t1 = time.time()
for pos, prob, rstate in sampler.sample(pos, iterations=1000):
    pass
dt = time.time() - t1
print("time : " + str(dt))
sampler.reset()

t1 = time.time()
for pos, prob, rstate in poolsampler.sample(ppos, iterations=1000):
    pass
dt = time.time() - t1
print("threads=10 time : " + str(dt))
poolsampler.reset()

t1 = time.time()
for pos, prob, rstate in vecsampler.sample(vpos, iterations=1000):
    pass
dt = time.time() - t1
print("bcast=True time : " + str(dt))

##################################################################

#plotting sampled density
try:
    import matplotlib.pyplot as pl
except ImportError:
    print("Try installing matplotlib to generate some sweet plots...")
else:
    H, xedges, yedges = np.histogram2d(vecsampler.flatchain[:, 0],
            vecsampler.flatchain[:, 1], bins=(128, 128))
    extent = [yedges[0], yedges[-1], xedges[-1], xedges[0]]
    pl.imshow(H, interpolation='nearest', extent=extent, aspect="auto")

vecsampler.reset()

##################################################################

# Run 100 steps as a burn-in.
pos, prob, state = sampler.run_mcmc(p0, 100)

# Reset the chain to remove the burn-in samples.
sampler.reset()

# Starting from the final position in the burn-in chain, sample for 1000
# steps.
sampler.run_mcmc(pos, 1000, rstate0=state)

# Print out the mean acceptance fraction. In general, acceptance_fraction
# has an entry for each walker so, in this case, it is a 250-dimensional
# vector.
print("Mean acceptance fraction:", np.mean(sampler.acceptance_fraction))

# If you have installed acor (http://github.com/dfm/acor), you can estimate
# the autocorrelation time for the chain. The autocorrelation time is also
# a vector with 10 entries (one for each dimension of parameter space).
try:
    print("Autocorrelation time:", sampler.acor)
except ImportError:
    print("You can install acor: http://github.com/dfm/acor")

# Finally, you can plot the projected histograms of the samples using
# matplotlib as follows (as long as you have it installed).
try:
    import matplotlib.pyplot as pl
except ImportError:
    print("Try installing matplotlib to generate some sweet plots...")
else:
    pl.hist(sampler.flatchain[:, 0], 100)
    pl.show()
