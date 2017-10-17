#!/usr/bin/env python
"""
Sample code for sampling a multivariate Gaussian using emcee.

"""

from __future__ import print_function
import numpy as np
import emcee

# First, define the probability distribution that you would like to sample.
def lnprob(x, mu, icov):
    diff = x-mu
    return -np.dot(diff,np.dot(icov,diff))/2.0

# We'll sample a 10-dimensional Gaussian...
ndim = 10
# ...with randomly chosen mean position...
means = np.random.rand(ndim)
# ...and a positive definite, non-trivial covariance matrix.
cov  = 0.5-np.random.rand(ndim**2).reshape((ndim, ndim))
cov  = np.triu(cov)
cov += cov.T - np.diag(cov.diagonal())
cov  = np.dot(cov,cov)

# Invert the covariance matrix first.
icov = np.linalg.inv(cov)

# We'll sample with 50 walkers.
nwalkers = 50

# Choose an initial set of positions for the walkers.
p0 = [np.random.rand(ndim) for i in range(nwalkers)]

# Initialize the sampler with the chosen specs.
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[means, icov])

# Run 5000 steps as a burn-in.
pos, prob, state = sampler.run_mcmc(p0, 5000)

# Reset the chain to remove the burn-in samples.
sampler.reset()

# Starting from the final position in the burn-in chain, sample for 100000
# steps.
sampler.run_mcmc(pos, 100000, rstate0=state)

# Print out the mean acceptance fraction. In general, acceptance_fraction
# has an entry for each walker so, in this case, it is a 50-dimensional
# vector.
print("Mean acceptance fraction:", np.mean(sampler.acceptance_fraction))

# Estimate the integrated autocorrelation time for the time series in each
# parameter.
print("Autocorrelation time:", sampler.get_autocorr_time())

# Finally, you can plot the projected histograms of the samples using
# matplotlib as follows (as long as you have it installed).
try:
    import matplotlib.pyplot as pl
except ImportError:
    print("Try installing matplotlib to generate some sweet plots...")
else:
    pl.hist(sampler.flatchain[:,0], 100)
    pl.show()
