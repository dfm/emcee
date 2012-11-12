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

# We'll sample a 50-dimensional Gaussian...
ndim = 50
# ...with randomly chosen mean position...
means = np.random.rand(ndim)
# ...and a positive definite, non-trivial covariance matrix.
cov  = 0.5-np.random.rand(ndim**2).reshape((ndim, ndim))
cov  = np.triu(cov)
cov += cov.T - np.diag(cov.diagonal())
cov  = np.dot(cov,cov)

# Invert the covariance matrix first.
icov = np.linalg.inv(cov)

# We'll sample with 250 walkers.
nwalkers = 250

# Choose an initial set of positions for the walkers.
p0 = [np.random.rand(ndim) for i in xrange(nwalkers)]

# Initialize the sampler with the chosen specs.
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[means, icov])

# Run 100 steps as a burn-in.
pos, prob, state = sampler.run_mcmc(p0, 100)

# Reset the chain to remove the burn-in samples.
sampler.reset()

# Starting from the final position in the burn-in chain, sample for 1000
# steps.
sampler.run_mcmc(pos, 1000, rstate0=state)

# Print out the mean acceptance fraction. In general, acceptance_fraction
# has an entry for each walker so, in this case, it is a 100-dimensional
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
    pl.hist(sampler.flatchain[:,0], 100)
    pl.show()

