"""
Demo of the proposed "blobs" feature.

"""

import os
import sys


# Adding the development version of `emcee` to the path so that this works
# even if you have an older (stable) version of the module installed on your
# path. NOTE: This should be removed once we merge this feature.
sys.path = [os.path.abspath(os.path.join(__file__, '..', '..')), ] + sys.path


import emcee
import numpy as np


# This is a dumb-ass log-probability function that also returns a random
# "blob" (this can be any arbitrary---preferably picklable---Python object
# that is associated with this particular position in parameter space.
def lnprob(p):
    return -0.5 * np.sum(p ** 2), np.random.rand()


# Set up the sampler and randomly select a starting position.
nwalkers, ndim = 100, 50
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
p0 = emcee.EnsembleSampler.sampleBall(np.zeros(ndim), np.random.rand(ndim),
        nwalkers)

# Sample for a few iterations.
niterations = 500
for pos, lp, rstate, blob in sampler.sample(p0, iterations=niterations):
    print blob

# The blobs are stored in the `blobs` object. This object is a list (of
# length `niterations`) of lists (of length `nwalkers`).
print sampler.blobs
