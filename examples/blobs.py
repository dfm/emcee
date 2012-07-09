import os
import sys


sys.path = [os.path.abspath(os.path.join(__file__, '..', '..')), ] + sys.path


import emcee
import numpy as np


def lnprob(p):
    return -0.5 * np.sum(p ** 2), np.random.rand()


nwalkers, ndim = 100, 50
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)

p0 = emcee.EnsembleSampler.sampleBall(np.zeros(ndim), np.random.rand(ndim),
        nwalkers)

for pos, lp, rstate, blob in sampler.sample(p0, iterations=500):
    print blob
