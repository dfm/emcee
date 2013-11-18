#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
import time
from emcee import PTSampler


#####################################################################
########Sampling Multi-Modal Gaussian timing#########################
#####################################################################

# mu1 = [1, 1], mu2 = [-1, -1]
mu1 = np.ones(2)
mu2 = -np.ones(2)

# Width of 0.1 in each dimension
sigma1inv = np.diag([100.0, 100.0])
sigma2inv = np.diag([100.0, 100.0])


def logl(x):
    dx1 = x - mu1
    dx2 = x - mu2

    return np.logaddexp(-np.dot(dx1, np.dot(sigma1inv, dx1)) / 2.0,
                        -np.dot(dx2, np.dot(sigma2inv, dx2)) / 2.0)


# Use a flat prior
def logp(x):
    return 0.0


#vectorized logl accepts subenseble of walkers
def vec_logl(x):
    dx1 = x.T - mu1[:, np.newaxis]
    dx2 = x.T - mu2[:, np.newaxis]

    return np.logaddexp(-0.5 * np.einsum('i..., i...', dx1, np.dot(sigma1inv, dx1)),
                        -0.5 * np.einsum('i..., i...', dx2, np.dot(sigma2inv, dx2)))


#vectorized logp accepts subenseble of walkers
def vec_logp(x):
    return np.zeros(np.shape(x)[0])


ntemps = 20
nwalkers = 100
ndim = 2


sampler = PTSampler(ntemps, nwalkers, ndim, logl, logp)
#poolsampler = PTSampler(ntemps, nwalkers, ndim, logl, logp, threads=8)
vecsampler = PTSampler(ntemps, nwalkers, ndim, vec_logl, vec_logp, bcast=True)

p0 = np.random.uniform(low=-1.0, high=1.0, size=(ntemps, nwalkers, ndim))

# Burn in.
for p, lnprob, lnlike in sampler.sample(p0, iterations=1000):
    pass
sampler.reset()

#for pp, plnprob, plnlike in poolsampler.sample(p0, iterations=1000):
#    pass
#poolsampler.reset()

for vp, vlnprob, vlnlike in vecsampler.sample(p0, iterations=1000):
    pass
vecsampler.reset()

##################################################################
######Timing for Multivariate Normal density sampling#############
##################################################################

t1 = time.time()
for p, lnprob, lnlike in sampler.sample(p, lnprob0=lnprob,
                                           lnlike0=lnlike,
                                           iterations=10000, thin=10):
    pass
dt = time.time() - t1
print "time : " + str(dt)
sampler.reset()

#it takes ~ 2500s on mobile i3 Sandy Bridge
#t1 = time.time()
#for p, lnprob, lnlike in poolsampler.sample(pp, lnprob0=plnprob,
#                                           lnlike0=plnlike,
#                                           iterations=10000, thin=10):
#    pass
#dt = time.time() - t1
#print "pool time : " + str(dt)
#poolsampler.reset()

t1 = time.time()
for p, lnprob, lnlike in vecsampler.sample(vp, lnprob0=vlnprob,
                                           lnlike0=vlnlike,
                                           iterations=10000, thin=10):
    pass
dt = time.time() - t1
print "bcast=True time : " + str(dt)

#####################################################################

assert vecsampler.chain.shape == (ntemps, nwalkers, 1000, ndim)

# Chain has shape (ntemps, nwalkers, nsteps, ndim)
# Zero temperature mean:
mu0 = np.mean(np.mean(vecsampler.chain[0, ...], axis=0), axis=0)

# Longest autocorrelation length (over any temperature)
max_acl = np.max(vecsampler.acor)

#plotting sampled density
try:
    import matplotlib.pyplot as pl
except ImportError:
    print("Try installing matplotlib to generate some sweet plots...")
else:
    H, xedges, yedges = np.histogram2d(vecsampler.chain[0][..., 0].flatten(),
            vecsampler.chain[0][..., 1].flatten(), bins=(128, 128))
    extent = [yedges[0], yedges[-1], xedges[-1], xedges[0]]
    pl.imshow(H, interpolation='nearest', extent=extent, aspect="auto")

vecsampler.reset()
