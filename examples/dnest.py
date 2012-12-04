#!/usr/bin/env python
"""
Run this example with:
mpirun -np 2 python examples/mpi.py

"""

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as pl

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), u".."))

import emcee


def lnlike(x):
    logl1 = - 0.5 * np.sum(x ** 2) \
            - 0.5 * len(x) * np.log(2 * np.pi)
    return logl1


def lnprior(x):
    if np.all(x < 10) and np.all(x > -10):
        return 0.0
    return -np.inf


ndim = 2
nwalkers = 100
p0 = [10 - 20 * np.random.rand(ndim) for i in xrange(nwalkers)]

# Initialize the sampler with the chosen specs.
sampler = emcee.DNestSampler(nwalkers, ndim, lnprior, lnlike)

sampler.build_levels(p0, 1000, nlevels=8)

lstars = sampler.levels.lstars
pl.plot(-np.arange(0, len(lstars)), lstars, "ok")
M = np.linspace(-len(lstars) + 1, -1, 1000)
pl.plot(M, -np.log(2 * np.pi) - 200 * np.exp(M) / np.pi)
pl.savefig("sup.png")
