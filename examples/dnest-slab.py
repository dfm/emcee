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
    logl1 = - 0.5 * np.sum(x ** 2) / 0.1 / 0.1 \
            - 0.5 * len(x) * np.log(2 * np.pi * 0.1 * 0.1)
    logl2 = - 0.5 * np.sum((x - 0.031) ** 2) / 0.01 / 0.01 \
            - 0.5 * len(x) * np.log(2 * np.pi * 0.01 * 0.01) + 100.0
    return np.logaddexp(logl1, logl2)


def lnprior(x):
    if np.all(x < 0.5) and np.all(x > -0.5):
        return 0.0
    return -np.inf


ndim = 20
nwalkers = 100
p0 = [0.5 - np.random.rand(ndim) for i in xrange(nwalkers)]

# Initialize the sampler with the chosen specs.
sampler = emcee.DNestSampler(nwalkers, ndim, lnprior, lnlike)

sampler.build_levels(p0, 1000, nlevels=100)

lstars = sampler.levels.lstars
pl.plot(-np.arange(0, len(lstars)), lstars, "ok")
pl.savefig("slab.png")
