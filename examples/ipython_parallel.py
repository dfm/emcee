#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This example demonstrates how you can use emcee with IPython parallel for
distributed computation of your probability function. First, start up an
ipcluster in one terminal and then execute this file.

"""

from __future__ import division, print_function

import emcee
import numpy as np
from IPython.parallel import Client


def lnprob(x):
    return -0.5 * np.sum(x ** 2)

# Set up the interface to the ipcluster.
c = Client()
view = c[:]
view.push({"lnprob": lnprob})
view.execute("import numpy as np")

# Set up the sampler.
ndim = 10
nwalkers = 100
p0 = [np.random.rand(ndim) for i in xrange(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=view)

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
print("Mean acceptance fraction: ", np.mean(sampler.acceptance_fraction))
