#!/usr/bin/env python
"""
Run this example with:
mpirun -np 2 python examples/mpi.py

"""

from __future__ import print_function

import sys
import numpy as np
import emcee
from emcee.utils import MPIPool


def lnprob(x):
    return -0.5 * np.sum(x ** 2)

ndim = 10
nwalkers = 200
p0 = [np.random.rand(ndim) for i in xrange(nwalkers)]
pool = MPIPool()

if not pool.is_master():
    # Wait for instructions from the master process.
    pool.wait()
    sys.exit(0)

# Initialize the sampler with the chosen specs.
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool)

# Run 100 steps as a burn-in.
pos, prob, state = sampler.run_mcmc(p0, 1)

# Reset the chain to remove the burn-in samples.
sampler.reset()

# Starting from the final position in the burn-in chain, sample for 1000
# steps.
sampler.run_mcmc(pos, 1000, rstate0=state)

# Close the processes.
pool.close()

# Print out the mean acceptance fraction. In general, acceptance_fraction
# has an entry for each walker so, in this case, it is a 100-dimensional
# vector.
print(u"Mean acceptance fraction: ", np.mean(sampler.acceptance_fraction))

# Finally, you can plot the projected histograms of the samples using
# matplotlib as follows (as long as you have it installed).
try:
    import matplotlib.pyplot as pl
except ImportError:
    print(u"Try installing matplotlib to generate some sweet plots...")
else:
    pl.hist(sampler.flatchain[:, 0], 100)
    pl.show()
