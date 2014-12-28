#!/usr/bin/env python
"""
Sample code for sampling a multivariate Gaussian using emcee.

"""

from __future__ import print_function

import emcee
import numpy as np


# First, define the probabilistic model.
class MyModel(emcee.BaseWalker):
    def lnpriorfn(self, x):
        return 0.0

    def lnlikefn(self, x):
        return -0.5 * np.sum(x ** 2)


# We'll sample a 5 dimensional Gaussian using 64 walkers.
ndim, nwalkers = 5, 64

# We'll start by choosing the initial coordinates for the walkers.
coords = np.random.randn(nwalkers, ndim)

# Initialize the ensemble using these coordinates. NOTE: the probabilities
# of the walkers will be evaluated immediately.
ensemble = emcee.Ensemble(MyModel(), coords)

# Initialize and the sampler.
sampler = emcee.Sampler()

# Run the burn-in.
ensemble = sampler.run(ensemble, 100, store=False)

# Run the production chain.
sampler.run(ensemble, 1000)

# Print out the mean acceptance fraction. In general, acceptance_fraction
# has an entry for each walker so, in this case, it is a 100-dimensional
# vector.
print("Mean acceptance fraction:", np.mean(sampler.acceptance_fraction))

# Finally, you can plot the projected histograms of the samples using
# triangle.py (https://github.com/dfm/triangle.py)
try:
    import triangle

except ImportError:
    print("Install triangle.py (https://github.com/dfm/triangle.py) for plots")

else:
    fig = triangle.corner(sampler.get_coords(flat=True))
    fig.savefig("quickstart-corner.png")
