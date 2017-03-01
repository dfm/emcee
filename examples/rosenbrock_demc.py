#!/usr/bin/env python
"""
Sample code for sampling the Rosenbrock density using emcee.

"""

import numpy as np
import emcee
import scipy
import matplotlib
matplotlib.use('QT4Agg')
import matplotlib.pyplot as plt
import click, corner

# Define the Density
class Rosenbrock(object):
    def __init__(self):
        self.a1 = 100.0
        self.a2 = 20.0

    def __call__(self, p):
        return -(self.a1 * (p[1] - p[0] ** 2) ** 2 + (1 - p[0]) ** 2) / self.a2


num_walkers = 10
num_steps = 50000
burn_in = 500
l_labels = ['x', 'y']

# Make an initial guess for the positions. This is a pretty bad guess!
p0 = np.random.rand(num_walkers * 2).reshape(num_walkers, 2)


# Instantiate the class
rosenbrock = Rosenbrock()


# The sampler object
#sampler = emcee.EnsembleSampler(num_walkers, 2, rosenbrock, threads=1)
sampler = emcee.DESampler(num_walkers, 2, rosenbrock, threads=4)


# Sample, outputting to a file
with click.progressbar(sampler.sample(p0, iterations=num_steps, ), length=num_steps) as mcmc_sampler:
	for pos, lnprob, state in mcmc_sampler:
		pass



samples = sampler.chain[:, burn_in:, :].reshape((-1, 2))
fig = corner.corner(samples, labels=l_labels, quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt='.3f', title_kwargs={'fontsize': 12})


print sampler.acceptance_fraction
try:
    print sampler.acor
except:
    print 'Not enough steps - try increasing chain size...'
plt.show()


