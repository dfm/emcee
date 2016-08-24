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
from scipy.stats import multivariate_normal

# Define the Density
class DoubleGaussian(object):
    def __init__(self):
        self.u1 = [-5.0, -5.0]
        self.u2 = [10.0, 10.0]
        self.s1 = [[1.0, 0], [0, 2.0]]
        self.s2 = [[2.0, 0], [0, 1.0]]

    def __call__(self, p):
        return np.log(multivariate_normal.pdf(p, self.u1, self.s1) + multivariate_normal.pdf(p, self.u2, self.s2))


num_walkers = 40
num_steps = 4000
burn_in = 1000
num_dim = 2
l_labels = ['x', 'y']
l_value_guesses = [0, 0]
l_std_guesses = [.1, .1]

# Make an initial guess for the positions. This is a pretty bad guess!
a_starting_pos = emcee.utils.sample_ball(l_value_guesses, l_std_guesses, size=num_walkers)


# Instantiate the class
double_gaus = DoubleGaussian()


# The sampler object
#sampler = emcee.EnsembleSampler(num_walkers, num_dim, double_gaus, threads=4)
sampler = emcee.DESampler(num_walkers, 2, double_gaus, threads=4)


# Sample, outputting to a file
with click.progressbar(sampler.sample(a_starting_pos, iterations=num_steps, ), length=num_steps) as mcmc_sampler:
	for pos, lnprob, state in mcmc_sampler:
		pass



samples = sampler.chain[:, burn_in:, :].reshape((-1, 2))
fig = corner.corner(samples, labels=l_labels, show_titles=True, title_fmt='.3f', title_kwargs={'fontsize': 12})


print sampler.acceptance_fraction
try:
    print sampler.acor
except:
    print 'Not enough steps - try increasing chain size...'
plt.show()


