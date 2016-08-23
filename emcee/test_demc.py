#!/usr/bin/env python
# encoding: utf-8
"""
Defines various nose unit tests

"""

import numpy as np
import scipy
import matplotlib
matplotlib.use('QT4Agg')
import matplotlib.pyplot as plt
import click, corner
from scipy.stats import norm

from .desampler import DifferentialEvolutionSampler
from .utils import sample_ball

a_data = np.random.normal(size=1000)

num_pars = 2
num_walkers = 64
num_steps = 5000
burn_in = 500

def ln_likelihood(a_pars):
    #print 'likelihood'
    ln_likelihood = np.sum(norm.logpdf(a_data, loc=a_pars[0], scale=a_pars[1]))
    if np.isnan(ln_likelihood):
        return -np.inf
    else:
        return ln_likelihood

sampler = DifferentialEvolutionSampler(num_walkers, num_pars, ln_likelihood)

l_value_guesses = [1., 4.]
l_std_guesses = [0.1, 2]
l_labels = ['mean', 'std']
a_starting_pos = sample_ball(l_value_guesses, l_std_guesses, size=num_walkers)

with click.progressbar(sampler.sample(a_starting_pos, iterations=num_steps, ), length=num_steps) as mcmc_sampler:
	for pos, lnprob, state in mcmc_sampler:
		pass


samples = sampler.chain[:, burn_in:, :].reshape((-1, num_pars))
fig = corner.corner(samples, labels=l_labels, quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt='.3f', title_kwargs={'fontsize': 12})


print sampler.acceptance_fraction
try:
    print sampler.acor
except:
    print 'Not enough steps - try increasing chain size...'
plt.show()