#!/usr/bin/env python
# encoding: utf-8
"""
Sample code for sampling the Rosenbrock density using emcee.

"""

import numpy as np

import emcee

# Define the Density
class Rosenbrock(object):
    def __init__(self):
        self.a1 = 100.0
        self.a2 = 20.0

    def __call__(self, p):
        return -(self.a1 * (p[1]-p[0]**2)**2 + (1-p[0])**2)/self.a2

nwalkers = 100

# Make an initial guess for the positions. This is a pretty bad guess!
p0 = np.random.rand(nwalkers*2).reshape(nwalkers,2)

# Instantiate the class
rosenbrock = Rosenbrock()

# The sampler object
sampler = emcee.EnsembleSampler(nwalkers, 2, rosenbrock)

# Sample, outputting to a file
f = open("rosenbrock.out", "w")
for pos, lnprob, rstate in sampler.sample(np.array(p0), iterations=500):
    # Write the current position to a file, one line per walker
    f.write("\n".join(["\t".join([str(q) for q in p]) for p in pos]))
    f.write("\n")
f.close()

