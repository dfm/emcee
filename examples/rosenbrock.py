#!/usr/bin/env python
"""
Sample code for sampling the Rosenbrock density using emcee.

"""

import numpy as np
import emcee
import time


# Define the Density
class Rosenbrock(object):
    def __init__(self):
        self.a1 = 100.0
        self.a2 = 20.0

    def __call__(self, p):
        return -(self.a1 * (p[1] - p[0] ** 2) ** 2 + (1 - p[0]) ** 2) / self.a2


#vectorized Density for bcast=True option in constructor
class vecRosenbrock(Rosenbrock):
    def __init__(self):
        super(vecRosenbrock, self).__init__()

    def __call__(self, p):
        p = np.array(p).T
        return super(vecRosenbrock, self).__call__(p)


nwalkers = 100


# Make an initial guess for the positions. This is a pretty bad guess!
p0 = np.random.rand(nwalkers * 2).reshape(nwalkers, 2)


# Instantiate the class
rosenbrock = Rosenbrock()
vecrosenbrock = vecRosenbrock()


# The sampler object
sampler = emcee.EnsembleSampler(nwalkers, 2, rosenbrock)

# using multiprocessing
poolsampler = emcee.EnsembleSampler(nwalkers, 2, rosenbrock, threads=10)

# or using vectorized density
vecsampler = emcee.EnsembleSampler(nwalkers, 2, vecrosenbrock, bcast=True)


##################################################################
#############Timing for Rosenbrock density sampling###############
##################################################################

t1 = time.time()
for pos, prob, rstate in sampler.sample(p0, iterations=2000):
    pass
dt = time.time() - t1
print "time : " + str(dt)
sampler.reset()

t1 = time.time()
for pos, prob, rstate in poolsampler.sample(p0, iterations=2000):
    pass
dt = time.time() - t1
print "threads=10 time : " + str(dt)
poolsampler.reset()

t1 = time.time()
for pos, prob, rstate in vecsampler.sample(p0, iterations=2000):
    pass
dt = time.time() - t1
print "bcast=True time : " + str(dt)

#plotting sampled density
try:
    import matplotlib.pyplot as pl
except ImportError:
    print("Try installing matplotlib to generate some sweet plots...")
else:
    H, xedges, yedges = np.histogram2d(vecsampler.flatchain[:, 0],
            vecsampler.flatchain[:, 1], bins=(128, 128))
    extent = [yedges[0], yedges[-1], xedges[-1], xedges[0]]
    pl.imshow(H, interpolation='nearest', extent=extent, aspect="auto")

vecsampler.reset()

##################################################################

p0 = np.random.rand(nwalkers * 2).reshape(nwalkers, 2)

# Sample, outputting to a file
f = open("rosenbrock.out", "w")
for pos, prob, rstate in sampler.sample(p0, iterations=2000):
    # Write the current position to a file, one line per walker
    f.write("\n".join(["\t".join([str(q) for q in p]) for p in pos]))
    f.write("\n")
f.close()
