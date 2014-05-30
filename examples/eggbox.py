#!/usr/bin/env python
"""
Sample code for sampling the Eggbox density [Feroz et al. (2008)][].

Thanks to Phil Marshall (Oxford) who coded this example.

[Feroz et al. (2008)]: http://arxiv.org/pdf/0809.3437.pdf

"""

from __future__ import print_function
import numpy as np
import emcee


# Define the posterior density to be sampled:
class Eggbox(object):
    def __init__(self):
        self.tmax = 10.0 * np.pi
        self.constant = np.log(1.0 / (self.tmax * self.tmax))

    def logprior(self, t):
        if (t[0] > self.tmax or t[0] < -self.tmax or \
            t[1] > self.tmax or t[1] < -self.tmax):
            return -np.inf
        else:
            return self.constant

    def loglhood(self, t):
        return (2.0 + np.cos(t[0] / 2.0) * np.cos(t[1] / 2.0)) ** 5.0

    def __call__(self, t):
        return self.logprior(t) + self.loglhood(t)


# Now, set up and run the sampler:

nwalkers = 500

# Make an initial guess for the positions - uniformly
# distributed between +/- 10pi:

p0 = 10.0 * np.pi * (2.0 * np.random.rand(nwalkers * 2) - 1.0)
p0 = p0.reshape(nwalkers, 2)

# Instantiate the class
logposterior = Eggbox()

# The sampler object:
sampler = emcee.EnsembleSampler(nwalkers, 2, logposterior, threads=10)

# Burn in.
pos, prob, state = sampler.run_mcmc(p0, 100)

# Clear the burn in.
sampler.reset()

# Sample, outputting to a file
fn = "eggbox.out"
f = open(fn, "w")
f.close()
for pos, prob, rstate in sampler.sample(pos, prob, state, iterations=1000):
    # Write the current position to a file, one line per walker
    f = open(fn, "w")
    f.write("\n".join(["\t".join([str(q) for q in p]) for p in pos]))
    f.write("\n")
    f.close()

# Plot it up.
try:
    import matplotlib.pyplot as pl
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    print("Try installing matplotlib to generate some sweet plots...")
else:
    pl.figure()
    for k in range(nwalkers):
        pl.plot(sampler.chain[k, :, 0])
    pl.xlabel("time")
    pl.savefig("eggbox_time.png")

    pl.figure(figsize=(8, 8))
    x, y = sampler.flatchain[:, 0], sampler.flatchain[:, 1]
    pl.plot(x, y, "ok", ms=1, alpha=0.1)
    pl.savefig("eggbox_2d.png")

    fig = pl.figure()
    ax = fig.add_subplot(111, projection="3d")

    for k in range(nwalkers):
        x, y = sampler.chain[k, :, 0], sampler.chain[k, :,1]
        z = sampler.lnprobability[k, :]
        ax.scatter(x, y, z, marker="o", c="k", alpha=0.5, s=10)
    pl.savefig("eggbox_3d.png")

