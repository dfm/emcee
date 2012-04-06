#!/usr/bin/env python
"""
Sample code for sampling the Eggbox density (Feroz et al 2008) using emcee.

"""

import numpy as np
import emcee

# ----------------------------------------------------------------------------
# Define the posterior density to be sampled:
class Eggbox(object):
    def __init__(self):
        self.tmax = 10.0*np.pi
        self.constant = np.log(1.0/(self.tmax*self.tmax))
    
    def logprior(self,t):
        if (t[0] > self.tmax or t[0] < -self.tmax or \
            t[1] > self.tmax or t[1] < -self.tmax):
          return -np.inf
        else:
          return self.constant
    
    def loglhood(self,t):
        return (2.0 + np.cos(t[0]/2.0)*np.cos(t[1]/2.0))**5.0
    
    def __call__(self, t):
        return self.logprior(t) + self.loglhood(t)

# ----------------------------------------------------------------------------

# Now, set up and run the sampler:

nwalkers = 100

# Make an initial guess for the positions - uniformly 
# distributed between +/- 10pi:

p0 = 10.0*np.pi*(2.0*np.random.rand(nwalkers*2)-1.0)
p0 = p0.reshape(nwalkers,2)

# Instantiate the class
logposterior = Eggbox()

# The sampler object:
sampler = emcee.EnsembleSampler(nwalkers, 2, logposterior, threads=10)

# Sample, outputting to a file
f = open("eggbox.out", "w")
for pos, prob, rstate in sampler.sample(p0, iterations=2000):
    # Write the current position to a file, one line per walker
    f.write("\n".join(["\t".join([str(q) for q in p]) for p in pos]))
    f.write("\n")
f.close()

