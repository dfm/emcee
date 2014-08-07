#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as pl

from emcee import autocorr
from emcee import proposals
from emcee.sampler import Sampler

np.random.seed(123)


def lnprior(p):
    return 0.0


class Rosenbrock(object):
    def __init__(self):
        self.a1 = 100.0
        self.a2 = 20.0

    def __call__(self, p):
        return -(self.a1 * (p[1] - p[0] ** 2) ** 2 + (1 - p[0]) ** 2) / self.a2

ndim = 10
lnlike = Rosenbrock()
# sampler = Sampler(lnprior, lnlike, proposals.StretchProposal())
sampler = Sampler(lnprior, lnlike, proposals.GaussianProposal(ndim, 1.0))
p0 = 0.1 * np.random.randn(32, ndim)

print("Production")
for i, state in enumerate(sampler.sample(p0, nstep=1e8)):
    pass

print(sampler.acceptance_fraction)
print(sampler.get_autocorr_time())
print(autocorr.integrated_time(sampler.chain[:, 1], axis=0))
# f = autocorr.function(sampler.chain[:, 0], axis=0)
# pl.plot(f)
# pl.savefig("blah.png")
assert 0
# print(sampler.chain.shape)
# print(np.mean(sampler.chain, axis=(0, 1)))
# print(np.std(sampler.chain, axis=(0, 1)))

pl.hist(sampler.chain[:, :, 0].flatten(), 50, histtype="step", color="k",
        normed=True)
x = np.linspace(-6, 6, 5000)
pl.plot(x, np.exp(-0.5 * x**2) / np.sqrt(2*np.pi))

pl.savefig("dude.png")

pl.clf()
pl.plot(sampler.chain[:, :, 0])
pl.savefig("dude2.png")
