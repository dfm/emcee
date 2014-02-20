#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import time
import numpy as np

import emcee
from emcee.schedule import Schedule
from emcee.state import MemoizedState
from emcee.proposals import GaussianProposal


class MyState(MemoizedState):
    def __call__(self, coords):
        return self.lnprob(coords)

    def lnprob(self, coords):
        return -0.5 * np.sum(coords**2)


N = 100
ndim, nwalkers = 10, 100
p0 = [np.random.rand(ndim) for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, MyState(None))

strt = time.time()
sampler.run_mcmc(p0, N)
print(time.time() - strt)


schedule = Schedule(MyState(None), [GaussianProposal(np.eye(ndim))])
ensemble = [np.random.rand(ndim) for i in range(nwalkers)]

samples = []
strt = time.time()
for i, ensemble in enumerate(schedule.sample(ensemble)):
    # samples += [s.get_coords() for s in ensemble]
    if i >= N:
        break
print(time.time() - strt)

# samples = np.array(samples)
# print(samples.shape)
