from __future__ import print_function

__all__ = ["DNestSampler"]

import numpy as np

from .sampler import Sampler
from .ensemble import EnsembleSampler


class DNestLevel(object):

    def __init__(self, lstar, lnpriorfn, lnlikefn):
        self.lstar = lstar

    def __call__(self, p):
        pr = self.lnpriorfn(p)
        if np.isinf(pr) or np.exp(self.lnlikefn(p)) < self.lstar:
            return -np.inf
        return pr


class DNestSampler(Sampler):

    def __init__(dim, lnpriorfn, lnlikefn):
        super(EnsembleSampler, self).__init__(dim, lnpriorfn)
        self.levels = []
        self.lnpriorfn = lnpriorfn
        self.lnlikefn = lnlikefn

    def build_levels(self):
        self.levels = [DNestLevel(0.0, lnpriorfn, lnlikefn)]
        lnpostfn =
        sampler = EnsembleSampler(self.dim,
