# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["DefaultBackend"]

import copy
import numpy as np


class DefaultBackend(object):

    def __init__(self, store_walkers=False):
        self.store_walkers = store_walkers
        self.reset()

    def reset(self):
        """
        Clear the chain and reset it to its default state.

        """
        # Clear the chain dimensions.
        self.niter = 0
        self.size = 0
        self.nwalkers, self.ndim = None, None

        # Clear the chain wrappers.
        self._coords = None
        self._lnprior = None
        self._lnlike = None
        self._walkers = []

    def check_dimensions(self, ens):
        if self.nwalkers is None:
            self.nwalkers = ens.nwalkers
        if self.ndim is None:
            self.ndim = ens.ndim
        if self.nwalkers != ens.nwalkers or self.ndim != ens.ndim:
            raise ValueError("Dimension mismatch")

    def extend(self, n):
        k, d = self.nwalkers, self.ndim
        self.size = l = self.niter + n
        if self._coords is None:
            self._coords = np.empty((l, k, d), dtype=np.float64)
            self._lnprior = np.empty((l, k), dtype=np.float64)
            self._lnlike = np.empty((l, k), dtype=np.float64)
            self._acceptance = np.zeros(k, dtype=np.uint64)
        else:
            self._coords = np.resize(self._coords, (l, k, d))
            self._lnprior = np.resize(self._lnprior, (l, k))
            self._lnlike = np.resize(self._lnlike, (l, k))

    def update(self, ensemble):
        i = self.niter
        if i >= self.size:
            self.extend(i - self.size + 1)
        self._coords[i] = ensemble.coords
        self._lnprior[i] = ensemble.lnprior
        self._lnlike[i] = ensemble.lnlike
        self._acceptance += ensemble.acceptance
        if self.store_walkers:
            self._walkers.append(copy.deepcopy(ensemble.walkers))
        self.niter += 1

    @property
    def coords(self):
        return None if self._coords is None else self._coords[:self.niter]

    @property
    def lnprior(self):
        return self._lnprior[:self.niter]

    @property
    def lnlike(self):
        return self._lnlike[:self.niter]

    @property
    def lnprob(self):
        return self.lnprior + self.lnlike

    @property
    def walkers(self):
        if not self.store_walkers:
            raise AttributeError("You need to store the walkers using the "
                                 "'store_walkers' keyword argument to the "
                                 "DefaultBackend class")
        return self._walkers

    @property
    def acceptance(self):
        return self._acceptance

    @property
    def acceptance_fraction(self):
        return self.acceptance / float(self.niter)
