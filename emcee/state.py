# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["State"]

import copy


class State(object):

    def __init__(self, lnprob_fn, coords, lnprior, lnlike):
        self.lnprob_fn = lnprob_fn

        self.coords = coords
        self.lnprior = lnprior
        self.lnlike = lnlike

    def copy(self):
        return State(self.lnprob_fn, copy.copy(self.coords), None, None)

    def __len__(self):
        return len(self.coords)

    @property
    def shape(self):
        return self.coords.shape

    def __add__(self, other):
        new = self.copy()
        if hasattr(other, "coords"):
            new.coords += other.coords
        else:
            new.coords += other
        return new

    def __sub__(self, other):
        new = self.copy()
        if hasattr(other, "coords"):
            new.coords -= other.coords
        else:
            new.coords -= other
        return new

    def __rsub__(self, other):
        new = self.copy()
        if hasattr(other, "coords"):
            new.coords[:] = other.coords - new.coords
        else:
            new.coords[:] = other - new.coords
        return new

    def __mul__(self, other):
        new = self.copy()
        if hasattr(other, "coords"):
            new.coords *= other.coords
        else:
            new.coords *= other
        return new

    def __rmul__(self, other):
        new = self.copy()
        if hasattr(other, "coords"):
            new.coords *= other.coords
        else:
            new.coords *= other
        return new

    def __getitem__(self, s):
        lp = self.lnprior[s] if self.lnprior is not None else None
        ll = self.lnlike[s] if self.lnlike is not None else None
        return State(self.lnprob_fn, self.coords[s], lp, ll)

    def compute_lnprob(self):
        self.lnprior, self.lnlike = self.lnprob_fn(self.coords)

    @property
    def lnprob(self):
        return self.lnprior + self.lnlike

    def update(self, other, m=None):
        if m is None:
            self.coords[:] = other.coords
            self.lnprior[:] = other.lnprior
            self.lnlike[:] = other.lnlike
            return

        self.coords[m] = other.coords[m]
        self.lnprior[m] = other.lnprior[m]
        self.lnlike[m] = other.lnlike[m]
