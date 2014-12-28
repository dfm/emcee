# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["BaseWalker", "SimpleWalker"]

import copy
import numpy as np


class BaseWalker(object):
    """
    The abstract base class for each walker implementation.

    """
    def propose(self, coords):
        """
        This function is called when a proposal is updating the walker
        coordinates.

        :param coords:
            The new coordinates of the walker.

        """
        new = copy.copy(self)
        new._compute(coords)
        return new

    def _compute(self, coords):
        self._coords = coords
        self._lnprior, self._lnlike, self._lnprob = self.lnprobfn(self._coords)

    def lnpriorfn(self, coords, *args):
        """
        This method must be implemented by subclasses to return the natural
        logarithm of the prior function up to a constant.

        :param coords:
            The coordinates where the prior should be evaluated.

        """
        raise NotImplementedError("Subclasses must implement a ln-prior "
                                  "function")

    def lnlikefn(self, coords, *args):
        """
        This method must be implemented by subclasses to return the natural
        logarithm of the likelihood function up to a constant.

        :param coords:
            The coordinates where the likelihood should be evaluated.

        """
        raise NotImplementedError("Subclasses must implement a ln-likelihood "
                                  "function")

    def lnprobfn(self, coords, *args):
        """
        This method computes the natural logarithm of the posterior
        probability up to a constant. User shouldn't usually need to touch
        this implementation. Override the :func:`lnpriorfn` and
        :func:`lnlikefn` methods instead.

        :param coords:
            The coordinates where the probability should be evaluated.

        """
        lp = self.lnpriorfn(coords, *args)
        if not np.isfinite(lp):
            return -np.inf, -np.inf, -np.inf
        ll = self.lnlikefn(coords, *args)
        if not np.isfinite(ll):
            return lp, -np.inf, -np.inf
        return lp, ll, lp + ll

    @property
    def coords(self):
        """The coordinate vector of the walker."""
        return self._coords

    @coords.setter
    def coords(self, coords):
        # Check the dimensions of the coordinate vector.
        coords = np.atleast_1d(coords).astype(np.float64)
        if not len(coords.shape) == 1:
            raise ValueError("Invalid coordinate dimensions")
        self._compute(coords)

    @property
    def lnprior(self):
        """The ln-prior of the walker up to a constant."""
        return self._lnprior

    @property
    def lnlike(self):
        """The ln-likelihood of the walker up to a constant."""
        return self._lnlike

    @property
    def lnprob(self):
        """The ln-probability of the walker up to a constant."""
        return self._lnprob

    def __len__(self):
        return len(self.coords)


class SimpleWalker(BaseWalker):
    """
    A simple extension of the :class:`BaseWalker` subclass that accepts
    functions that evaluate the ln-prior and ln-likelihood functions.

    :param coords:
        The initial coordinate vector for the walker.

    :param lnpriorfn:
        A callable that evaluates the ln-prior at a given coordinate vector.

    :param lnlikefn:
        A callable that evaluates the ln-likelihood at a given coordinate
        vector.

    """
    def __init__(self, lnpriorfn, lnlikefn, *args, **kwargs):
        self._lnpriorfn = lnpriorfn
        self._lnlikefn = lnlikefn
        self.args = args
        self.kwargs = kwargs

    def lnpriorfn(self, coords):
        return self._lnpriorfn(coords, *(self.args), **(self.kwargs))

    def lnlikefn(self, coords):
        return self._lnlikefn(coords, *(self.args), **(self.kwargs))
