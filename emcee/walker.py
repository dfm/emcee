# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["BaseWalker", "SimpleWalker"]

import numpy as np


class BaseWalker(object):
    """
    The abstract base class for each walker implementation.

    :param coords:
        The initial coordinate vector for the walker.

    """
    def __init__(self, coords, *args, **kwargs):
        self._init_args = args
        self._init_kwargs = kwargs
        self.coords = coords

    def propose(self, coords):
        """
        This function is called when a proposal is updating the walker
        coordinates.

        :param coords:
            The new coordinates of the walker.

        """
        return self.__class__(coords, *(self._init_args),
                              **(self._init_kwargs))

    def _compute(self):
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
        self._coords = np.atleast_1d(coords).astype(np.float64)
        if not len(self._coords.shape) == 1:
            raise ValueError("Invalid coordinate dimensions")
        self._compute()

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
    def __init__(self, coords, lnpriorfn, lnlikefn, *args, **kwargs):
        self._lnpriorfn = lnpriorfn
        self._lnlikefn = lnlikefn
        super(SimpleWalker, self).__init__(coords, *args, **kwargs)

    def propose(self, coords):
        """
        This function is called when a proposal is updating the walker
        coordinates.

        :param coords:
            The new coordinates of the walker.

        """
        return self.__class__(coords, self._lnpriorfn, self._lnlikefn,
                              *(self._init_args), **(self._init_kwargs))

    def lnpriorfn(self, coords, *args):
        return self._lnpriorfn(coords, *args)

    def lnlikefn(self, coords, *args):
        return self._lnlikefn(coords, *args)
