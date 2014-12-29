# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["BaseWalker", "SimpleWalker"]

import numpy as np


class BaseWalker(object):
    """
    The abstract base class for each walker implementation.

    """

    stateful_attributes = ["_coords", "_lnprior", "_lnlike", "_lnprob"]

    def propose(self, coords):
        """
        This function is called when a proposal is updating the walker
        coordinates.

        :param coords:
            The new coordinates of the walker.

        """
        new = self.new()
        new.compute(coords)
        return new

    def new(self):
        """
        Return a new instance of the walker. The default implementation is
        similar to a shallow copy after removing the "stateful" attributes.
        These are attributes that always depend on the coordinates (things
        like the ln-probability) so you wouldn't want to copy them to a new
        instance. This gets a *tiny* performance improvement over direct use
        of ``copy.copy`` but we'll call it a lot of times so maybe it's worth
        it. You might want to implement a more efficient copy function if you
        know more about your class. If your model has other stateful
        attributes, you can also just append them to the class-level
        ``stateful_attributes`` list.

        """
        # Update the __dict__ to skip the stateful attributes.
        d = dict(self.__dict__)
        for k in self.stateful_attributes:
            d.pop(k, None)

        # Create the new instance.
        cls = self.__class__
        new = cls.__new__(cls)

        # Only copy across the attributes that we need.
        new.__dict__.update(d)
        return new

    def compute(self, coords):
        """
        Update the walker to a specified coordinate vector and compute the
        probability at this location.

        :param coords:
            The new coordinate vector.

        """
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
        self.compute(coords)

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

    :param lnprior_or_lnprob:
        A callable that evaluates the ln-prior at a given coordinate vector.
        If ``lnlikefn`` isn't given, this function is assumed to evaluate the
        log-probability up to a constant.

    :param lnlikefn: (optional)
        A callable that evaluates the ln-likelihood at a given coordinate
        vector.

    """
    def __init__(self, lnprior_or_lnprob, lnlikefn=None):
        if lnlikefn is None:
            self._lnpriorfn = _default_lnprior
            self._lnlikefn = lnprior_or_lnprob
        else:
            self._lnpriorfn = lnprior_or_lnprob
            self._lnlikefn = lnlikefn

    def lnpriorfn(self, coords):
        return self._lnpriorfn(coords)

    def lnlikefn(self, coords):
        return self._lnlikefn(coords)


def _default_lnprior(*args, **kwargs):
    return 0.0
