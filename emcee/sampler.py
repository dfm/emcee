#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The base sampler class implementing various helpful functions.

"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["Sampler"]

import numpy as np


class Sampler(object):
    """
    An abstract sampler object that implements various helper functions

    :param dim:
        The number of dimensions in the parameter space.

    :param lnpostfn:
        A function that takes a vector in the parameter space as input and
        returns the natural logarithm of the posterior probability for that
        position.

    :param args: (optional)
        A list of extra positional arguments for ``lnpostfn``. ``lnpostfn``
        will be called with the sequence ``lnpostfn(p, *args, **kwargs)``.

    :param kwargs: (optional)
        A list of extra keyword arguments for ``lnpostfn``. ``lnpostfn``
        will be called with the sequence ``lnpostfn(p, *args, **kwargs)``.

    """
    def __init__(self, dim, lnprobfn, args=[], kwargs={}):
        self.dim = dim
        self.lnprobfn = lnprobfn
        self.args = args
        self.kwargs = kwargs

        # This is a random number generator that we can easily set the state
        # of without affecting the numpy-wide generator
        self._random = np.random.mtrand.RandomState()

        self.reset()

    @property
    def random_state(self):
        """
        The state of the internal random number generator. In practice, it's
        the result of calling ``get_state()`` on a
        ``numpy.random.mtrand.RandomState`` object. You can try to set this
        property but be warned that if you do this and it fails, it will do
        so silently.

        """
        return self._random.get_state()

    @random_state.setter  # NOQA
    def random_state(self, state):
        """
        Try to set the state of the random number generator but fail silently
        if it doesn't work. Don't say I didn't warn you...

        """
        try:
            self._random.set_state(state)
        except:
            pass

    @property
    def acceptance_fraction(self):
        """
        The fraction of proposed steps that were accepted.

        """
        return self.naccepted / self.iterations

    @property
    def chain(self):
        """
        A pointer to the Markov chain.

        """
        return self._chain

    @property
    def flatchain(self):
        """
        Alias of ``chain`` provided for compatibility.

        """
        return self._chain

    @property
    def lnprobability(self):
        """
        A list of the log-probability values associated with each step in
        the chain.

        """
        return self._lnprob

    @property
    def acor(self):
        return self.get_autocorr_time()

    def get_autocorr_time(self, window=50):
        raise NotImplementedError("The acor method must be implemented "
                                  "by subclasses")

    def get_lnprob(self, p):
        """Return the log-probability at the given position."""
        return self.lnprobfn(p, *self.args, **self.kwargs)

    def reset(self):
        """
        Clear ``chain``, ``lnprobability`` and the bookkeeping parameters.

        """
        self.iterations = 0
        self.naccepted = 0
        self._last_run_mcmc_result = None

    def clear_chain(self):
        """An alias for :func:`reset` kept for backwards compatibility."""
        return self.reset()

    def sample(self, *args, **kwargs):
        raise NotImplementedError("The sampling routine must be implemented "
                                  "by subclasses")

    def run_mcmc(self, pos0, N, rstate0=None, lnprob0=None, **kwargs):
        """
        Iterate :func:`sample` for ``N`` iterations and return the result.

        :param pos0:
            The initial position vector.  Can also be None to resume from
            where :func:``run_mcmc`` left off the last time it executed.

        :param N:
            The number of steps to run.

        :param lnprob0: (optional)
            The log posterior probability at position ``p0``. If ``lnprob``
            is not provided, the initial value is calculated.

        :param rstate0: (optional)
            The state of the random number generator. See the
            :func:`random_state` property for details.

        :param kwargs: (optional)
            Other parameters that are directly passed to :func:`sample`.

        This returns the results of the final sample in whatever form
        :func:`sample` yields.  Usually, that's:
        ``pos``, ``lnprob``, ``rstate``, ``blobs`` (blobs optional)
        """
        if pos0 is None:
            if self._last_run_mcmc_result is None:
                raise ValueError("Cannot have pos0=None if run_mcmc has never "
                                 "been called.")
            pos0 = self._last_run_mcmc_result[0]
            if lnprob0 is None:
                rstate0 = self._last_run_mcmc_result[1]
            if rstate0 is None:
                rstate0 = self._last_run_mcmc_result[2]

        for results in self.sample(pos0, lnprob0, rstate0, iterations=N,
                                   **kwargs):
            pass

        # store so that the ``pos0=None`` case will work.  We throw out the blob
        # if it's there because we don't need it
        self._last_run_mcmc_result = results[:3]

        return results
