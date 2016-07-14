#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A vanilla Metropolis-Hastings sampler

"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["MHSampler"]

import numpy as np

from . import autocorr
from .sampler import Sampler


# === MHSampler ===
class MHSampler(Sampler):
    """
    The most basic possible Metropolis-Hastings style MCMC sampler

    :param cov:
        The covariance matrix to use for the proposal distribution.

    :param dim:
        Number of dimensions in the parameter space.

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
    def __init__(self, cov, *args, **kwargs):
        super(MHSampler, self).__init__(*args, **kwargs)
        self.cov = cov

    def reset(self):
        super(MHSampler, self).reset()
        self._chain = np.empty((0, self.dim))
        self._lnprob = np.empty(0)

    def sample(self, p0, lnprob=None, randomstate=None, thin=1,
               storechain=True, iterations=1):
        """
        Advances the chain ``iterations`` steps as an iterator

        :param p0:
            The initial position vector.

        :param lnprob0: (optional)
            The log posterior probability at position ``p0``. If ``lnprob``
            is not provided, the initial value is calculated.

        :param rstate0: (optional)
            The state of the random number generator. See the
            :func:`random_state` property for details.

        :param iterations: (optional)
            The number of steps to run.

        :param thin: (optional)
            If you only want to store and yield every ``thin`` samples in the
            chain, set thin to an integer greater than 1.

        :param storechain: (optional)
            By default, the sampler stores (in memory) the positions and
            log-probabilities of the samples in the chain. If you are
            using another method to store the samples to a file or if you
            don't need to analyse the samples after the fact (for burn-in
            for example) set ``storechain`` to ``False``.

        At each iteration, this generator yields:

        * ``pos`` - The current positions of the chain in the parameter
          space.

        * ``lnprob`` - The value of the log posterior at ``pos`` .

        * ``rstate`` - The current state of the random number generator.

        """

        self.random_state = randomstate

        p = np.array(p0)
        if lnprob is None:
            lnprob = self.get_lnprob(p)

        # Resize the chain in advance.
        if storechain:
            N = int(iterations / thin)
            self._chain = np.concatenate((self._chain,
                                          np.zeros((N, self.dim))), axis=0)
            self._lnprob = np.append(self._lnprob, np.zeros(N))

        i0 = self.iterations
        # Use range instead of xrange for python 3 compatability
        for i in range(int(iterations)):
            self.iterations += 1

            # Calculate the proposal distribution.
            q = self._random.multivariate_normal(p, self.cov)
            newlnprob = self.get_lnprob(q)
            diff = newlnprob - lnprob

            # M-H acceptance ratio
            if diff < 0:
                diff = np.exp(diff) - self._random.rand()

            if diff > 0:
                p = q
                lnprob = newlnprob
                self.naccepted += 1

            if storechain and i % thin == 0:
                ind = i0 + int(i / thin)
                self._chain[ind, :] = p
                self._lnprob[ind] = lnprob

            # Heavy duty iterator action going on right here...
            yield p, lnprob, self.random_state

    @property
    def acor(self):
        """
        An estimate of the autocorrelation time for each parameter (length:
        ``dim``).

        """
        return self.get_autocorr_time()

    def get_autocorr_time(self, low=10, high=None, step=1, c=10, fast=False):
        """
        Compute an estimate of the autocorrelation time for each parameter
        (length: ``dim``).

        :param low: (Optional[int])
            The minimum window size to test.
            (default: ``10``)
        :param high: (Optional[int])
            The maximum window size to test.
            (default: ``x.shape[axis] / (2*c)``)
        :param step: (Optional[int])
            The step size for the window search.
            (default: ``1``)
        :param c: (Optional[float])
            The minimum number of autocorrelation times needed to trust the
            estimate.
            (default: ``10``)
        :param fast: (Optional[bool])
            If ``True``, only use the first ``2^n`` (for the largest power)
            entries for efficiency.
            (default: False)
        """
        return autocorr.integrated_time(self.chain, axis=0, low=low,
                                        high=high, step=step, c=c, fast=fast)
