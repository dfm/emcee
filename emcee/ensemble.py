#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
An affine invariant Markov chain Monte Carlo (MCMC) sampler.

Goodman & Weare, Ensemble Samplers With Affine Invariance
   Comm. App. Math. Comp. Sci., Vol. 5 (2010), No. 1, 65–80

"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["EnsembleSampler"]

import logging

import numpy as np

from . import autocorr
from .moves import StretchMove


class EnsembleSampler(object):
    """
    A generalized Ensemble sampler that uses 2 ensembles for parallelization.
    The ``__init__`` function will raise an ``AssertionError`` if
    ``nwalkers < 2 * dim`` (and you haven't set the ``live_dangerously`` parameter)
    or if ``nwalkers`` is odd.

    **Warning**: The :attr:`chain` member of this object has the shape:
    ``(nwalkers, nlinks, dim)`` where ``nlinks`` is the number of steps
    taken by the chain and ``nwalkers`` is the number of walkers.  Use the
    :attr:`flatchain` property to get the chain flattened to
    ``(nlinks, dim)``. For users of pre-1.0 versions, this shape is
    different so be careful!

    :param nwalkers:
        The number of Goodman & Weare "walkers".

    :param dim:
        Number of dimensions in the parameter space.

    :param lnpostfn:
        A function that takes a vector in the parameter space as input and
        returns the natural logarithm of the posterior probability for that
        position.

    :param a: (optional)
        The proposal scale parameter. (default: ``2.0``)

    :param args: (optional)
        A list of extra positional arguments for ``lnpostfn``. ``lnpostfn``
        will be called with the sequence ``lnpostfn(p, *args, **kwargs)``.

    :param kwargs: (optional)
        A list of extra keyword arguments for ``lnpostfn``. ``lnpostfn``
        will be called with the sequence ``lnpostfn(p, *args, **kwargs)``.

    :param postargs: (optional)
        Alias of ``args`` for backwards compatibility.

    :param threads: (deprecated; ignored)
        The number of threads to use for parallelization. If ``threads == 1``,
        then the ``multiprocessing`` module is not used but if
        ``threads > 1``, then a ``Pool`` object is created and calls to
        ``lnpostfn`` are run in parallel.

    :param pool: (optional)
        An alternative method of using the parallelized algorithm. If
        provided, the value of ``threads`` is ignored and the
        object provided by ``pool`` is used for all parallelization. It
        can be any object with a ``map`` method that follows the same
        calling sequence as the built-in ``map`` function.

    :param runtime_sortingfn: (deprecated; ignored)
        A function implementing custom runtime load-balancing. See
        :ref:`loadbalance` for more information.

    """
    def __init__(self, nwalkers, dim, lnpostfn, a=None,
                 pool=None, moves=None,
                 args=None, kwargs=None,
                 # Deprecated...
                 postargs=None, threads=None,  live_dangerously=None,
                 runtime_sortingfn=None):
        if threads is not None:
            logging.warn("the 'threads' argument is deprecated; "
                         "use 'pool' instead")

        if runtime_sortingfn is not None:
            logging.warn("the 'runtime_sortingfn' argument is deprecated")

        if moves is None:
            self._moves = [StretchMove()]
            self._weights = [1.0]
        elif isinstance(moves, Iterable):
            try:
                self._moves, self._weights = zip(*moves)
            except TypeError:
                self._moves = moves
                self._weights = np.ones(len(moves))
        else:
            self._moves = [moves]
            self._weights = [1.0]
        self._weights = np.atleast_1d(self._weights).astype(float)
        self._weights /= np.sum(self._weights)

        self.dim = dim
        self.k = nwalkers
        self.pool = pool

        # This is a random number generator that we can easily set the state
        # of without affecting the numpy-wide generator
        self._random = np.random.mtrand.RandomState()

        self.reset()

        # Do a little bit of _magic_ to make the likelihood call with
        # ``args`` and ``kwargs`` pickleable.
        self.lnprobfn = _function_wrapper(lnpostfn, args, kwargs)

        # assert self.k % 2 == 0, "The number of walkers must be even."
        # if not live_dangerously:
        #     assert self.k >= 2 * self.dim, (
        #         "The number of walkers needs to be more than twice the "
        #         "dimension of your parameter space unless you know what "
        #         "you're getting yourself into...")

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

    def clear_blobs(self):
        """
        Clear the ``blobs`` list.

        """
        self._blobs = []

    def reset(self):
        """
        Clear the ``chain`` and ``lnprobability`` array. Also reset the
        bookkeeping parameters.

        """
        self.iterations = 0
        self.naccepted = 0
        self._last_run_mcmc_result = None

        self.naccepted = np.zeros(self.k)
        self._chain = np.empty((self.k, 0, self.dim))
        self._lnprob = np.empty((self.k, 0))

        # Initialize list for storing optional metadata blobs.
        self.clear_blobs()

    def __getstate__(self):
        # In order to be generally picklable, we need to discard the pool
        # object before trying.
        d = self.__dict__
        d.pop("pool", None)
        return d

    # def __setstate__(self, state):
    #     self.__dict__ = state

    def sample(self, p0, lnprob0=None, rstate0=None, blobs0=None,
               iterations=1, thin=1, storechain=True, mh_proposal=None):
        """
        Advance the chain ``iterations`` steps as a generator.

        :param p0:
            A list of the initial positions of the walkers in the
            parameter space. It should have the shape ``(nwalkers, dim)``.

        :param lnprob0: (optional)
            The list of log posterior probabilities for the walkers at
            positions given by ``p0``. If ``lnprob is None``, the initial
            values are calculated. It should have the shape ``(k, dim)``.

        :param rstate0: (optional)
            The state of the random number generator.
            See the :attr:`Sampler.random_state` property for details.

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

        :param mh_proposal: (optional)
            A function that returns a list of positions for ``nwalkers``
            walkers given a current list of positions of the same size. See
            :class:`utils.MH_proposal_axisaligned` for an example.

        At each iteration, this generator yields:

        * ``pos`` - A list of the current positions of the walkers in the
          parameter space. The shape of this object will be
          ``(nwalkers, dim)``.

        * ``lnprob`` - The list of log posterior probabilities for the
          walkers at positions given by ``pos`` . The shape of this object
          is ``(nwalkers,)``.

        * ``rstate`` - The current state of the random number generator.

        * ``blobs`` - (optional) The metadata "blobs" associated with the
          current position. The value is only returned if ``lnpostfn``
          returns blobs too.

        """
        # Try to set the initial value of the random number generator. This
        # fails silently if it doesn't work but that's what we want because
        # we'll just interpret any garbage as letting the generator stay in
        # it's current state.
        self.random_state = rstate0

        p = np.array(p0)
        halfk = int(self.k / 2)

        # If the initial log-probabilities were not provided, calculate them
        # now.
        lnprob = lnprob0
        blobs = blobs0
        if lnprob is None:
            lnprob, blobs = self.compute_log_prob(p)
            # lnprob, blobs = self._get_lnprob(p)

        # Check to make sure that the probability function didn't return
        # ``np.nan``.
        if np.any(np.isnan(lnprob)):
            raise ValueError("The initial lnprob was NaN.")

        # Store the initial size of the stored chain.
        initial_size = self._chain.shape[1]

        # Check that the thin keyword is reasonable.
        thin = int(thin)
        if thin <= 0:
            raise ValueError("Invalid thinning argument")

        # Here, we resize chain in advance for performance. This actually
        # makes a pretty big difference.
        if storechain:
            N = iterations // thin
            self._chain = np.concatenate((self._chain,
                                          np.zeros((self.k, N, self.dim))),
                                         axis=1)
            self._lnprob = np.concatenate((self._lnprob,
                                           np.zeros((self.k, N))), axis=1)

        for i in range(int(iterations)):
            self.iterations += 1

            # Choose a random move
            move = self._random.choice(self._moves, p=self._weights)

            # Propose
            p, lnprob, blobs, accepted = move.propose(
                p, lnprob, blobs, self.compute_log_prob, self._random)
            self.naccepted += accepted

            # Save the results
            if storechain and (i + 1) % thin == 0:
                ind = initial_size + (i + 1) // thin - 1
                self._chain[:, ind, :] = p
                self._lnprob[:, ind] = lnprob
                if blobs is not None:
                    self._blobs.append(list(blobs))

            # Yield the result as an iterator so that the user can do all
            # sorts of fun stuff with the results so far.
            if blobs is not None:
                # This is a bit of a hack to keep things backwards compatible.
                yield p, lnprob, self.random_state, blobs
            else:
                yield p, lnprob, self.random_state

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

        This method returns the most recent result from :func:`sample`. The
        particular values vary from sampler to sampler, but the result is
        generally a tuple ``(pos, lnprob, rstate)`` or ``(pos, lnprob, rstate,
        blobs)`` where ``pos`` is the most recent position
        vector (or ensemble thereof), ``lnprob`` is the most recent
        log posterior probability (or ensemble thereof), ``rstate`` is the
        state of the random number generator, and the optional ``blobs`` are
        user-provided large data blobs.
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

        for results in self.sample(pos0, lnprob0, rstate0=rstate0, iterations=N,
                                   **kwargs):
            pass

        # Store so that the ``pos0=None`` case will work.  We throw out the
        # blob if it's there because we don't need it
        self._last_run_mcmc_result = results[:3]

        return results

    def compute_log_prob(self, coords=None):
        """
        Calculate the vector of log-probability for the walkers.

        :param pos: (optional)
            The position vector in parameter space where the probability
            should be calculated. This defaults to the current position
            unless a different one is provided.

        This method returns:

        * ``lnprob`` - A vector of log-probabilities with one entry for each
          walker in this sub-ensemble.

        * ``blob`` - The list of meta data returned by the ``lnpostfn`` at
          this position or ``None`` if nothing was returned.

        """
        if coords is None:
            p = self.pos
        else:
            p = coords

        # Check that the parameters are in physical ranges.
        if np.any(np.isinf(p)):
            raise ValueError("At least one parameter value was infinite.")
        if np.any(np.isnan(p)):
            raise ValueError("At least one parameter value was NaN.")

        # If the `pool` property of the sampler has been set (i.e. we want
        # to use `multiprocessing`), use the `pool`'s map method. Otherwise,
        # just use the built-in `map` function.
        if self.pool is not None:
            M = self.pool.map
        else:
            M = map

        # Run the log-probability calculations (optionally in parallel).
        results = list(M(self.lnprobfn, [p[i] for i in range(len(p))]))

        try:
            lnprob = np.array([float(l[0]) for l in results])
            blob = [l[1] for l in results]
        except (IndexError, TypeError):
            lnprob = np.array([float(l) for l in results])
            blob = None

        # Check for lnprob returning NaN.
        if np.any(np.isnan(lnprob)):
            # Print some debugging stuff.
            print("NaN value of lnprob for parameters: ")
            for pars in p[np.isnan(lnprob)]:
                print(pars)

            # Finally raise exception.
            raise ValueError("lnprob returned NaN.")

        return lnprob, blob

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
    def blobs(self):
        """
        Get the list of "blobs" produced by sampling. The result is a list
        (of length ``iterations``) of ``list`` s (of length ``nwalkers``) of
        arbitrary objects. **Note**: this will actually be an empty list if
        your ``lnpostfn`` doesn't return any metadata.

        """
        return self._blobs

    @property
    def flatchain(self):
        """
        A shortcut for accessing chain flattened along the zeroth (walker)
        axis.

        """
        s = self.chain.shape
        return self.chain.reshape(s[0] * s[1], s[2])

    @property
    def lnprobability(self):
        """
        A list of the log-probability values associated with each step in
        the chain.

        """
        return self._lnprob

    @property
    def flatlnprobability(self):
        """
        A shortcut to return the equivalent of ``lnprobability`` but aligned
        to ``flatchain`` rather than ``chain``. The shape is
        ``(k * iterations)``.

        """
        return self.lnprobability.flatten()

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
        return autocorr.integrated_time(np.mean(self.chain, axis=0), axis=0,
                                        low=low, high=high, step=step, c=c,
                                        fast=fast)


class _function_wrapper(object):
    """
    This is a hack to make the likelihood function pickleable when ``args``
    or ``kwargs`` are also included.

    """
    def __init__(self, f, args, kwargs):
        self.f = f
        self.args = [] if args is None else args
        self.kwargs = {} if kwargs is None else kwargs

    def __call__(self, x):
        try:
            return self.f(x, *self.args, **self.kwargs)
        except:
            import traceback
            print("emcee: Exception while calling your likelihood function:")
            print("  params:", x)
            print("  args:", self.args)
            print("  kwargs:", self.kwargs)
            print("  exception:")
            traceback.print_exc()
            raise
