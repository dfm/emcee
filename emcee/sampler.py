# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["Sampler"]

import traceback
import numpy as np


class Sampler(object):

    def __init__(self, lnprior_fn, lnlike_fn, proposal, args=[], kwargs={}):
        # Wrap the ln-priod and ln-likelihood functions for pickling.
        self.lnprob_fn = _lnprob_fn_wrapper(lnprior_fn, lnlike_fn, args,
                                            kwargs)

        # Save the proposal or list of proposals.
        try:
            len(proposal)
        except TypeError:
            self.schedule = [proposal]
        else:
            self.schedule = proposal

        # Initialize a clean sampler.
        self.reset()

    def reset(self):
        # The sampler starts at the zeroth step.
        self.step = 0

        # Initialize the metadata lists as None; they will be created before
        # the first step.
        self._chain = None
        self._lnprior = None
        self._lnlike = None
        self.naccepted = None

    @property
    def chain(self):
        return self._chain[:self.step]

    @property
    def lnprior(self):
        return self._lnprior[:self.step]

    @property
    def lnlike(self):
        return self._lnlike[:self.step]

    @property
    def lnprob(self):
        return self.lnprior + self.lnlike

    @property
    def acceptance_fraction(self):
        return self.naccepted / self.step

    def initialize_chain(self, initial_coords):
        # Parse the dimensions.
        initial_coords = np.atleast_2d(initial_coords)
        nwalkers, ndim = initial_coords.shape

        self.ensemble_size = nwalkers
        self._chain = np.empty((0, nwalkers, ndim))
        self._lnprior = np.empty((0, nwalkers))
        self._lnlike = np.empty((0, nwalkers))
        self.naccepted = np.zeros(nwalkers, dtype=int)

    def grow_chain(self, N):
        _, nwalkers, ndim = self._chain.shape
        self._chain = np.concatenate((self._chain[:self.step],
                                     np.empty((N, nwalkers, ndim))), axis=0)
        self._lnprior = np.concatenate((self._lnprior[:self.step],
                                        np.empty((N, nwalkers))), axis=0)
        self._lnlike = np.concatenate((self._lnlike[:self.step],
                                       np.empty((N, nwalkers))), axis=0)

    def sample(self, initial_coords, initial_lnprior=None, initial_lnlike=None,
               nstep=None, chunksize=128, mapper=map):
        # Wrap the lnprob function in an ensemble mapper.
        lnprob_fn = _ensemble_lnprob(self.lnprob_fn, mapper)

        # Compute the initial lnprobability.
        if initial_lnprior is None or initial_lnlike is None:
            initial_lnprior, initial_lnlike = lnprob_fn(initial_coords)

        # Sanity check the metadata lists and all of the dimensions.
        if self.step > 0:
            if (self._chain is None or
                    len(initial_coords) != self.ensemble_size):
                raise RuntimeError("The chain dimensions are incompatible "
                                   "with the initial coordinates")

        # Initialize the metadata containers.
        else:
            self.initialize_chain(initial_coords)

        # Resize the metadata objects.
        if nstep is not None:
            self.grow_chain(nstep)

        # Start sampling.
        initial = int(self.step)
        while True:
            # Grow the chain if needed.
            if self.step >= len(self._chain):
                self.grow_chain(chunksize)

            # Iterate over proposals.
            p = self.schedule[self.step % len(self.schedule)]

            # Run the update.
            acc = p.update(lnprob_fn, initial_coords, initial_lnprior,
                           initial_lnlike, self._chain[self.step],
                           self._lnprior[self.step],
                           self._lnlike[self.step])

            # Update the acceptance count.
            self.naccepted += acc

            yield (self._chain[self.step], self._lnprior[self.step],
                   self._lnlike[self.step])

            initial_coords = self._chain[self.step]
            initial_lnprior = self._lnprior[self.step]
            initial_lnlike = self._lnlike[self.step]

            # Break when the requested number of steps is reached.
            self.step += 1
            if nstep is not None and self.step - initial >= nstep:
                break


class _ensemble_lnprob(object):

    def __init__(self, lnprob, mapper):
        self.lnprob = lnprob
        self.mapper = mapper

    def __call__(self, xs):
        return map(np.array, zip(*(self.mapper(self.lnprob, xs))))


class _lnprob_fn_wrapper(object):

    def __init__(self, lnprior, lnlike, args, kwargs):
        self.lnprior = lnprior
        self.lnlike = lnlike
        self.args = args
        self.kwargs = kwargs

    def __call__(self, x):
        # Compute the lnprior.
        try:
            lp = self.lnprior(x, *self.args, **self.kwargs)
        except:
            print("emcee: Exception while calling your lnprior function:")
            print("  params:", x)
            print("  args:", self.args)
            print("  kwargs:", self.kwargs)
            print("  exception:")
            traceback.print_exc()
            raise

        # Check if the prior is finite and return otherwise.
        if not np.isfinite(lp):
            return -np.inf, -np.inf

        # Compute the lnlikelihood.
        try:
            ll = self.lnlike(x, *self.args, **self.kwargs)
        except:
            print("emcee: Exception while calling your lnlikelihood function:")
            print("  params:", x)
            print("  args:", self.args)
            print("  kwargs:", self.kwargs)
            print("  exception:")
            traceback.print_exc()
            raise

        # Always return -infinity instead of NaN.
        ll = ll if np.isfinite(ll) else -np.inf

        return lp, ll
