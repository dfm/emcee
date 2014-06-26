# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["Sampler"]

import traceback
import numpy as np

from . import autocorr
from .state import State


class Sampler(object):

    def __init__(self, proposal):
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
        self.naccepted = None

    @property
    def chain(self):
        return self._chain[:self.step]

    @property
    def acceptance_fraction(self):
        return self.naccepted / self.step

    def get_state(self, base, step=None):
        if step is None:
            step = self.step

        # Get the current state or create a new one.
        state = self._chain[step]
        if state is None:
            state = self._chain[step] = base.copy()

        return state

    def initialize_chain(self, initial_coords):
        nwalkers = len(initial_coords)
        self.ensemble_size = nwalkers
        self._chain = np.empty(0, dtype=object)
        self.naccepted = np.zeros(nwalkers, dtype=int)

    def grow_chain(self, N):
        self._chain = np.concatenate((self._chain[:self.step],
                                     np.empty(N, dtype=object)), axis=0)

    def sample(self, initial_state, nstep=None, chunksize=128):
        if self.step <= 0:
            self.initialize_chain(initial_state)

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
            out_state = self.get_state(initial_state)
            acc = p.update(initial_state, out_state)

            # Update the acceptance count.
            self.step += 1
            self.naccepted += acc
            initial_state = out_state

            yield initial_state

            # Break when the requested number of steps is reached.
            if nstep is not None and self.step - initial >= nstep:
                break


class SimpleSampler(Sampler):

    def __init__(self, lnprior_fn, lnlike_fn, proposal, args=[], kwargs={}):
        # Wrap the ln-priod and ln-likelihood functions for pickling.
        self.lnprob_fn = _lnprob_fn_wrapper(lnprior_fn, lnlike_fn, args,
                                            kwargs)
        super(SimpleSampler, self).__init__(proposal)

    @property
    def lnprior(self):
        return self._lnprior[:self.step]

    @property
    def lnlike(self):
        return self._lnlike[:self.step]

    @property
    def lnprob(self):
        return self.lnprior + self.lnlike

    def get_autocorr_time(self, window=50, fast=False):
        return autocorr.integrated_time(np.mean(self.chain, axis=1), axis=0,
                                        window=window, fast=fast)

    def get_state(self, base, step=None):
        if step is None:
            step = self.step
        return State(base.lnprob_fn, self._chain[step], self._lnprior[step],
                     self._lnlike[step])

    def initialize_chain(self, state):
        # Parse the dimensions.
        initial_coords = np.atleast_2d(state.coords)
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
               mapper=map, **kwargs):
        # Wrap the lnprob function in an ensemble mapper.
        lnprob_fn = _ensemble_lnprob(self.lnprob_fn, mapper)

        # Set up the initial state.
        initial_state = State(lnprob_fn, initial_coords, initial_lnprior,
                              initial_lnlike)

        # Compute the initial lnprobability.
        if initial_lnprior is None or initial_lnlike is None:
            initial_state.compute_lnprob()

        return super(SimpleSampler, self).sample(initial_state, **kwargs)


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
