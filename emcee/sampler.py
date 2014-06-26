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
        self.chain = None
        self.lnprior = None
        self.lnlike = None
        self.naccepted = None

    def sample(self, initial_coords, initial_lnprior=None, initial_lnlike=None,
               nstep=None, chunksize=128, mapper=map):
        # Parse the dimensions.
        initial_coords = np.atleast_2d(initial_coords)
        nwalkers, ndim = initial_coords.shape

        # Wrap the lnprob function in an ensemble mapper.
        lnprob_fn = _ensemble_lnprob(self.lnprob_fn, mapper)

        # Compute the initial lnprobability.
        if initial_lnprior is None or initial_lnlike is None:
            initial_lnprior, initial_lnlike = lnprob_fn(initial_coords)

        # Sanity check the metadata lists and all of the dimensions.
        if self.step > 0:
            if self.chain is None or nwalkers != self.ensemble_size:
                raise RuntimeError("The chain dimensions are incompatible "
                                   "with the initial coordinates")

        # Initialize the metadata containers.
        else:
            self.chain = np.empty((0, nwalkers, ndim))
            self.lnprior = np.empty((0, nwalkers))
            self.lnlike = np.empty((0, nwalkers))
            self.naccepted = np.zeros(nwalkers, dtype=int)

        # Resize the metadata objects.
        assert nstep is not None
        self.chain = np.concatenate((self.chain[:self.step],
                                    np.empty((nstep, nwalkers, ndim))), axis=0)
        self.lnprior = np.concatenate((self.lnprior[:self.step],
                                       np.empty((nstep, nwalkers))), axis=0)
        self.lnlike = np.concatenate((self.lnlike[:self.step],
                                      np.empty((nstep, nwalkers))), axis=0)

        # Start sampling.
        initial = int(self.step)
        while True:
            p = self.schedule[self.step % len(self.schedule)]

            acc = p.update(lnprob_fn, initial_coords, initial_lnprior,
                           initial_lnlike, self.chain[self.step],
                           self.lnprior[self.step],
                           self.lnlike[self.step])
            self.naccepted += acc

            yield self.chain[self.step]

            initial_coords = self.chain[self.step]
            initial_lnprior = self.lnprior[self.step]
            initial_lnlike = self.lnlike[self.step]

            # Break when the requested number of steps is reached.
            self.step += 1
            if self.step - initial >= nstep:
                break

        print(self.naccepted / self.step)


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
