# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["Sampler"]

import logging
from functools import wraps
from collections import Iterable
from .backends import DefaultBackend


def _check_run(f):
    @wraps(f)
    def func(self, *args, **kwargs):
        if self.backend.coords is None:
            raise AttributeError("You need to run the chain first")
        return f(self, *args, **kwargs)
    return func


class Sampler(object):

    def __init__(self, schedule, backend=None):
        # Save the schedule. This should be a list of proposals.
        if isinstance(schedule, Iterable):
            self.schedule = schedule
        else:
            self.schedule = [schedule]

        # Set up the backend.
        if backend is None:
            self.backend = DefaultBackend()

        # Set the chain to the original untouched state.
        self.reset()

    def reset(self):
        """
        Clear the chain and reset it to its default state.

        """
        self.backend.reset()

    def sample(self, ensemble, niter=None, store=None):
        """
        Starting from a given ensemble, start sampling as an iterator yielding
        each updated ensemble.

        :param ensemble:
            The starting :class:`Ensemble`.

        :param niter: (optional)
            The number of steps to run. If not given, the iterator will run
            forever.

        :param store: (optional)
            If ``True``, save the chain using the

        """
        # Set the default backend behavior if not overridden.
        if niter is not None:
            store = True if store is None else store
        else:
            store = False if store is None else store

        # Warn the user about trying to store the chain without setting the
        # number of iterations.
        if niter is None and store:
            logging.warn("Storing the chain without specifying the total "
                         "number of iterations is very inefficient")

        # Check the ensemble dimensions.
        if store:
            self.backend.check_dimensions(ensemble)
        else:
            self.backend.reset()

        # Extend the chain to the right length.
        if store and niter is not None:
            self.backend.extend(niter)

        # Start the generator.
        i = 0
        while True:
            for p in self.schedule:
                ensemble = p.update(ensemble)
                if store:
                    self.backend.update(ensemble)
                yield ensemble

                # Finish the chain if the total number of steps was reached.
                i += 1
                if niter is not None and i >= niter:
                    return

    @property
    @_check_run
    def coords(self):
        return self.backend.coords

    @property
    @_check_run
    def lnprior(self):
        return self.backend.lnprior

    @property
    @_check_run
    def lnlike(self):
        return self.backend.lnlike

    @property
    @_check_run
    def lnprob(self):
        return self.backend.lnprob

    @property
    @_check_run
    def walkers(self):
        return self.backend.walkers

    @property
    @_check_run
    def acceptance_fraction(self):
        return self.backend.acceptance_fraction
