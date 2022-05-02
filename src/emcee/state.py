# -*- coding: utf-8 -*-

from copy import deepcopy

import numpy as np

__all__ = ["State"]


class State(object):
    """The state of the ensemble during an MCMC run

    For backwards compatibility, this will unpack into ``coords, log_prob,
    (blobs), random_state`` when iterated over (where ``blobs`` will only be
    included if it exists and is not ``None``).

    Args:
        coords (ndarray[nwalkers, ndim]): The current positions of the walkers
            in the parameter space.
        log_prob (ndarray[nwalkers, ndim], Optional): Log posterior
            probabilities for the  walkers at positions given by ``coords``.
        blobs (Optional): The metadata “blobs” associated with the current
            position. The value is only returned if lnpostfn returns blobs too.
        random_state (Optional): The current state of the random number
            generator.
    """

    __slots__ = "coords", "log_prob", "blobs", "random_state"

    def __init__(
        self, coords, log_prob=None, blobs=None, random_state=None, copy=False
    ):
        dc = deepcopy if copy else lambda x: x

        if hasattr(coords, "coords"):
            self.coords = dc(coords.coords)
            self.log_prob = dc(coords.log_prob)
            self.blobs = dc(coords.blobs)
            self.random_state = dc(coords.random_state)
            return

        self.coords = dc(np.atleast_2d(coords))
        self.log_prob = dc(log_prob)
        self.blobs = dc(blobs)
        self.random_state = dc(random_state)

    def __len__(self):
        if self.blobs is None:
            return 3
        return 4

    def __repr__(self):
        return "State({0}, log_prob={1}, blobs={2}, random_state={3})".format(
            self.coords, self.log_prob, self.blobs, self.random_state
        )

    def __iter__(self):
        if self.blobs is None:
            return iter((self.coords, self.log_prob, self.random_state))
        return iter(
            (self.coords, self.log_prob, self.random_state, self.blobs)
        )

    def __getitem__(self, index):
        if index < 0:
            return self[len(self) + index]
        if index == 0:
            return self.coords
        elif index == 1:
            return self.log_prob
        elif index == 2:
            return self.random_state
        elif index == 3 and self.blobs is not None:
            return self.blobs
        raise IndexError("Invalid index '{0}'".format(index))
