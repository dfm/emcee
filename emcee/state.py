# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["State"]

import numpy as np


class State(object):
    """An ensemble MCMC sampler

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

    def __init__(self, coords, log_prob=None, blobs=None, random_state=None):
        if hasattr(coords, "coords"):
            self.coords = coords.coords
            self.log_prob = coords.log_prob
            self.blobs = coords.blobs
            self.random_state = coords.random_state
            return

        self.coords = np.atleast_2d(coords)
        self.log_prob = log_prob
        self.blobs = blobs
        self.random_state = random_state

    def __repr__(self):
        return "State({0}, log_prob={1}, blobs={2}, random_state={3})".format(
            self.coords, self.log_prob, self.blobs, self.random_state
        )

    def __iter__(self):
        if self.blobs is None:
            return iter((self.coords, self.log_prob, self.random_state))
        return iter((self.coords, self.log_prob, self.random_state,
                     self.blobs))
