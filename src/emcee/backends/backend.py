# -*- coding: utf-8 -*-

import numpy as np

from .. import autocorr
from ..state import State
from .base import BackendBase

__all__ = ["Backend"]


class Backend(BackendBase):
    """A simple default backend that stores the chain in memory"""

    def __init__(self, dtype=None):
        self._initialized = False
        if dtype is None:
            dtype = np.float
        self.dtype = dtype

    def reset(self, nwalkers, ndim):
        """Clear the state of the chain and empty the backend

        Args:
            nwakers (int): The size of the ensemble
            ndim (int): The number of dimensions

        """
        self.nwalkers = int(nwalkers)
        self.ndim = int(ndim)
        self._iteration = 0
        self.accepted = np.zeros(self.nwalkers, dtype=self.dtype)
        self.chain = np.empty((0, self.nwalkers, self.ndim), dtype=self.dtype)
        self.log_prob = np.empty((0, self.nwalkers), dtype=self.dtype)
        self.blobs = None
        self._random_state = None
        self._initialized = True

    def has_blobs(self):
        """Returns ``True`` if the model includes blobs"""
        return self.blobs is not None

    @property
    def iteration(self):
        """Return the iteration number."""
        return self._iteration

    @property
    def initialized(self):
        """Return true if backend has been initialized."""
        return self._initialized

    @property
    def shape(self):
        """The dimensions of the ensemble ``(nwalkers, ndim)``"""
        return self.nwalkers, self.ndim

    def _get_value(self, name, flat, thin, discard):
        if name == "blobs" and not self.has_blobs():
            return None

        v = getattr(self, name)[discard + thin - 1 : self.iteration : thin]
        if flat:
            s = list(v.shape[1:])
            s[0] = np.prod(v.shape[:2])
            return v.reshape(s)
        return v

    def grow(self, ngrow, blobs):
        """Expand the storage space by some number of samples

        Args:
            ngrow (int): The number of steps to grow the chain.
            blobs: The current list of blobs. This is used to compute the
                dtype for the blobs array.

        """
        self._check_blobs(blobs)
        i = ngrow - (len(self.chain) - self.iteration)
        a = np.empty((i, self.nwalkers, self.ndim), dtype=self.dtype)
        self.chain = np.concatenate((self.chain, a), axis=0)
        a = np.empty((i, self.nwalkers), dtype=self.dtype)
        self.log_prob = np.concatenate((self.log_prob, a), axis=0)
        if blobs is not None:
            dt = np.dtype((blobs[0].dtype, blobs[0].shape))
            a = np.empty((i, self.nwalkers), dtype=dt)
            if self.blobs is None:
                self.blobs = a
            else:
                self.blobs = np.concatenate((self.blobs, a), axis=0)

    def save_step(self, state, accepted):
        """Save a step to the backend

        Args:
            state (State): The :class:`State` of the ensemble.
            accepted (ndarray): An array of boolean flags indicating whether
                or not the proposal for each walker was accepted.

        """
        self._check(state, accepted)

        self.chain[self.iteration, :, :] = state.coords
        self.log_prob[self.iteration, :] = state.log_prob
        if state.blobs is not None:
            self.blobs[self.iteration, :] = state.blobs
        self.accepted += accepted
        self._random_state = state.random_state
        self._iteration += 1

    @property
    def random_state(self):
        """Return the random state."""
        return self._random_state
