# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["Backend"]

import numpy as np


class Backend(object):

    def reset(self, nwalkers, ndim):
        self.nwalkers = nwalkers
        self.ndim = ndim
        self.iteration = 0
        self.accepted = np.zeros(self.nwalkers, dtype=int)
        self.chain = np.empty((0, self.nwalkers, self.ndim))
        self.log_prob = np.empty((0, self.nwalkers))
        self.blobs = None
        self.random_state = None

    def has_blobs(self):
        return self.blobs is not None

    def get_value(self, name, flat=False, thin=1, discard=0):
        if self.iteration <= 0:
            raise AttributeError("you must run the sampler with "
                                 "'store == True' before accessing the "
                                 "results")

        if name == "blobs" and not self.has_blobs():
            return None

        v = getattr(self, name)[discard+thin-1:self.iteration:thin]
        if flat:
            s = list(v.shape[1:])
            s[0] = np.prod(v.shape[:2])
            return v.reshape(s)
        return v

    @property
    def shape(self):
        return self.nwalkers, self.ndim

    def _check_blobs(self, blobs):
        has_blobs = self.has_blobs()
        if has_blobs and blobs is None:
            raise ValueError("inconsistent use of blobs")
        if self.iteration > 0 and blobs is not None and not has_blobs:
            raise ValueError("inconsistent use of blobs")

    def grow(self, N, blobs):
        """Expand the storage space by ``N``"""
        self._check_blobs(blobs)
        a = np.empty((N, self.nwalkers, self.ndim))
        self.chain = np.concatenate((self.chain, a), axis=0)
        a = np.empty((N, self.nwalkers))
        self.log_prob = np.concatenate((self.log_prob, a), axis=0)
        if blobs is not None:
            dt = np.dtype((blobs[0].dtype, blobs[0].shape))
            a = np.empty((N, self.nwalkers), dtype=dt)
            if self.blobs is None:
                self.blobs = a
            else:
                self.blobs = np.concatenate((self.blobs, a), axis=0)

    def _check(self, coords, log_prob, blobs, accepted):
        """Check the dimensions of a proposed state"""
        self._check_blobs(blobs)
        nwalkers, ndim = self.shape
        has_blobs = self.has_blobs()
        if coords.shape != (nwalkers, ndim):
            raise ValueError("invalid coordinate dimensions; expected {0}"
                             .format((nwalkers, ndim)))
        if log_prob.shape != (nwalkers, ):
            raise ValueError("invalid log probability size; expected {0}"
                             .format(nwalkers))
        if blobs is not None and not has_blobs:
            raise ValueError("unexpected blobs")
        if blobs is None and has_blobs:
            raise ValueError("expected blobs, but none were given")
        if blobs is not None and len(blobs) != nwalkers:
            raise ValueError("invalid blobs size; expected {0}"
                             .format(nwalkers))
        if accepted.shape != (nwalkers, ):
            raise ValueError("invalid acceptance size; expected {0}"
                             .format(nwalkers))

    def save_step(self, coords, log_prob, blobs, accepted, random_state):
        """Save a step to the backend"""
        self._check(coords, log_prob, blobs, accepted)

        self.chain[self.iteration, :, :] = coords
        self.log_prob[self.iteration, :] = log_prob
        if blobs is not None:
            self.blobs[self.iteration, :] = blobs
        self.accepted += accepted
        self.random_state = random_state
        self.iteration += 1

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        pass
