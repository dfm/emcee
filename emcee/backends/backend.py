# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["Backend"]

import numpy as np

from .. import autocorr


class Backend(object):

    def __init__(self):
        self.initialized = False

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

    def get_chain(self, **kwargs):
        """Get the stored chain of MCMC samples

        Args:
            flat (Optional[bool]): Flatten the chain across the ensemble.
                (default: ``False``)
            thin (Optional[int]): Take only every ``thin`` steps from the
                chain. (default: ``1``)
            discard (Optional[int]): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)

        Returns:
            array[..., nwalkers, ndim]: The MCMC samples.

        """
        return self.get_value("chain", **kwargs)

    def get_blobs(self, **kwargs):
        """Get the chain of blobs for each sample in the chain

        Args:
            flat (Optional[bool]): Flatten the chain across the ensemble.
                (default: ``False``)
            thin (Optional[int]): Take only every ``thin`` steps from the
                chain. (default: ``1``)
            discard (Optional[int]): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)

        Returns:
            array[..., nwalkers]: The chain of blobs.

        """
        return self.get_value("blobs", **kwargs)

    def get_log_prob(self, **kwargs):
        """Get the chain of log probabilities evaluated at the MCMC samples

        Args:
            flat (Optional[bool]): Flatten the chain across the ensemble.
                (default: ``False``)
            thin (Optional[int]): Take only every ``thin`` steps from the
                chain. (default: ``1``)
            discard (Optional[int]): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)

        Returns:
            array[..., nwalkers]: The chain of log probabilities.

        """
        return self.get_value("log_prob", **kwargs)

    def get_autocorr_time(self, discard=0, thin=1, **kwargs):
        """Compute an estimate of the autocorrelation time for each parameter

        Args:
            thin (Optional[int]): Use only every ``thin`` steps from the
                chain. The returned estimate is multiplied by ``thin`` so the
                estimated time is in units of steps, not thinned steps.
                (default: ``1``)
            discard (Optional[int]): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)

        Other arguments are passed directly to
        :func:`emcee.autocorr.integrated_time`.

        Returns:
            array[ndim]: The integrated autocorrelation time estimate for the
                chain for each parameter.

        """
        x = self.get_chain(discard=discard, thin=thin)
        return thin * autocorr.integrated_time(x, **kwargs)

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
