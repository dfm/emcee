# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["GaussianProposal"]

import numpy as np


class GaussianProposal(object):

    def __init__(self, cov):
        # Parse the proposal type.
        try:
            float(cov)

        except TypeError:
            cov = np.atleast_1d(cov)
            if len(cov.shape) == 1:
                # A diagonal proposal was given.
                self.ndim = len(cov)
                self.proposal = _diagonal_proposal(np.sqrt(cov))

            elif len(cov.shape) == 2 and cov.shape[0] == cov.shape[1]:
                # The full, square covariance matrix was given.
                self.ndim = cov.shape[0]
                self.proposal = _proposal(cov)

            else:
                raise ValueError("Invalid proposal scale dimensions")

        else:
            # This was a scalar proposal.
            self.ndim = None
            self.proposal = _isotropic_proposal(np.sqrt(cov))

    def update(self, ensemble):
        # Check to make sure that the dimensions match.
        ndim = ensemble.ndim
        if self.ndim is not None and self.ndim != ndim:
            raise ValueError("Dimension mismatch in proposal")

        # Compute the proposal.
        q = self.proposal(ensemble.random, ensemble.coords)
        new_walkers = ensemble.propose(q)

        # Loop over the walkers and update them accordingly.
        ensemble.acceptance[:] = False
        for i, w in enumerate(new_walkers):
            lnpdiff = w.lnprob - ensemble.walkers[i].lnprob
            if lnpdiff > 0.0 or ensemble.random.rand() < np.exp(lnpdiff):
                ensemble.walkers[i] = w
                ensemble.acceptance[i] = True

        # Update the ensemble's coordinates and log-probabilities.
        ensemble.update()
        return ensemble


class _isotropic_proposal(object):

    def __init__(self, scale):
        self.scale = scale

    def __call__(self, rng, x0):
        return x0 + self.scale * rng.randn(*(x0.shape))


class _diagonal_proposal(object):

    def __init__(self, scale):
        self.scale = scale

    def __call__(self, rng, x0):
        return x0 + self.scale * rng.randn(*(x0.shape))


class _proposal(object):

    def __init__(self, cov):
        self.cov = cov
        self.zero = np.zeros(len(cov))

    def __call__(self, rng, x0):
        return x0 + rng.multivariate_normal(self.zero, self.cov, size=len(x0))
