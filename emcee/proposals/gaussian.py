# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["GaussianProposal"]

import numpy as np

from .base import Proposal


class GaussianProposal(Proposal):

    def __init__(self, ndim, cov, **kwargs):
        # Parse the various allowed forms for the covariance matrix.
        try:
            l = len(cov)
        except TypeError:
            # A scalar was given.
            cov = np.diag(float(cov) * np.ones(ndim))
        else:
            cov = np.atleast_1d(cov)

            # The diagonal was given.
            if len(cov.shape) == 1 and l == ndim:
                cov = np.diag(cov)

            # The full matrix was given.
            elif cov.shape == (ndim, ndim):
                cov = np.array(cov)

            # This is an invalid matrix.
            else:
                raise ValueError("Proposal covariance matrix "
                                 "dimension mismatch")

        # The matrix must be symmetric.
        i = np.tril_indices_from(cov)
        if np.any(cov[i] - cov.T[i] > np.finfo(float).eps):
            raise ValueError("A symmetric proposal matrix is required")

        # Save the matrix for later.
        self.cov = cov
        self.zero = np.zeros(ndim)

        # Call the superclass initialization.
        super(GaussianProposal, self).__init__(**kwargs)

    def update(self, state_in, state_out):
        # Generate a proposal coordinate.
        q = self.random.multivariate_normal(self.zero, self.cov, len(state_in))
        state_out.update(state_in + q)

        # Compute the lnprior and lnlikelihood at the new positions.
        state_out.compute_lnprob()

        # Compute the acceptance probability and accept or reject on that
        # basis.
        acc_lp = state_out.lnprob - state_in.lnprob
        nacc = acc_lp < 0.0
        nacc[nacc] = self.random.rand(sum(nacc)) >= np.exp(acc_lp[nacc])

        # Update the output coordinates.
        state_out.update(state_in, nacc)

        return ~nacc
