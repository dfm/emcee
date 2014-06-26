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

    def update(self, ens_lnprob_fn, coords_in, lnprior_in, lnlike_in,
               coords_out, lnprior_out, lnlike_out):
        # Parse the dimensions of the input coordinates.
        coords_in = np.atleast_2d(coords_in)
        nens, ndim = coords_in.shape
        if ndim != len(self.cov):
            raise ValueError("Dimension mismatch between coordinates and "
                             "proposal")

        # Generate a proposal coordinate.
        q = self.random.multivariate_normal(self.zero, self.cov, size=nens)
        q += coords_in

        # Compute the lnprior and lnlikelihood at the new positions.
        lnprior, lnlike = ens_lnprob_fn(q)

        # Compute the acceptance probability and accept or reject on that
        # basis.
        acc_lp = (lnprior + lnlike) - (lnprior_in + lnlike_in)
        nacc = acc_lp < 0.0
        nacc[nacc] = self.random.rand(sum(nacc)) >= np.exp(acc_lp[nacc])

        # Update the output coordinates.
        q[nacc] = coords_in[nacc, :]
        lnprior[nacc] = lnprior_in[nacc]
        lnlike[nacc] = lnlike_in[nacc]

        # Save the output coordinates in place.
        coords_out[:] = q
        lnprior_out[:] = lnprior
        lnlike_out[:] = lnlike

        return ~nacc
