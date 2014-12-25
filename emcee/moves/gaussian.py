# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["GaussianMove"]

import numpy as np

from .mh import MHMove


class GaussianMove(MHMove):
    """
    A Metropolis step with a Gaussian proposal function.

    :param cov:
        The covariance of the proposal function. This can be a scalar, vector,
        or matrix and the proposal will be assumed isotropic, axis-aligned, or
        general respectively.

    """
    def __init__(self, cov):
        # Parse the proposal type.
        try:
            float(cov)

        except TypeError:
            cov = np.atleast_1d(cov)
            if len(cov.shape) == 1:
                # A diagonal proposal was given.
                ndim = len(cov)
                proposal = _diagonal_proposal(np.sqrt(cov))

            elif len(cov.shape) == 2 and cov.shape[0] == cov.shape[1]:
                # The full, square covariance matrix was given.
                ndim = cov.shape[0]
                proposal = _proposal(cov)

            else:
                raise ValueError("Invalid proposal scale dimensions")

        else:
            # This was a scalar proposal.
            ndim = None
            proposal = _isotropic_proposal(np.sqrt(cov))

        super(GaussianMove, self).__init__(proposal, ndim=ndim)


class _isotropic_proposal(object):

    def __init__(self, scale):
        self.scale = scale

    def __call__(self, rng, x0):
        return x0 + self.scale * rng.randn(*(x0.shape)), 0.0


class _diagonal_proposal(object):

    def __init__(self, scale):
        self.scale = scale

    def __call__(self, rng, x0):
        return x0 + self.scale * rng.randn(*(x0.shape)), 0.0


class _proposal(object):

    def __init__(self, cov):
        self.cov = cov
        self.zero = np.zeros(len(cov))

    def __call__(self, rng, x0):
        return (x0+rng.multivariate_normal(self.zero, self.cov, size=len(x0)),
                0.0)
