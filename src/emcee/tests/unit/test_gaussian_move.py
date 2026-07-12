# -*- coding: utf-8 -*-

import numpy as np

from emcee.moves import GaussianMove


def test_full_covariance_proposal_is_independent_per_walker():
    # With a full (non-diagonal-only-shaped) covariance matrix, each
    # walker must receive its own independent proposal step. Before the
    # fix, rng.multivariate_normal was called without `size`, so a
    # single vector was drawn and broadcast identically to every
    # walker.
    ndim = 2
    nwalkers = 8
    cov = np.array([[1.0, 0.3], [0.3, 1.0]])
    move = GaussianMove(cov, mode="vector")

    rng = np.random.RandomState(0)
    x0 = np.random.RandomState(1).randn(nwalkers, ndim)

    xnew, factors = move.get_proposal(x0, rng)

    assert xnew.shape == x0.shape
    deltas = xnew - x0
    # Not every walker's step should be identical.
    assert not np.allclose(deltas, deltas[0])
