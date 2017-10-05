# -*- coding: utf-8 -*-

from __future__ import division, print_function

import pytest
import numpy as np

from emcee import moves

__all__ = ["test_live_dangerously"]


def test_live_dangerously(nwalkers=32, nsteps=3000, seed=1234):
    # Set up the random number generator.
    np.random.seed(seed)
    coords = np.random.randn(nwalkers, 2 * nwalkers)
    proposal = moves.StretchMove()

    # Test to make sure that the error is thrown if there aren't enough
    # walkers.
    with pytest.raises(RuntimeError):
        proposal.propose(coords, np.random.randn(nwalkers), None,
                         lambda x: (np.zeros(nwalkers), None), np.random)

    # Living dangerously...
    proposal.live_dangerously = True
    proposal.propose(coords, np.random.randn(nwalkers), None,
                     lambda x: (np.zeros(nwalkers), None), np.random)
