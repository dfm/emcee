# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["test_live_dangerously"]

import numpy as np

from ... import moves, Ensemble
from ..common import NormalWalker


def test_live_dangerously(nwalkers=32, nsteps=3000, seed=1234):
    # Set up the random number generator.
    rnd = np.random.RandomState()
    rnd.seed(seed)
    coords = rnd.randn(nwalkers, 2 * nwalkers)
    ensemble = Ensemble(NormalWalker, coords, 1.0, random=rnd)
    proposal = moves.StretchMove()

    # Test to make sure that the error is thrown if there aren't enough
    # walkers.
    try:
        proposal.update(ensemble)
    except RuntimeError:
        pass
    else:
        assert False, "This should raise a RuntimeError"

    # Living dangerously...
    proposal.live_dangerously = True
    proposal.update(ensemble)
