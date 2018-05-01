# -*- coding: utf-8 -*-

from __future__ import division, print_function

import warnings

import pytest
import numpy as np

from emcee import moves
from emcee.state import State

__all__ = ["test_live_dangerously"]


def test_live_dangerously(nwalkers=32, nsteps=3000, seed=1234):
    warnings.filterwarnings("error")

    # Set up the random number generator.
    np.random.seed(seed)
    state = State(np.random.randn(nwalkers, 2 * nwalkers),
                  log_prob=np.random.randn(nwalkers))
    proposal = moves.StretchMove()

    # Test to make sure that the error is thrown if there aren't enough
    # walkers.
    with pytest.raises(RuntimeError):
        proposal.propose(state, lambda x: (np.zeros(len(x)), None), None,
                         np.random)

    # Living dangerously...
    proposal.live_dangerously = True
    proposal.propose(state, lambda x: (np.zeros(len(x)), None), None,
                     np.random)
