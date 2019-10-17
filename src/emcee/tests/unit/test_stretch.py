# -*- coding: utf-8 -*-

import warnings

import numpy as np
import pytest

from emcee import moves
from emcee.model import Model
from emcee.state import State

__all__ = ["test_live_dangerously"]


def test_live_dangerously(nwalkers=32, nsteps=3000, seed=1234):
    warnings.filterwarnings("error")

    # Set up the random number generator.
    np.random.seed(seed)
    state = State(
        np.random.randn(nwalkers, 2 * nwalkers),
        log_prob=np.random.randn(nwalkers),
    )
    model = Model(None, lambda x: (np.zeros(len(x)), None), map, np.random)
    proposal = moves.StretchMove()

    # Test to make sure that the error is thrown if there aren't enough
    # walkers.
    with pytest.raises(RuntimeError):
        proposal.propose(model, state)

    # Living dangerously...
    proposal.live_dangerously = True
    proposal.propose(model, state)
