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


def _propose_with_loop(move, model, state):
    # Reference implementation: the original per-walker accept/reject loop.
    nwalkers, ndim = state.coords.shape
    move.setup(state.coords)
    accepted = np.zeros(nwalkers, dtype=bool)
    all_inds = np.arange(nwalkers)
    inds = all_inds % move.nsplits
    if move.randomize_split:
        model.random.shuffle(inds)
    for split in range(move.nsplits):
        S1 = inds == split
        sets = [state.coords[inds == j] for j in range(move.nsplits)]
        q, factors = move.get_proposal(
            sets[split], sets[:split] + sets[split + 1 :], model.random
        )
        new_log_probs, new_blobs = model.compute_log_prob_fn(q)
        for j, f, nlp in zip(all_inds[S1], factors, new_log_probs):
            if f + nlp - state.log_prob[j] > np.log(model.random.rand()):
                accepted[j] = True
        new_state = State(q, log_prob=new_log_probs, blobs=new_blobs)
        state = move.update(state, new_state, accepted, S1)
    return state, accepted


@pytest.mark.parametrize("nwalkers", [32, 33])
@pytest.mark.parametrize("nsplits", [2, 3])
def test_vectorized_accept_matches_loop(nwalkers, nsplits, ndim=5):
    def log_prob(x):
        lp = -0.5 * np.sum(x**2, axis=1)
        lp[np.abs(x[:, 0]) > 1.5] = -np.inf
        return lp, None

    rng = np.random.RandomState(42)
    coords = rng.uniform(-1, 1, size=(nwalkers, ndim))
    states = [
        State(coords.copy(), log_prob=log_prob(coords)[0]) for _ in range(2)
    ]
    rngs = [np.random.RandomState(1234) for _ in range(2)]
    move = moves.StretchMove(nsplits=nsplits)

    for _ in range(50):
        new, acc = move.propose(Model(None, log_prob, map, rngs[0]), states[0])
        ref, acc_ref = _propose_with_loop(
            move, Model(None, log_prob, map, rngs[1]), states[1]
        )
        np.testing.assert_array_equal(acc, acc_ref)
        np.testing.assert_array_equal(new.coords, ref.coords)
        np.testing.assert_array_equal(new.log_prob, ref.log_prob)
        assert rngs[0].get_state()[2] == rngs[1].get_state()[2]
        np.testing.assert_array_equal(
            rngs[0].get_state()[1], rngs[1].get_state()[1]
        )
        states = [new, ref]
