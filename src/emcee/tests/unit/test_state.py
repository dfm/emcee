# -*- coding: utf-8 -*-

import numpy as np
import pytest

from emcee import EnsembleSampler
from emcee.state import State


def check_rstate(a, b):
    assert all(np.allclose(a_, b_) for a_, b_ in zip(a[1:], b[1:]))


def test_back_compat(seed=1234):
    np.random.seed(seed)
    coords = np.random.randn(16, 3)
    log_prob = np.random.randn(len(coords))
    blobs = np.random.randn(len(coords))
    rstate = np.random.get_state()

    state = State(coords, log_prob, blobs, rstate)
    c, l, r, b = state
    assert np.allclose(coords, c)
    assert np.allclose(log_prob, l)
    assert np.allclose(blobs, b)
    check_rstate(rstate, r)

    state = State(coords, log_prob, None, rstate)
    c, l, r = state
    assert np.allclose(coords, c)
    assert np.allclose(log_prob, l)
    check_rstate(rstate, r)


def test_overwrite(seed=1234):
    np.random.seed(seed)

    def ll(x):
        return -0.5 * np.sum(x**2)

    nwalkers = 64
    p0 = np.random.normal(size=(nwalkers, 1))
    init = np.copy(p0)

    sampler = EnsembleSampler(nwalkers, 1, ll)
    sampler.run_mcmc(p0, 10)
    assert np.allclose(init, p0)


def test_indexing(seed=1234):
    np.random.seed(seed)
    coords = np.random.randn(16, 3)
    log_prob = np.random.randn(len(coords))
    blobs = np.random.randn(len(coords))
    rstate = np.random.get_state()

    state = State(coords, log_prob, blobs, rstate)
    np.testing.assert_allclose(state[0], state.coords)
    np.testing.assert_allclose(state[1], state.log_prob)
    check_rstate(state[2], state.random_state)
    np.testing.assert_allclose(state[3], state.blobs)
    np.testing.assert_allclose(state[-1], state.blobs)
    with pytest.raises(IndexError):
        state[4]

    state = State(coords, log_prob, random_state=rstate)
    np.testing.assert_allclose(state[0], state.coords)
    np.testing.assert_allclose(state[1], state.log_prob)
    check_rstate(state[2], state.random_state)
    check_rstate(state[-1], state.random_state)
    with pytest.raises(IndexError):
        state[3]
