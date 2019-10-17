# -*- coding: utf-8 -*-

import numpy as np

from emcee import EnsembleSampler
from emcee.state import State


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
    assert all(np.allclose(a, b) for a, b in zip(rstate[1:], r[1:]))

    state = State(coords, log_prob, None, rstate)
    c, l, r = state
    assert np.allclose(coords, c)
    assert np.allclose(log_prob, l)
    assert all(np.allclose(a, b) for a, b in zip(rstate[1:], r[1:]))


def test_overwrite(seed=1234):
    np.random.seed(seed)

    def ll(x):
        return -0.5 * np.sum(x ** 2)

    nwalkers = 64
    p0 = np.random.normal(size=(nwalkers, 1))
    init = np.copy(p0)

    sampler = EnsembleSampler(nwalkers, 1, ll)
    sampler.run_mcmc(p0, 10)
    assert np.allclose(init, p0)
