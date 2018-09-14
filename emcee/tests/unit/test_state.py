# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
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
