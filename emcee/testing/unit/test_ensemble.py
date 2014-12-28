# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["test_invalid_init", "test_inf_init", "test_same_init",
           "test_invalid_acceptance"]

import numpy as np

from ... import Ensemble
from ..common import UniformWalker


def test_invalid_init(nwalkers=32, ndim=5, seed=1234):
    np.random.seed(seed)
    coords = np.ones(nwalkers) + 0.001 * np.random.randn(nwalkers)
    try:
        Ensemble(UniformWalker, coords)
    except ValueError:
        pass
    else:
        assert False, "Ensemble should fail with invalid dimensions."

    coords = np.ones((nwalkers, ndim, 3))
    coords += 0.001 * np.random.randn(*(coords.shape))
    try:
        Ensemble(UniformWalker(), coords)
    except ValueError:
        pass
    else:
        assert False, "Ensemble should fail with invalid dimensions."


def test_inf_init(nwalkers=32, ndim=5, seed=1234):
    np.random.seed(seed)
    coords = 2*np.ones((nwalkers, ndim))+0.1*np.random.randn(nwalkers, ndim)

    try:
        Ensemble(UniformWalker(), coords)
    except ValueError:
        pass
    else:
        assert False, \
            "If the initial ensemble coordinates are -inf, we should get a " \
            "ValueError."


def test_same_init(nwalkers=32, ndim=5, seed=1234):
    coords = np.zeros((nwalkers, ndim))

    try:
        Ensemble(UniformWalker(), coords)
    except ValueError:
        pass
    else:
        assert False, \
            "If any of the initial walkers have identical coordinates, we " \
            "should get a ValueError."


def test_invalid_acceptance(nwalkers=32, ndim=5, seed=1234):
    np.random.seed(seed)
    coords = np.zeros((nwalkers, ndim)) + 0.1*np.random.randn(nwalkers, ndim)
    ensemble = Ensemble(UniformWalker(), coords)
    ensemble.walkers[0]._lnlike = -np.inf

    try:
        ensemble.update()
    except RuntimeError:
        pass
    else:
        assert False, \
            "Invalid proposals should *never* be accepted."
